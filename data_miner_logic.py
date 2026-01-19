import ccxt
import pandas as pd
import numpy as np
import time
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

class DataManager:
    """
    Binance Futures 데이터 수집을 담당하는 클래스
    """
    def __init__(self):
        # 바이낸스 선물 시장 연결 (인증 불필요)
        self.exchange = ccxt.binance({
            'options': {'defaultType': 'future'}
        })

    def fetch_data(self, ticker, start_str, end_str):
        """
        주어진 기간 동안의 5분봉 데이터를 페이지네이션을 통해 모두 가져옵니다.
        사용자 입력(KST)을 UTC로 변환하여 요청하고, 결과는 다시 KST로 복구합니다.
        
        Args:
            ticker (str): 심볼 (예: 'ETH/USDT')
            start_str (str): 시작 시간 (KST 기준 YYYY.MM.DD.HH)
            end_str (str): 종료 시간 (KST 기준 YYYY.MM.DD.HH)
        """
        try:
            # 1. 입력된 KST 시간을 datetime 객체로 변환
            kst_start = datetime.strptime(start_str, "%Y.%m.%d.%H")
            kst_end = datetime.strptime(end_str, "%Y.%m.%d.%H")
            
            # 2. API 요청을 위해 UTC로 변환 (KST - 9시간)
            utc_start = kst_start - timedelta(hours=9)
            utc_end = kst_end - timedelta(hours=9)
            
            # CCXT용 타임스탬프(ms) 생성
            since = self.exchange.parse8601(utc_start.isoformat())
            end_ts = self.exchange.parse8601(utc_end.isoformat())
            
        except Exception as e:
            raise ValueError(f"날짜 형식이 올바르지 않습니다. (YYYY.MM.DD.HH): {e}")

        all_ohlcv = []
        limit = 1000
        current_since = since
        
        while current_since < end_ts:
            try:
                ohlcv = self.exchange.fetch_ohlcv(ticker, timeframe='5m', since=current_since, limit=limit)
                if not ohlcv:
                    break
                
                all_ohlcv.extend(ohlcv)
                last_candle_time = ohlcv[-1][0]
                current_since = last_candle_time + (5 * 60 * 1000)
                
                if last_candle_time >= end_ts:
                    break
                    
                time.sleep(0.1)
            except Exception as e:
                print(f"데이터 수집 중 오류: {e}")
                time.sleep(1)
        
        # 3. 데이터프레임 변환 및 시간대 복구
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # UTC 타임스탬프를 datetime으로 변환 후 KST(+9)로 시프트
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms') + timedelta(hours=9)
        df.set_index('timestamp', inplace=True)
        
        # 중복 제거
        df = df[~df.index.duplicated(keep='first')]
        
        # KST 기준으로 최종 필터링
        df = df[(df.index >= kst_start) & (df.index <= kst_end)]
        
        return df

class Analyzer:
    """
    지표 계산, 이벤트 스캔, 시각화 및 라벨링을 담당하는 클래스
    """
    def __init__(self):
        # 시각화 저장 폴더 생성
        self.chart_dir = 'verification_charts'
        os.makedirs(self.chart_dir, exist_ok=True)

    def calculate_indicators(self, df):
        """
        모든 기술적 지표를 계산합니다. (HTF EMA 포함)
        """
        df = df.copy()
        
        # 1. HTF EMA (상위 타임프레임 EMA)
        # 중요: Look-ahead bias 방지를 위해 Shift(1) 적용
        timeframes = {'15min': '15m', '1h': '1h', '4h': '4h'}
        
        for tf, suffix in timeframes.items():
            # 리샘플링
            resampled = df['close'].resample(tf).last()
            
            # 224 EMA 계산
            ema_htf = resampled.ewm(span=224, adjust=False).mean()
            
            # CRITICAL: 미래 데이터 참조 방지를 위해 1칸 Shift (현재 5분봉 시점에서는 이전 상위봉의 마감값만 알 수 있음)
            ema_htf_shifted = ema_htf.shift(1)
            
            # 5분봉 타임라인으로 매핑 (Forward Fill)
            # reindex를 사용하여 5분봉 인덱스에 맞추고 빈 값은 앞의 값으로 채움
            df[f'ema_224_{suffix}'] = ema_htf_shifted.reindex(df.index).ffill()

        # 2. Basic EMA (5분봉 기준)
        for span in [5, 20, 60, 224]:
            df[f'ema_{span}'] = df['close'].ewm(span=span, adjust=False).mean()

        # 3. Momentum / Energy 지표
        
        # RSI (14)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD (12, 26, 9)
        k = df['close'].ewm(span=12, adjust=False).mean()
        d = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = k - d
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # ATR (14)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(window=14).mean()
        
        # OBV
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        
        # Volume EMA (20)
        df['vol_ema_20'] = df['volume'].ewm(span=20, adjust=False).mean()
        
        return df

    def scan_and_label(self, df, threshold_pct, detection_window_min, hold_time_min, ticker, tier, market_regime, history_len, sl_percent=2.0, rr_ratio=2.0):
        """
        이벤트를 스캔하고 라벨링(Type 1, 2, 3) 및 시각화를 수행합니다.
        
        [V4.0 업데이트] Risk:Reward (R:R) 로직 적용
        - TP (Target Price) = 진입가 ± (SL% * R:R)
        - SL 도달 시: Type 2 (Loss)
        - TP 도달 시: Type 1 (Win)
        - 시간 초과: Type 3 (Draw)
        """
        results = []
        
        # 윈도우 및 홀드 캔들 수 계산 (5분봉 기준)
        window_candles = int(detection_window_min / 5)
        hold_candles = int(hold_time_min / 5)
        
        # XGBoost용 추가 지표 사전 계산
        df = df.copy()
        df['vol_ratio'] = np.where(df['vol_ema_20'] == 0, 0, df['volume'] / df['vol_ema_20'])
        
        # EMA Z-Score 계산
        df['ema_z_score'] = (df['close'] - df['ema_20']) / df['atr'].replace(0, np.nan)
        df['ema_z_score'] = df['ema_z_score'].fillna(0)
        
        target_features = ['close', 'rsi', 'macd', 'obv', 'vol_ratio', 'ema_z_score']
        
        # history_len은 사용자 입력값 사용
        start_idx = max(history_len, window_candles) + 50 
        
        thresh = threshold_pct / 100.0
        sl_rate = sl_percent / 100.0
        
        print(f"스캔 시작: {len(df)} 캔들 분석 중... (SL: {sl_percent}%, R:R: {rr_ratio})")
        
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        timestamps = df.index
        
        i = start_idx
        last_idx = len(df) - hold_candles - 50 
        
        next_scan_idx = 0 
        
        while i < last_idx:
            # Cool-down 로직
            if i < next_scan_idx:
                i += 1
                continue
                
            current_close = closes[i]
            past_close = closes[i - window_candles]
            
            if past_close == 0: 
                i += 1
                continue
                
            # 변동률 계산
            pct_change = (current_close - past_close) / past_close
            
            # 이벤트 감지 (Impulse)
            if abs(pct_change) >= thresh:
                
                direction = 1 if pct_change > 0 else -1
                entry_price = current_close
                
                # R:R 기반 TP/SL 계산
                if direction == 1: # Long
                    sl_price = entry_price * (1 - sl_rate)
                    tp_price = entry_price * (1 + (sl_rate * rr_ratio))
                else: # Short
                    sl_price = entry_price * (1 + sl_rate)
                    tp_price = entry_price * (1 - (sl_rate * rr_ratio))
                
                # 결과 판정 (캔들 단위 시뮬레이션)
                event_type = "Type 3 (Draw/Timeout)" # 기본값
                
                # 미래 윈도우 순회
                for h in range(1, hold_candles + 1):
                    idx = i + h
                    # 고가/저가 확인 (꼬리 포함)
                    curr_high = highs[idx]
                    curr_low = lows[idx]
                    
                    if direction == 1: # Long
                        # SL 체크 (Low가 SL 터치)
                        if curr_low <= sl_price:
                            event_type = "Type 2 (Loss)"
                            break # 종료
                        # TP 체크 (High가 TP 터치)
                        if curr_high >= tp_price:
                            event_type = "Type 1 (Win)"
                            break # 종료
                    else: # Short
                        # SL 체크 (High가 SL 터치)
                        if curr_high >= sl_price:
                            event_type = "Type 2 (Loss)"
                            break
                        # TP 체크 (Low가 TP 터치)
                        if curr_low <= tp_price:
                            event_type = "Type 1 (Win)"
                            break
                
                # --- Feature Flattening ---
                row_data = {
                    'timestamp': timestamps[i],
                    'ticker': ticker,
                    'coin_tier': tier,
                    'market_regime': market_regime,
                    'event_type': event_type,
                    'direction': 'Long' if direction == 1 else 'Short',
                    'pct_change': pct_change,
                    'entry_price': entry_price,
                    'sl_price': sl_price, # [New]
                    'tp_price': tp_price, # [New]
                    'target_rr': rr_ratio # [New]
                }
                
                slice_start = i - history_len + 1
                slice_end = i + 1
                
                for feature in target_features:
                    feat_values = df[feature].iloc[slice_start:slice_end].values
                    if len(feat_values) != history_len: continue
                    for lag in range(history_len):
                        val = feat_values[(history_len - 1) - lag]
                        row_data[f'{feature}_lag_{lag}'] = val
                
                results.append(row_data)
                
                # --- Event Visualization ---
                try:
                    direction_str = 'Long' if direction == 1 else 'Short'
                    self._save_verification_chart(
                        df, i, hold_candles, ticker, event_type, history_len, 
                        direction_str, sl_price, sl_percent, tp_price, rr_ratio
                    )
                except Exception as e:
                    print(f"차트 저장 실패: {e}")

                # Cool-down 적용
                next_scan_idx = i + window_candles
                
            i += 1

        return pd.DataFrame(results)

    def _save_verification_chart(self, df, current_idx, hold_candles, ticker, event_type, history_len, direction_str, sl_price, sl_percent, tp_price, rr_ratio):
        """
        이벤트 발생 시점의 검증 차트 저장 (SL & TP 라인 포함)
        """
        # 시각화 범위
        plot_start = max(0, current_idx - 350)
        plot_end = min(len(df), current_idx + hold_candles + 50)
        
        subset = df.iloc[plot_start:plot_end]
        timestamps = subset.index
        closes = subset['close'].values
        
        rel_idx = current_idx - plot_start
        
        plt.figure(figsize=(12, 7))
        
        # 1. Close Price
        plt.plot(timestamps, closes, color='black', linewidth=1, label='Close Price')
        
        # --- EMAs ---
        if 'ema_224' in subset.columns:
            plt.plot(timestamps, subset['ema_224'], color='blue', linestyle='-', linewidth=1.5, label='5m 224 EMA')
        if 'ema_224_15m' in subset.columns:
            plt.plot(timestamps, subset['ema_224_15m'], color='green', linestyle='--', linewidth=1.5, label='15m 224 EMA')
        if 'ema_224_1h' in subset.columns:
            plt.plot(timestamps, subset['ema_224_1h'], color='hotpink', linestyle='-', linewidth=2.0, label='1h 224 EMA')
        if 'ema_224_4h' in subset.columns:
            plt.plot(timestamps, subset['ema_224_4h'], color='purple', linestyle='-', linewidth=2.5, label='4h 224 EMA')
        
        # 2. Zones
        feat_start_rel = max(0, rel_idx - history_len)
        plt.axvspan(timestamps[feat_start_rel], timestamps[rel_idx], color='yellow', alpha=0.2, label='Feature Zone')
        
        hold_end_rel = min(len(subset)-1, rel_idx + hold_candles)
        plt.axvspan(timestamps[rel_idx], timestamps[hold_end_rel], color='green', alpha=0.1, label='Hold Zone')
        
        # 3. Lines (SL & TP)
        # SL Line (Red Dashed)
        plt.hlines(y=sl_price, xmin=timestamps[rel_idx], xmax=timestamps[hold_end_rel], 
                   colors='red', linestyles='--', linewidth=1.5, label=f'SL (-{sl_percent}%)')
        
        # TP Line (Green Dashed) [New]
        plt.hlines(y=tp_price, xmin=timestamps[rel_idx], xmax=timestamps[hold_end_rel], 
                   colors='green', linestyles='--', linewidth=1.5, label=f'TP (R:R {rr_ratio})')

        # 4. Markers
        if direction_str == 'Long':
            plt.scatter([timestamps[rel_idx]], [closes[rel_idx]], color='green', marker='^', s=100, zorder=10, label='Long Entry')
        else:
            plt.scatter([timestamps[rel_idx]], [closes[rel_idx]], color='red', marker='v', s=100, zorder=10, label='Short Entry')
        
        # 5. Formatting
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%y.%m.%d.%H.%M'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45, ha='right')

        t_str = df.index[current_idx].strftime("%Y%m%d_%H%M")
        
        # 파일명 단순화 (Type_1_Win, Type_2_Loss)
        short_event = event_type.split('(')[0].strip().replace(" ", "_")
        if "Win" in event_type: short_event += "_Win"
        elif "Loss" in event_type: short_event += "_Loss"
        else: short_event += "_Draw"
            
        title = f"{ticker} | {t_str} | {event_type} | [{direction_str}]"
        
        plt.title(title)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = f"{ticker.replace('/', '')}_{t_str}_{short_event}.png"
        filepath = os.path.join(self.chart_dir, filename)
        
        plt.savefig(filepath)
        plt.close()
