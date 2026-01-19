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
        
        Args:
            ticker (str): 심볼 (예: 'ETH/USDT')
            start_str (str): 시작 시간 (YYYY.MM.DD.HH)
            end_str (str): 종료 시간 (YYYY.MM.DD.HH)
        
        Returns:
            pd.DataFrame: OHLCV 데이터
        """
        # 시간 포맷 파싱
        try:
            since = self.exchange.parse8601(datetime.strptime(start_str, "%Y.%m.%d.%H").isoformat())
            end_ts = self.exchange.parse8601(datetime.strptime(end_str, "%Y.%m.%d.%H").isoformat())
        except Exception as e:
            raise ValueError(f"날짜 형식이 올바르지 않습니다. (YYYY.MM.DD.HH): {e}")

        all_ohlcv = []
        limit = 1000  # 바이낸스 최대 요청 개수
        
        current_since = since
        
        while current_since < end_ts:
            try:
                # 데이터 가져오기 (5분봉)
                ohlcv = self.exchange.fetch_ohlcv(ticker, timeframe='5m', since=current_since, limit=limit)
                
                if not ohlcv:
                    break
                
                all_ohlcv.extend(ohlcv)
                
                # 다음 요청을 위해 시간 업데이트 (마지막 캔들 시간 + 5분)
                last_candle_time = ohlcv[-1][0]
                current_since = last_candle_time + (5 * 60 * 1000)
                
                # 종료 시간 도달 확인
                if last_candle_time >= end_ts:
                    break
                    
                # API 레이트 리밋 조절
                time.sleep(0.1)
                
            except Exception as e:
                print(f"데이터 수집 중 오류: {e}")
                time.sleep(1) # 오류 시 잠시 대기 후 재시도
        
        # 데이터프레임 변환
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # 중복 제거 및 시간 범위 자르기
        df = df[~df.index.duplicated(keep='first')]
        df = df[(df.index >= pd.to_datetime(start_str, format="%Y.%m.%d.%H")) & 
                (df.index <= pd.to_datetime(end_str, format="%Y.%m.%d.%H"))]
        
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

    def scan_and_label(self, df, threshold_pct, detection_window_min, hold_time_min, ticker, tier, market_regime, history_len):
        """
        이벤트를 스캔하고 라벨링(Type 1, 2, 3) 및 시각화를 수행합니다.
        
        업데이트 사항:
        1. Type 3 (Bottom Consolidation) 로직 추가
        2. 이벤트 발생 시 검증 차트(PNG) 저장 기능 추가
        3. [New] 데이터 카테고리화 (Tier, Market) 및 가변적 히스토리 길이 적용
        
        Args:
            df (pd.DataFrame): 지표가 계산된 데이터프레임
            threshold_pct (float): 감지 임계값 (%)
            detection_window_min (int): 감지 윈도우 (분)
            hold_time_min (int): 보유 시간 (분)
            tier (str): 코인 체급
            market_regime (str): 장세 정보
            history_len (int): 과거 데이터 조회 길이 (Look-back Window)
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
        
        # history_len은 사용자 입력값 사용 (기존 하드코딩 288 제거)
        start_idx = max(history_len, window_candles) + 50 # 시각화 여유분 확보를 위해 약간 더 뒤에서 시작 가능
        
        thresh = threshold_pct / 100.0
        
        print(f"스캔 시작: {len(df)} 캔들 분석 중...")
        
        closes = df['close'].values
        timestamps = df.index
        
        i = start_idx
        last_idx = len(df) - hold_candles - 50 # 시각화 여유분 고려
        
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
            
            # 이벤트 감지
            if abs(pct_change) >= thresh:
                
                event_type = "Unknown"
                direction = 1 if pct_change > 0 else -1
                entry_price = current_close
                
                # 미래 데이터 윈도우 추출 (Hold Time)
                future_window = closes[i+1 : i+1+hold_candles]
                
                if len(future_window) < hold_candles:
                    i += 1
                    continue
                
                if direction == 1: # 상승 임펄스 (Long)
                    move_size = entry_price - past_close
                    retrace_threshold = entry_price - (move_size * 0.7) # 70% 되돌림
                    
                    min_price = np.min(future_window)
                    
                    if min_price < retrace_threshold:
                        event_type = "Type 2 (Fake-out)"
                    else:
                        event_type = "Type 1 (Success)"
                        
                else: # 하락 임펄스 (Short)
                    move_size = past_close - entry_price # 하락폭 (양수)
                    
                    # 기준값 계산
                    max_price = np.max(future_window)
                    min_price = np.min(future_window)
                    
                    retrace_threshold_fakeout = entry_price + (move_size * 0.7) # 70% 되돌림 (Type 2 기준)
                    
                    # Type 3 (Bottom Consolidation) 조건
                    # 1. 반등(V-bounce)이 약함: 되돌림이 50% 이하
                    cond_no_bounce = max_price <= entry_price + (move_size * 0.5)
                    # 2. 추가 하락이 제한적: 초기 임펄스의 20% 이상 더 떨어지지 않음
                    cond_no_crash = min_price >= entry_price - (move_size * 0.2)
                    
                    if max_price > retrace_threshold_fakeout:
                        event_type = "Type 2 (Fake-out)"
                    elif cond_no_bounce and cond_no_crash:
                        event_type = "Type 3 (Bottom Consolidation)"
                    else:
                        # 위 조건에 해당하지 않으면(예: 그대로 더 폭락하거나, 적당한 반등 후 횡보 등)
                        # 여기서는 단순 추세 지속(Success)으로 분류
                        event_type = "Type 1 (Success)"
                
                # --- Feature Flattening ---
                row_data = {
                    'timestamp': timestamps[i],
                    'ticker': ticker,
                    'coin_tier': tier,         # [New]
                    'market_regime': market_regime, # [New]
                    'event_type': event_type,
                    'direction': 'Long' if direction == 1 else 'Short',
                    'pct_change': pct_change,
                    'entry_price': entry_price
                }
                
                slice_start = i - history_len + 1
                slice_end = i + 1
                
                for feature in target_features:
                    feat_values = df[feature].iloc[slice_start:slice_end].values
                    if len(feat_values) != history_len: continue
                    # 동적 히스토리 길이 적용
                    for lag in range(history_len):
                        val = feat_values[(history_len - 1) - lag]
                        row_data[f'{feature}_lag_{lag}'] = val
                
                results.append(row_data)
                
                # --- Event Visualization (차트 저장) ---
                try:
                    direction_str = 'Long' if direction == 1 else 'Short'
                    self._save_verification_chart(df, i, hold_candles, ticker, event_type, history_len, direction_str)
                except Exception as e:
                    print(f"차트 저장 실패: {e}")

                # Cool-down 적용
                next_scan_idx = i + window_candles
                
            i += 1

        return pd.DataFrame(results)

    def _save_verification_chart(self, df, current_idx, hold_candles, ticker, event_type, history_len, direction_str):
        """
        이벤트 발생 시점의 검증 차트를 생성하고 저장합니다.
        
        Args:
            current_idx (int): 이벤트 발생 캔들 인덱스 (T)
            history_len (int): 피처 추출 구간 길이
            direction_str (str): 매매 방향 ('Long' 또는 'Short')
        """
        # 시각화 범위 설정 (과거 350개 ~ 미래 50개 + 홀드기간)
        plot_start = max(0, current_idx - 350)
        plot_end = min(len(df), current_idx + hold_candles + 50)
        
        subset = df.iloc[plot_start:plot_end]
        timestamps = subset.index
        closes = subset['close'].values
        
        # 현재 시점(T)의 상대적 인덱스 찾기
        # 전체 df에서의 인덱스가 current_idx이므로, subset 내에서는:
        rel_idx = current_idx - plot_start
        
        plt.figure(figsize=(12, 7)) # 가로 폭을 약간 넓힘
        
        # 1. Close Price Plot (X축을 Timestamp로 변경)
        plt.plot(timestamps, closes, color='black', linewidth=1, label='Close Price')
        
        # --- Multi-Timeframe EMA Lines 추가 ---
        # 1. 5m 224 EMA (Blue, Solid, 1.5)
        if 'ema_224' in subset.columns:
            plt.plot(timestamps, subset['ema_224'], color='blue', linestyle='-', linewidth=1.5, label='5m 224 EMA')
            
        # 2. 15m 224 EMA (Green, Dashed, 1.5)
        if 'ema_224_15m' in subset.columns:
            plt.plot(timestamps, subset['ema_224_15m'], color='green', linestyle='--', linewidth=1.5, label='15m 224 EMA')
            
        # 3. 1h 224 EMA (Hotpink, Solid, 2.0)
        if 'ema_224_1h' in subset.columns:
            plt.plot(timestamps, subset['ema_224_1h'], color='hotpink', linestyle='-', linewidth=2.0, label='1h 224 EMA')
            
        # 4. 4h 224 EMA (Purple, Solid, 2.5)
        if 'ema_224_4h' in subset.columns:
            plt.plot(timestamps, subset['ema_224_4h'], color='purple', linestyle='-', linewidth=2.5, label='4h 224 EMA')
        
        # 2. Feature Extraction Zone (Yellow, Past N)
        # T - history_len ~ T
        feat_start_rel = max(0, rel_idx - history_len)
        plt.axvspan(timestamps[feat_start_rel], timestamps[rel_idx], color='yellow', alpha=0.2, label='Feature Zone')
        
        # 3. Hold Time Zone (Green, Future)
        # T ~ T + Hold
        hold_end_rel = min(len(subset)-1, rel_idx + hold_candles)
        plt.axvspan(timestamps[rel_idx], timestamps[hold_end_rel], color='green', alpha=0.1, label='Hold Zone')
        
        # 4. Event Point (Directional Marker)
        if direction_str == 'Long':
            plt.scatter([timestamps[rel_idx]], [closes[rel_idx]], color='green', marker='^', s=100, zorder=10, label='Long Entry')
        else:
            plt.scatter([timestamps[rel_idx]], [closes[rel_idx]], color='red', marker='v', s=100, zorder=10, label='Short Entry')
        
        # 5. X-axis Formatting (Date/Time)
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%y.%m.%d.%H.%M'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45, ha='right')

        # 타이틀 및 파일명 정리
        t_str = df.index[current_idx].strftime("%Y%m%d_%H%M")
        clean_event = event_type.replace(" ", "_").replace("(", "").replace(")", "")
        title = f"{ticker} | {t_str} | {event_type} | [{direction_str}]"
        
        plt.title(title)
        plt.legend(loc='best') # 범례 위치 자동 최적화
        plt.grid(True, alpha=0.3)
        plt.tight_layout() # 레이아웃 자동 조정 (라벨 잘림 방지)
        
        # 파일 저장
        filename = f"{ticker.replace('/', '')}_{t_str}_{clean_event}.png"
        filepath = os.path.join(self.chart_dir, filename)
        
        plt.savefig(filepath)
        plt.close() # 메모리 누수 방지
