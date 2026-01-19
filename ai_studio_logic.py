import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg') # GUI 스레드 충돌 방지 (백그라운드 저장용)
import matplotlib.pyplot as plt
import os
import glob
import joblib
import re
import numpy as np
from datetime import timedelta
from data_miner_logic import DataManager, Analyzer

class AIStudioEngine:
    def __init__(self):
        self.model = None
        self.reports_dir = "reports"
        os.makedirs(self.reports_dir, exist_ok=True)

    def load_and_merge_data(self, folder_path):
        """
        지정된 폴더 내의 모든 CSV 파일을 읽어 하나의 DataFrame으로 병합합니다.
        """
        all_files = glob.glob(os.path.join(folder_path, "*.csv"))
        
        if not all_files:
            raise FileNotFoundError("선택한 폴더에 CSV 파일이 없습니다.")

        df_list = []
        for filename in all_files:
            try:
                df = pd.read_csv(filename)
                df_list.append(df)
            except Exception as e:
                print(f"파일 로드 오류 ({filename}): {e}")

        if not df_list:
            raise ValueError("데이터를 로드할 수 없습니다.")

        # 모든 데이터 병합
        merged_df = pd.concat(df_list, axis=0, ignore_index=True)
        return merged_df

    def preprocess_data(self, df):
        """
        데이터 전처리: 타겟 생성, 인코딩, 피처 선택
        """
        # 1. Target Labeling
        # event_type에 'Type 1' 또는 'Type 3'가 포함되면 1 (Buy), 아니면 0 (Pass)
        if 'event_type' not in df.columns:
             raise ValueError("'event_type' 컬럼이 데이터에 존재하지 않습니다.")
        
        df['target'] = df['event_type'].apply(lambda x: 1 if isinstance(x, str) and ('Type 1' in x or 'Type 3' in x) else 0)

        # 2. Meta-Data Encoding
        # coin_tier: 첫 번째 숫자 추출 (예: "1. TOP..." -> 1)
        if 'coin_tier' in df.columns:
            def extract_tier(val):
                if pd.isna(val): return 0
                match = re.search(r'\d+', str(val))
                return int(match.group()) if match else 0
            df['coin_tier_encoded'] = df['coin_tier'].apply(extract_tier)
        else:
            df['coin_tier_encoded'] = 0 # Default

        # market_regime: 매핑 (Bull=1, Bear=-1, Mixed/Sideways=0)
        regime_map = {
            'Bull': 1, 'Bear': -1, 
            'Mixed': 0, 'Sideways': 0, 
            'Bullish': 1, 'Bearish': -1 # 추가 변형 대응
        }
        if 'market_regime' in df.columns:
            df['market_regime_encoded'] = df['market_regime'].map(regime_map).fillna(0).astype(int)
        else:
            df['market_regime_encoded'] = 0

        # 3. Feature Selection
        # '_lag_'가 포함된 컬럼 + 인코딩된 메타 데이터
        feature_cols = [col for col in df.columns if '_lag_' in col]
        feature_cols.extend(['coin_tier_encoded', 'market_regime_encoded'])
        
        # 데이터 정제 (NaN 제거)
        df_clean = df[feature_cols + ['target']].dropna()
        
        return df_clean, feature_cols

    def train_model(self, df, feature_cols):
        """
        XGBoost 모델 학습 및 결과 리포트 생성
        """
        X = df[feature_cols]
        y = df['target']

        # 학습/테스트 데이터 분리
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

        # 클래스 불균형 처리 (scale_pos_weight)
        # neg_count / pos_count
        pos_count = y_train.sum()
        neg_count = len(y_train) - pos_count
        weight = neg_count / pos_count if pos_count > 0 else 1

        # 모델 생성 및 학습
        self.model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            scale_pos_weight=weight,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        self.model.fit(X_train, y_train)

        # 평가
        y_pred = self.model.predict(X_test)
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        result_text = "=== 학습 결과 리포트 ===\n"
        result_text += f"Train Set: {len(X_train)}, Test Set: {len(X_test)}\n"
        result_text += f"Scale Pos Weight: {weight:.2f}\n\n"
        result_text += "Classification Report:\n"
        result_text += report
        result_text += "\nConfusion Matrix:\n"
        result_text += str(conf_matrix)

        # Feature Importance Plot 저장
        self._save_feature_importance(feature_cols)

        return self.model, result_text

    def _save_feature_importance(self, feature_cols):
        """
        Feature Importance 차트를 reports 폴더에 저장
        """
        if self.model is None:
            return

        plt.figure(figsize=(10, 8))
        xgb.plot_importance(self.model, max_num_features=20)
        plt.title("XGBoost Feature Importance")
        plt.tight_layout()
        
        save_path = os.path.join(self.reports_dir, "feature_importance.png")
        plt.savefig(save_path)
        plt.close() # 메모리 해제

    def save_model(self, filepath):
        """
        모델 저장 (JSON 형식 권장)
        """
        if self.model:
            self.model.save_model(filepath)
            return True
        return False

    def load_model(self, filepath):
        """
        모델 불러오기
        """
        self.model = xgb.XGBClassifier()
        self.model.load_model(filepath)

    def run_backtest(self, test_data_path):
        """
        [Legacy] 단순 파일 기반 백테스팅
        """
        if self.model is None:
            raise ValueError("모델이 로드되지 않았습니다.")

        # 데이터 로드 (단일 파일)
        df = pd.read_csv(test_data_path)
        
        # 전처리
        df_clean, feature_cols = self.preprocess_data(df)
        
        # 예측
        X_test = df_clean[feature_cols]
        y_true = df_clean['target']
        
        y_pred = self.model.predict(X_test)
        
        report = classification_report(y_true, y_pred)
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        result_text = f"=== 백테스팅 결과 ({os.path.basename(test_data_path)}) ===\n"
        result_text += report
        result_text += "\nConfusion Matrix:\n"
        result_text += str(conf_matrix)
        
        return result_text

    def run_simulation(self, ticker, start_date, end_date, hold_candles, tier_str, regime_str):
        """
        [V3.0] 실전 시뮬레이션 백테스팅
        """
        if self.model is None:
            raise ValueError("모델이 먼저 로드되어야 합니다.")

        # 1. 모델에서 필요한 History Length 추론
        feature_names = self.model.feature_names_in_
        lag_pattern = re.compile(r'_lag_(\d+)')
        lags = [int(lag_pattern.search(f).group(1)) for f in feature_names if lag_pattern.search(f)]
        history_len = max(lags) + 1 if lags else 12 # 기본값 12
        print(f"모델 분석: 감지된 History Length = {history_len}")

        # 2. 데이터 수집
        dm = DataManager()
        # 데이터가 부족하면 지표 계산 시 NaN이 발생하므로 앞뒤로 넉넉히 가져오는 것이 좋으나,
        # 여기서는 사용자가 요청한 기간을 최대한 존중하며 fetch.
        # 단, EMA 224 계산을 위해 앞부분 데이터가 충분해야 정확함. 
        # API 구조상 요청한 start_date부터 가져오므로, 초기 300개 정도는 지표 워밍업으로 간주하고 스킵하는 것이 안전.
        
        df = dm.fetch_data(ticker, start_date, end_date)
        if df.empty:
            return "데이터를 가져올 수 없습니다. 기간이나 티커를 확인해주세요."

        # 3. 지표 계산
        analyzer = Analyzer()
        df = analyzer.calculate_indicators(df)

        # 4. Feature Engineering (Vectorized)
        # 모델이 학습된 피처 컬럼을 그대로 생성해야 함.
        # target_features는 data_miner_logic을 참고 (혹은 모델 피처명에서 역추적)
        base_features = ['close', 'rsi', 'macd', 'obv', 'vol_ratio', 'ema_z_score']
        
        # vol_ratio, ema_z_score 등은 calculate_indicators에서 생성되지 않으므로 여기서 추가 계산 필요
        # Analyzer.scan_and_label 로직 일부 차용
        df['vol_ratio'] = np.where(df['vol_ema_20'] == 0, 0, df['volume'] / df['vol_ema_20'])
        df['ema_z_score'] = (df['close'] - df['ema_20']) / df['atr'].replace(0, np.nan)
        df['ema_z_score'] = df['ema_z_score'].fillna(0)

        # Lag Feature 생성
        for feat in base_features:
            for i in range(history_len):
                df[f'{feat}_lag_{i}'] = df[feat].shift(i)

        # Meta Data Encoding
        # Tier Parsing
        match = re.search(r'\d+', str(tier_str))
        tier_val = int(match.group()) if match else 0
        df['coin_tier_encoded'] = tier_val

        # Regime Mapping
        regime_map = {'Bull': 1, 'Bear': -1, 'Mixed': 0, 'Sideways': 0, 'Bullish': 1, 'Bearish': -1}
        regime_val = regime_map.get(regime_str, 0)
        df['market_regime_encoded'] = regime_val

        # NaN 제거 (초기 데이터)
        # 실제 시뮬레이션을 위해 원본 인덱스 보존 필요
        valid_df = df.dropna().copy()
        
        if valid_df.empty:
            return "지표 계산 후 유효한 데이터가 없습니다. 기간을 더 길게 설정하세요."

        # 모델 입력 준비 (컬럼 순서 맞추기)
        try:
            X = valid_df[feature_names]
        except KeyError as e:
            return f"피처 매칭 실패: {e}\n모델과 데이터의 피처가 일치하지 않습니다."

        # 5. AI Inference
        # 확률 예측
        probs = self.model.predict_proba(X)[:, 1] # Class 1 (Success) Probability
        valid_df['ai_prob'] = probs

        # 6. Simulation Loop
        trades = []
        cooldown_idx = 0
        total_pnl = 0
        
        # 결과 저장 폴더
        backtest_dir = "backtest_results"
        os.makedirs(backtest_dir, exist_ok=True)

        log_text = "=== 실전 시뮬레이션 로그 ===\n"
        
        # 인덱스 접근을 위해 reset_index (Timestamp는 컬럼으로)
        sim_df = valid_df.reset_index()
        
        for i in range(len(sim_df)):
            if i < cooldown_idx:
                continue
            
            row = sim_df.iloc[i]
            prob = row['ai_prob']
            
            # 진입 조건: AI 확신 50% 이상
            if prob > 0.5:
                entry_time = row['timestamp']
                entry_price = row['close']
                
                # Exit 시점 계산
                exit_idx = i + hold_candles
                
                if exit_idx >= len(sim_df):
                    break # 데이터 끝
                
                exit_row = sim_df.iloc[exit_idx]
                exit_time = exit_row['timestamp']
                exit_price = exit_row['close']
                
                # PnL 계산 (Long Only 가정)
                pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                total_pnl += pnl_pct
                
                result_str = "WIN" if pnl_pct > 0 else "LOSS"
                
                trade_info = {
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'exit_time': exit_time,
                    'exit_price': exit_price,
                    'pnl': pnl_pct,
                    'result': result_str,
                    'prob': prob
                }
                trades.append(trade_info)
                
                log_text += f"[{entry_time}] BUY @ {entry_price:.4f} -> [{exit_time}] SELL @ {exit_price:.4f} | PnL: {pnl_pct:.2f}% ({result_str}) | AI: {prob:.2f}\n"
                
                # 차트 저장
                self._save_trade_chart(sim_df, i, exit_idx, ticker, pnl_pct, result_str, backtest_dir)
                
                # 쿨다운 (포지션 보유 중 진입 금지)
                cooldown_idx = exit_idx + 1

        # 요약 리포트
        win_count = len([t for t in trades if t['pnl'] > 0])
        total_trades = len(trades)
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
        
        summary = "\n=== 시뮬레이션 요약 ===\n"
        summary += f"총 거래 횟수: {total_trades}회\n"
        summary += f"승률: {win_rate:.2f}%\n"
        summary += f"누적 수익률: {total_pnl:.2f}%\n"
        
        return summary + "\n" + log_text

    def _save_trade_chart(self, sim_df, entry_idx, exit_idx, ticker, pnl, result, save_dir):
        """
        매매 시점 차트 생성 및 저장
        """
        # 앞뒤 여유분
        start_view = max(0, entry_idx - 50)
        end_view = min(len(sim_df), exit_idx + 30)
        
        subset = sim_df.iloc[start_view:end_view]
        
        plt.figure(figsize=(10, 6))
        plt.plot(subset['timestamp'], subset['close'], color='black', label='Price')
        
        # 진입/청산 마커
        entry_data = sim_df.iloc[entry_idx]
        exit_data = sim_df.iloc[exit_idx]
        
        plt.scatter(entry_data['timestamp'], entry_data['close'], color='green', marker='^', s=120, label='Entry', zorder=5)
        plt.scatter(exit_data['timestamp'], exit_data['close'], color='red', marker='x', s=120, label='Exit', zorder=5)
        
        # 구간 표시
        plt.axvspan(entry_data['timestamp'], exit_data['timestamp'], color='blue', alpha=0.1)
        
        plt.title(f"{ticker} | PnL: {pnl:.2f}% | {result}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        filename = f"Trade_{entry_data['timestamp'].strftime('%Y%m%d_%H%M')}_{result}.png"
        plt.savefig(os.path.join(save_dir, filename))
        plt.close()

