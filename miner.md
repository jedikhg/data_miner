# Data Miner & AI Studio Project Documentation

## 1. Data Miner V4.0 (Latest)
데이터 마이닝 및 학습 데이터 생성 엔진

### 주요 기능
- **KST (한국 표준시) 지원**: 사용자 입력을 KST로 처리하고 내부적으로 UTC 변환 후 바이낸스 API 호출. 결과물은 다시 KST로 복구.
- **다중 타임프레임 EMA (HTF)**: 5분봉 데이터에 15m, 1h, 4h 기준의 224 EMA를 계산하여 피처로 활용. (Look-ahead bias 방지 로직 포함)
- **R:R (Risk:Reward) 로직 (V4.0)**:
  - 손절(SL %)과 목표 손익비(R:R)를 설정하여 학습 데이터 라벨링.
  - 홀딩 기간 내 캔들 단위 시뮬레이션을 통해 승/패/무승부 판정.
  - **Type 1 (Win)**: 손절선 터치 전 목표가(TP) 도달.
  - **Type 2 (Loss)**: 목표가 도달 전 손절선(SL) 터치.
  - **Type 3 (Draw)**: 기간 내 둘 다 도달하지 못함 (시간 초과).
- **데이터 카테고리화**: `coin_tier` (체급) 및 `market_regime` (장세) 정보를 메타 데이터로 인코딩하여 AI 학습용 피처로 포함.
- **시각화 강화**: 차트에 진입점, 홀딩 구역, 피처 추출 구역, **SL 라인(Red)**, **TP 라인(Green)** 표시.

---

## 2. AI Studio V3.0 (Latest)
XGBoost 기반 모델 학습 및 실전 시뮬레이션 도구

### 주요 기능
- **XGBoost 모델 학습**:
  - `scale_pos_weight`를 통한 클래스 불균형 자동 처리.
  - 학습 완료 시 **Feature Importance** 차트 자동 생성 (`reports/`).
- **자동 피처 탐지**: 로드된 모델(.json)에서 사용된 `history_len` (lag 개수)을 자동으로 분석하여 매칭.
- **실전 시뮬레이션 엔진 (V3.0)**:
  - 과거 데이터 폴더가 아닌, **실제 티커와 날짜**를 입력받아 실시간 지표 계산 및 AI 예측 수행.
  - **AI 확률 > 0.5** 시 진입 시뮬레이션.
  - 누적 수익률, 승률 계산 및 개별 거래 차트 저장 (`backtest_results/`).
- **GUI (PySide6)**:
  - Worker Thread 패턴으로 학습 및 시뮬레이션 중 GUI 프리징 방지.
  - 실시간 로그 윈도우 지원.

---

## 3. 기술적 세부 사항 (Tech Stack)
- **Language**: Python 3.10+
- **GUI Framework**: PySide6
- **ML Framework**: XGBoost, Scikit-learn
- **Data**: Pandas, CCXT (Binance Futures)
- **Visualization**: Matplotlib (Agg Backend for Thread-safety)

---

## 4. 파일 구조
- `data_miner_gui.py`: 마이너 메인 UI
- `data_miner_logic.py`: 데이터 수집, 지표 계산, 라벨링 엔진
- `ai_studio_gui.py`: AI 학습 및 시뮬레이터 UI
- `ai_studio_logic.py`: XGBoost 학습 및 시뮬레이션 로직
- `verification_charts/`: 마이너 생성 검증 차트 폴더
- `backtest_results/`: 시뮬레이터 결과 차트 폴더
- `reports/`: 모델 학습 리포트 및 피처 중요도 폴더
