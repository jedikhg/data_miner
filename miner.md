# Data Miner Project Specification

이 프로젝트는 바이낸스 선물 시장에서 특정 조건(임펄스)을 만족하는 가격 변동 데이터를 수집하고, 기술적 지표를 계산하여 이벤트별로 라벨링(Type 1, 2, 3) 및 시각화하는 도구입니다.

## 1. 시스템 구조

프로젝트는 크게 GUI 레이어와 로직 레이어로 구분됩니다.

- **`data_miner_gui.py`**: PySide6 기반의 사용자 인터페이스.
    - 파라미터 입력: Ticker, Start/End Date, Threshold, Window, Hold Time.
    - **[New] 데이터 분류**: Coin Tier, Market Regime 선택 가능.
    - **[New] 유연성**: Look-back Window(과거 패턴 길이) 설정 가능.
    - `QThread`를 사용하여 메인 스레드 차단 없이 백그라운드에서 분석 로직(`Worker`)을 실행합니다.
- **`data_miner_logic.py`**: 핵심 데이터 처리 모듈.
    - `DataManager`: 바이낸스 API를 통한 OHLCV 데이터 수집.
    - `Analyzer`: 지표 계산, 이벤트 스캔, 라벨링, 시각화 담당.
    - **[New] 동적 처리**: 사용자 설정 `history_len`에 따라 피처 추출 및 시각화 범위가 자동 조정됨.
- **`ai_studio.py`**: (신규 추가된 파일 - 상세 내용 분석 필요, 현재 프로젝트 구조상 독립적인 AI 관련 모듈로 추정됨)
- **`verification_charts/`**: 감지된 이벤트의 시각화 결과(PNG)가 저장되는 디렉토리.

## 2. 주요 기능 상세

### A. 데이터 수집 (`DataManager`)
- **대상**: 바이낸스 선물 (Binance Futures)
- **시간대**: 5분봉(5m) 기준
- **특징**: 페이지네이션을 통해 사용자가 설정한 기간(`YYYY.MM.DD.HH`) 전체 데이터를 안정적으로 수집합니다.

### B. 지표 계산 (`Analyzer.calculate_indicators`)
- **HTF EMA (상위 타임프레임)**: 15분, 1시간, 4시간 봉 기준의 224 EMA를 계산하여 현재 5분봉 데이터에 매핑합니다. (Look-ahead bias 방지를 위해 Shift(1) 적용)
- **기본 EMA**: 5분봉 기준 5, 20, 60, 224 EMA
- **모멘텀/에너지**: RSI(14), MACD(12, 26, 9), ATR(14), OBV, 거래량 EMA(20)
- **기타**: 거래량 비율(Vol Ratio), EMA Z-Score

### C. 이벤트 스캔 및 라벨링 (`Analyzer.scan_and_label`)
사용자가 설정한 임계값(`Threshold`) 이상의 급격한 변동이 발생했을 때 이를 감지하고 이후 흐름에 따라 분류합니다.

1.  **Type 1 (Success)**: 임펄스 발생 후 추세가 유지되거나 더 진행되는 경우.
2.  **Type 2 (Fake-out)**: 임펄스 발생 후 초기 상승/하락폭의 70% 이상을 되돌리는 경우.
3.  **Type 3 (Bottom Consolidation)**: (하락 임펄스 전용) 큰 하락 후 반등이 약하며(50% 이하), 추가 하락도 제한적인 바닥권 횡보 구간.

**[New] 메타데이터 주입**:
- 각 이벤트 결과에 `coin_tier` (코인 체급) 및 `market_regime` (장세) 정보가 포함됩니다.

### D. 시각화 (`Analyzer._save_verification_chart`)
- **X축 포맷**: 숫자 인덱스 대신 직관적인 **날짜/시간 포맷 (`%y.%m.%d.%H.%M`)** 사용.
- **추세선**: 5분, 15분, 1시간, 4시간 봉의 224 EMA를 함께 그려 멀티 타임프레임 추세 확인 가능.
- **영역 표시**:
    - `Feature Zone` (Yellow): 이벤트 발생 전 `history_len` 만큼의 구간.
    - `Hold Zone` (Green): 이벤트 발생 후 `Hold Time` 만큼의 구간.
- **백엔드**: Matplotlib의 `Agg` 백엔드 사용 (Non-interactive mode).

## 3. 기술 스택
- **Language**: Python 3.x
- **GUI**: PySide6 (Qt for Python)
- **Data Analysis**: Pandas, Numpy
- **API**: CCXT (Binance)
- **Visualization**: Matplotlib (Agg backend, DateFormatter)

## 4. 실행 방법
1. `requirements.txt`에 명시된 라이브러리 설치
2. `python data_miner_gui.py` 실행
3. 심볼(예: ETH/USDT), 기간, 임계값, **체급(Tier), 장세(Market), 과거 패턴 길이** 등 설정 후 'Start Scanning' 클릭