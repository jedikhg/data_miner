import sys
import os
from datetime import datetime
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QProgressBar,
    QMessageBox,
    QFormLayout,
    QComboBox,
)
from PySide6.QtCore import QThread, Signal, Slot

# 로직 모듈 임포트
from data_miner_logic import DataManager, Analyzer

class Worker(QThread):
    """
    백그라운드에서 데이터 수집 및 분석을 수행하는 스레드
    """
    log_signal = Signal(str)
    progress_signal = Signal(int)
    finished_signal = Signal(str)
    error_signal = Signal(str)

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        try:
            ticker = self.params['ticker']
            start_date = self.params['start_date']
            end_date = self.params['end_date']
            threshold = float(self.params['threshold'])
            window = int(self.params['window'])
            hold_time = int(self.params['hold_time'])
            
            # 신규 파라미터 추출
            tier = self.params['tier']
            market = self.params['market']
            history_len = int(self.params['history_len'])

            self.log_signal.emit(f"=== 작업 시작: {ticker} ===")
            self.progress_signal.emit(10)

            # 1. 데이터 수집
            self.log_signal.emit(f"데이터 수집 중... ({start_date} ~ {end_date})")
            dm = DataManager()
            df = dm.fetch_data(ticker, start_date, end_date)
            
            if df.empty:
                raise Exception("데이터를 찾을 수 없습니다.")
                
            self.log_signal.emit(f"데이터 수집 완료: {len(df)} 캔들")
            self.progress_signal.emit(40)

            # 2. 지표 계산
            self.log_signal.emit("기술적 지표 계산 중 (HTF 포함)...")
            analyzer = Analyzer()
            df_features = analyzer.calculate_indicators(df)
            self.progress_signal.emit(70)

            # 3. 이벤트 스캔 및 라벨링
            self.log_signal.emit("이벤트 스캔 및 라벨링 분석 중...")
            # 파라미터 전달 업데이트
            results_df = analyzer.scan_and_label(
                df_features, threshold, window, hold_time, ticker,
                tier, market, history_len
            )
            
            self.progress_signal.emit(90)
            
            # 4. 결과 저장
            if not results_df.empty:
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{ticker.replace('/', '')}_Impulse_Scan_{timestamp_str}.csv"
                results_df.to_csv(filename, index=False)
                msg = f"완료! 결과 저장됨: {filename} (총 {len(results_df)} 건 감지)"
            else:
                msg = "완료! 감지된 이벤트가 없습니다."

            self.log_signal.emit(msg)
            self.progress_signal.emit(100)
            self.finished_signal.emit(msg)

        except Exception as e:
            import traceback
            error_msg = f"오류 발생: {str(e)}\n{traceback.format_exc()}"
            self.error_signal.emit(error_msg)

class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Binance Impulse Scanner")
        self.resize(600, 600)
        
        # 메인 위젯 설정
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # 입력 폼
        form_layout = QFormLayout()
        
        self.ticker_input = QLineEdit("ETH/USDT")
        self.start_input = QLineEdit("2024.01.01.00") # 예시 날짜
        self.end_input = QLineEdit("2024.01.05.00")
        self.threshold_input = QLineEdit("3.0")
        self.window_input = QLineEdit("60")
        self.hold_input = QLineEdit("120")
        
        # 신규 입력 필드 추가
        self.tier_input = QComboBox()
        self.tier_input.addItems(['1. TOP (BTC/ETH)', '2. Major', '3. General', '4. Meme/Shit', '5. New Listing'])
        
        self.market_input = QComboBox()
        self.market_input.addItems(['Mixed (복합)', 'Bull (불장)', 'Bear (하락장)', 'Sideways (횡보장)'])
        
        self.history_input = QLineEdit("288")
        
        form_layout.addRow("Ticker (심볼):", self.ticker_input)
        form_layout.addRow("Start Date (YYYY.MM.DD.HH):", self.start_input)
        form_layout.addRow("End Date (YYYY.MM.DD.HH):", self.end_input)
        form_layout.addRow("Impulse Threshold (%):", self.threshold_input)
        form_layout.addRow("Detection Window (분):", self.window_input)
        form_layout.addRow("Hold Time (분):", self.hold_input)
        
        # 신규 필드 레이아웃 추가
        form_layout.addRow("코인 체급 (Tier):", self.tier_input)
        form_layout.addRow("장세 (Market):", self.market_input)
        form_layout.addRow("과거 패턴 길이 (캔들 수):", self.history_input)
        
        layout.addLayout(form_layout)

        # 실행 버튼
        self.start_btn = QPushButton("Start Scanning")
        self.start_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        self.start_btn.clicked.connect(self.start_scan)
        layout.addWidget(self.start_btn)

        # 프로그레스 바
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        # 로그 창
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")

    def start_scan(self):
        params = {
            'ticker': self.ticker_input.text(),
            'start_date': self.start_input.text(),
            'end_date': self.end_input.text(),
            'threshold': self.threshold_input.text(),
            'window': self.window_input.text(),
            'hold_time': self.hold_input.text(),
            # 신규 파라미터 수집
            'tier': self.tier_input.currentText(),
            'market': self.market_input.currentText(),
            'history_len': self.history_input.text()
        }
        
        # 버튼 비활성화
        self.start_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log("스캔 작업을 시작합니다...")

        # 워커 스레드 시작
        self.worker = Worker(params)
        self.worker.log_signal.connect(self.log)
        self.worker.progress_signal.connect(self.progress_bar.setValue)
        self.worker.finished_signal.connect(self.scan_finished)
        self.worker.error_signal.connect(self.scan_error)
        self.worker.start()

    @Slot(str)
    def scan_finished(self, msg):
        self.start_btn.setEnabled(True)
        QMessageBox.information(self, "완료", msg)

    @Slot(str)
    def scan_error(self, msg):
        self.start_btn.setEnabled(True)
        self.log(msg)
        QMessageBox.critical(self, "오류", "작업 중 오류가 발생했습니다. 로그를 확인하세요.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec())
