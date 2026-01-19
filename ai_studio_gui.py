import sys
import os
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                               QTextEdit, QTabWidget, QGroupBox, QLineEdit, 
                               QComboBox, QSpinBox, QFormLayout)
from PySide6.QtCore import Qt, QThread, Signal
from ai_studio_logic import AIStudioEngine

class TrainingWorker(QThread):
    """
    백그라운드에서 학습을 수행하는 워커 스레드
    """
    log_signal = Signal(str)
    finished_signal = Signal(str) # 결과 텍스트 전달
    error_signal = Signal(str)

    def __init__(self, engine, data_folder):
        super().__init__()
        self.engine = engine
        self.data_folder = data_folder

    def run(self):
        try:
            self.log_signal.emit(f"데이터 로드 시작: {self.data_folder}")
            merged_df = self.engine.load_and_merge_data(self.data_folder)
            self.log_signal.emit(f"데이터 로드 완료. 총 행 수: {len(merged_df)}")

            self.log_signal.emit("데이터 전처리 중...")
            df_clean, features = self.engine.preprocess_data(merged_df)
            self.log_signal.emit(f"전처리 완료. 학습 데이터 수: {len(df_clean)}, 피처 수: {len(features)}")

            self.log_signal.emit("XGBoost 모델 학습 시작...")
            _, report = self.engine.train_model(df_clean, features)
            
            self.log_signal.emit("학습 완료.")
            self.finished_signal.emit(report)

        except Exception as e:
            self.error_signal.emit(str(e))

class SimulationWorker(QThread):
    """
    백그라운드에서 시뮬레이션을 수행하는 워커 스레드
    """
    log_signal = Signal(str)
    finished_signal = Signal(str)
    error_signal = Signal(str)

    def __init__(self, engine, ticker, start_date, end_date, hold_candles, tier, regime):
        super().__init__()
        self.engine = engine
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.hold_candles = hold_candles
        self.tier = tier
        self.regime = regime

    def run(self):
        try:
            self.log_signal.emit(f"시뮬레이션 시작: {self.ticker} ({self.start_date} ~ {self.end_date})")
            result_log = self.engine.run_simulation(
                self.ticker, self.start_date, self.end_date, 
                self.hold_candles, self.tier, self.regime
            )
            self.finished_signal.emit(result_log)
        except Exception as e:
            self.error_signal.emit(str(e))

class AIStudioGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Studio V3.0 - Real-world Simulation")
        self.resize(1000, 800)
        
        self.engine = AIStudioEngine()
        self.worker = None
        self.sim_worker = None

        self.setup_ui()

    def setup_ui(self):
        # Main Widget & Layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # Title
        title_label = QLabel("AI Studio V3.0 - AI Trader Simulation")
        title_label.setStyleSheet("font-size: 20px; font-weight: bold; margin-bottom: 10px; color: #333;")
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)

        # Tabs
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Tab 1: Training
        self.tab_training = QWidget()
        self.setup_training_tab()
        self.tabs.addTab(self.tab_training, "모델 학습 (Training)")

        # Tab 2: Simulation (Backtesting)
        self.tab_backtest = QWidget()
        self.setup_backtest_tab()
        self.tabs.addTab(self.tab_backtest, "실전 시뮬레이션 (Simulation)")

    def setup_training_tab(self):
        layout = QVBoxLayout(self.tab_training)

        # 1. Data Selection
        gb_data = QGroupBox("학습 데이터 설정")
        gb_layout = QHBoxLayout()
        
        self.folder_path_edit = QLineEdit()
        self.folder_path_edit.setPlaceholderText("학습용 CSV 파일들이 있는 폴더를 선택하세요.")
        self.btn_select_folder = QPushButton("폴더 선택")
        self.btn_select_folder.clicked.connect(self.select_data_folder)

        gb_layout.addWidget(self.folder_path_edit)
        gb_layout.addWidget(self.btn_select_folder)
        gb_data.setLayout(gb_layout)
        layout.addWidget(gb_data)

        # 2. Action
        self.btn_start_train = QPushButton("학습 시작 (Start Training)")
        self.btn_start_train.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        self.btn_start_train.clicked.connect(self.start_training)
        layout.addWidget(self.btn_start_train)

        # 3. Save Model
        gb_save = QGroupBox("모델 저장")
        gb_save_layout = QHBoxLayout()
        self.btn_save_model = QPushButton("학습된 모델 저장 (.json)")
        self.btn_save_model.clicked.connect(self.save_trained_model)
        self.btn_save_model.setEnabled(False) # 학습 전 비활성화
        gb_save_layout.addWidget(self.btn_save_model)
        gb_save.setLayout(gb_save_layout)
        layout.addWidget(gb_save)

        # 4. Log
        self.log_train = QTextEdit()
        self.log_train.setReadOnly(True)
        layout.addWidget(QLabel("학습 로그 및 결과:"))
        layout.addWidget(self.log_train)

    def setup_backtest_tab(self):
        layout = QVBoxLayout(self.tab_backtest)

        # 1. Model Selection
        gb_model = QGroupBox("1. 모델 로드")
        gb_model_layout = QHBoxLayout()
        self.model_path_edit = QLineEdit()
        self.btn_select_model = QPushButton("모델 파일 선택 (.json)")
        self.btn_select_model.clicked.connect(self.select_model_file)
        gb_model_layout.addWidget(self.model_path_edit)
        gb_model_layout.addWidget(self.btn_select_model)
        gb_model.setLayout(gb_model_layout)
        layout.addWidget(gb_model)

        # 2. Simulation Parameters
        gb_sim = QGroupBox("2. 시뮬레이션 환경 설정")
        form_layout = QFormLayout()

        self.input_ticker = QLineEdit("ETH/USDT")
        
        # 날짜 기본값 설정
        self.input_start_date = QLineEdit("2024.01.01.00")
        self.input_end_date = QLineEdit("2024.01.31.23")
        
        self.combo_tier = QComboBox()
        self.combo_tier.addItems(["1. TOP (Major)", "2. High (Mid-Major)", "3. Mid", "4. Low", "5. Trash"])
        
        self.combo_regime = QComboBox()
        self.combo_regime.addItems(["Bull", "Bear", "Mixed", "Sideways"])
        
        self.spin_hold_time = QSpinBox()
        self.spin_hold_time.setRange(12, 2880) # 1시간 ~ 10일
        self.spin_hold_time.setValue(288) # 기본 1일 (5분 * 288 = 1440분)
        self.spin_hold_time.setSuffix(" Candles (5m)")

        form_layout.addRow("티커 (Ticker):", self.input_ticker)
        form_layout.addRow("시작 날짜 (YYYY.MM.DD.HH):", self.input_start_date)
        form_layout.addRow("종료 날짜 (YYYY.MM.DD.HH):", self.input_end_date)
        form_layout.addRow("코인 체급 (Tier):", self.combo_tier)
        form_layout.addRow("장세 (Market Regime):", self.combo_regime)
        form_layout.addRow("익절/손절 보유 시간:", self.spin_hold_time)
        
        gb_sim.setLayout(form_layout)
        layout.addWidget(gb_sim)

        # 3. Action
        self.btn_run_sim = QPushButton("시뮬레이션 실행 (Run Simulation)")
        self.btn_run_sim.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 12px;")
        self.btn_run_sim.clicked.connect(self.run_simulation)
        layout.addWidget(self.btn_run_sim)

        # 4. Log
        self.log_sim = QTextEdit()
        self.log_sim.setReadOnly(True)
        layout.addWidget(QLabel("시뮬레이션 결과 리포트:"))
        layout.addWidget(self.log_sim)

    # --- Training Logic ---
    def select_data_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "데이터 폴더 선택")
        if folder:
            self.folder_path_edit.setText(folder)
            self.log_train.append(f"선택된 폴더: {folder}")

    def start_training(self):
        folder = self.folder_path_edit.text()
        if not folder:
            self.log_train.append("오류: 데이터 폴더를 먼저 선택해주세요.")
            return

        self.btn_start_train.setEnabled(False)
        self.log_train.append("=== 학습 프로세스 시작 ===")
        
        self.worker = TrainingWorker(self.engine, folder)
        self.worker.log_signal.connect(self.update_train_log)
        self.worker.finished_signal.connect(self.on_training_finished)
        self.worker.error_signal.connect(self.on_training_error)
        self.worker.start()

    def update_train_log(self, message):
        self.log_train.append(message)

    def on_training_finished(self, report):
        self.log_train.append(report)
        self.log_train.append("Feature Importance 차트가 'reports/feature_importance.png'에 저장되었습니다.")
        self.btn_start_train.setEnabled(True)
        self.btn_save_model.setEnabled(True)

    def on_training_error(self, error_msg):
        self.log_train.append(f"오류 발생: {error_msg}")
        self.btn_start_train.setEnabled(True)

    def save_trained_model(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "모델 저장", "", "JSON Files (*.json)")
        if file_path:
            if self.engine.save_model(file_path):
                self.log_train.append(f"모델이 성공적으로 저장되었습니다: {file_path}")
            else:
                self.log_train.append("모델 저장 실패.")

    # --- Simulation Logic ---
    def select_model_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "모델 파일 선택", "", "JSON Files (*.json)")
        if file_path:
            self.model_path_edit.setText(file_path)
            # 모델 로드 시도
            try:
                self.engine.load_model(file_path)
                self.log_sim.append(f"모델 로드 완료: {os.path.basename(file_path)}")
            except Exception as e:
                self.log_sim.append(f"모델 로드 오류: {e}")

    def run_simulation(self):
        model_path = self.model_path_edit.text()
        if not model_path:
            self.log_sim.append("오류: 먼저 모델 파일을 선택하고 로드해주세요.")
            return

        ticker = self.input_ticker.text()
        start = self.input_start_date.text()
        end = self.input_end_date.text()
        hold = self.spin_hold_time.value()
        tier = self.combo_tier.currentText()
        regime = self.combo_regime.currentText()

        self.btn_run_sim.setEnabled(False)
        self.log_sim.clear()
        self.log_sim.append("=== 시뮬레이션 초기화 중... ===")

        self.sim_worker = SimulationWorker(self.engine, ticker, start, end, hold, tier, regime)
        self.sim_worker.log_signal.connect(self.log_sim.append)
        self.sim_worker.finished_signal.connect(self.on_sim_finished)
        self.sim_worker.error_signal.connect(self.on_sim_error)
        self.sim_worker.start()

    def on_sim_finished(self, result):
        self.log_sim.append(result)
        self.log_sim.append("시뮬레이션 완료. 결과 차트는 'backtest_results' 폴더를 확인하세요.")
        self.btn_run_sim.setEnabled(True)

    def on_sim_error(self, error):
        self.log_sim.append(f"치명적 오류 발생: {error}")
        self.btn_run_sim.setEnabled(True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AIStudioGUI()
    window.show()
    sys.exit(app.exec())