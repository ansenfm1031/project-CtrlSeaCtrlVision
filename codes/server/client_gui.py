import sys
import os
import json
import base64
import time
from datetime import datetime

# PyQt6 Core Imports
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
    QSplitter, QGroupBox, QLabel, QTextEdit, 
    QGridLayout, QSizePolicy
)
from PyQt6.QtGui import QFont, QFontDatabase, QImage, QPixmap
from PyQt6.QtCore import Qt, QObject, pyqtSignal, QThread, QSize

# MQTT Imports
# NOTE: Make sure to install: pip install paho-mqtt
import paho.mqtt.client as mqtt

# --- Global Configuration (Based on User's Files) ---
# MQTT Broker IP (10.10.14.73)
MQTT_BROKER = "10.10.14.73"
MQTT_PORT = 1883

# Base Topic Prefix
TOPIC_BASE = "project"

# 🔑 MQTT Credentials 
MQTT_USERNAME = "PYQT_USER"
# 🚨 이 값은 PYQT_USER의 실제 비밀번호로 변경하십시오!
MQTT_PASSWORD = "sksk" 

# 1. IMU Data Topic (client_IMU)
TOPIC_IMU = f"{TOPIC_BASE}/imu/RAW" 

# 2. Camera Topics (client_PE, client_AD)
TOPIC_CAM_PE = f"{TOPIC_BASE}/vision/PE/RAW"
TOPIC_CAM_AD = f"{TOPIC_BASE}/vision/AD/RAW"

# 3. Log/Event Topic 
TOPIC_LOGS = f"{TOPIC_BASE}/log/RAW" 

def safe_b64decode(data: str):
    """깨진 Base64 문자열을 자동 복구하여 디코딩."""
    data = data.strip().replace('\n', '').replace('\r', '')
    missing_padding = len(data) % 4
    if missing_padding:
        data += '=' * (4 - missing_padding)
    try:
        return base64.b64decode(data)
    except Exception as e:
        print(f"[Decode Error] {e}")
        return b''
    
# === 색상 테마 정의 ===
COLOR_MAP = {
    "IMU": "#58a6ff",      # 파란색
    "AD": "#e76f51",       # 주황색
    "PE": "#9d4edd",       # 보라색
    "SERVER": "#2a9d8f",   # 초록색
    "STT": "#2a9d8f",
    "LLM": "#2a9d8f",
    "DEFAULT": "#a8a8a8"
}

# --- MQTT Client (Running in a separate thread for PyQt) ---
class MqttClient(QObject):
    # PyQt의 이벤트 루프에서 안전하게 실행되도록 시그널 사용
    message_signal = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.client = mqtt.Client(client_id="PYQT_Dashboard_Client")
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        
        self.client.username_pw_set(username=MQTT_USERNAME, password=MQTT_PASSWORD)

    def on_connect(self, client, userdata, flags, rc):

        client.subscribe("project/vision/AD/VIDEO")

        if rc == 0:
            print("MQTT Broker Connected Successfully.")
            client.subscribe(TOPIC_IMU)
            client.subscribe(TOPIC_CAM_AD)
            client.subscribe(TOPIC_CAM_PE)
            client.subscribe(TOPIC_LOGS)
            print(f"Subscribed to: {TOPIC_IMU}, {TOPIC_CAM_AD}, {TOPIC_CAM_PE}, {TOPIC_LOGS}")
        else:
            print(f"MQTT Connection Failed with code {rc}. Please check broker IP and port or credentials.")

    def on_message(self, client, userdata, msg):
        topic = msg.topic
        try:
            payload = msg.payload.decode()
            self.message_signal.emit(topic, payload)
        except Exception as e:
            # 이 에러는 payload 디코딩 자체의 문제입니다.
            print(f"Error decoding payload for topic {topic}: {e}")

    def connect_and_loop(self, broker, port, keepalive=60):
        try:
            self.client.connect(broker, port, keepalive)
            self.client.loop_start() 
        except Exception as e:
            print(f"Connection error: {e}. Check network connection to {broker}.")
            
# --- Main Application Class ---
class MarineDashboardApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Marine Server 실시간 통합 대시보드")
        self.setMinimumSize(1200, 800)
        
        self.imu_data = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}
        self.imu_labels = {} 

        self.init_ui()
        self.mqtt_client = self.setup_mqtt()

    # --- UI Initialization ---
    def init_ui(self):
        font_family = "Nanum Gothic"
        if font_family not in QFontDatabase.families():
            font_family = "DejaVu Sans"
        self.setFont(QFont(font_family, 10))

        main_h_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setLayout(QHBoxLayout(self))
        self.layout().addWidget(main_h_splitter)

        # ----------------------------------------------------
        # A. 왼쪽 창: DB 실시간 로그 (events 테이블)
        # ----------------------------------------------------
        left_log_widget = QGroupBox("DB 실시간 시스템 로그 (events)")
        left_vbox = QVBoxLayout(left_log_widget)
        
        self.db_log_widget = QTextEdit()
        self.db_log_widget.setReadOnly(True)
        self.db_log_widget.setFont(QFont("Monospace", 9))
        self.db_log_widget.setStyleSheet("background-color: #0d1117; color: #58a6ff;") 
        left_vbox.addWidget(self.db_log_widget)

        main_h_splitter.addWidget(left_log_widget)
        main_h_splitter.setSizes([400, 800]) 

        # ----------------------------------------------------
        # B. 오른쪽 창: IMU 및 카메라 (수직 분할)
        # ----------------------------------------------------
        right_main_vbox = QWidget()
        right_vbox = QVBoxLayout(right_main_vbox)

        # 2. 오른쪽 상단: IMU 모듈 실시간 값 (30% 높이)
        imu_group = QGroupBox("IMU 모듈 실시간 센서 데이터 (project/IMU/RAW)")
        imu_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        
        imu_grid = QGridLayout()
        self._setup_imu_display(imu_grid)
        imu_group.setLayout(imu_grid)
        
        # 3. 오른쪽 하단: 카메라 피드 (70% 높이)
        camera_group = QGroupBox("실시간 카메라 피드 (AD & PE)")
        camera_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        camera_hbox = QHBoxLayout(camera_group)
        
        # client_AD 카메라
        self.cam_ad_label = QLabel("client_AD 카메라 피드 (Waiting...)")
        self.cam_ad_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cam_ad_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.cam_ad_label.setStyleSheet("border: 2px solid #2a9d8f; background-color: black; color: white;")
        
        # client_PE 카메라
        self.cam_pe_label = QLabel("client_PE 카메라 피드 (Waiting...)")
        self.cam_pe_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cam_pe_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.cam_pe_label.setStyleSheet("border: 2px solid #e76f51; background-color: black; color: white;")

        camera_hbox.addWidget(self.cam_ad_label)
        camera_hbox.addWidget(self.cam_pe_label)
        
        right_vbox.addWidget(imu_group, 3)    
        right_vbox.addWidget(camera_group, 7) 

        main_h_splitter.addWidget(right_main_vbox)

    # --- IMU UI Setup Helper ---
    def _setup_imu_display(self, grid_layout):
        data_keys = [
            ("Roll (X축 회전)", "roll", "#2a9d8f"),
            ("Pitch (Y축 회전)", "pitch", "#e9c46a"),
            ("Yaw (Z축 회전)", "yaw", "#f4a261"),
        ]

        for col, (title, key, color) in enumerate(data_keys):
            title_label = QLabel(f"<b>{title}:</b>")
            grid_layout.addWidget(title_label, 0, col * 2, alignment=Qt.AlignmentFlag.AlignRight)

            value_label = QLabel("0.00")
            value_label.setObjectName(f"imu_{key}") 
            value_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
            value_label.setStyleSheet(f"color: {color}; padding: 5px;")
            
            grid_layout.addWidget(value_label, 0, col * 2 + 1, alignment=Qt.AlignmentFlag.AlignLeft)
            self.imu_labels[key] = value_label 
        
        grid_layout.setColumnStretch(1, 1)
        grid_layout.setColumnStretch(3, 1)
        grid_layout.setColumnStretch(5, 1)


    # --- MQTT Setup and Handlers ---
    def setup_mqtt(self):
        client = MqttClient(self)
        client.message_signal.connect(self.on_mqtt_message) 
        client.connect_and_loop(MQTT_BROKER, MQTT_PORT)
        return client

    def on_mqtt_message(self, topic, payload):
        # 1️⃣ IMU Data Processing
        if topic == TOPIC_IMU:
            try:
                data = json.loads(payload)
                print(f"IMU Data Received & Parsed: {data}")
                self.update_imu_ui(data)
            except json.JSONDecodeError as e:
                print(f"IMU: JSON Decode Error: {e} | Payload: {payload}")
                self.db_log_widget.insertPlainText(f"[ERROR] IMU JSON Error: {e}\n")

        # 2️⃣ AD VIDEO (Base64 → 이미지 표시)
        elif topic == "project/vision/AD/VIDEO":
            self.update_camera_feed(self.cam_ad_label, payload)

        # 3️⃣ Vision AD RAW
        elif topic == TOPIC_CAM_AD:
            self.update_camera_feed(self.cam_ad_label, payload)

        # 4️⃣ Vision PE RAW
        elif topic == TOPIC_CAM_PE:
            self.update_camera_feed(self.cam_pe_label, payload)

        # 5️⃣ Log Data Processing (project/log/RAW)
        elif topic == TOPIC_LOGS:
            try:
                log = json.loads(payload)
                self.update_log_ui(log)
            except json.JSONDecodeError:
                self.update_log_ui({
                    "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "module": "SYS", "action": "RAW", "payload": payload
                })


    def update_imu_ui(self, data):
        # 🚨 여기서 오류가 발생할 가능성이 높습니다.
        for key in self.imu_data.keys():
            if key in data and key in self.imu_labels:
                try:
                    # 데이터가 문자열로 넘어왔을 수 있으므로 str()을 먼저 적용
                    value = float(str(data[key])) 
                    self.imu_labels[key].setText(f"{value:.2f}")
                except (ValueError, TypeError) as e:
                    # 👈 변환 오류 시 터미널에 로그 출력
                    print(f"IMU: Value Error for key '{key}': {e} | Value: {data[key]}")
                    self.imu_labels[key].setText("CONV ERR")


    def update_log_ui(self, log):
        """
        서버에서 수신한 로그(JSON dict)를 색상, 포맷에 맞게 표시
        """
        try:
            ts = log.get('ts', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            module = log.get('module', 'UNKNOWN').upper()
            action = log.get('action', 'EVENT')
            color = COLOR_MAP.get(module, COLOR_MAP["DEFAULT"])

            # 기본 메시지
            message = log.get('payload', '')

            # --- IMU 로그 ---
            if module == "IMU" and all(k in log for k in ["roll", "pitch", "yaw"]):
                msg = f"Roll={float(log['roll']):.2f}°  Pitch={float(log['pitch']):.2f}°  Yaw={float(log['yaw']):.2f}°"

            # --- Vision 로그 (AD/PE) ---
            elif module in ["AD", "PE"] and isinstance(log.get("detections"), list):
                det_lines = []
                for det in log["detections"]:
                    obj = det.get("object", "Unknown")
                    risk = int(det.get("risk", 0))
                    desc = det.get("desc", "")
                    # 위험도 강조 색상 처리
                    if risk >= 3:
                        det_lines.append(f"<b><span style='color:#ff4d4d'>{obj} 위험도 {risk}</span></b> - {desc}")
                    else:
                        det_lines.append(f"{obj} 위험도 {risk} - {desc}")
                msg = " / ".join(det_lines)

            # --- STT/LLM 로그 ---
            elif module in ["STT", "LLM", "SERVER"]:
                msg = message

            # --- 일반 로그 ---
            else:
                msg = f"{action} → {message}"

            # 최종 문자열 구성
            formatted = f"<span style='color:{color}'>[{ts}] ({module}) {msg}</span><br>"

            # QTextEdit에 추가
            self.db_log_widget.insertHtml(formatted)
            self.db_log_widget.moveCursor(self.db_log_widget.textCursor().MoveOperation.End)

            # 자동 스크롤 유지
            self.db_log_widget.verticalScrollBar().setValue(
                self.db_log_widget.verticalScrollBar().maximum()
            )

        except Exception as e:
            print(f"[GUI-ERROR] update_log_ui 처리 중 오류: {e}")
            # 예외 발생 시 기본 문자열로 출력
            fallback = f"[{datetime.now().strftime('%H:%M:%S')}] {log}\n"
            self.db_log_widget.insertPlainText(fallback)


    def update_camera_feed(self, label, base64_data):
        try:
            image_data = safe_b64decode(base64_data)
            image = QImage.fromData(image_data)
            
            if image.isNull():
                label.setText(f"Invalid Image Data from {label.objectName()}")
                return
            
            # 크기 조정 (현재 크기를 가져와서 사용)
            # **주의**: VNC 환경에서 size()가 0을 반환할 수 있으므로, 최소 크기를 설정합니다.
            target_size = label.size()
            if target_size.width() <= 0 or target_size.height() <= 0:
                 target_size = QSize(400, 300) # Fallback size

            pixmap = QPixmap.fromImage(image)
            pixmap = pixmap.scaled(target_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            label.setPixmap(pixmap)
            label.setText("") 

        except Exception as e:
            label.setText(f"Decode/Display Error: {e}")
            print(f"Camera Feed Error: {e}")

# --- Application Entry Point ---
if __name__ == '__main__':
    if os.environ.get('XDG_RUNTIME_DIR') is None and 'root' in os.environ.get('HOME', ''):
        os.environ['XDG_RUNTIME_DIR'] = '/tmp/runtime-root'

    try:
        import paho.mqtt.client
    except ImportError:
        print("ERROR: 'paho-mqtt' 라이브러리가 설치되지 않았습니다. 'pip install paho-mqtt'를 실행하세요.")
        sys.exit(1)

    app = QApplication(sys.argv)
    ex = MarineDashboardApp()
    ex.show()
    sys.exit(app.exec())


