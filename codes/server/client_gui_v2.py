import sys
import os
import json
import base64
from datetime import datetime

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
    QSplitter, QGroupBox, QLabel, QTextEdit, 
    QGridLayout, QSizePolicy, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
)
from PyQt6.QtGui import QFont, QFontDatabase, QImage, QPixmap
from PyQt6.QtCore import Qt, QObject, pyqtSignal, QSize

import paho.mqtt.client as mqtt

# --- Global Configuration ---
MQTT_BROKER = "10.10.14.73"
MQTT_PORT = 1883

TOPIC_BASE = "project/vision"
MQTT_USERNAME = "PYQT_USER"
MQTT_PASSWORD = "sksk"

TOPIC_IMU = "project/imu/RAW"
TOPIC_CAM_PE = f"{TOPIC_BASE}/FALL/VIDEO"   # PE.py의 비디오 스트림
TOPIC_PE_RAW = f"{TOPIC_BASE}/PE/RAW"       # 낙상 감지 RAW 로그
TOPIC_PE_ALERT = f"{TOPIC_BASE}/PE/ALERT"   # 낙상 감지 ALERT 로그
TOPIC_CAM_AD = f"{TOPIC_BASE}/AD/RAW"
TOPIC_VIDEO_AD = f"{TOPIC_BASE}/AD/VIDEO"
TOPIC_LOGS = f"{TOPIC_BASE}/log/RAW"

def safe_b64decode(data: str):
    data = data.strip().replace('\n', '').replace('\r', '')
    missing_padding = len(data) % 4
    if missing_padding:
        data += '=' * (4 - missing_padding)
    try:
        return base64.b64decode(data)
    except Exception as e:
        print(f"[Decode Error] {e}")
        return b''

COLOR_MAP = {
    "IMU": "#58a6ff",
    "AD": "#e76f51",
    "PE": "#9d4edd",
    "SERVER": "#2a9d8f",
    "STT": "#2a9d8f",
    "LLM": "#2a9d8f",
    "DEFAULT": "#a8a8a8"
}

# --- MQTT Client ---
class MqttClient(QObject):
    message_signal = pyqtSignal(str, str)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.client = mqtt.Client(client_id="PYQT_Dashboard_Client")
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.username_pw_set(username=MQTT_USERNAME, password=MQTT_PASSWORD)

    def on_connect(self, client, userdata, flags, rc):
        client.subscribe(TOPIC_VIDEO_AD)
        if rc == 0:
            print("MQTT Broker Connected Successfully.")
            client.subscribe(TOPIC_IMU)
            client.subscribe(TOPIC_CAM_AD)
            client.subscribe(TOPIC_VIDEO_AD)
            client.subscribe(TOPIC_CAM_PE)     # FALL/VIDEO
            client.subscribe(TOPIC_PE_RAW)     # 낙상 RAW
            client.subscribe(TOPIC_PE_ALERT)   # 낙상 ALERT
            client.subscribe(TOPIC_LOGS)
            print(f"Subscribed → {TOPIC_IMU}, {TOPIC_VIDEO_AD}, {TOPIC_CAM_PE}, {TOPIC_PE_RAW}, {TOPIC_PE_ALERT}, {TOPIC_LOGS}")
        else:
            print(f"MQTT Connection Failed with code {rc}.")

    def on_message(self, client, userdata, msg):
        topic = msg.topic
        try:
            payload = msg.payload.decode()
            self.message_signal.emit(topic, payload)
        except Exception as e:
            print(f"Error decoding payload for topic {topic}: {e}")

    def connect_and_loop(self, broker, port, keepalive=60):
        try:
            self.client.connect(broker, port, keepalive)
            self.client.loop_start()
        except Exception as e:
            print(f"Connection error: {e}")

# --- Main GUI ---
class MarineDashboardApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Marine Server 실시간 통합 대시보드")
        self.setMinimumSize(1200, 800)

        self.imu_data = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}
        self.imu_labels = {}

        # QGraphicsScene/PixmapItem 저장용
        self.ad_scene = None
        self.ad_pixmap_item = None
        self.pe_scene = None
        self.pe_pixmap_item = None

        self.init_ui()
        self.mqtt_client = self.setup_mqtt()

    # --- UI 구성 ---
    def init_ui(self):
        font_family = "Nanum Gothic" if "Nanum Gothic" in QFontDatabase.families() else "DejaVu Sans"
        self.setFont(QFont(font_family, 10))
        main_h_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setLayout(QHBoxLayout(self))
        self.layout().addWidget(main_h_splitter)

        # --- 좌측 로그 창 ---
        left_log_widget = QGroupBox("DB 실시간 시스템 로그 (events)")
        left_vbox = QVBoxLayout(left_log_widget)
        self.db_log_widget = QTextEdit()
        self.db_log_widget.setReadOnly(True)
        self.db_log_widget.setFont(QFont("Monospace", 9))
        self.db_log_widget.setStyleSheet("background-color: #0d1117; color: #58a6ff;")
        left_vbox.addWidget(self.db_log_widget)
        main_h_splitter.addWidget(left_log_widget)
        main_h_splitter.setSizes([400, 800])

        # --- 우측 (IMU + 카메라) ---
        right_main = QWidget()
        right_vbox = QVBoxLayout(right_main)

        # IMU 데이터
        imu_group = QGroupBox("IMU 모듈 실시간 센서 데이터 (project/IMU/RAW)")
        imu_grid = QGridLayout()
        self._setup_imu_display(imu_grid)
        imu_group.setLayout(imu_grid)

        # 카메라 (QGraphicsView 사용)
        camera_group = QGroupBox("실시간 카메라 피드 (AD & PE)")
        camera_hbox = QHBoxLayout(camera_group)

        # AD 카메라
        self.cam_ad_view = QGraphicsView()
        self.cam_ad_view.setScene(QGraphicsScene())
        self.ad_scene = self.cam_ad_view.scene()
        self.ad_pixmap_item = QGraphicsPixmapItem()
        self.ad_scene.addItem(self.ad_pixmap_item)
        self.cam_ad_view.setStyleSheet("border: 2px solid #2a9d8f; background-color: black;")

        # PE 카메라
        self.cam_pe_view = QGraphicsView()
        self.cam_pe_view.setScene(QGraphicsScene())
        self.pe_scene = self.cam_pe_view.scene()
        self.pe_pixmap_item = QGraphicsPixmapItem()
        self.pe_scene.addItem(self.pe_pixmap_item)
        self.cam_pe_view.setStyleSheet("border: 2px solid #e76f51; background-color: black;")

        camera_hbox.addWidget(self.cam_ad_view)
        camera_hbox.addWidget(self.cam_pe_view)

        right_vbox.addWidget(imu_group, 3)
        right_vbox.addWidget(camera_group, 7)
        main_h_splitter.addWidget(right_main)

    # --- IMU UI ---
    def _setup_imu_display(self, grid):
        data_keys = [
            ("Roll (X축 회전)", "roll", "#2a9d8f"),
            ("Pitch (Y축 회전)", "pitch", "#e9c46a"),
            ("Yaw (Z축 회전)", "yaw", "#f4a261"),
        ]
        for col, (title, key, color) in enumerate(data_keys):
            t_label = QLabel(f"<b>{title}:</b>")
            grid.addWidget(t_label, 0, col*2, alignment=Qt.AlignmentFlag.AlignRight)
            v_label = QLabel("0.00")
            v_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
            v_label.setStyleSheet(f"color: {color}; padding: 5px;")
            grid.addWidget(v_label, 0, col*2 + 1)
            self.imu_labels[key] = v_label

    # --- MQTT 설정 ---
    def setup_mqtt(self):
        client = MqttClient(self)
        client.message_signal.connect(self.on_mqtt_message)
        client.connect_and_loop(MQTT_BROKER, MQTT_PORT)
        return client

    # --- 메시지 처리 ---
    def on_mqtt_message(self, topic, payload):
        if topic == TOPIC_IMU:
            try:
                data = json.loads(payload)
                self.update_imu_ui(data)
            except json.JSONDecodeError:
                print(f"[IMU] JSON Error")

        elif topic in [TOPIC_VIDEO_AD, TOPIC_CAM_AD]:
            self.update_camera_view(self.ad_pixmap_item, payload)

        elif topic == TOPIC_CAM_PE:  # ✅ 낙상 영상
            self.update_camera_view(self.pe_pixmap_item, payload)

        elif topic in [TOPIC_LOGS, TOPIC_PE_RAW, TOPIC_PE_ALERT]:  # ✅ 낙상 로그/알람
            try:
                log = json.loads(payload)
                self.update_log_ui(log)
            except json.JSONDecodeError:
                self.update_log_ui({
                    "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "module": "SYS",
                    "action": "RAW",
                    "payload": payload
                })

    # --- IMU UI 업데이트 ---
    def update_imu_ui(self, data):
        for key in self.imu_data:
            if key in data:
                try:
                    val = float(str(data[key]))
                    self.imu_labels[key].setText(f"{val:.2f}")
                except Exception as e:
                    print(f"[IMU Error] {key}: {e}")
                    self.imu_labels[key].setText("ERR")

    # --- 로그 UI 업데이트 ---
    def update_log_ui(self, log):
        try:
            ts = log.get('ts', datetime.now().strftime("%H:%M:%S"))
            module = log.get('module', 'UNKNOWN').upper()
            action = log.get('action', 'EVENT')
            color = COLOR_MAP.get(module, COLOR_MAP["DEFAULT"])
            msg = log.get('payload', '')
            formatted = f"<span style='color:{color}'>[{ts}] ({module}) {action} → {msg}</span><br>"
            self.db_log_widget.insertHtml(formatted)
            self.db_log_widget.moveCursor(self.db_log_widget.textCursor().MoveOperation.End)
        except Exception as e:
            print(f"[LogUI Error] {e}")

    # --- 카메라 업데이트 (QGraphicsView용) ---
    def update_camera_view(self, pixmap_item, base64_data):
        try:
            img_data = safe_b64decode(base64_data)
            qimg = QImage.fromData(img_data)
            if qimg.isNull():
                return
            pix = QPixmap.fromImage(qimg).scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)
            pixmap_item.setPixmap(pix)
            pixmap_item.scene().update()  # 즉시 렌더링
        except Exception as e:
            print(f"[Camera Feed Error] {e}")

# --- Entry Point ---
if __name__ == '__main__':
    if os.environ.get('XDG_RUNTIME_DIR') is None and 'root' in os.environ.get('HOME', ''):
        os.environ['XDG_RUNTIME_DIR'] = '/tmp/runtime-root'

    app = QApplication(sys.argv)
    ex = MarineDashboardApp()
    ex.show()
    sys.exit(app.exec())
