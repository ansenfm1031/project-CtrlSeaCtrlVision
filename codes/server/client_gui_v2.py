import sys
import os
import json
import base64
import time
from datetime import datetime

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
    QSplitter, QGroupBox, QLabel, QTextEdit, 
    QGridLayout, QSizePolicy, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
)
from PyQt6.QtGui import QFont, QFontDatabase, QImage, QPixmap
from PyQt6.QtCore import Qt, QObject, pyqtSignal, QTimer

import paho.mqtt.client as mqtt

# --- MQTT Configuration ---
MQTT_BROKER = "10.10.14.73"
MQTT_PORT = 1883
MQTT_USERNAME = "PYQT_USER"
MQTT_PASSWORD = "sksk"

TOPIC_BASE = "project"
TOPIC_IMU = f"{TOPIC_BASE}/IMU/RAW"
TOPIC_CAM_AD = f"{TOPIC_BASE}/vision/AD/RAW"
TOPIC_CAM_PE = f"{TOPIC_BASE}/vision/PE/RAW"
TOPIC_LOGS = f"{TOPIC_BASE}/log/RAW"
TOPIC_VIDEO_AD = f"{TOPIC_BASE}/vision/AD/VIDEO"  # 추가됨

COLOR_MAP = {
    "IMU": "#58a6ff",
    "AD": "#e76f51",
    "PE": "#9d4edd",
    "SERVER": "#2a9d8f",
    "STT": "#2a9d8f",
    "LLM": "#2a9d8f",
    "DEFAULT": "#a8a8a8"
}

# ===========================
# MQTT Client
# ===========================
class MqttClient(QObject):
    message_signal = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.client = mqtt.Client(client_id="PYQT_Dashboard_Client")
        self.client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("✅ MQTT Broker Connected.")
            client.subscribe(TOPIC_IMU)
            client.subscribe(TOPIC_CAM_AD)
            client.subscribe(TOPIC_CAM_PE)
            client.subscribe(TOPIC_LOGS)
            client.subscribe(TOPIC_VIDEO_AD)
            print(f"Subscribed to {TOPIC_IMU}, {TOPIC_CAM_AD}, {TOPIC_CAM_PE}, {TOPIC_LOGS}, {TOPIC_VIDEO_AD}")
        else:
            print(f"❌ MQTT Connection Failed. rc={rc}")

    def on_message(self, client, userdata, msg):
        try:
            self.message_signal.emit(msg.topic, msg.payload.decode())
        except Exception as e:
            print(f"[MQTT ERROR] {e}")

    def connect_and_loop(self):
        try:
            self.client.connect(MQTT_BROKER, MQTT_PORT, 60)
            self.client.loop_start()
        except Exception as e:
            print(f"❌ MQTT Connection Error: {e}")

# ===========================
# Main Dashboard UI
# ===========================
class MarineDashboardApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("⚓ Marine Server 실시간 통합 대시보드")
        self.setMinimumSize(1200, 800)

        self.imu_data = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}
        self.imu_labels = {}

        # GraphicsScene용 멤버 변수
        self.cam_ad_scene = None
        self.cam_ad_item = None
        self.cam_pe_scene = None
        self.cam_pe_item = None

        self.init_ui()
        self.mqtt_client = self.setup_mqtt()

    # ===========================
    # UI Initialization
    # ===========================
    def init_ui(self):
        font_family = "Nanum Gothic" if "Nanum Gothic" in QFontDatabase.families() else "DejaVu Sans"
        self.setFont(QFont(font_family, 10))

        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        layout = QHBoxLayout(self)
        layout.addWidget(main_splitter)

        # -------- Left (DB Logs) --------
        left_group = QGroupBox("DB 실시간 시스템 로그 (events)")
        left_layout = QVBoxLayout(left_group)
        self.db_log_widget = QTextEdit()
        self.db_log_widget.setReadOnly(True)
        self.db_log_widget.setFont(QFont("Monospace", 9))
        self.db_log_widget.setStyleSheet("background-color:#0d1117; color:#58a6ff;")
        left_layout.addWidget(self.db_log_widget)
        main_splitter.addWidget(left_group)

        # -------- Right (IMU + Camera) --------
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # IMU Section
        imu_group = QGroupBox("IMU 모듈 실시간 센서 데이터 (project/IMU/RAW)")
        imu_layout = QGridLayout()
        self._setup_imu_display(imu_layout)
        imu_group.setLayout(imu_layout)

        # Camera Section (QGraphicsView 기반)
        cam_group = QGroupBox("실시간 카메라 피드 (AD & PE)")
        cam_layout = QHBoxLayout(cam_group)

        # AD 카메라
        self.cam_ad_view = QGraphicsView()
        self.cam_ad_scene = QGraphicsScene()
        self.cam_ad_item = QGraphicsPixmapItem()
        self.cam_ad_scene.addItem(self.cam_ad_item)
        self.cam_ad_view.setScene(self.cam_ad_scene)
        self.cam_ad_view.setStyleSheet("border:2px solid #2a9d8f; background:black;")

        # PE 카메라
        self.cam_pe_view = QGraphicsView()
        self.cam_pe_scene = QGraphicsScene()
        self.cam_pe_item = QGraphicsPixmapItem()
        self.cam_pe_scene.addItem(self.cam_pe_item)
        self.cam_pe_view.setScene(self.cam_pe_scene)
        self.cam_pe_view.setStyleSheet("border:2px solid #e76f51; background:black;")

        cam_layout.addWidget(self.cam_ad_view)
        cam_layout.addWidget(self.cam_pe_view)

        right_layout.addWidget(imu_group, 3)
        right_layout.addWidget(cam_group, 7)
        main_splitter.addWidget(right_widget)

    def _setup_imu_display(self, grid):
        for col, (title, key, color) in enumerate([
            ("Roll (X축 회전)", "roll", "#2a9d8f"),
            ("Pitch (Y축 회전)", "pitch", "#e9c46a"),
            ("Yaw (Z축 회전)", "yaw", "#f4a261"),
        ]):
            label_t = QLabel(f"<b>{title}:</b>")
            label_v = QLabel("0.00")
            label_v.setFont(QFont("Arial", 14, QFont.Weight.Bold))
            label_v.setStyleSheet(f"color:{color}; padding:5px;")
            self.imu_labels[key] = label_v
            grid.addWidget(label_t, 0, col * 2, Qt.AlignmentFlag.AlignRight)
            grid.addWidget(label_v, 0, col * 2 + 1, Qt.AlignmentFlag.AlignLeft)

    # ===========================
    # MQTT Setup
    # ===========================
    def setup_mqtt(self):
        client = MqttClient(self)
        client.message_signal.connect(self.on_mqtt_message)
        client.connect_and_loop()
        return client

    # ===========================
    # MQTT Message Handler
    # ===========================
    def on_mqtt_message(self, topic, payload):
        if topic == TOPIC_IMU:
            try:
                data = json.loads(payload)
                self.update_imu_ui(data)
            except Exception as e:
                print(f"IMU Error: {e}")

        elif topic == TOPIC_VIDEO_AD:
            self.update_camera_feed(self.cam_ad_item, payload)

        elif topic == TOPIC_CAM_PE:
            self.update_camera_feed(self.cam_pe_item, payload)

        elif topic == TOPIC_LOGS:
            try:
                self.update_log_ui(json.loads(payload))
            except:
                pass

    # ===========================
    # Update Functions
    # ===========================
    def update_imu_ui(self, data):
        for key, label in self.imu_labels.items():
            try:
                val = float(data.get(key, 0.0))
                label.setText(f"{val:.2f}")
            except:
                label.setText("ERR")

    def update_log_ui(self, log):
        ts = log.get("ts", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        module = log.get("module", "UNKNOWN").upper()
        action = log.get("action", "EVENT")
        msg = log.get("payload", "")
        color = COLOR_MAP.get(module, COLOR_MAP["DEFAULT"])
        formatted = f"<span style='color:{color}'>[{ts}] ({module}) {action} → {msg}</span><br>"
        self.db_log_widget.insertHtml(formatted)
        self.db_log_widget.moveCursor(self.db_log_widget.textCursor().MoveOperation.End)

    def update_camera_feed(self, graphics_item, base64_data):
        try:
            img_data = base64.b64decode(base64_data)
            qimg = QImage.fromData(img_data)
            if qimg.isNull():
                return
            pixmap = QPixmap.fromImage(qimg)
            pixmap = pixmap.scaled(
                640, 480,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation   # ✅ 속도 우선
            )
            graphics_item.setPixmap(pixmap)

            # 🚀 바로 다시 그리기 (렌더 큐 지연 방지)
            graphics_item.scene().update()
        except Exception as e:
            print(f"[Camera Feed Error] {e}")

# ===========================
# Main Entry Point
# ===========================
if __name__ == "__main__":
    if os.environ.get("XDG_RUNTIME_DIR") is None:
        os.environ["XDG_RUNTIME_DIR"] = "/tmp/runtime-root"

    app = QApplication(sys.argv)
    window = MarineDashboardApp()
    window.show()
    sys.exit(app.exec())

