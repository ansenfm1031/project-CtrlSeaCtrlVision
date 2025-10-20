import sys
import os
import json
import base64
from datetime import datetime

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
    QSplitter, QGroupBox, QLabel, QTextEdit, 
    QGridLayout, QSizePolicy, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QTabWidget, QGraphicsEllipseItem, QGraphicsItem
)
from PyQt6.QtGui import QFont, QFontDatabase, QImage, QPixmap, QBrush, QPen, QColor, QPainter, QPolygonF
from PyQt6.QtCore import Qt, QObject, pyqtSignal, QSize, QRectF, QPointF

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
TOPIC_LOGS = f"project/log/RAW"
TOPIC_LOGBOOK = "project/log/LOGBOOK"

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
            # client.subscribe(TOPIC_IMU)
            client.subscribe(TOPIC_CAM_AD)
            client.subscribe(TOPIC_VIDEO_AD)
            client.subscribe(TOPIC_CAM_PE)     # FALL/VIDEO
            client.subscribe(TOPIC_PE_RAW)     # 낙상 RAW
            client.subscribe(TOPIC_PE_ALERT)   # 낙상 ALERT
            client.subscribe(TOPIC_LOGS)
            client.subscribe(TOPIC_LOGBOOK)
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

# --- 롤(Roll) 시각화: 배의 뒷모습 ---
class RollIndicator(QGraphicsItem):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.angle = 0.0
        # 회전 중심을 아이템의 중심(0, 0)으로 설정
        self.setTransformOriginPoint(0, 0) 

    def boundingRect(self):
        # 아이템이 차지하는 공간 (고정 크기)
        return QRectF(-60, -60, 120, 120)

    def paint(self, painter: QPainter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # 캔버스 중심: (0, 0)
        center = QPointF(0, 0)
        
        # 1. 배경 (수평선)
        painter.setPen(QPen(QColor(150, 150, 150), 1))
        painter.drawLine(-60, 0, 60, 0) 
        
        # 2. 선박의 단면 (직사각형)
        # 롤 각도에 따라 회전
        self.setRotation(self.angle)
        
        painter.setBrush(QBrush(QColor(42, 157, 143, 200))) # 청록색 (배의 몸체)
        painter.setPen(QPen(QColor(244, 162, 97), 2)) # 주황색 테두리
        
        # 선박 몸체: 너비 100, 높이 30
        ship_rect = QRectF(-50, -15, 100, 30)
        painter.drawRect(ship_rect)
        
        # 3. 중앙 기준점 표시 (선박 몸체가 회전하더라도 중심에 고정)
        self.setRotation(0) # 기준점은 회전하지 않도록 리셋
        painter.setBrush(QBrush(QColor(255, 0, 0)))
        painter.drawEllipse(center, 3, 3)

    def set_roll(self, roll_angle):
        self.angle = roll_angle
        self.update() # 화면 갱신 요청

# --- 피치(Pitch) 시각화: 배의 옆모습 ---
class PitchIndicator(QGraphicsItem):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.angle = 0.0
        # 회전 중심을 아이템의 중심(0, 0)으로 설정
        self.setTransformOriginPoint(0, 0) 

    def boundingRect(self):
        return QRectF(-60, -60, 120, 120)

    def paint(self, painter: QPainter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        center = QPointF(0, 0)
        
        # 1. 배경 (수평선)
        painter.setPen(QPen(QColor(150, 150, 150), 1))
        painter.drawLine(-60, 0, 60, 0) 
        
        # 2. 선박의 옆모습 (삼각형과 직사각형 조합)
        self.setRotation(self.angle)
        
        # 몸체
        painter.setBrush(QBrush(QColor(233, 196, 106, 200))) # 황토색
        painter.setPen(QPen(QColor(244, 162, 97), 2))
        
        # 직사각형 (선박 본체)
        rect_body = QRectF(-50, -15, 100, 20)
        painter.drawRect(rect_body)
        
        # 삼각형 (선박 선수)
        bow_points = QPolygonF([
            QPointF(50, -15), 
            QPointF(60, -5), 
            QPointF(50, 5)
        ])
        painter.drawPolygon(bow_points)
        
        # 3. 중앙 기준점 표시
        self.setRotation(0)
        painter.setBrush(QBrush(QColor(255, 0, 0)))
        painter.drawEllipse(center, 3, 3)

    def set_pitch(self, pitch_angle):
        self.angle = pitch_angle
        self.update()

# --- 요(Yaw) 시각화: 나침반 ---
class YawIndicator(QGraphicsItem):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.angle = 0.0
        self.setTransformOriginPoint(0, 0)

    def boundingRect(self):
        return QRectF(-60, -60, 120, 120)

    def paint(self, painter: QPainter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # 1. 배경 (나침반 원)
        painter.setBrush(QBrush(QColor(30, 30, 30)))
        painter.setPen(QPen(QColor(150, 150, 150), 2))
        painter.drawEllipse(QRectF(-55, -55, 110, 110))

        # 2. 나침반 눈금 및 방향 표시
        painter.setPen(QPen(QColor(200, 200, 200), 1))
        # N, E, S, W
        painter.drawText(QPointF(-5, -50), "N") 
        painter.drawText(QPointF(45, 5), "E")
        painter.drawText(QPointF(-5, 55), "S")
        painter.drawText(QPointF(-55, 5), "W")
        
        # 3. 방향 지시기 (Yaw에 따라 회전)
        # Yaw는 보통 북쪽 기준 0~360도이므로, 시계 방향 회전을 위해 마이너스 값을 사용
        self.setRotation(-self.angle) 

        # 빨간색/흰색 바늘
        needle_points = QPolygonF([
            QPointF(-5, 0), QPointF(5, 0), QPointF(0, -50)
        ])
        
        # 빨간색 (북쪽 방향)
        painter.setBrush(QBrush(QColor(255, 0, 0)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPolygon(needle_points)
        
        # 흰색 (남쪽 방향)
        white_points = QPolygonF([
            QPointF(-5, 0), QPointF(5, 0), QPointF(0, 50)
        ])
        painter.setBrush(QBrush(QColor(255, 255, 255)))
        painter.drawPolygon(white_points)
        
        # 4. 중앙 나사
        painter.setBrush(QBrush(QColor(100, 100, 100)))
        painter.drawEllipse(QPointF(0, 0), 5, 5)

    def set_yaw(self, yaw_angle):
        self.angle = yaw_angle
        self.update()

# --- Main GUI ---
class MarineDashboardApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Marine Server 실시간 통합 대시보드")
        self.setMinimumSize(1200, 800)

        self.imu_data = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}
        self.imu_labels = {}

        # IMU 시각화용 뷰/아이템 참조 저장 딕셔너리 추가
        self.imu_views = {}
        self.imu_items = {}

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
        left_log_widget = QGroupBox("데이터 로그 보기")
        tab_widget = QTabWidget()

        # 🟦 시스템 로그 탭
        self.db_log_widget = QTextEdit()
        self.db_log_widget.setReadOnly(True)
        self.db_log_widget.setFont(QFont("Monospace", 9))
        self.db_log_widget.setStyleSheet("background-color: #0d1117; color: #58a6ff;")

        # 🟧 항해일지 탭
        self.voyage_log_widget = QTextEdit()
        self.voyage_log_widget.setReadOnly(True)
        self.voyage_log_widget.setFont(QFont("Monospace", 9))
        self.voyage_log_widget.setStyleSheet("background-color: #0d1117; color: #9d4edd;")

        # 탭 구성
        tab_widget.addTab(self.db_log_widget, "시스템 로그")
        tab_widget.addTab(self.voyage_log_widget, "최근 항해일지")

        # 그룹 박스에 추가
        left_vbox = QVBoxLayout(left_log_widget)
        left_vbox.addWidget(tab_widget)

        # 메인 스플리터에 추가
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
        self.cam_ad_view.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.cam_ad_view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.cam_ad_view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.cam_ad_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # PE 카메라
        self.cam_pe_view = QGraphicsView()
        self.cam_pe_view.setScene(QGraphicsScene())
        self.pe_scene = self.cam_pe_view.scene()
        self.pe_pixmap_item = QGraphicsPixmapItem()
        self.pe_scene.addItem(self.pe_pixmap_item)
        self.cam_pe_view.setStyleSheet("border: 2px solid #e76f51; background-color: black;")
        self.cam_pe_view.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.cam_pe_view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.cam_pe_view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.cam_pe_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        camera_hbox.addWidget(self.cam_ad_view)
        camera_hbox.addWidget(self.cam_pe_view)

        right_vbox.addWidget(imu_group, 4)
        right_vbox.addWidget(camera_group, 6)
        main_h_splitter.addWidget(right_main)

    # --- IMU UI ---
    def _setup_imu_display(self, grid):
        data_keys = [
            ("좌우 기울어진 각도 (Roll)", "roll", RollIndicator, "#2a9d8f"),
            ("앞뒤 기울어진 각도 (Pitch)", "pitch", PitchIndicator, "#e9c46a"),
            ("쳐다보는 방향 (Yaw)", "yaw", YawIndicator, "#f4a261"),
        ]

        row_idx = 0
        for col, (title, key, IndicatorClass, color) in enumerate(data_keys):
            
            # 0행: 제목 레이블
            t_label = QLabel(f"<b>{title}:</b>")
            grid.addWidget(t_label, 0, col*2, alignment=Qt.AlignmentFlag.AlignRight)
            
            # 1행: 값 레이블 (숫자 표시)
            v_label = QLabel("0.00°")
            v_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
            v_label.setStyleSheet(f"color: {color}; padding: 5px 0px 5px 5px;") 
            grid.addWidget(v_label, 0, col*2 + 1)
            self.imu_labels[key] = v_label # 텍스트 값은 여기에 저장
            
            # 2행: 시각화 뷰 (QGraphicsView)
            scene = QGraphicsScene()
            # 💡 커스텀 아이템 생성 및 장면에 추가
            indicator_item = IndicatorClass() 
            scene.addItem(indicator_item)
            
            view = QGraphicsView(scene)
            view.setFixedSize(130, 130) # 시각화 영역 크기 고정
            view.setSceneRect(indicator_item.boundingRect()) # 아이템 크기에 맞춰 장면 설정
            view.fitInView(indicator_item, Qt.AspectRatioMode.KeepAspectRatio) # 뷰에 맞춤
            view.setStyleSheet(f"border: 2px solid {color}; background-color: #0d1117;")
            
            # 뷰와 아이템을 딕셔너리에 저장하여 외부에서 접근 가능하도록 함
            self.imu_views[key] = view
            self.imu_items[key] = indicator_item
            
            # 2행: 시각화 뷰를 그리드에 추가 (총 2칸 차지)
            grid.addWidget(view, 1, col*2, 1, 2, alignment=Qt.AlignmentFlag.AlignCenter) 
            
            # 3행: 설명 레이블
            desc_label = QLabel("데이터 없음")
            desc_label.setFont(QFont("Arial", 10))
            desc_label.setStyleSheet(f"color: {color}; font-style: italic; padding: 2px; border: 1px solid {color}; border-radius: 3px;") 
            grid.addWidget(desc_label, 2, col*2, 1, 2, alignment=Qt.AlignmentFlag.AlignCenter) 
            self.imu_labels[f'{key}_desc'] = desc_label 
            
            grid.setRowStretch(1, 1) # 시각화 뷰 행에 공간 할당

    # --- MQTT 설정 ---
    def setup_mqtt(self):
        client = MqttClient(self)
        client.message_signal.connect(self.on_mqtt_message)
        client.connect_and_loop(MQTT_BROKER, MQTT_PORT)
        return client

    # --- 메시지 처리 ---
    def on_mqtt_message(self, topic, payload):
        # if topic == TOPIC_IMU:
        #     try:
        #         data = json.loads(payload)
        #         self.update_imu_ui(data)
        #     except json.JSONDecodeError:
        #         print(f"[IMU] JSON Error")

        # elif topic in [TOPIC_VIDEO_AD, TOPIC_CAM_AD]:
        #     self.update_camera_view(self.ad_pixmap_item, payload)

        # elif topic == TOPIC_CAM_PE:  # 낙상 영상
        #     self.update_camera_view(self.pe_pixmap_item, payload)
        
        # elif topic == TOPIC_LOGBOOK:  # 항해일지
        #     try:
        #         data = json.loads(payload)
        #         self.update_logbook_tab(data)
        #     except Exception as e:
        #         print(f"[LOGBOOK Error] {e}")

        # elif topic in [TOPIC_LOGS, TOPIC_PE_RAW, TOPIC_PE_ALERT, TOPIC_PE_RAW]: 
        #     try:
        #         log = json.loads(payload)
        #         self.update_log_ui(log)
        #     except json.JSONDecodeError:
        #         # JSON 형식이 아닌 일반 로그 (STT 등)도 처리할 수 있도록 보강
        #         self.update_log_ui({
        #             "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        #             "module": "SYS",
        #             "action": "RAW",
        #             "payload": payload
        #         })
        if topic in [TOPIC_VIDEO_AD, TOPIC_CAM_AD]:
            self.update_camera_view(self.ad_pixmap_item, payload)

        elif topic == TOPIC_CAM_PE:  # 낙상 영상
            self.update_camera_view(self.pe_pixmap_item, payload)
        
        elif topic == TOPIC_LOGBOOK:  # 항해일지
            try:
                data = json.loads(payload)
                self.update_logbook_tab(data)
            except Exception as e:
                print(f"[LOGBOOK Error] {e}")

        # 🚨 3. TOPIC_LOGS (project/log/RAW)에서 IMU 데이터 처리 로직을 추가합니다.
        # TOPIC_PE_RAW가 중복되어 있으니 하나로 정리하고 TOPIC_LOGS와 함께 묶습니다.
        elif topic in [TOPIC_LOGS, TOPIC_PE_RAW, TOPIC_PE_ALERT]: 
            try:
                log = json.loads(payload)
                
                # 💡 IMU 데이터라면 IMU UI도 업데이트
                # 🚨🚨🚨 이 조건문이 정확해야 합니다. 🚨🚨🚨
                if log.get('module') == "IMU" and log.get('action') == "RAW":
                    # log 자체가 IMU 데이터 페이로드이므로 바로 전달
                    self.update_imu_ui(log)
                    
                # 💡 모든 로그 데이터 (IMU 포함)를 시스템 로그 창에 출력
                self.update_log_ui(log) 
                
            except json.JSONDecodeError:
                # JSON 형식이 아닌 일반 로그 (STT 등)도 처리할 수 있도록 보강
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
                    self.imu_labels[key].setText(f"{val:.2f}°")

                    # 시각화 아이템 업데이트
                    if key in self.imu_items:
                        item = self.imu_items[key]
                        if key == 'roll':
                            item.set_roll(val)
                        elif key == 'pitch':
                            item.set_pitch(val)
                        elif key == 'yaw':
                            item.set_yaw(val)

                except Exception as e:
                    print(f"[IMU Error] {key}: {e}")
                    self.imu_labels[key].setText("ERR")
            
            # 설명 필드 업데이트 (새로 추가)
            desc_key = f'{key}_desc'
            if desc_key in self.imu_labels and desc_key in data:
                # 서버에서 가공한 직관적인 설명 텍스트를 바로 표시
                self.imu_labels[desc_key].setText(str(data[desc_key]))

    # --- 로그 UI 업데이트 ---
    def update_log_ui(self, log):
        """시스템 로그 탭에 사람이 읽기 좋은 형태로 출력"""
        try:
            ts = datetime.now().strftime("%H:%M:%S")
            module = log.get('module', 'UNKNOWN').upper()
            action = log.get('action', '').upper()
            level = log.get('level', '').upper()
            color = COLOR_MAP.get(module, COLOR_MAP["DEFAULT"])

            # --- payload 처리 ---
            msg_payload = log.get('payload', '')
            if isinstance(msg_payload, str):
                try:
                    msg_payload = json.loads(msg_payload)
                except Exception:
                    pass

            # 중첩 payload 제거
            if isinstance(msg_payload, dict) and "payload" in msg_payload:
                inner = msg_payload.get("payload")
                if isinstance(inner, dict):
                    msg_payload = inner

            # --- message 추출 ---
            msg = ""
            if isinstance(msg_payload, dict):
                msg = msg_payload.get('message', '')
            elif isinstance(msg_payload, list):
                msg = f"목록 {len(msg_payload)}건 수신"
            else:
                msg = str(msg_payload) or "상태 데이터 수신 완료."
            msg = " " + msg

            module_color = COLOR_MAP.get(module, COLOR_MAP["DEFAULT"])
            base_color = "#E6E6E6"  # 전체 텍스트 기본색

            if 'AD' in module:
                module_color = "#FF6600"
                module_text = "AD"
            elif 'PE' in module:
                module_color = "#9A71CF"
                module_text = "PE"
            elif 'STT' in module:
                module_color = "#06D6A0"
                module_text = "STT"
            elif 'LLM' in module:
                module_color = "#25DA0D"
                module_text = "LLM"
            elif 'IMU' in module:
                module_color = "#25ACD4"
                module_text = "IMU"
            else:
                module_color = "#A8A8A8"
                module_text = module

            if 'CRITICAL' in level or 'ALERT' in action:
                level_color = "#FF4C4C"
                level_text = "긴급"
            elif 'WARNING' in level:
                level_color = "#FFD166"
                level_text = "주의"
            elif 'INFO' in level or 'RAW' in action:
                level_color = "#FCFCFC"
                level_text = "정보"
            else:
                level_color = "#A8A8A8"
                level_text = "안전"

            # --- 최종 출력 ---
            formatted = (
                f"<pre style='color:{base_color}; font-family:monospace;'>"
                f"[{ts}]  "
                f"<span style='color:{module_color}; font-weight:bold;'>{module:<6}</span>"
                f"<span style='color:{level_color};'>[{level_text:^4}]</span>  "
                f"{msg}</pre><br>"
            )

            self.db_log_widget.insertHtml(formatted)
            self.db_log_widget.moveCursor(self.db_log_widget.textCursor().MoveOperation.End)

        except Exception as e:
            error_msg = f"<span style='color:red'>[LogUI Fatal Error] {e}</span><br>"
            self.db_log_widget.insertHtml(error_msg)
            print(f"[LogUI Error] {e}")
    
    def update_logbook_tab(self, data):
        """
        LOGBOOK 토픽 수신 시 항해일지 탭에 출력
        """
        try:
            entries = data.get("entries", [])
            if not entries:
                self.voyage_log_widget.setPlainText("최근 항해일지 데이터가 없습니다.")
                return

            text_lines = []
            for e in entries:
                text_lines.append(
                    f"[{e['log_dt']}] "
                    f"풍향: {e['wind_dir']} / 풍속: {e['wind_spd']} m/s / "
                    f"날씨: {e['weather']} / "
                    f"항로상태: {'ON' if e['on_route'] else 'OFF'}\n"
                    f"운항 메모: {e['on_notes']}\n"
                    f"특이사항: {e['ex_notes']}\n"
                    "-----------------------------------------"
                )

            self.voyage_log_widget.setPlainText("\n".join(text_lines))

        except Exception as e:
            print(f"[update_logbook_tab Error] {e}")
            self.voyage_log_widget.setPlainText(f"항해일지 데이터 표시 중 오류: {e}")


    # --- 카메라 업데이트 (QGraphicsView용) ---
    def update_camera_view(self, pixmap_item, base64_data):
        try:
            img_data = safe_b64decode(base64_data)
            qimg = QImage.fromData(img_data)
            if qimg.isNull():
                return

            pix = QPixmap.fromImage(qimg)
            pixmap_item.setPixmap(pix)

            # 🔹 장면 즉시 갱신
            scene = pixmap_item.scene()
            scene.update()

            # 🔹 화면 비율 맞춤 자동 스케일
            view = pixmap_item.scene().views()[0]
            view.fitInView(pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

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
