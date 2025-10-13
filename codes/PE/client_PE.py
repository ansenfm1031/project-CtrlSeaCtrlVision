import os
import cv2
import time
import torch
import argparse
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import paho.mqtt.client as mqtt
import json

# ============================================
# MQTT 설정 (서버와 동일해야 합니다)
# ============================================
MQTT_BROKER = "10.10.14.73" 
MQTT_PORT = 1883
TOPIC_BASE = "project/"
RAW_TOPIC = TOPIC_BASE + "VISION/RAW"
ALERT_TOPIC = TOPIC_BASE + "VISION/ALERT"

# ============================================
# 설정
# ============================================
DEBUG_MODE = False  # True로 하면 상세 로그

# 위험구역 설정 (절대 좌표)
USE_RATIO = False
DANGER_X_MIN = None
DANGER_X_MAX = 200
DANGER_Y_MIN = None
DANGER_Y_MAX = None

# 비율 방식 (USE_RATIO = True일 때만 사용)
DANGER_X_RATIO_MIN = None
DANGER_X_RATIO_MAX = 0.3
DANGER_Y_RATIO_MIN = None
DANGER_Y_RATIO_MAX = None

ZONE_WARNING_TIME = 3
ZONE_ALERT_TIME = 5
SHOW_DANGER_AREA = True
DANGER_AREA_COLOR = (0, 0, 255)

# 넘어짐 판단 설정
FALL_CONFIDENCE_THRESHOLD = 0.65
FALL_FRAMES = 3 # 이 프레임 수만큼 연속되어야 낙상 확정

# MQTT RAW 데이터 발행 주기 (프레임)
RAW_PUBLISH_INTERVAL = 15

# ============================================
# MoveNet 포즈 추정 모델
# ============================================

class MoveNetPose:
    """MoveNet Thunder - 라즈베리파이5 최적화"""
    
    def __init__(self, model_type='thunder', device='cpu'):
        print(f"Loading MoveNet {model_type}...")
        self.model_type = model_type
        self.input_size = 256 if model_type == 'thunder' else 192
        self.use_tflite = False
        
        # TFLite 파일을 우선 시도
        model_path = f'movenet_{model_type}.tflite'
        if os.path.exists(model_path):
            print(f"Found local TFLite model: {model_path}")
            try:
                # TFLite 모델 로드
                self.interpreter = tf.lite.Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.use_tflite = True
                print(f"✅ Loaded TFLite model successfully!")
                return
            except Exception as e:
                print(f"❌ TFLite loading failed: {e}")
                
        # TFLite 로드 실패 시, 최소한의 기능은 유지 (TFLite 로드가 성공해야 작동)
        if not self.use_tflite:
             print("⚠️ MoveNet TFLite model not found or failed to load. Pose estimation disabled.")

    
    def predict(self, frame, bboxes, scores=None):
        """
        프레임 안의 여러 사람에 대한 자세를 예측합니다.
        """
        if bboxes is None or len(bboxes) == 0:
            return []

        poses = []
        # TFLite Interpreter가 로드되었는지 확인
        if not hasattr(self, 'interpreter') or not self.use_tflite:
            return []

        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
                
            crop_img = frame[y1:y2, x1:x2]
            crop_height, crop_width, _ = crop_img.shape
            
            input_image = tf.convert_to_tensor(crop_img, dtype=tf.uint8)
            
            # 모델 입력에 맞게 리사이즈 및 패딩
            resized_img = tf.image.resize_with_pad(input_image, self.input_size, self.input_size)
            input_img_tensor = tf.cast(resized_img, dtype=tf.uint8)
            input_batch = tf.expand_dims(input_img_tensor, axis=0)
            
            # TFLite Inference
            input_details = self.interpreter.get_input_details()
            self.interpreter.set_tensor(input_details[0]['index'], input_batch)
            self.interpreter.invoke()
            
            output_details = self.interpreter.get_output_details()
            keypoints_with_scores = np.squeeze(self.interpreter.get_tensor(output_details[0]['index']))
            
            # 좌표 변환 로직 (Movenet의 출력 좌표를 원본 이미지 좌표로 변환)
            if crop_height > crop_width:
                scale_factor = self.input_size / crop_height
                new_width = crop_width * scale_factor
                padd_x = (self.input_size - new_width) / 2
                padd_y = 0
                scale_x = new_width / crop_width
                scale_y = self.input_size / crop_height
            else:
                scale_factor = self.input_size / crop_width
                new_height = crop_height * scale_factor
                padd_x = 0
                padd_y = (self.input_size - new_height) / 2
                scale_x = self.input_size / crop_width
                scale_y = new_height / crop_height

            keypoints = np.zeros((17, 3), dtype=np.float32)
            
            norm_y = keypoints_with_scores[:, 0]
            norm_x = keypoints_with_scores[:, 1]
            
            # 변환된 좌표를 원본 bbox 위치에 맞게 조정
            keypoints[:, 0] = ((norm_x * self.input_size - padd_x) / scale_x) + x1
            keypoints[:, 1] = ((norm_y * self.input_size - padd_y) / scale_y) + y1
            keypoints[:, 2] = keypoints_with_scores[:, 2] # Confidence score
            
            proposal_score = float(np.mean(keypoints[:, 2]))
            
            poses.append({
                'keypoints': keypoints,  # [17, 3] 형태로 통일
                'proposal_score': proposal_score,
                'bbox': bbox
            })
        return poses

# ============================================
# 룰 기반 낙상 감지
# ============================================
def estimate_motion(prev_kp, curr_kp):
    """평균 키포인트 이동량 (걷기 인식용)"""
    if prev_kp is None or len(prev_kp) == 0 or prev_kp.shape != curr_kp.shape:
        return 0.0
    
    # 신뢰도 체크
    valid = (prev_kp[:, 2] > 0.2) & (curr_kp[:, 2] > 0.2)
    if np.sum(valid) < 5:
        return 0.0
    
    # 유효한 키포인트만 사용하여 이동량 계산
    diffs = np.linalg.norm(curr_kp[valid, :2] - prev_kp[valid, :2], axis=1)
    motion = float(np.mean(diffs))
    
    if DEBUG_MODE:
        print(f"    Motion calculation: {np.sum(valid)} valid points, motion={motion:.2f}")
    
    return motion

def calculate_body_angle(keypoints):
    """몸의 기울기 각도 (0° = 완전 수직, 90° = 완전 수평)"""
    if len(keypoints) < 13:
        return None

    valid_shoulder = []
    valid_hip = []

    # 어깨 중심점 계산
    if keypoints[5][2] > 0.2: # Left Shoulder
        valid_shoulder.append(keypoints[5][:2])
    if keypoints[6][2] > 0.2: # Right Shoulder
        valid_shoulder.append(keypoints[6][:2])
        
    # 골반 중심점 계산
    if keypoints[11][2] > 0.2: # Left Hip
        valid_hip.append(keypoints[11][:2])
    if keypoints[12][2] > 0.2: # Right Hip
        valid_hip.append(keypoints[12][:2])

    if len(valid_shoulder) == 0 or len(valid_hip) == 0:
        return None

    shoulder_center = np.mean(valid_shoulder, axis=0)
    hip_center = np.mean(valid_hip, axis=0)

    dx = hip_center[0] - shoulder_center[0]
    dy = shoulder_center[1] - hip_center[1] # y축은 아래로 갈수록 커지므로 어깨 y - 골반 y로 계산
    
    # 수직(y축)과의 각도 계산 (0도: 수직, 90도: 수평)
    angle = np.degrees(np.arctan2(abs(dx), abs(dy)))
    return angle


def get_body_aspect_ratio(keypoints):
    """몸의 가로/세로 비율"""
    valid_points = keypoints[keypoints[:, 2] > 0.15]
    
    if len(valid_points) < 3:
        return None
    
    x_coords = valid_points[:, 0]
    y_coords = valid_points[:, 1]
    
    width = x_coords.max() - x_coords.min()
    height = y_coords.max() - y_coords.min()
    
    if height < 5:
        return None
    
    return width / height


def detect_fall_rule_based(keypoints, prev_keypoints=None):
    """향상된 룰 기반 상태 인식"""
    
    angle = calculate_body_angle(keypoints)
    ratio = get_body_aspect_ratio(keypoints)
    motion = estimate_motion(prev_keypoints, keypoints)

    conf = float(np.mean(keypoints[:, 2]))

    if angle is None or ratio is None:
        return 'Unknown', conf, {}
    
    details = {"angle": f"{angle:.1f}", "ratio": f"{ratio:.2f}", "motion": f"{motion:.1f}"}
    
    if DEBUG_MODE:
        print(f"  DEBUG >>> Angle: {angle:.1f}°, Ratio: {ratio:.2f}, Motion: {motion:.1f}")

    # 1. 눕거나 넘어진 상태 (수평에 가까움, angle > 50)
    if angle > 50:
        if ratio > 1.0:
            if motion < 7:
                return 'Lying Down', conf * 0.95, details
            else:
                # 높은 이동량 + 수평 자세 = 넘어지는 중 (Fall Down)
                return 'Fall Down', conf * 1.0, details
        else:
            return 'Unknown', conf, details

    # 2. 앉은 상태 (어느 정도 기울어짐, 10 <= angle < 50)
    elif 10 <= angle < 50 and ratio > 0.5:
        return 'Sitting', conf * 0.9, details
    
    # 3. 서 있거나 걷는 상태 (거의 수직, angle < 10)
    elif angle < 10:
        if motion > 2:
            return 'Walking', conf, details
        else:
            return 'Standing', conf, details
    
    else: 
        return 'Unknown', conf, details


# ============================================
# 위험구역 함수
# ============================================

def get_location_details(bbox):
    """바운딩 박스를 이용해 중심 x, 하단 y 좌표를 반환"""
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    bottom_y = y2
    return int(center_x), int(bottom_y)

def is_in_danger_zone(bbox, frame_width, frame_height):
    """바운딩박스가 위험구역에 있는지 확인"""
    center_x, bottom_y = get_location_details(bbox)
    
    if USE_RATIO:
        x_min = int(frame_width * DANGER_X_RATIO_MIN) if DANGER_X_RATIO_MIN is not None else 0
        x_max = int(frame_width * DANGER_X_RATIO_MAX) if DANGER_X_RATIO_MAX is not None else frame_width
        y_min = int(frame_height * DANGER_Y_RATIO_MIN) if DANGER_Y_RATIO_MIN is not None else 0
        y_max = int(frame_height * DANGER_Y_RATIO_MAX) if DANGER_Y_RATIO_MAX is not None else frame_height
    else:
        x_min = DANGER_X_MIN if DANGER_X_MIN is not None else 0
        x_max = DANGER_X_MAX if DANGER_X_MAX is not None else frame_width
        y_min = DANGER_Y_MIN if DANGER_Y_MIN is not None else 0
        y_max = DANGER_Y_MAX if DANGER_Y_MAX is not None else frame_height
    
    in_danger_x = True
    if x_min is not None and center_x < x_min:
        in_danger_x = False
    if x_max is not None and center_x > x_max:
        in_danger_x = False
    
    in_danger_y = True
    if y_min is not None and bottom_y < y_min:
        in_danger_y = False
    if y_max is not None and bottom_y > y_max:
        in_danger_y = False
    
    return in_danger_x and in_danger_y


def draw_danger_area(frame):
    """위험구역 시각화"""
    if not SHOW_DANGER_AREA:
        return frame
    
    h, w = frame.shape[:2]
    overlay = frame.copy()
    
    if USE_RATIO:
        x_min = int(w * DANGER_X_RATIO_MIN) if DANGER_X_RATIO_MIN is not None and DANGER_X_RATIO_MIN >= 0 else 0
        x_max = int(w * DANGER_X_RATIO_MAX) if DANGER_X_RATIO_MAX is not None and DANGER_X_RATIO_MAX <= 1 else w
        y_min = int(h * DANGER_Y_RATIO_MIN) if DANGER_Y_RATIO_MIN is not None and DANGER_Y_RATIO_MIN >= 0 else 0
        y_max = int(h * DANGER_Y_RATIO_MAX) if DANGER_Y_RATIO_MAX is not None and DANGER_Y_RATIO_MAX <= 1 else h
    else:
        x_min = DANGER_X_MIN if DANGER_X_MIN is not None else 0
        x_max = DANGER_X_MAX if DANGER_X_MAX is not None else w
        y_min = DANGER_Y_MIN if DANGER_Y_MIN is not None else 0
        y_max = DANGER_Y_MAX if DANGER_Y_MAX is not None else h

    # x_min, x_max, y_min, y_max를 프레임 크기 내로 조정
    x_min = max(0, min(x_min, w))
    x_max = max(0, min(x_max, w))
    y_min = max(0, min(y_min, h))
    y_max = max(0, min(y_max, h))
    
    cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), DANGER_AREA_COLOR, -1)
    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), DANGER_AREA_COLOR, 3)
    cv2.putText(frame, "DANGER ZONE", (x_min + 10, y_min + 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, DANGER_AREA_COLOR, 2)
    
    return frame


def draw_zone_warnings(frame, zone_warnings):
    """여러 위험구역 경고를 순차적으로 표시"""
    y_offset = 50
    
    for i, (track_id, elapsed_time, status) in enumerate(zone_warnings):
        if status == 'warning':
            color = (0, 255, 255)
            text = f"WARNING! Worker #{track_id} in danger zone {elapsed_time:.1f}s"
        elif status == 'danger':
            color = (0, 0, 255)
            text = f"DANGER! Worker #{track_id} in danger zone {elapsed_time:.1f}s"
        else:
            continue
        
        y_pos = y_offset + (i * 35)
        cv2.rectangle(frame, (10, y_pos), (650, y_pos + 30), color, -1)
        cv2.putText(frame, text, (15, y_pos + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return frame


# ============================================
# 간단한 트래커 (IoU 기반)
# ============================================

class SimpleTracker:
    """간단한 IoU 기반 트래커"""
    
    def __init__(self, max_age=50):
        self.tracks = {}
        self.next_id = 1
        self.max_age = max_age
    
    def update(self, detections):
        """
        detections: List of {'bbox': [x1,y1,x2,y2], 'keypoints': array, 'score': float}
        """
        # 기존 트랙 나이 증가
        for track_id in list(self.tracks.keys()):
            self.tracks[track_id]['age'] += 1
            if self.tracks[track_id]['age'] > self.max_age:
                del self.tracks[track_id]
        
        # IoU 매칭
        for det in detections:
            best_iou = 0
            best_id = None
            
            for track_id, track in self.tracks.items():
                iou = self._calculate_iou(det['bbox'], track['bbox'])
                if iou > best_iou and iou > 0.3:
                    best_iou = iou
                    best_id = track_id
            
            if best_id is not None:
                # 기존 트랙 업데이트
                self.tracks[best_id]['bbox'] = det['bbox']
                self.tracks[best_id]['keypoints'].append(det['keypoints'])
                if len(self.tracks[best_id]['keypoints']) > 30:
                    self.tracks[best_id]['keypoints'].pop(0)
                self.tracks[best_id]['age'] = 0
            else:
                # 새 트랙 생성
                self.tracks[self.next_id] = {
                    'bbox': det['bbox'],
                    'keypoints': [det['keypoints']],
                    'age': 0
                }
                self.next_id += 1
        
        # 현재 활성 트랙 ID 리스트 반환
        return list(self.tracks.keys())
    
    def _calculate_iou(self, box1, box2):
        """IoU 계산"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-6)


# ============================================
# YOLOv8 검출기
# ============================================

class YOLOv8_Detector:
    def __init__(self, model_name='yolov8n.pt', conf_thres=0.65, device='cpu'):
        self.model = YOLO(model_name)
        self.conf_thres = conf_thres
        self.device = device
    
    def detect(self, frame):
        """사람 검출"""
        results = self.model.predict(
            frame,
            conf=self.conf_thres,
            classes=[0],  # person only
            device=self.device,
            verbose=False
        )
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            return [], []
        
        boxes = results[0].boxes
        bboxes = []
        scores = []
        
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            
            bboxes.append([x1, y1, x2, y2])
            scores.append(conf)
        
        return np.array(bboxes, dtype=np.float32), np.array(scores, dtype=np.float32)


# ============================================
# 스켈레톤 그리기
# ============================================

SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # 얼굴
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 팔
    (5, 11), (6, 12), (11, 12),  # 몸통
    (11, 13), (13, 15), (12, 14), (14, 16)  # 다리
]

def draw_skeleton(frame, keypoints):
    """스켈레톤 그리기"""
    # 연결선 그리기
    for start_idx, end_idx in SKELETON_CONNECTIONS:
        if keypoints[start_idx][2] > 0.3 and keypoints[end_idx][2] > 0.3:
            start = tuple(keypoints[start_idx][:2].astype(int))
            end = tuple(keypoints[end_idx][:2].astype(int))
            cv2.line(frame, start, end, (0, 255, 0), 2)
    
    # 키포인트 점 그리기
    for i, kp in enumerate(keypoints):
        if kp[2] > 0.3:
            cv2.circle(frame, tuple(kp[:2].astype(int)), 4, (0, 0, 255), -1)
    
    return frame

# ============================================
# MQTT 발행 함수
# ============================================
def on_connect(client, userdata, flags, rc):
    """MQTT 연결 콜백"""
    if rc == 0:
        print("✅ MQTT Connected successfully.")
    else:
        print(f"❌ MQTT Connection failed with code {rc}")

def publish_mqtt_message(client, topic, payload):
    """JSON 메시지를 MQTT로 발행"""
    try:
        json_payload = json.dumps(payload, ensure_ascii=False)
        client.publish(topic, json_payload, qos=0)
        if DEBUG_MODE:
            print(f"[MQTT SEND] {topic}: {json_payload}")
    except Exception as e:
        print(f"[MQTT ERROR] Failed to publish to {topic}: {e}")


# ============================================
# 메인
# ============================================

def main():
    parser = argparse.ArgumentParser(description='Raspberry Pi 5 Optimized Fall Detection')
    parser.add_argument('--camera', type=str, default='0', help='Camera source or video path')
    parser.add_argument('--device', type=str, default='cpu', help='cpu only for RPi5')
    parser.add_argument('--model', type=str, default='thunder', choices=['thunder', 'lightning'])
    parser.add_argument('--save_out', type=str, default='', help='Save output video')
    parser.add_argument('--show_skeleton', action='store_true', help='Show skeleton')
    args = parser.parse_args()
    
    print("="*60)
    print("Raspberry Pi 5 Optimized Fall Detection System (MQTT PE Client)")
    print("- Pose: MoveNet " + args.model.title())
    print("- Detection: Rule-based")
    print("- Device: CPU")
    print(f"- MQTT Broker: {MQTT_BROKER}:{MQTT_PORT}")
    print("="*60)
    
    # 1. 모델 로드
    print("\n1️⃣ Loading models...")
    detector = YOLOv8_Detector(model_name='yolov8n.pt', conf_thres=0.5, device='cpu')
    pose_model = MoveNetPose(model_type=args.model)
    
    # 2. 트래커
    tracker = SimpleTracker(max_age=50)
    
    # 3. MQTT 클라이언트 초기화
    mqtt_client = mqtt.Client()
    mqtt_client.on_connect = on_connect
    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start()
    except Exception as e:
        print(f"❌ Failed to connect to MQTT broker: {e}")
        # 계속 진행 (비디오 처리는 가능하도록)
        
    # 4. 카메라
    print("\n2️⃣ Opening camera...")
    cam_source = args.camera
    if cam_source.isdigit():
        cap = cv2.VideoCapture(int(cam_source))
    else:
        cap = cv2.VideoCapture(cam_source)
    
    if not cap.isOpened():
        print("❌ Cannot open camera! Check source path or index.")
        mqtt_client.loop_stop()
        return
    
    print("✅ Camera opened")
    
    # 비디오 저장
    writer = None
    if args.save_out:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.save_out, fourcc, fps, (width, height))
    
    # 상태 변수
    fall_counters = {}
    zone_timers = {}
    
    # Alert 전송 상태 (중복 전송 방지)
    alert_sent_fall = {}
    alert_sent_zone = {}

    fps_time = time.time()
    frame_count = 0
    
    print("\n3️⃣ Starting detection... (Press 'q' to quit)\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of stream or video.")
            break
        
        frame_count += 1
        h, w = frame.shape[:2]
        
        # 위험구역 그리기
        frame = draw_danger_area(frame)
        
        # 사람 검출
        bboxes, scores = detector.detect(frame)
        
        # 포즈 추정 및 트래킹 준비
        detections = []
        if len(bboxes) > 0:
            poses = pose_model.predict(frame, bboxes, scores)
            for pose, bbox in zip(poses, bboxes):
                detections.append({
                    'bbox': bbox,
                    'keypoints': pose['keypoints'],
                    'score': pose['proposal_score']
                })
        
        # 트래킹
        current_tracks = tracker.update(detections)
        current_time = time.time()
        
        # RAW 데이터용 리스트 및 위험구역 경고 리스트
        raw_detections_list = []
        zone_warnings = []
        is_person_detected = len(current_tracks) > 0
        
        # 각 트랙 처리
        for track_id in current_tracks:
            track = tracker.tracks[track_id]
            bbox = track['bbox']
            keypoints_list = track['keypoints']
            
            if len(keypoints_list) < 1:
                continue
            
            x1, y1, x2, y2 = bbox.astype(int)
            center_x, bottom_y = get_location_details(bbox)
            
            # 1. 위험구역 체크 (ZONE)
            in_zone = is_in_danger_zone(bbox, w, h)
            
            # A. ZONE TIMER 업데이트
            if in_zone:
                if track_id not in zone_timers:
                    zone_timers[track_id] = current_time
                    alert_sent_zone[track_id] = False # 새로 진입 시 알림 상태 초기화
                
                elapsed = current_time - zone_timers[track_id]
                
                # B. ZONE ALERT (LEVEL 5)
                if elapsed >= ZONE_ALERT_TIME and not alert_sent_zone.get(track_id, False):
                    # ALERT 메시지 발행
                    alert_payload = {
                        "module": "VISION",
                        "message": f"DANGER ZONE ALERT: Worker #{track_id} has been in the high-risk area for {elapsed:.1f}s. Location: ({center_x}, {bottom_y})",
                        "level": 5,
                        "details": [{"track_id": track_id, "object_type": "Person", "action": "InDangerZone", "location": f"Center: ({center_x}, {bottom_y})"}]
                    }
                    publish_mqtt_message(mqtt_client, ALERT_TOPIC, alert_payload)
                    alert_sent_zone[track_id] = True
                    print(f"[ALERT SENT] Zone alert for Worker #{track_id}")
                    
                    zone_warnings.append((track_id, elapsed, 'danger'))
                elif elapsed >= ZONE_WARNING_TIME:
                    zone_warnings.append((track_id, elapsed, 'warning'))
            else:
                # C. ZONE LEAVE
                if track_id in zone_timers:
                    del zone_timers[track_id]
                    alert_sent_zone[track_id] = False # 존을 나갔으므로 알림 상태 초기화

            # 2. 낙상 감지 (FALL)
            current_kp = keypoints_list[-1]
            prev_kp = keypoints_list[-2] if len(keypoints_list) >= 2 else None
            
            action_name, confidence, details = detect_fall_rule_based(current_kp, prev_kp)
            
            # A. FALL COUNTER 업데이트
            if track_id not in fall_counters:
                fall_counters[track_id] = 0
                alert_sent_fall[track_id] = False
            
            is_fall_action = action_name in ['Fall Down', 'Lying Down'] and confidence >= FALL_CONFIDENCE_THRESHOLD
            
            if is_fall_action:
                fall_counters[track_id] += 1
            else:
                fall_counters[track_id] = 0
                alert_sent_fall[track_id] = False # 정상 상태로 돌아오면 알림 상태 초기화
            
            # B. FALL ALERT (LEVEL 5)
            if fall_counters[track_id] >= FALL_FRAMES and not alert_sent_fall.get(track_id, False):
                # ALERT 메시지 발행
                alert_payload = {
                    "module": "VISION",
                    "message": f"CRITICAL FALL DETECTED: Worker #{track_id} is {action_name.lower()} at location ({center_x}, {bottom_y}). Immediate assistance required.",
                    "level": 5,
                    "details": [{"track_id": track_id, "object_type": "Person", "action": action_name, "confidence": float(confidence), "location": f"Center: ({center_x}, {bottom_y})"}]
                }
                publish_mqtt_message(mqtt_client, ALERT_TOPIC, alert_payload)
                alert_sent_fall[track_id] = True
                print(f"[ALERT SENT] Fall alert for Worker #{track_id}")

            # 3. 시각화 및 RAW 데이터 기록
            
            # 상태 및 색상 결정
            if fall_counters[track_id] >= FALL_FRAMES:
                action_display = f'FALL: {confidence*100:.1f}%'
                clr = (0, 0, 255)  # 🔴 확정 낙상
            elif action_name in ['Fall Down', 'Lying Down']:
                action_display = f'{action_name}: {confidence*100:.1f}%'
                clr = (0, 0, 255)  # 🔴 일시적 낙상도 빨강으로
            elif action_name == 'Standing':
                action_display = f'{action_name}: {confidence*100:.1f}%'
                clr = (0, 255, 0)  # 🟢 정상
            elif action_name == 'Sitting':
                action_display = f'{action_name}: {confidence*100:.1f}%'
                clr = (0, 255, 255)  # 🟡 앉음
            else:
                action_display = f'{action_name}: {confidence*100:.1f}%'
                clr = (255, 255, 255)  # ⚪ Unknown
            
            # 시각화
            cv2.rectangle(frame, (x1, y1), (x2, y2), clr, 2)
            cv2.putText(frame, str(track_id), (center_x, y1 - 35),
                       cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(frame, action_display, (x1 + 5, y1 + 20),
                       cv2.FONT_HERSHEY_COMPLEX, 0.5, clr, 2)
            cv2.putText(frame, f'Loc:({center_x},{bottom_y})', (x1 + 5, y1 + 40),
                       cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
            
            # 스켈레톤
            if args.show_skeleton:
                frame = draw_skeleton(frame, current_kp)
            
            # RAW 데이터 리스트에 추가
            raw_detections_list.append({
                "track_id": int(track_id),
                "object_type": "Person",
                "action": action_name,
                "confidence": float(confidence),
                "x_center": center_x,
                "y_bottom": bottom_y,
                "in_danger_zone": in_zone
            })
        
        # 4. RAW 데이터 발행 (주기적으로)
        if frame_count % RAW_PUBLISH_INTERVAL == 0 and is_person_detected:
            raw_payload = {
                "module": "VISION",
                "detections": raw_detections_list,
                "person_detected": is_person_detected
            }
            publish_mqtt_message(mqtt_client, RAW_TOPIC, raw_payload)


        # 위험구역 경고 표시 (여러 사람 동시 표시)
        if zone_warnings:
            frame = draw_zone_warnings(frame, zone_warnings)
        
        # FPS 표시
        fps = 1.0 / (time.time() - fps_time + 1e-6)
        fps_time = time.time()
        cv2.putText(frame, f'Frame: {frame_count}, FPS: {fps:.1f}',
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 화면 표시
        cv2.imshow('Fall Detection + Danger Zone (MQTT PE Client)', frame)
        
        # 비디오 저장
        if writer:
            writer.write(frame)
        
        # 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 정리
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("Program terminated. Closing MQTT connection.")
    mqtt_client.loop_stop()
    mqtt_client.disconnect()
    print("="*60)


if __name__ == '__main__':
    main()

