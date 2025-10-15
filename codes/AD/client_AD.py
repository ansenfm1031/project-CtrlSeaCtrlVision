import paho.mqtt.client as mqtt
import cv2
import numpy as np
from openvino.runtime import Core
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
import os
import time
import sys
import json 
from datetime import datetime, timezone
# ====================================================
# 0. 고정 카메라 할당을 위한 모듈 임포트 (수정된 부분 1/2)
# camera_init_robust.py 파일이 이 스크립트와 같은 경로에 있어야 합니다.
# ====================================================
# find_camera_by_vid_pid 함수를 임포트합니다.
from camera_init_robust import find_camera_by_vid_pid 

# ====================================================
# 1. 환경 설정 및 상수 정의
# ====================================================

# MQTT 설정
BROKER = "10.10.14.73"
PORT = 1883
TOPIC_BASE = "project/vision" # 토픽 접두사

# AD 모듈 명확히 지정 및 토픽 분리
AD_MODULE = "AD"
RAW_TOPIC = TOPIC_BASE + "/" + AD_MODULE + "/RAW"
ALERT_TOPIC = TOPIC_BASE + "/" + AD_MODULE + "/ALERT" # 경고 토픽도 AD 전용으로 분리
def now_str():
    """ISO 8601 형식의 현재 UTC 시각을 반환합니다."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# =======================
# 2. 모델 경로 설정 및 유효성 검사
# (모든 모델 파일은 이 스크립트와 같은 디렉토리에 있어야 합니다.)
# =======================
det_xml = "models/Detection.xml"
det_bin = "models/Detection.bin"

cls_xml = "models/Classification.xml"
cls_bin = "models/Classification.bin"

DEPLOYMENT_FILE = "models/deployed_obstacle_detector.pt"

ALL_MODEL_PATHS = [det_xml, det_bin, cls_xml, cls_bin, DEPLOYMENT_FILE]
for path in ALL_MODEL_PATHS:
    if not os.path.exists(path):
        print(f"[{now_str()}] ❌ CRITICAL: 모델 파일을 찾을 수 없습니다: {path}")
        sys.exit(1) # 모델 파일이 없으면 즉시 종료

# =======================
# 3. 전역 객체 및 상수
# =======================
ie = Core()
det_compiled = None
det_input_layer = None
det_output_layer = None
cls_compiled = None
cls_input_layer = None
cls_output_layer = None
cls_h, cls_w = 0, 0
cap = None
deployed_model = None

class_names = ["Buoy", "Reef", "Island", "Ship", "Lighthouse"]
last_frame_boxes = [] # NMS/스무딩을 위한 이전 프레임 박스
OPTIMAL_THRESHOLD = 0.7 # Anomaly Detection 임계값
MODEL_INPUT_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

inference_transforms = transforms.Compose([
    transforms.Resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])


# =======================
# 4. 유틸리티 함수 (NMS, IoU, Preprocessing)
# (기존 코드와 동일)
# =======================

def nms(boxes, scores, score_threshold=0.5, iou_threshold=0.5):
    """Non-Maximum Suppression"""
    if not boxes: return [], []
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold, iou_threshold)
    if len(indices) > 0:
        indices = indices.flatten()
        return [boxes[i] for i in indices], [scores[i] for i in indices]
    return [], []

def iou(box1, box2):
    """Intersection over Union 계산"""
    # box1, box2: [x, y, w, h]
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union_area = w1 * h1 + w2 * h2 - inter_area
    return inter_area / union_area if union_area > 0 else 0

def enhance_low_light(image):
    """CLAHE를 이용한 저조도 개선"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_img

def dehaze(image):
    """Dark Channel Prior를 이용한 Dehazing"""
    min_channel = np.min(image, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dark_channel = cv2.erode(min_channel, kernel)
    A = np.max(dark_channel)
    t = 1 - 0.95 * dark_channel / A
    t = np.clip(t, 0.1, 1)
    J = np.empty_like(image, dtype=np.float32)
    for c in range(3):
        J[:,:,c] = (image[:,:,c].astype(np.float32) - A) / t + A
    J = np.clip(J, 0, 255).astype(np.uint8)
    return J

# =======================
# 5. 초기화 함수 (모든 모델 로드 및 카메라 초기화)
# =======================

def initialize_vision():
    """OpenVINO 모델, PyTorch 모델 및 카메라를 초기화합니다."""
    global det_compiled, det_input_layer, det_output_layer
    global cls_compiled, cls_input_layer, cls_output_layer
    global cls_h, cls_w, cap, deployed_model
    
    try:
        # OpenVINO 모델 로드 (생략하지 않음, 기존 코드와 동일)
        det_model = ie.read_model(det_xml, det_bin)
        det_compiled = ie.compile_model(det_model, "CPU")
        det_input_layer = det_compiled.input(0)
        det_output_layer = det_compiled.output(0)
        print(f"[{now_str()}] ✅ OpenVINO Detection 모델 로드 완료.")

        cls_model = ie.read_model(cls_xml, cls_bin)
        cls_compiled = ie.compile_model(cls_model, "CPU")
        cls_input_layer = cls_compiled.input(0)
        cls_output_layer = cls_compiled.output(0)
        _, _, cls_h, cls_w = cls_input_layer.shape
        print(f"[{now_str()}] ✅ OpenVINO Classification 모델 로드 완료.")

        deployed_model = torch.jit.load(DEPLOYMENT_FILE, map_location='cpu')
        deployed_model.eval()
        print(f"[{now_str()}] ✅ PyTorch Anomaly 모델 로드 완료.")

        # ============================================================
        # 🚨 카메라 초기화 로직 수정 (수정된 부분 2/2) 🚨
        # find_camera_by_vid_pid를 사용하여 AD 카메라의 고정 인덱스를 찾습니다.
        # ============================================================
        print(f"[{now_str()}] INFO System :: AD 카메라 고유 ID 기반 인덱스 검색 중...")
        
        # AD 인덱스만 추출하고, PE 인덱스는 무시합니다.
        AD_CAMERA_INDEX, _ = find_camera_by_vid_pid()
        
        if AD_CAMERA_INDEX == -1:
            raise RuntimeError("AD 카메라 (VID:PID)를 찾을 수 없습니다. 연결 상태를 확인하거나 camera_init_robust.py의 설정을 확인하세요.")

        print(f"[{now_str()}] ✅ AD 카메라 고정 인덱스 확보: {AD_CAMERA_INDEX}")
        
        # 확보된 인덱스로 카메라를 엽니다.
        cap = cv2.VideoCapture(AD_CAMERA_INDEX)

        if cap.isOpened():
            # 해상도 및 속성 설정
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            print(f"[{now_str()}] ✅ 웹캠 열기 성공: 인덱스 {AD_CAMERA_INDEX}")
        else:
             # cv2.VideoCapture가 인덱스에 실패하면 오류 발생
            raise RuntimeError(f"웹캠 인덱스 {AD_CAMERA_INDEX}를 열 수 없습니다.")
        
    except Exception as e:
        print(f"[{now_str()}] ❌ CRITICAL: 초기화 실패 - {e}")
        sys.exit(1)


# =======================
# 6. 메인 추론 및 발행 함수
# (기존 코드와 동일)
# =======================

def run_inference_and_publish(client):
    """
    1. 이미지 캡처 및 전처리 (저조도 개선, Dehazing)
    2. OpenVINO Detection/Classification
    3. PyTorch Anomaly Detection
    4. MQTT로 결과 발행
    """
    global last_frame_boxes
    
    start_time = time.time() # 시연용: FPS 측정 시작
    
    # 1. 프레임 캡처
    ret, frame = cap.read()
    if not ret:
        print(f"[{now_str()}] ❌ ERROR: 프레임 캡처 실패. 카메라 연결을 확인하세요.")
        time.sleep(0.1)
        return

    # --------------------------
    # 1-1) 전처리: 저조도 개선 및 Dehazing
    # --------------------------
    enhanced = enhance_low_light(frame)
    dehazed = dehaze(enhanced)

    # --------------------------
    # 2) OpenVINO Detection (장애물 감지)
    # --------------------------
    resized = cv2.resize(dehazed, (640, 640))
    # OpenVINO 입력 형태: BxCxHxW
    input_image = np.expand_dims(resized.transpose(2, 0, 1), 0).astype(np.float32)
    det_results = det_compiled([input_image])[det_output_layer][0]

    boxes, scores = [], []
    for det in det_results:
        # OpenVINO 출력 포맷에 따라 (x_min, y_min, x_max, y_max, conf)
        x_min, y_min, x_max, y_max, conf = det
        if conf > 0.5:
            # 원본 이미지 크기로 좌표 복원
            x_min = int(x_min / 640 * dehazed.shape[1])
            y_min = int(y_min / 640 * dehazed.shape[0])
            x_max = int(x_max / 640 * dehazed.shape[1])
            y_max = int(y_max / 640 * dehazed.shape[0])
            # NMS를 위해 [x, y, w, h] 형태로 저장
            boxes.append([x_min, y_min, x_max - x_min, y_max - y_min]) 
            scores.append(float(conf))

    filtered_boxes, _ = nms(boxes, scores)

    # --------------------------
    # 3) Frame smoothing (NMS 기반)
    # --------------------------
    smoothed_boxes = []
    # 이전 프레임과 IoU 0.7 이상인 박스는 이전 박스 위치를 사용 (흔들림 방지)
    for box in filtered_boxes:
        matched = False
        for prev_box in last_frame_boxes:
            if iou(box, prev_box) > 0.7:
                smoothed_boxes.append(prev_box)
                matched = True
                break
        if not matched:
            smoothed_boxes.append(box)
    last_frame_boxes = smoothed_boxes.copy() # 다음 프레임을 위해 저장

    # --------------------------
    # 4) Classification & Anomaly Check
    # --------------------------
    detections = []
    anomaly_detected = False
    critical_ship_detected = False # 🚨 CRITICAL 충돌 위험을 유발할 수 있는 배 감지
    
    for (x, y, w, h) in smoothed_boxes:
        # Classification을 위한 영역 추출
        crop = dehazed[max(0, y-5):min(dehazed.shape[0], y+h+5), max(0, x-5):min(dehazed.shape[1], x+w+5)]
        if crop.size == 0: continue

        # Classification (OpenVINO)
        cls_resized = cv2.resize(crop, (cls_w, cls_h))
        cls_resized = cv2.cvtColor(cls_resized, cv2.COLOR_BGR2RGB)
        cls_input = np.expand_dims(cls_resized.transpose(2, 0, 1), 0).astype(np.float32) / 255.0

        cls_result = cls_compiled([cls_input])[cls_output_layer]
        class_id = int(np.argmax(cls_result))
        score_cls = float(np.max(cls_result))
        label_name = class_names[class_id]
        
        # 🚨 일반 'Ship'을 감지했을 때 충돌 위험으로 판단
        if label_name in ['Ship']: 
             # 실제 충돌 위험 로직 (예: 객체 크기, 위치, 속도)이 없으므로 임시로 'Ship' 감지 시 CRITICAL로 설정합니다.
             critical_ship_detected = True

        # Anomaly Detection (PyTorch) - 탐지된 객체 영역에만 적용
        pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        input_tensor = inference_transforms(pil_crop).unsqueeze(0).to('cpu')
        
        with torch.no_grad():
            anomaly_score = deployed_model(input_tensor).item() 
        
        is_anomaly = anomaly_score > OPTIMAL_THRESHOLD
        if is_anomaly:
            anomaly_detected = True

        detections.append({
            "object_type": label_name,
            "confidence": round(score_cls, 2),
            "anomaly": is_anomaly,
            "anomaly_score": round(anomaly_score, 4),
            "box": [x, y, w, h] # x, y, width, height
        })

    # --------------------------
    # 5) MQTT 발행
    # --------------------------
    
    # 5-1. 기본 RAW 데이터 (모든 탐지 결과 포함)
    raw_data = {
        "timestamp": now_str(),
        "module": AD_MODULE, # 🚨 수정: 모듈 이름 명시
        "level": "INFO",     # 🚨 수정: INFO 레벨 명시
        "detections": detections,
        "total_count": len(detections),
        "anomaly_count": sum(1 for d in detections if d['anomaly']),
    }
    raw_payload = json.dumps(raw_data)
    client.publish(RAW_TOPIC, raw_payload, qos=0)
    # 시연용 로그: RAW 데이터 발행 
    end_time = time.time()
    fps = 1.0 / (end_time - start_time + 1e-6)
    print(f"[{now_str()}] INFO PUB :: {RAW_TOPIC} → Sent {len(detections)} detections. (FPS: {fps:.1f})")


    # 5-2. 경고 이벤트 (Anomaly나 중요 객체 감지 시)
    if anomaly_detected or critical_ship_detected:
        
        # 🚨 레벨 및 메시지 결정 로직 🚨
        if anomaly_detected or critical_ship_detected:
            # 이상 징후나 충돌 위험(Ship 감지)이 있으면 CRITICAL
            alert_level = "CRITICAL"
            alert_summary = f"🚨 긴급 충돌/이상 징후! 총 {len(detections)}개 객체 중 {sum(1 for d in detections if d['anomaly'])}개 이상 징후."
            
        elif any(d['object_type'] in ['Animal', 'Reef'] for d in detections):
            # Animal 또는 Reef는 항해에 주의가 필요하므로 WARNING
            alert_level = "WARNING"
            alert_summary = f"⚠️ 항해 주의! {', '.join(set(d['object_type'] for d in detections if d['object_type'] in ['Animal', 'Reef']))} 감지됨."
        else:
            return # 경고 발행 필요 없음

        alert_data = {
            "timestamp": now_str(),
            "module": AD_MODULE, # 🚨 수정: 모듈 이름 명시
            "level": alert_level, # 🚨 수정: 결정된 레벨 적용
            "message": alert_summary,
            "details": [d for d in detections if d['anomaly'] or d['object_type'] in ['Ship', 'Animal', 'Reef']],
        }
        alert_payload = json.dumps(alert_data)
        client.publish(ALERT_TOPIC, alert_payload, qos=1)
        # 시연용 로그: 경고 발행
        print(f"[{now_str()}] 📢 {alert_level} PUB :: {ALERT_TOPIC} → {alert_summary}")
        
    
    time.sleep(0.01) # 0.01초 대기 (CPU 점유율 관리)


# ====================================================
# 7. 메인 실행 함수
# (기존 코드와 동일)
# ====================================================

def main():
    # 1. 모델 및 카메라 초기화
    initialize_vision()

    # 2. MQTT 클라이언트 생성 및 연결
    client = mqtt.Client()
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print(f"[{now_str()}] INFO MQTT :: Client connected successfully (RC: {rc})")
        else:
            print(f"[{now_str()}] ❌ CRITICAL: MQTT Connection failed (RC: {rc})")
            sys.exit(1)
            
    client.on_connect = on_connect # 콜백 설정
    
    try:
        client.connect(BROKER, PORT, 60)
        client.loop_start() 
        print(f"[{now_str()}] INFO MQTT :: Client attempting connection to {BROKER}:{PORT}")
    except Exception as e:
        print(f"[{now_str()}] ❌ CRITICAL: MQTT 연결 실패: {e}")
        sys.exit(1)
    
    # 3. 메인 루프
    try:
        while True:
            # GPU를 사용하는 경우 cv2.waitKey(1) 대신 time.sleep을 사용해야 합니다.
            # cv2.imshow는 제거했으므로 time.sleep(0.01) 정도를 추가하여 CPU 점유율을 관리합니다.
            run_inference_and_publish(client)
            
    except KeyboardInterrupt:
        print(f"\n[{now_str()}] INFO System :: Vision client stopped by user (Ctrl+C).")
    except Exception as e:
        print(f"\n[{now_str()}] ❌ ERROR System :: An unexpected error occurred: {e}")
    finally:
        client.loop_stop()
        client.disconnect() 
        print(f"[{now_str()}] INFO MQTT :: Client disconnected.")
        if cap is not None:
            cap.release()
        sys.exit(0)

if __name__ == "__main__":
    main()
