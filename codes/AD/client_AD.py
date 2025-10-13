import paho.mqtt.client as mqtt
import cv2
import numpy as np
from openvino.runtime import Core # OpenVINO 런타임 Core로 명시적으로 변경
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
# 1. 환경 설정 및 상수 정의
# ====================================================

# MQTT 설정
BROKER = "10.10.14.73" # 브로커 주소를 사용자의 환경에 맞게 설정하세요
PORT = 1883
TOPIC_BASE = "project/vision" # 토픽 접두사

def now_str():
    """ISO 8601 형식의 현재 UTC 시각을 반환합니다."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# =======================
# 2. 모델 경로 설정 및 유효성 검사
# (모든 모델 파일은 이 스크립트와 같은 디렉토리에 있어야 합니다.)
# =======================
det_xml = "Detection.xml"
det_bin = "Detection.bin"

cls_xml = "Classification.xml"
cls_bin = "Classification.bin"

DEPLOYMENT_FILE = "deployed_obstacle_detector.pt"  # PyTorch Anomaly Detection TorchScript 모델

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

class_names = ["Buoy", "Reef", "Island", "Ship", "Bridge", "Dockside", "Animal"]
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
# 5. 초기화 함수 (모든 모델 로드)
# =======================

def initialize_vision():
    """OpenVINO 모델, PyTorch 모델 및 카메라를 초기화합니다."""
    global det_compiled, det_input_layer, det_output_layer
    global cls_compiled, cls_input_layer, cls_output_layer
    global cls_h, cls_w, cap, deployed_model
    
    try:
        # OpenVINO Detection
        det_model = ie.read_model(det_xml, det_bin)
        det_compiled = ie.compile_model(det_model, "CPU")
        det_input_layer = det_compiled.input(0)
        det_output_layer = det_compiled.output(0)
        print(f"[{now_str()}] ✅ OpenVINO Detection 모델 로드 완료.")

        # OpenVINO Classification
        cls_model = ie.read_model(cls_xml, cls_bin)
        cls_compiled = ie.compile_model(cls_model, "CPU")
        cls_input_layer = cls_compiled.input(0)
        cls_output_layer = cls_compiled.output(0)
        _, _, cls_h, cls_w = cls_input_layer.shape
        print(f"[{now_str()}] ✅ OpenVINO Classification 모델 로드 완료.")

        # PyTorch Anomaly Detection (TorchScript)
        deployed_model = torch.jit.load(DEPLOYMENT_FILE, map_location='cpu')
        deployed_model.eval()
        print(f"[{now_str()}] ✅ PyTorch Anomaly 모델 로드 완료.")

        # 웹캠 열기: **카메라 초기화 다중 시도 로직 (안정성 확보)**
        # 0번 인덱스를 우선 시도하고, V4L2 백엔드를 명시하여 안정성을 높입니다.
        
        # --- [!!! 카메라 인덱스 시도 목록 !!!] ---
        capture_attempts = [
            (0, cv2.CAP_V4L2), (0, 0), # 0번 인덱스에 V4L2/기본 백엔드 시도 (가장 안정적)
            (1, cv2.CAP_V4L2), (1, 0), # 1번 인덱스
            (2, cv2.CAP_V4L2), (2, 0), # 2번 인덱스
            (3, cv2.CAP_V4L2), (3, 0), # 3번 인덱스
            (4, cv2.CAP_V4L2), (4, 0), # 4번 인덱스
            (5, cv2.CAP_V4L2), (5, 0)  # 5번 인덱스
        ]
        # --------------------------------------------------------
        
        cap = None
        for index, api_preference in capture_attempts:
            # 백엔드 명시 시도 (api_preference = cv2.CAP_V4L2) 또는 기본 백엔드 시도
            if api_preference != 0:
                cap = cv2.VideoCapture(index, api_preference)
            else:
                cap = cv2.VideoCapture(index)

            if cap.isOpened():
                # **해상도 및 속성 설정:** 안정적인 캡처를 위해 해상도를 명시적으로 설정합니다.
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                print(f"[{now_str()}] ✅ 웹캠 열기 성공: 인덱스 {index}, 백엔드 {api_preference}")
                break # 성공하면 루프 종료
            
        if cap is None or not cap.isOpened():
            # 이 메시지를 보게 되면, 인덱스 0 외의 다른 인덱스를 시도해야 하거나 권한 문제가 남아있는 것입니다.
            raise RuntimeError("웹캠을 열 수 없습니다. 0~5번 인덱스 및 V4L2 백엔드 시도 실패.")
        
        # 웹캠 열기 성공 메시지는 루프 내부에서 출력됩니다.

    except Exception as e:
        print(f"[{now_str()}] ❌ CRITICAL: 초기화 실패 - {e}")
        sys.exit(1)


# =======================
# 6. 메인 추론 및 발행 함수
# =======================

def run_inference_and_publish(client):
    """
    1. 이미지 캡처 및 전처리 (저조도 개선, Dehazing)
    2. OpenVINO Detection/Classification
    3. PyTorch Anomaly Detection
    4. MQTT로 결과 발행
    """
    global last_frame_boxes
    
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

        # Anomaly Detection (PyTorch) - 탐지된 객체 영역에만 적용
        pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        input_tensor = inference_transforms(pil_crop).unsqueeze(0).to('cpu')
        
        with torch.no_grad():
            # anomaly_score는 TorchScript 모델의 출력에 따라 수정이 필요할 수 있습니다.
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
    raw_payload = json.dumps({
        "timestamp": now_str(),
        "detections": detections,
        "total_count": len(detections),
        "anomaly_count": sum(1 for d in detections if d['anomaly']),
    })
    client.publish(f"{TOPIC_BASE}/RAW", raw_payload, qos=0)
    print(f"[{now_str()}] INFO PUB :: {TOPIC_BASE}/RAW → Sent {len(detections)} detections.")

    # 5-2. 경고 이벤트 (Anomaly나 중요 객체 감지 시)
    if anomaly_detected or any(d['object_type'] in ['Ship', 'Animal'] for d in detections):
        
        alert_summary = f"위험 감지: 총 {len(detections)}개 객체 중 {sum(1 for d in detections if d['anomaly'])}개가 이상 징후."
        
        alert_payload = json.dumps({
            "level": 5 if anomaly_detected else 3,
            "message": alert_summary,
            "details": [d for d in detections if d['anomaly'] or d['object_type'] in ['Ship', 'Animal']]
        })
        client.publish(f"{TOPIC_BASE}/ALERT", alert_payload, qos=1) # QOS 1로 중요 경고 전송
        print(f"[{now_str()}] ⚠️ ALERT PUB :: {TOPIC_BASE}/ALERT → {alert_summary}")
        
    
    time.sleep(0.5) # 0.5초당 1회 추론 및 발행 (약 2 FPS)


# ====================================================
# 7. 메인 실행 함수
# ====================================================

def main():
    # 1. 모델 및 카메라 초기화
    initialize_vision()

    # 2. MQTT 클라이언트 생성 및 연결
    client = mqtt.Client()
    try:
        # DeprecationWarning 제거: Callback API version 1 is deprecated, update to latest version
        # paho-mqtt의 최신 버전은 context managers나 loop_forever()를 권장하지만, 
        # 기존 코드 스타일 유지를 위해 warning은 무시하고 진행합니다.
        client.connect(BROKER, PORT, 60)
        client.loop_start() 
        print(f"[{now_str()}] INFO MQTT :: Client connected to {BROKER}:{PORT}")
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
        print(f"\n[{now_str()}] INFO System :: Vision client stopped by user.")
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
