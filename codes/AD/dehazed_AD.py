import cv2
import numpy as np
from openvino.runtime import Core
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
import os
import time

# =======================
# 1. 모델 경로 설정
# =======================
det_xml = "/home/ubuntu26/workspace/AD_Dataset/Detection_Model/Detection.xml"
det_bin = "/home/ubuntu26/workspace/AD_Dataset/Detection_Model/Detection.bin"

cls_xml = "/home/ubuntu26/workspace/AD_Dataset/Detection_Model/Classification.xml"
cls_bin = "/home/ubuntu26/workspace/AD_Dataset/Detection_Model/Classification.bin"

for path in [det_xml, det_bin, cls_xml, cls_bin]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ 모델 파일을 찾을 수 없습니다: {path}")

# =======================
# 2. OpenVINO 모델 로드
# =======================
ie = Core()

det_model = ie.read_model(det_xml, det_bin)
det_compiled = ie.compile_model(det_model, "CPU")
det_input_layer = det_compiled.input(0)
det_output_layer = det_compiled.output(0)

cls_model = ie.read_model(cls_xml, cls_bin)
cls_compiled = ie.compile_model(cls_model, "CPU")
cls_input_layer = cls_compiled.input(0)
cls_output_layer = cls_compiled.output(0)

_, _, cls_h, cls_w = cls_input_layer.shape

# =======================
# 3. 라벨 이름 정의
# =======================
class_names = ["Buoy", "Reef", "Island", "Ship", "Bridge", "Dockside", "Animal"]

# =======================
# 4. NMS 함수 정의
# =======================
def nms(boxes, scores, score_threshold=0.5, iou_threshold=0.5):
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold, iou_threshold)
    if len(indices) > 0:
        indices = indices.flatten()
        return [boxes[i] for i in indices], [scores[i] for i in indices]
    return [], []

# =======================
# 5. IoU 계산 함수
# =======================
def iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union_area = w1 * h1 + w2 * h2 - inter_area
    return inter_area / union_area if union_area > 0 else 0

# =======================
# 6. PyTorch Anomaly Detection 모델 로드
# =======================
DEPLOYMENT_FILE = "deployed_obstacle_detector.pt"  # TorchScript 모델
OPTIMAL_THRESHOLD = 0.7
MODEL_INPUT_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

try:
    deployed_model = torch.jit.load(DEPLOYMENT_FILE, map_location='cpu')
    deployed_model.eval()
    print(f"✅ 배포 모델 '{DEPLOYMENT_FILE}' 로드 완료.")
except Exception as e:
    print(f"❌ 오류: 배포 파일 로드 실패. {e}")
    exit()

inference_transforms = transforms.Compose([
    transforms.Resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

# =======================
# 7. 이미지 전처리 함수들 (저조도 개선, Dehazing)
# =======================
def enhance_low_light(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_img

def dehaze(image):
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
# 8. 웹캠 열기
# =======================
cap = cv2.VideoCapture(0)  # 카메라 인덱스 0번 (웹캠)
if not cap.isOpened():
    print("❌ 웹캠을 열 수 없습니다.")
    exit()

print("✅ 웹캠 열기 성공. 'q'를 눌러 종료합니다.")

last_frame_boxes = []
frame_count = 0
start_time = time.time()

# =======================
# 9. 실시간 루프
# =======================
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # --------------------------
    # 1) 저조도 개선 (enhanced)
    # --------------------------
    enhanced = enhance_low_light(frame)

    # --------------------------
    # 2) Dehazing 적용
    # --------------------------
    dehazed = dehaze(enhanced)

    # --------------------------
    # 3) OpenVINO Detection
    # --------------------------
    resized = cv2.resize(dehazed, (640, 640))
    input_image = np.expand_dims(resized.transpose(2, 0, 1), 0).astype(np.float32)
    det_results = det_compiled([input_image])[det_output_layer][0]

    boxes, scores = [], []
    for det in det_results:
        x_min, y_min, x_max, y_max, conf = det
        if conf > 0.5:
            x_min = int(x_min / 640 * dehazed.shape[1])
            y_min = int(y_min / 640 * dehazed.shape[0])
            x_max = int(x_max / 640 * dehazed.shape[1])
            y_max = int(y_max / 640 * dehazed.shape[0])
            boxes.append([x_min, y_min, x_max - x_min, y_max - y_min])
            scores.append(float(conf))

    filtered_boxes, filtered_scores = nms(boxes, scores)

    # --------------------------
    # 4) Frame smoothing (NMS)
    # --------------------------
    smoothed_boxes = []
    for box in filtered_boxes:
        matched = False
        for prev_box in last_frame_boxes:
            if iou(box, prev_box) > 0.7:
                smoothed_boxes.append(prev_box)
                matched = True
                break
        if not matched:
            smoothed_boxes.append(box)
    last_frame_boxes = smoothed_boxes.copy()

    # --------------------------
    # 5) Classification + Visualization
    # --------------------------
    for (x, y, w, h) in smoothed_boxes:
        crop = dehazed[max(0, y-5):min(dehazed.shape[0], y+h+5), max(0, x-5):min(dehazed.shape[1], x+w+5)]
        if crop.size == 0:
            continue
        cls_resized = cv2.resize(crop, (cls_w, cls_h))
        cls_resized = cv2.cvtColor(cls_resized, cv2.COLOR_BGR2RGB)
        cls_input = np.expand_dims(cls_resized.transpose(2, 0, 1), 0).astype(np.float32) / 255.0

        cls_result = cls_compiled([cls_input])[cls_output_layer]
        class_id = int(np.argmax(cls_result))
        score_cls = float(np.max(cls_result))
        label_name = class_names[class_id]

        cv2.rectangle(dehazed, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{label_name}: {score_cls:.2f}"
        cv2.putText(dehazed, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # --------------------------
    # 6) 실시간 출력
    # --------------------------
    cv2.imshow("Processed Video", dehazed)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        print(f"FPS: {fps:.2f}")
        start_time = time.time()
        frame_count = 0

# =======================
# 7. 종료 처리
# =======================
cap.release()
cv2.destroyAllWindows()
