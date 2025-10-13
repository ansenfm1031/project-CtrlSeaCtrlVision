import paho.mqtt.client as mqtt
import pymysql
from datetime import datetime, timezone
from gtts import gTTS
import os
from openai import OpenAI
import sys
import re 
import json 

# === DB 연결 (MariaDB) ===
DB_HOST = "localhost"
DB_USER = "marine_user"
DB_PASSWORD = "sksk"
DB_NAME = "marine_system"

# === MQTT 설정 ===
BROKER = "0.0.0.0"
PORT = 1883
TOPIC_BASE = "project/"   # 모듈 로그 접두사 (예: project/VISION/EVT)
COMMAND_TOPIC = "command/" # 서버 명령 접두사 (예: command/summary)

# === OpenAI 클라이언트 설정 ===
client_llm = OpenAI() # 키는 환경 변수에서 자동 로드됩니다.

# === 유틸리티 ===
def now_str():
    """UTC 시각을 'YYYY-MM-DD HH:MM:SS' 형식으로 반환합니다."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

# === DB 연결 함수 (연결이 끊어졌을 경우를 대비) ===
def get_db_connection():
    """DB 연결 객체를 생성하고 반환합니다. 연결 실패 시 None 반환."""
    try:
        # 전역 상수 DB_HOST, DB_USER 등을 사용합니다.
        db = pymysql.connect(
            host=DB_HOST, user=DB_USER, password=DB_PASSWORD, 
            database=DB_NAME, charset="utf8mb4"
        )
        return db
    except Exception as e:
        print(f"[DB-ERROR] 연결 실패: {e}")
        return None
    
# === 키=값; 형태의 문자열을 딕셔너리로 파싱 (함수 내용 그대로 사용) ===
def parse_payload_to_dict(payload: str) -> dict:
    """'키=값;키=값' 형태의 문자열을 딕셔너리로 파싱합니다."""
    # 클라이언트가 JSON을 보낸다면 JSON 파싱을 시도합니다.
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        # JSON이 아니면 기존 키=값; 로직을 유지합니다. (혹시 모를 다른 클라이언트 대응)
        data = {}
        if "|" in payload:
            payload = payload.split("|", 1)[-1].strip()
        pairs = payload.split(';')
        for pair in pairs:
            if '=' in pair:
                k, v = pair.split('=', 1)
                data[k.strip()] = v.strip()
        return data

def clean_tts_text(text: str) -> str:
    """
    TTS 재생을 위해 불필요한 마크다운 문자를 제거하되, 한글/구두점은 유지합니다.
    """
    cleaned_text = text.replace('**', '').replace('*', '').replace('#', '')
    # 한글, 영문, 숫자, 공백, 자주 쓰는 구두점만 남기고 모두 제거
    cleaned_text = re.sub(r'[^\w\s\.\,\!\?ㄱ-ㅎㅏ-ㅣ가-힣]', ' ', cleaned_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

# === DB 연결 초기화 (함수 정의 후 실행되어야 함) ===
DB_CONN = get_db_connection()
if DB_CONN is None:
    print("[CRITICAL] DB 연결 실패. 서버를 종료합니다.")
    sys.exit(1)
CURSOR = DB_CONN.cursor()

# === DB 저장 함수 (DB_CONN, CURSOR 사용) ===
def save_event_log(module: str, action: str, full_payload: str):
    """events 테이블에 일반 로그, STT, IMU 위험 로그를 저장"""
    try:
        now = now_str()
        sql = "INSERT INTO events (module, action, payload, ts) VALUES (%s, %s, %s, %s)"
        CURSOR.execute(sql, (module, action, full_payload, now))
        DB_CONN.commit()
        print(f"[{now}] [DB-OK] Log saved to events: ({module}) {action}")
    except Exception as e:
        print(f"[{now}] [DB-ERROR] events 테이블 저장 실패: {e}")

def save_vision_data(module: str, action: str, payload_dict: dict):
    """vision_data 테이블에 VISION/POSE 결과를 저장"""
    try:
        now = now_str()
        
        object_type = action 
        risk_level = int(payload_dict.get('level', 0) or payload_dict.get('risk', 0))
        description = payload_dict.get('posture') or payload_dict.get('zone') or object_type
        # json.dumps() 사용 시 한글이 깨지지 않도록 ensure_ascii=False 옵션을 추가했습니다.
        detail_json = json.dumps(payload_dict, ensure_ascii=False) 
        
        sql = """
            INSERT INTO vision_data 
            (ts, module, object_type, risk_level, description, detail_json) 
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        CURSOR.execute(sql, (now, module, object_type, risk_level, description, detail_json))
        DB_CONN.commit()
        print(f"[{now}] [DB-OK] Data saved to vision_data: ({module}/{object_type}) Risk:{risk_level}")
    except Exception as e:
        print(f"[{now}] [DB-ERROR] vision_data 테이블 저장 실패: {e}")

def save_imu_raw_data(payload_dict: dict):
    """imu_data 테이블에 연속적인 Pitch/Roll/Yaw 데이터를 저장"""
    try:
        now = now_str()
        
        # 클라이언트가 보낸 roll, pitch, yaw 키를 사용합니다.
        roll = float(payload_dict.get('roll', 0.0))
        pitch = float(payload_dict.get('pitch', 0.0))
        yaw = float(payload_dict.get('yaw', 0.0))
        
        sql = "INSERT INTO imu_data (ts, pitch, roll, yaw) VALUES (%s, %s, %s, %s)"
        # 순서를 Pitch, Roll, Yaw 순으로 맞추는 것이 좋습니다 (DB 테이블 순서에 따라)
        CURSOR.execute(sql, (now, pitch, roll, yaw)) 
        DB_CONN.commit()
        print(f"[{now}] [DB-OK] Raw data saved to imu_data: R:{roll:.2f} P:{pitch:.2f} Y:{yaw:.2f}")
    except Exception as e:
        print(f"[{now}] [DB-ERROR] imu_data 테이블 저장 실패: {e}")

# === LLM/TTS 로직 함수 (DB_CONN, CURSOR 사용) ===

def query_llm(prompt: str) -> str:
    """OpenAI API를 사용하여 LLM에 질문하고 응답을 받습니다."""
    try:
        messages = [
             {"role": "system", "content": "너는 선박 항해 보조관이야. 로그를 분석하여 간결하고 명확하게 한국어로 브리핑해줘. 답변 시 마크다운 기호(\\#, \\*, \\- 등)를 절대 사용하지 말고, 문장 끝에 마침표를 제외한 쉼표나 기타 구두점의 사용을 최소화하며 평문으로만 응답해야 해."},
             {"role": "user", "content": prompt}
        ]
        response = client_llm.chat.completions.create(
             model="gpt-4o-mini",
             messages=messages,
             max_tokens=300,
             temperature=0.7,
        )
        result = response.choices[0].message.content
        print("[LLM OK] Response received.")
        return result
    except Exception as e:
        print(f"[LLM ERROR] {e}")
        return "⚠️ LLM 요청 중 오류 발생."

# === 로그 불러오기 ===
def fetch_logs(minutes=10):
    """DB에서 최근 minutes분 동안의 이벤트를 가져옵니다."""
    try:
        sql = """
            SELECT ts, module, action, payload
            FROM events
            WHERE ts >= NOW() - INTERVAL %s MINUTE
            ORDER BY ts ASC
        """
        # 전역 커서 CURSOR를 사용합니다.
        CURSOR.execute(sql, (minutes,)) 
        rows = CURSOR.fetchall()
        if not rows:
            return [f"최근 {minutes}분 동안 이벤트가 없습니다."]
        logs = [f"[{r[0]}] ({r[1]}) {r[2]} → {r[3]}" for r in rows]
        print(f"[DB] Retrieved {len(logs)} logs for summary")
        return logs
    except Exception as e:
        print(f"[DB-ERROR] fetch_logs: {e}")
        return ["로그 불러오기 실패."]
    
# === LLM 요약 ===
def summarize_logs(logs):
    """로그 목록을 LLM에 전달하여 요약 보고서를 생성합니다."""
    text = "\n".join(logs)
    prompt = f"""
    다음은 선박 항해 로그입니다:
    {text}

    위 로그를 분석하여 다음 4가지 정보를 반드시 포함하여 간결하게 요약 및 브리핑해줘:
    1. 선박의 일반적인 상태 (엔진, 속도 등)
    2. 최근 10분간 발생한 주요 이벤트 또는 특이사항
    3. 카메라나 레이더를 통한 위험 감지 상황 (안개, 타 선박, 사람 등)
    4. 조치된 사항이나 필요한 추가 조치

    응답은 항목별 요약 없이 하나의 문단 형태로 한국어로 작성하고, 마크다운 기호(\\#, \\*, \\- 등)는 절대 사용하지 마세요.
    """
    print("[LLM] Summarizing logs using GPT-4o mini...")
    summary = query_llm(prompt)
    print("[SUMMARY]\n", summary)
    return summary
    
# === TTS 변환 및 재생 ===
def text_to_speech(text, filename="summary.mp3"):
    """텍스트를 gTTS로 MP3 파일로 변환 후 mpv를 사용하여 재생합니다."""
    try:
        clean_text = clean_tts_text(text)
        tts = gTTS(text=clean_text, lang="ko")
        tts.save(filename)
        # mpv --no-terminal --volume=100 --speed=1.3 명령을 통해 재생 (Linux/macOS 환경 가정)
        os.system(f"mpv --no-terminal --volume=100 --speed=1.3 {filename}") 
        print("[TTS] Summary spoken successfully.")
    except Exception as e:
        print(f"[TTS Error] {e}")

# === MQTT 콜백 함수 (메인 로직) ===
def on_connect(client, userdata, flags, rc):
    """브로커 연결 시 호출되며, 토픽을 구독합니다."""
    if rc == 0:
        print("[OK] Connected to broker")
        # TOPIC_BASE와 COMMAND_TOPIC을 사용하여 구독
        client.subscribe(TOPIC_BASE + "#") 
        client.subscribe(COMMAND_TOPIC + "#")
        print(f"[SUB] Subscribed to {TOPIC_BASE}# and {COMMAND_TOPIC}#")
    else:
        print("[FAIL] Connection failed, code:", rc)

# === [데이터 라우터] RAW 데이터 저장 및 임계값 초과 시 events 테이블에 경고를 기록하는 핵심 로직.===
def process_and_save_data(msg):
    """
    수신된 MQTT 메시지를 분석하여 알맞은 테이블에 저장하고, 
    필요 시 이벤트를 생성합니다.
    """
    
    # 1. 토픽 파싱
    topic = msg.topic
    payload = msg.payload.decode('utf-8')
    # parse_payload_to_dict 함수는 JSON 파싱을 수행한다고 가정
    payload_dict = parse_payload_to_dict(payload)
    
    # 토픽에서 모듈/액션 추출 (예: project/IMU/RAW -> module=IMU, action=RAW)
    parts = topic.split('/') # ['project', 'vision', 'RAW'] 또는 ['project', 'IMU', 'RAW']
    
    # 토픽이 최소한 3단계 (project/module/action) 이상이어야 함
    if len(parts) < 3:
        print(f"[WARN] Skipping short topic: {topic}")
        return

    topic_base = parts[0] + '/' + parts[1] # 'project/vision' 또는 'project/IMU'
    module = parts[1].upper() # 'VISION' 또는 'IMU'
    action = parts[2].upper() # 'RAW' 또는 'ALERT' 등
    
    # 디버깅 및 서버 요구사항 맞추기
    
    # Vision 토픽일 경우 (기존 로직)
    if module == "VISION":
        # 🟢 VISION 모듈: save_vision_data 함수에 필수 인수 3개를 전달합니다.
        try:
            # 파싱된 topic_base ('project/vision'), action ('RAW'), payload_dict를 전달합니다.
            save_vision_data(topic_base, action, payload_dict) 
            
        except TypeError as e:
            # save_vision_data 호출 시 TypeError가 발생하면, 인수가 여전히 틀렸다는 뜻입니다.
            print(f"[CRITICAL ERROR] save_vision_data 호출 실패: {e}. server.py의 함수 정의를 확인하세요.")
            
    # IMU 토픽일 경우 (추가된 로직)
    elif module == "IMU":
        # 🟢 IMU 모듈: save_imu_data 함수에 필수 인수 3개를 전달합니다.
        # save_imu_data(topic_base, action, payload_dict) 시그니처를 가정합니다.
        try:
            # 파싱된 topic_base ('project/IMU'), action ('RAW'), payload_dict를 전달합니다.
            # 'save_imu_data' 함수가 server.py에 정의되어 있다고 가정합니다.
            save_imu_data(topic_base, action, payload_dict)
            
        except TypeError as e:
            # save_imu_data 호출 시 TypeError가 발생하면, 인수가 여전히 틀렸다는 뜻입니다.
            print(f"[CRITICAL ERROR] save_imu_data 호출 실패: {e}. server.py의 함수 정의를 확인하세요.")
            
    # 기타 모듈이 추가될 경우 elif를 사용하여 확장 가능
            
    # ... (다른 모듈 처리 로직이 있었다면 이 아래에 위치)
    
    return

    # =======================================================
    # 2. 데이터 라우팅 및 저장 (수정 및 확장 필요)
    # =======================================================
    
    # IMU 원시 데이터 (IMU/RAW) -> imu_data 저장 및 이벤트 감지
    if module == "IMU" and action == "RAW": 
        
        # 2-1. 원시 데이터 저장 (imu_data)
        save_imu_raw_data(payload_dict)
        print(f"[DB] Saved IMU RAW data to imu_data table.")
        
        # 2-2. 이벤트 감지 및 생성 (events)
        roll_angle = payload_dict.get('roll', 0.0)
        RISK_THRESHOLD = 20.0 # 임계값 설정 (예: 20도)

        if abs(roll_angle) > RISK_THRESHOLD:
            # Roll 각도가 위험 임계값을 초과했을 때만 events에 기록
            event_type = "TILT_ALERT"
            event_message = f"심한 기울기 감지: Roll {roll_angle:.2f}도 초과."
            
            # save_event_log 함수는 events 테이블에 저장해야 합니다.
            save_event_log("IMU", event_type, event_message)
            print(f"[EVENT] RISK ALERT generated for Roll: {roll_angle:.2f}")

    # VISION 원시 데이터 (VISION/RAW) -> vision_data 저장 및 이벤트 감지
    elif module == "VISION" and action == "RAW":
        
        # 2-3. 원시 데이터 저장 (vision_data)
        save_vision_data(payload_dict)
        print(f"[DB] Saved VISION RAW data to vision_data table.")
        
        # 2-4. 이벤트 감지 및 생성 (events)
        # 예시: payload_dict 안에 'detected_person' 키가 있고 값이 True일 경우
        is_person_detected = payload_dict.get('person_detected', False) 
        
        if is_person_detected:
            event_type = "INTRUSION_ALERT"
            event_message = "배 앞 갑판에 사람 접근 감지됨."
            save_event_log("VISION", event_type, event_message)
            print(f"[EVENT] Intruder Alert generated.")

    # 그 외 일반 시스템/STT 이벤트 -> events 테이블
    else: 
        # STT, DB 로그, 기타 관리 목적의 로그는 events에 바로 저장
        save_event_log(module, action, payload)
        print(f"[LOG] Saved general log to events table. Module: {module}")
        
# === [MQTT 콜백] 명령어 처리 후 데이터 라우팅을 'process_and_save_data'로 위임하는 진입점. ===
def on_message(client, userdata, msg):
    """메시지가 수신될 때 호출되며, 토픽에 따라 데이터 저장 또는 명령을 처리합니다."""
    now = now_str() 
    payload = msg.payload.decode()
    topic = msg.topic
    print(f"[{now}] {topic} → {payload}") 

    # 1. === 명령어/요약 트리거 처리 ===
    if topic.startswith(COMMAND_TOPIC):
        if topic == f"{COMMAND_TOPIC}summary":
            print(f"[{now}] [CMD] Summary request received → Generating report...")
            logs = fetch_logs(10)
            summary = summarize_logs(logs)
            text_to_speech(summary)
            # LLM 결과 TTS 발화 후 DB에 기록
            save_event_log("LLM", "SAY", summary)
        elif topic == f"{COMMAND_TOPIC}query":
             # 사용자의 질의 요청 처리 (현재는 로그만 남김)
             save_event_log("SERVER", "CMD_QUERY", payload)
        return

    # 2. === 데이터 처리 로직을 새로운 함수로 위임 ===
    process_and_save_data(msg)
    

# === MQTT 클라이언트 및 메인 루프 ===
client = mqtt.Client(client_id="MarineServer")
client.on_connect = on_connect
client.on_message = on_message

# === 브로커 연결 ===
print("[INFO] Connecting to broker...")
client.connect(BROKER, PORT, 60)

# === 루프 ===
try:
    client.loop_forever()
except KeyboardInterrupt:
    print("\n[EXIT] Server stopped by user")
    client.disconnect()
    # 전역 연결 객체를 닫습니다. (이전에 사용했던 local 'cursor', 'db' 대신)
    CURSOR.close() 
    DB_CONN.close()
