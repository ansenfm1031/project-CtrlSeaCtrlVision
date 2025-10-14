import paho.mqtt.client as mqtt
import paho.mqtt.publish as publish # STT 스레드에서 publish.single 사용
import pymysql
from datetime import datetime, timezone
from gtts import gTTS
import os
from openai import OpenAI
import sys
import re 
import json 
import threading # STT 기능을 별도 스레드에서 실행하기 위함
import speech_recognition as sr # STT 기능 추가

# === DB 연결 (MariaDB) ===
DB_HOST = "localhost"
DB_USER = "marine_user"
DB_PASSWORD = "sksk"
DB_NAME = "marine_system"

# === MQTT 설정 ===
BROKER = "0.0.0.0"
PORT = 1883
TOPIC_BASE = "project/"   # 모듈 로그 접두사 (예: project/IMU/RAW)
COMMAND_TOPIC = "command/summary" # 항해일지 요약 명령
QUERY_TOPIC = "command/query" # 일반 질의 명령

# === OpenAI 클라이언트 설정 ===
client_llm = OpenAI() # 키는 환경 변수에서 자동 로드됩니다.

# === 유틸리티 ===
def now_str():
    """UTC 시각을 'YYYY-MM-DD HH:MM:SS.ffffff' (마이크로초) 형식으로 반환합니다."""
    # 초 단위가 아닌 마이크로초 단위까지 포함하여 고유성을 높입니다. (Duplicate Entry 방지)
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")

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
    
# === 키=값; 형태의 문자열을 딕셔너리로 파싱 ===
def parse_payload_to_dict(payload: str) -> dict:
    """'키=값;키=값' 형태의 문자열을 딕셔너리로 파싱합니다. JSON 우선 파싱."""
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        # JSON이 아니면 기존 키=값; 로직을 유지합니다. 
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
    """events 테이블에 일반 로그, STT, 모든 CRITICAL/WARNING 로그를 저장"""
    try:
        now = now_str()
        sql = "INSERT INTO events (module, action, payload, ts) VALUES (%s, %s, %s, %s)"
        CURSOR.execute(sql, (module, action, full_payload, now))
        DB_CONN.commit()
        print(f"[{now}] [DB-OK] Log saved to events: ({module}) {action}")
    except Exception as e:
        print(f"[{now}] [DB-ERROR] events 테이블 저장 실패: {e}")
# 'module' 인수를 사용하여 AD/PE/VISION을 명확히 구분
def save_vision_data(module: str, action: str, payload_dict: dict):
    """vision_data 테이블에 VISION/AD/PE 결과를 저장합니다."""
    try:
        now = now_str()
        
        # 'action'은 보통 'RAW'이지만, object_type으로 사용될 수 있음.
        object_type = payload_dict.get('type') or action 
        # 클라이언트 JSON payload에 'level' 또는 'risk' 키가 있다고 가정
        risk_level = int(payload_dict.get('level', 0) or payload_dict.get('risk', 0)) 
        description = payload_dict.get('posture') or payload_dict.get('zone') or object_type
        # json.dumps() 사용 시 한글이 깨지지 않도록 ensure_ascii=False 옵션을 추가했습니다.
        detail_json = json.dumps(payload_dict, ensure_ascii=False) 
        
        sql = """
            INSERT INTO vision_data 
            (ts, module, object_type, risk_level, description, detail_json) 
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        # module 인수로 받은 값을 사용 (AD, PE, VISION 중 하나)
        CURSOR.execute(sql, (now, module, object_type, risk_level, description, detail_json))
        DB_CONN.commit()
        print(f"[{now}] [DB-OK] Data saved to vision_data: ({module}/{object_type}) Risk:{risk_level}")
    except Exception as e:
        print(f"[DB-ERROR] vision_data 테이블 저장 실패: {e}")

def save_imu_raw_data(payload_dict: dict):
    """imu_data 테이블에 연속적인 Pitch/Roll/Yaw 데이터를 저장"""
    try:
        now = now_str()
        
        # 클라이언트가 보낸 roll, pitch, yaw 키를 사용합니다.
        roll = float(payload_dict.get('roll', 0.0) or payload_dict.get('roll_angle', 0.0)) 
        pitch = float(payload_dict.get('pitch', 0.0))
        yaw = float(payload_dict.get('yaw', 0.0))
        
        sql = "INSERT INTO imu_data (ts, pitch, roll, yaw) VALUES (%s, %s, %s, %s)"
        # 순서를 DB 테이블 순서에 따라 Pitch, Roll, Yaw 순으로 맞춥니다.
        CURSOR.execute(sql, (now, pitch, roll, yaw)) 
        DB_CONN.commit()
        print(f"[{now}] [DB-OK] Raw data saved to imu_data: R:{roll:.2f} P:{pitch:.2f} Y:{yaw:.2f}")
    except Exception as e:
        print(f"[DB-ERROR] imu_data 테이블 저장 실패: {e}")

# === LLM/TTS 로직 함수 (DB_CONN, CURSOR 사용) ===

def query_llm(prompt: str) -> str:
    """OpenAI API를 사용하여 LLM에 질문하고 응답을 받습니다."""
    try:
        # LLM 시스템 프롬프트: 답변 시 마크다운 기호를 사용하지 않고 평문으로만 응답하도록 강제
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

# === 로그 불러오기 및 IMU 통계 가져오기 ===
def fetch_logs(minutes=15):
    """DB에서 최근 minutes분 동안의 이벤트 로그와 IMU 통계를 가져옵니다."""
    logs = []
    imu_stats = {
        'max_roll': 0.0,
        'min_roll': 0.0,
        'latest_yaw': 0.0
    }
    
    try:
        # 1. 이벤트 로그 가져오기 (events 테이블)
        sql_events = """
            SELECT ts, module, action, payload
            FROM events
            WHERE ts >= NOW() - INTERVAL %s MINUTE
            ORDER BY ts ASC
        """
        CURSOR.execute(sql_events, (minutes,)) 
        rows = CURSOR.fetchall()
        logs = [f"[{r[0]}] ({r[1]}) {r[2]} → {r[3]}" for r in rows]
        print(f"[DB] Retrieved {len(logs)} event logs.")

        # 2. IMU 통계 가져오기 (imu_data 테이블)
        # 최대/최소 Roll (기울기)
        sql_roll = """
            SELECT MAX(roll), MIN(roll)
            FROM imu_data
            WHERE ts >= NOW() - INTERVAL %s MINUTE
        """
        CURSOR.execute(sql_roll, (minutes,))
        max_roll, min_roll = CURSOR.fetchone()
        imu_stats['max_roll'] = max_roll if max_roll is not None else 0.0
        imu_stats['min_roll'] = min_roll if min_roll is not None else 0.0

        # 최신 Yaw (현재 방향)
        sql_yaw = """
            SELECT yaw
            FROM imu_data
            WHERE ts >= NOW() - INTERVAL %s MINUTE
            ORDER BY ts DESC 
            LIMIT 1
        """
        CURSOR.execute(sql_yaw, (minutes,))
        latest_yaw_result = CURSOR.fetchone()
        imu_stats['latest_yaw'] = latest_yaw_result[0] if latest_yaw_result else 0.0
        
        print("[DB] Retrieved IMU statistics.")
        
    except Exception as e:
        print(f"[DB-ERROR] Log or IMU data fetching failed: {e}")
        logs = [f"최근 {minutes}분 동안 로그 불러오기 실패."]
        
    return logs, imu_stats
    
# === LLM 요약 (응답 스타일 강제) ===
def summarize_logs(logs, imu_stats, minutes):
    """로그 목록과 IMU 통계를 LLM에 전달하여 요약 보고서를 생성합니다."""
    text = "\n".join(logs)
    
    # LLM에게 전달할 IMU 통계 정보
    imu_context = f"""
    [선박 통계 (최근 {minutes}분)]:
    - 최대 롤(기울기): {imu_stats['max_roll']:.2f}도
    - 최소 롤(기울기): {imu_stats['min_roll']:.2f}도
    - 현재 추정 방향 (Yaw): {imu_stats['latest_yaw']:.2f}도
    """
    
    # LLM 사용자 프롬프트: 4가지 규칙을 명시적으로 요구
    prompt = f"""
    다음은 선박 통계와 항해 이벤트 로그입니다:

    {imu_context}
    
    [항해 이벤트 로그]:
    {text}

    위 정보를 분석하여 한국어로 간결하고 명확하게 브리핑해주세요. 응답은 오직 하나의 문단 형태로 작성해야 하며, 다음 4가지 정보를 반드시 포함해야 합니다:
    1. 선박의 일반적인 상태 (위 IMU 통계를 활용하여 최대 기울기 및 현재 방향 포함).
    2. 최근 {minutes}분간 'ALERT' 등 발생한 주요 이벤트 또는 특이사항.
    3. 카메라나 레이더 모듈(VISION, AD, PE)을 통해 감지된 위험 상황 관련 요약.
    4. 발생한 문제에 대해 조치된 사항이나 필요한 추가 조치. (로그에 조치 내용이 없으면 '현재 조치된 사항은 없습니다.' 등으로 언급).

    답변 시 마크다운 기호(\\#, \\*, \\- 등)는 절대 사용하지 말고, 문장 끝에 마침표를 제외한 쉼표나 기타 구두점의 사용을 최소화하며 평문으로만 응답해야 합니다.
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


# =======================================================================
# === [STT/음성 명령] 스레드 로직 추가 ===
# =======================================================================

def parse_speech_command(text: str) -> tuple[str, str]:
    """
    음성 텍스트를 분석하여 명령 토픽과 페이로드를 결정합니다.
    """
    text_lower = text.lower()
    
    # 1. 요약/보고 명령 감지
    summary_keywords = ["요약해줘", "보고해줘", "브리핑해줘", "일지", "요약"]
    if any(keyword in text_lower for keyword in summary_keywords):
        
        # '최근 N분'에서 N 추출
        match = re.search(r'(\d+)\s*(분|시간)', text_lower)
        minutes = 15 # 기본값: 15분
        
        if match:
            value = int(match.group(1))
            unit = match.group(2)
            
            if unit == "시간":
                minutes = value * 60
            else: # "분"
                minutes = value
        
        # 서버는 payload로 분(minutes) 값만 받습니다.
        return COMMAND_TOPIC, str(minutes)

    # 2. 일반 질문 명령
    else:
        # 일반 질문은 query 토픽으로 그대로 전송합니다.
        return QUERY_TOPIC, text


def stt_listening_loop():
    """마이크 입력을 받고 음성을 텍스트로 변환하여 MQTT로 전송하는 독립 루프입니다."""
    r = sr.Recognizer()

    # 마이크 설정 및 캘리브레이션
    try:
        # 16000Hz 샘플링 속도로 마이크를 엽니다.
        with sr.Microphone(sample_rate=16000) as source:
            print("[STT-THREAD] Ambient noise calibrating...")
            r.adjust_for_ambient_noise(source, duration=1.5)
            print("[STT-THREAD] Setup complete. Starting speech recognition loop...")
    except Exception as e:
        print(f"[CRITICAL] STT Initialization Error (Microphone): {e}")
        return # 스레드 종료

    # MQTT publish는 독립 스레드에서 publish.single을 사용합니다.
    mqtt_broker = BROKER 

    while True:
        try:
            with sr.Microphone(sample_rate=16000) as source:
                print("\n[STT-THREAD] Listening for command (Say '최근 N분 요약해줘')...")
                # 음성 인식 대기 (최대 10초 발화 길이 제한)
                audio = r.listen(source, timeout=None, phrase_time_limit=10) 
            
            print("[STT-THREAD] Recognizing speech...")
            # 구글 STT를 사용하여 한국어(ko-KR)로 인식
            text = r.recognize_google(audio, language="ko-KR") 
            print("[STT-THREAD] You said:", text)

            # 텍스트 분석 및 MQTT 명령 생성
            topic, payload = parse_speech_command(text)
            
            # MQTT 전송
            try:
                # STT 스레드에서 직접 메시지 발행
                publish.single(topic, payload=payload, hostname=mqtt_broker, qos=1)
                print(f"[STT-THREAD] MQTT Published: {topic} -> {payload}")
                # TTS 발화 후 DB에 기록 (이벤트 로깅)
                save_event_log("STT", "COMMAND", text)
            except Exception as e:
                print(f"[STT-THREAD] MQTT publish error: {e}")

        except sr.UnknownValueError:
            print("[STT-THREAD] Google Speech Recognition could not understand audio.")
        except sr.WaitTimeoutError:
            print("[STT-THREAD] No speech detected.")
        except sr.RequestError as e:
            print(f"[STT-THREAD] Could not request results from Google service; {e}")


# === MQTT 콜백 함수 (메인 로직) ===
def on_connect(client, userdata, flags, rc):
    """브로커 연결 시 호출되며, 토픽을 구독합니다."""
    if rc == 0:
        print("[OK] Connected to broker")
        # TOPIC_BASE와 COMMAND_TOPIC을 사용하여 구독
        client.subscribe(TOPIC_BASE + "#") 
        client.subscribe("command/#") # 모든 command/ 토픽 구독 (summary, query 포함)
        print(f"[SUB] Subscribed to {TOPIC_BASE}# and command/#")
    else:
        print("[FAIL] Connection failed, code:", rc)

# === [데이터 라우터] 핵심 로직 ===
def process_and_save_data(msg):
    """
    수신된 MQTT 메시지를 분석하여 알맞은 테이블에 저장하고, 
    필요 시 이벤트를 생성합니다.
    """
    
    # 1. 토픽 파싱
    topic = msg.topic
    payload = msg.payload.decode('utf-8')
    payload_dict = parse_payload_to_dict(payload)
    
    parts = topic.split('/') 
    
    if len(parts) < 3:
        # command/summary 같은 2단계 토픽은 아래 if topic.startswith(COMMAND_TOPIC)에서 처리됩니다.
        if not topic.startswith("command/"):
             print(f"[WARN] Skipping short topic: {topic}")
        return

    module = parts[1].upper()
    action = parts[2].upper()

    # =======================================================
    # 2. 데이터 라우팅 및 저장 (ALERT 우선 처리)
    # =======================================================
    
    # 2-1. 🚨 ALERT 토픽 처리 (CRITICAL/WARNING 레벨)
    if action == "ALERT":
        save_event_log(module, action, payload)
        print(f"[{now_str()}] [DB] ALERT log saved to events: {module}/{action}")
        return

    # 2-2. 🟢 RAW 토픽 처리 (INFO 레벨 - 연속 데이터)
    elif action == "RAW":
        if module == "IMU":
            save_imu_raw_data(payload_dict)
            print(f"[{now_str()}] [DB] Saved IMU RAW data to imu_data table.")
        
        elif module in ["VISION", "AD", "PE"]:
            save_vision_data(module, action, payload_dict)
            print(f"[{now_str()}] [DB] Saved {module} RAW data to vision_data table.")
            
        else:
            print(f"[{now_str()}] [WARN] Unknown RAW module: {module}. Data discarded.")
        return
        
    # 2-3. 기타 일반 시스템/STT 이벤트 (events 테이블)
    else: 
        save_event_log(module, action, payload)
        print(f"[{now_str()}] [LOG] Saved general log to events table. Module: {module}")
        
# === [MQTT 콜백] 명령어 처리 후 데이터 라우팅을 'process_and_save_data'로 위임하는 진입점. ===
def on_message(client, userdata, msg):
    """메시지가 수신될 때 호출되며, 토픽에 따라 데이터 저장 또는 명령을 처리합니다."""
    now = now_str() 
    payload = msg.payload.decode()
    topic = msg.topic
    print(f"[{now}] {topic} → {payload}") 

    # 1. === 명령어/요약 트리거 처리 (동적 시간 파싱) ===
    if topic.startswith("command/"):
        
        if topic == "command/summary":
            print(f"[{now}] [CMD] Summary request received → Generating report...")
            
            minutes = 15 # 기본값은 15분
            try:
                # payload는 '30'과 같은 문자열 분 단위이거나 'minutes=30' 형태
                minutes = int(payload.strip())
            except ValueError:
                # payload가 단순 숫자가 아닐 경우 무시하고 기본값 15분 유지
                pass 
            
            # 최소 1분 이상, 최대 180분(3시간)까지만 처리하도록 제한 (안전성 확보)
            minutes = max(1, min(minutes, 180)) 

            # 추출된 minutes 값으로 로그와 IMU 통계를 함께 가져옵니다.
            print(f"[{now}] Fetching logs for the last {minutes} minutes.")
            logs, imu_stats = fetch_logs(minutes) 
            
            # minutes 값을 summarize_logs 함수에 전달합니다.
            summary = summarize_logs(logs, imu_stats, minutes) 
            text_to_speech(summary)
            # LLM 결과 TTS 발화 후 DB에 기록
            save_event_log("LLM", "SAY", summary)

        elif topic == "command/query":
             # 일반 쿼리는 LLM에 바로 질의 후 답변을 TTS로 발화합니다.
             print(f"[{now}] [CMD] Query request received → {payload}")
             save_event_log("USER_STT", "QUERY", payload)
             
             # LLM 질의
             response = query_llm(payload)
             text_to_speech(response)
             save_event_log("LLM", "RESPONSE", response)

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
    # 1. STT 리스닝 스레드 시작
    stt_thread = threading.Thread(target=stt_listening_loop)
    stt_thread.daemon = True # 메인 스레드 종료 시 함께 종료
    stt_thread.start()
    
    # 2. 메인 MQTT 루프 실행 (STT와 동시 실행)
    client.loop_forever()
    
except KeyboardInterrupt:
    print("\n[EXIT] Server stopped by user")
    client.disconnect()
    CURSOR.close() 
    DB_CONN.close()
