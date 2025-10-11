import paho.mqtt.client as mqtt
import pymysql
from datetime import datetime, timezone
from gtts import gTTS
import os
from openai import OpenAI # OpenAI API 용
import re # 정규표현식 사용을 위해 추가

# === DB 연결 (MariaDB) ===
db = pymysql.connect(
    host="localhost",
    user="marine_user",       # 위에서 만든 사용자
    password="sksk",          # 설정한 비밀번호
    database="marine",
    charset="utf8mb4"
)
cursor = db.cursor()

# === MQTT 설정 ===
BROKER = "0.0.0.0"
PORT = 1883
TOPIC = "project/#"

# === OpenAI 클라이언트 설정 ===
client_llm = OpenAI() # 키는 환경 변수에서 자동 로드됩니다.

# === TTS 텍스트 전처리 함수 ===
def clean_tts_text(text: str) -> str:
    """
    TTS 재생을 위해 불필요한 마크다운 및 특수 공백 문자를 완전히 제거합니다.
    """
    # 1. 특수 마크다운 문자 제거
    # \*\* (볼드체), \*\* (기타 별표), \# (헤더), \- (목록 기호) 등
    cleaned_text = text.replace('**', '').replace('*', '').replace('#', '').replace('-', '')
    
    # 2. 유니코드 공백 및 제어 문자 제거 (가장 핵심적인 수정)
    # \s+는 일반적인 공백, 탭, 줄바꿈을 포함하지만, 다른 유니코드 공백도 정리해야 합니다.
    # [ \t\n\r\f\v] 외의 모든 비표준 공백 문자를 일반 공백으로 치환합니다.
    cleaned_text = re.sub(r'[\u2000-\u200A\u202F\u205F\u3000]', ' ', cleaned_text)
    
    # 3. 모든 연속된 공백(일반 공백, 탭, 줄바꿈)을 하나의 공백으로 치환
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    
    # 4. 양 끝의 공백 제거
    cleaned_text = cleaned_text.strip()
    
    return cleaned_text

# === LLM 질의응답 함수 ===
def query_llm(prompt: str) -> str:
    try:
        messages = [
            {"role": "system", "content": "너는 선박 항해 보조관이야. 사용자에게 요청받은 로그를 바탕으로 항해일지, 선박 상태, 위험 감지 상황 등을 **간결하고 구조적으로 한국어로 브리핑**해야 해."},
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
    try:
        sql = """
            SELECT ts, module, action, payload
            FROM events
            WHERE ts >= NOW() - INTERVAL %s MINUTE
            ORDER BY ts ASC
        """
        cursor.execute(sql, (minutes,))
        rows = cursor.fetchall()
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
    text = "\n".join(logs)
    prompt = f"""
    다음은 선박 항해 로그입니다:
    {text}

    위 로그를 간결하고 구조적으로 요약해줘. (한국어로)
    """
    print("[LLM] Summarizing logs using GPT-4o mini...")
    summary = query_llm(prompt)
    print("[SUMMARY]\n", summary)
    return summary
    
# === TTS 변환 및 재생 ===
def text_to_speech(text, filename="summary.mp3"):
    try:
        tts = gTTS(text=text, lang="ko")
        tts.save(filename)
        os.system(f"mpv --no-terminal --volume=100 --speed=1.3 {filename}") 
        print("[TTS] Summary spoken successfully.")
    except Exception as e:
        print(f"[TTS Error] {e}")


# === MQTT 콜백 ===
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("[OK] Connected to broker")
        client.subscribe("project/#")
        client.subscribe("command/#")
        print("[SUB] Subscribed to project/# and command/#")
    else:
        print("[FAIL] Connection failed, code:", rc)

def on_message(client, userdata, msg):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S") # UTC 기준
    payload = msg.payload.decode()
    topic = msg.topic
    print(f"[{now}] {topic} → {payload}")

    # 로그 DB 저장
    try:
        sql = "INSERT INTO events (module, action, payload, ts) VALUES (%s, %s, %s, %s)"
        # topic은 보통 "project/모듈명" 이므로 split
        module = topic.split("/")[1] if "/" in topic else topic
        action = "EVT"   # 기본값, 필요하면 파싱해서 변경 가능
        cursor.execute(sql, (module, action, payload, now))
        db.commit()
    except Exception as e:
        print(f"[DB-ERROR] {e}")

    # === 명령 트리거 ===
    if topic == "command/summary":
        print("[CMD] Summary request received → Generating report...")
        logs = fetch_logs(10)
        summary = summarize_logs(logs)
        print("[SUMMARY]\n", summary)
        text_to_speech(summary)

# === MQTT 클라이언트 생성 ===
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
    cursor.close()
    db.close()

