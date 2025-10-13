import paho.mqtt.client as mqtt
import smbus2
import time
import math
import sys
import json # JSON 포장을 위해 필요
from datetime import datetime, timezone

# ====================================================
# 1. 환경 설정 및 상수 정의
# ====================================================

# MQTT 설정
BROKER = "10.10.14.73"
PORT = 1883
TOPIC = "project/imu/roll"  # IMU 모듈의 고유 토픽

# MPU-6050 I2C 주소 및 레지스터 주소
MPU6050_ADDR = 0x68    # 모듈의 주소
I2C_BUS = 1            # 라즈베리 파이의 일반적인 I2C 버스 번호

# 레지스터 주소
PWR_MGMT_1   = 0x6B
GYRO_CONFIG  = 0x1B
ACCEL_YOUT_H = 0x3D
ACCEL_ZOUT_H = 0x3F
GYRO_XOUT_H  = 0x43

# 스케일 상수 및 필터 계수
ACCEL_SCALE = 16384.0
GYRO_SCALE  = 131.0
RAD_TO_DEG  = 57.2957795
ALPHA       = 0.98     # 상보 필터 계수

# ====================================================
# 2. 전역 변수 및 I2C 통신 함수
# ====================================================

bus = None
prev_angle = 0.0 # Filtered Roll 각도 저장
Gx_offset = 0.0  # 자이로스코프 오프셋 저장

def now_str():
    """ISO 8601 형식의 현재 UTC 시각을 반환합니다."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def read_word_2c(reg):
    """MPU-6050에서 16비트 데이터를 읽고 2의 보수 처리합니다."""
    global bus
    try:
        high = bus.read_byte_data(MPU6050_ADDR, reg)
        low  = bus.read_byte_data(MPU6050_ADDR, reg + 1)
    except IOError:
        print(f"[{now_str()}] ERROR I2C :: 데이터 읽기 실패. 센서 연결 확인 필요.")
        # I2C 통신 실패는 치명적이므로 시스템 종료
        sys.exit(1)
        
    val = (high << 8) + low
    
    # 2의 보수 처리 (음수 값 처리)
    if val >= 0x8000:
        return -((65535 - val) + 1)
    else:
        return val

def mpu6050_init():
    """MPU-6050 센서 초기화 및 I2C 버스 열기"""
    global bus
    try:
        bus = smbus2.SMBus(I2C_BUS)
        bus.write_byte_data(MPU6050_ADDR, PWR_MGMT_1, 0x00) # 슬립 모드 해제
        bus.write_byte_data(MPU6050_ADDR, GYRO_CONFIG, 0x00) # ±250 deg/s
        print(f"[{now_str()}] INFO  Sensor  :: MPU-6050 Initialized successfully.")
        time.sleep(0.1)
    except FileNotFoundError:
        print(f"[{now_str()}] ERROR System  :: I2C 버스 파일 ({I2C_BUS})을 찾을 수 없습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"[{now_str()}] ERROR Sensor  :: MPU-6050 초기화 오류: {e}")
        sys.exit(1)

# ====================================================
# 3. 각도 계산 및 필터링 함수
# ====================================================

def accel_roll(ay, az):
    """가속도계 데이터로 Roll 각도 계산"""
    return math.atan2(ay, az) * RAD_TO_DEG

def complementary_filter(accel_angle, gyro_rate, dt):
    """상보 필터를 적용하여 Roll 각도를 계산합니다."""
    global prev_angle
    
    # 자이로스코프 값 적분 (이전 각도 + 자이로 변화량)
    gyro_angle = prev_angle + gyro_rate * dt

    # 상보 필터 적용
    prev_angle = ALPHA * gyro_angle + (1.0 - ALPHA) * accel_angle

    return prev_angle

# ====================================================
# 4. 메인 실행 함수 (통합 로직)
# ====================================================

def main():
    global Gx_offset
    global prev_angle

    # 1. 센서 초기화
    mpu6050_init()

    # 2. MQTT 클라이언트 생성 및 연결
    client = mqtt.Client()
    try:
        client.connect(BROKER, PORT, 60)
        client.loop_start() # 백그라운드에서 네트워크 루프 시작 (논블로킹)
        print(f"[{now_str()}] INFO  MQTT    :: Client connected to {BROKER}:{PORT}")
    except Exception as e:
        print(f"[{now_str()}] ERROR MQTT    :: Connection failed: {e}")
        sys.exit(1)
    
    # 3. Gyro 오프셋 캘리브레이션
    print(f"[{now_str()}] INFO  Sensor  :: Calibrating GyroX offset. KEEP SENSOR STILL...")
    sum_Gx_raw = 0
    calibration_count = 100
    for _ in range(calibration_count):
        GyroX_raw = read_word_2c(GYRO_XOUT_H)
        sum_Gx_raw += GyroX_raw
        time.sleep(0.01)
    Gx_offset = (sum_Gx_raw / calibration_count) / GYRO_SCALE
    print(f"[{now_str()}] INFO  Sensor  :: GyroX Offset: {Gx_offset:.4f} deg/s")

    # 4. 초기 Roll 각도 설정
    AccY_init = read_word_2c(ACCEL_YOUT_H)
    AccZ_init = read_word_2c(ACCEL_ZOUT_H)
    Ay_init = AccY_init / ACCEL_SCALE
    Az_init = AccZ_init / ACCEL_SCALE
    prev_angle = accel_roll(Ay_init, Az_init)
    print(f"[{now_str()}] INFO  Sensor  :: Initial Filtered Roll Angle: {prev_angle:.2f} deg")

    # 5. 메인 측정 및 발행 루프
    last_time_s = time.time() 

    try:
        while True:
            current_time_s = time.time()
            dt = current_time_s - last_time_s
            last_time_s = current_time_s

            # 센서 데이터 읽기 및 처리
            AccY = read_word_2c(ACCEL_YOUT_H)
            AccZ = read_word_2c(ACCEL_ZOUT_H)
            GyroX = read_word_2c(GYRO_XOUT_H)
            Ay = AccY / ACCEL_SCALE
            Az = AccZ / ACCEL_SCALE
            Gx = GyroX / GYRO_SCALE - Gx_offset

            # 각도 계산 및 필터링
            accel_angle = accel_roll(Ay, Az)
            filtered_roll = complementary_filter(accel_angle, Gx, dt)

            # MQTT 메시지 생성 (JSON)
            msg_payload = json.dumps({
                "roll_angle": round(filtered_roll, 2),
                "dt": round(dt, 4),
                "timestamp": now_str()
            })
            
            # 발행
            result, mid = client.publish(TOPIC, msg_payload, qos=0)
            
            if result == mqtt.MQTT_ERR_SUCCESS:
                print(f"[{now_str()}] INFO  PUB     :: {TOPIC} → Roll: {filtered_roll:6.2f} deg | dt: {dt:.4f}s")
            else:
                print(f"[{now_str()}] ERROR PUB     :: Failed to publish (Error Code: {result})")
            
            time.sleep(1) # 50ms 대기 (주기 제어)
            
    except KeyboardInterrupt:
        print(f"\n[{now_str()}] INFO  System  :: Measurement stopped by user.")
    except Exception as e:
        print(f"\n[{now_str()}] ERROR System  :: An unexpected error occurred: {e}")
    finally:
        # MQTT 클라이언트 연결 해제 및 I2C 버스 닫기
        client.loop_stop()
        client.disconnect() 
        print(f"[{now_str()}] INFO  MQTT    :: Client disconnected.")
        if bus:
             bus.close()
        sys.exit(0)

if __name__ == "__main__":
    main()
