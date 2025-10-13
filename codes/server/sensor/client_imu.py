import paho.mqtt.client as mqtt
import smbus2
import time
import math
import sys
import json 
from datetime import datetime, timezone

# ====================================================
# 1. 환경 설정 및 상수 정의
# ====================================================

# MQTT 설정
BROKER = "10.10.14.73" # 브로커 주소를 사용자의 환경에 맞게 설정하세요
PORT = 1883
# 서버의 DB 로직이 원시 데이터를 구분할 수 있도록 'RAW' 액션으로 토픽 변경
TOPIC = "project/imu/raw" 

# MPU-6050 I2C 주소 및 레지스터 주소
MPU6050_ADDR = 0x68
I2C_BUS = 1

# 레지스터 주소
PWR_MGMT_1 = 0x6B
GYRO_CONFIG = 0x1B
# 가속도계 3축
ACCEL_XOUT_H = 0x3B 
ACCEL_YOUT_H = 0x3D
ACCEL_ZOUT_H = 0x3F
# 자이로스코프 3축
GYRO_XOUT_H = 0x43
GYRO_YOUT_H = 0x45 
GYRO_ZOUT_H = 0x47

# 스케일 상수 및 필터 계수
ACCEL_SCALE = 16384.0 # 가속도계 1g 스케일 (±2g 기준)
GYRO_SCALE = 131.0 # 자이로스코프 스케일 (±250 deg/s 기준)
RAD_TO_DEG = 57.2957795 # 라디안을 도로 변환
ALPHA = 0.98 # 상보 필터 계수 (자이로스코프 신뢰도 98%)

# ====================================================
# 2. 전역 변수 및 I2C 통신 함수
# ====================================================

bus = None
# 필터링된 Roll/Pitch 각도 및 Yaw 통합 각도
filtered_roll_angle = 0.0
filtered_pitch_angle = 0.0
integrated_yaw_angle = 0.0

# 자이로 오프셋 저장
Gx_offset = 0.0
Gy_offset = 0.0
Gz_offset = 0.0

def now_str():
    """ISO 8601 형식의 현재 UTC 시각을 반환합니다."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def read_word_2c(reg):
    """MPU-6050에서 16비트 데이터를 읽고 2의 보수 처리합니다."""
    global bus
    try:
        high = bus.read_byte_data(MPU6050_ADDR, reg)
        low = bus.read_byte_data(MPU6050_ADDR, reg + 1)
    except IOError:
        print(f"[{now_str()}] ERROR I2C :: 데이터 읽기 실패. 센서 연결 확인 필요.")
        sys.exit(1)
        
    val = (high << 8) + low
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
        bus.write_byte_data(MPU6050_ADDR, GYRO_CONFIG, 0x00) # 자이로 스케일 ±250 deg/s
        # 가속도계 스케일 ±2g 설정 (0x00)
        bus.write_byte_data(MPU6050_ADDR, 0x1C, 0x00)
        print(f"[{now_str()}] INFO Sensor :: MPU-6050 Initialized successfully.")
        time.sleep(0.1)
    except FileNotFoundError:
        print(f"[{now_str()}] ERROR System :: I2C 버스 파일 ({I2C_BUS})을 찾을 수 없습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"[{now_str()}] ERROR Sensor :: MPU-6050 초기화 오류: {e}")
        sys.exit(1)

# ====================================================
# 3. 각도 계산 및 필터링 함수
# ====================================================

def accel_roll(ay, az):
    """가속도계 데이터로 Roll 각도 계산 (X축 회전)"""
    return math.atan2(ay, az) * RAD_TO_DEG

def accel_pitch(ax, ay, az):
    """가속도계 데이터로 Pitch 각도 계산 (Y축 회전)"""
    # X축 가속도와 YZ 평면의 벡터 합 간의 각도를 계산하여 더 안정적인 Pitch 값을 얻습니다.
    return math.atan2(-ax, math.sqrt(ay*ay + az*az)) * RAD_TO_DEG

def complementary_filter(accel_angle, gyro_rate, prev_angle, dt):
    """상보 필터를 적용하여 Roll 또는 Pitch 각도를 계산합니다."""
    
    # 자이로스코프 값 적분 (이전 각도 + 자이로 변화량)
    gyro_angle = prev_angle + gyro_rate * dt

    # 상보 필터 적용: 자이로(단기) 98% + 가속도(장기) 2%
    filtered_angle = ALPHA * gyro_angle + (1.0 - ALPHA) * accel_angle

    return filtered_angle

# ====================================================
# 4. 메인 실행 함수 (통합 로직)
# ====================================================

def main():
    global Gx_offset, Gy_offset, Gz_offset
    global filtered_roll_angle, filtered_pitch_angle, integrated_yaw_angle

    # 1. 센서 초기화
    mpu6050_init()

    # 2. MQTT 클라이언트 생성 및 연결
    client = mqtt.Client()
    try:
        client.connect(BROKER, PORT, 60)
        client.loop_start() 
        print(f"[{now_str()}] INFO MQTT :: Client connected to {BROKER}:{PORT}")
    except Exception as e:
        print(f"[{now_str()}] ERROR MQTT :: Connection failed: {e}")
        sys.exit(1)
    
    # 3. Gyro 3축 오프셋 캘리브레이션
    print(f"[{now_str()}] INFO Sensor :: Calibrating Gyro 3-Axis offset. KEEP SENSOR STILL...")
    sum_Gx_raw, sum_Gy_raw, sum_Gz_raw = 0, 0, 0
    calibration_count = 100
    for _ in range(calibration_count):
        sum_Gx_raw += read_word_2c(GYRO_XOUT_H)
        sum_Gy_raw += read_word_2c(GYRO_YOUT_H)
        sum_Gz_raw += read_word_2c(GYRO_ZOUT_H)
        time.sleep(0.01)
        
    Gx_offset = (sum_Gx_raw / calibration_count) / GYRO_SCALE
    Gy_offset = (sum_Gy_raw / calibration_count) / GYRO_SCALE
    Gz_offset = (sum_Gz_raw / calibration_count) / GYRO_SCALE
    print(f"[{now_str()}] INFO Sensor :: Gyro Offsets (X/Y/Z): {Gx_offset:.2f}/{Gy_offset:.2f}/{Gz_offset:.2f} deg/s")

    # 4. 초기 Roll/Pitch 각도 설정
    # 초기 가속도계 데이터 읽기
    AccX_init = read_word_2c(ACCEL_XOUT_H)
    AccY_init = read_word_2c(ACCEL_YOUT_H)
    AccZ_init = read_word_2c(ACCEL_ZOUT_H)
    Ax_init, Ay_init, Az_init = AccX_init / ACCEL_SCALE, AccY_init / ACCEL_SCALE, AccZ_init / ACCEL_SCALE

    filtered_roll_angle = accel_roll(Ay_init, Az_init)
    filtered_pitch_angle = accel_pitch(Ax_init, Ay_init, Az_init)
    # Yaw는 초기화 시 0으로 설정
    integrated_yaw_angle = 0.0 
    print(f"[{now_str()}] INFO Sensor :: Initial Angles (R/P/Y): {filtered_roll_angle:.2f} / {filtered_pitch_angle:.2f} / {integrated_yaw_angle:.2f} deg")

    # 5. 메인 측정 및 발행 루프
    last_time_s = time.time() 

    try:
        while True:
            current_time_s = time.time()
            dt = current_time_s - last_time_s
            last_time_s = current_time_s

            # --- 5-1. 센서 데이터 읽기 ---
            AccX = read_word_2c(ACCEL_XOUT_H)
            AccY = read_word_2c(ACCEL_YOUT_H)
            AccZ = read_word_2c(ACCEL_ZOUT_H)
            GyroX = read_word_2c(GYRO_XOUT_H)
            GyroY = read_word_2c(GYRO_YOUT_H)
            GyroZ = read_word_2c(GYRO_ZOUT_H)

            # --- 5-2. 스케일 및 오프셋 적용 ---
            Ax, Ay, Az = AccX / ACCEL_SCALE, AccY / ACCEL_SCALE, AccZ / ACCEL_SCALE
            Gx = GyroX / GYRO_SCALE - Gx_offset # Roll Rate (X)
            Gy = GyroY / GYRO_SCALE - Gy_offset # Pitch Rate (Y)
            Gz = GyroZ / GYRO_SCALE - Gz_offset # Yaw Rate (Z)

            # --- 5-3. 각도 계산 및 필터링 ---
            # Roll (X축 회전)
            accel_roll_angle = accel_roll(Ay, Az)
            filtered_roll_angle = complementary_filter(accel_roll_angle, Gx, filtered_roll_angle, dt)

            # Pitch (Y축 회전)
            accel_pitch_angle = accel_pitch(Ax, Ay, Az)
            filtered_pitch_angle = complementary_filter(accel_pitch_angle, Gy, filtered_pitch_angle, dt)
             
            # Yaw (Z축 회전) - 자이로스코프 적분만 사용 (오차 누적 주의!)
            integrated_yaw_angle += Gz * dt 
            
            # Yaw 각도를 0~360 범위로 유지
            integrated_yaw_angle = integrated_yaw_angle % 360.0
            if integrated_yaw_angle < 0:
                integrated_yaw_angle += 360.0


            # --- 5-4. MQTT 메시지 생성 (JSON) ---
            # 서버의 DB 테이블에 맞추어 roll, pitch, yaw 키를 사용합니다.
            msg_payload = json.dumps({
                "roll": round(filtered_roll_angle, 2),
                "pitch": round(filtered_pitch_angle, 2),
                "yaw": round(integrated_yaw_angle, 2),
                "dt": round(dt, 4),
                "timestamp": now_str()
            })
            
            # --- 5-5. 발행 ---
            result, mid = client.publish(TOPIC, msg_payload, qos=0)
            
            if result == mqtt.MQTT_ERR_SUCCESS:
                print(f"[{now_str()}] INFO PUB :: {TOPIC} → R:{filtered_roll_angle:6.2f} P:{filtered_pitch_angle:6.2f} Y:{integrated_yaw_angle:6.2f} deg | dt: {dt:.4f}s")
            else:
                print(f"[{now_str()}] ERROR PUB :: Failed to publish (Error Code: {result})")
            
            time.sleep(1) # 1s 대기 (초당 1회 측정)
            
    except KeyboardInterrupt:
        print(f"\n[{now_str()}] INFO System :: Measurement stopped by user.")
    except Exception as e:
        print(f"\n[{now_str()}] ERROR System :: An unexpected error occurred: {e}")
    finally:
        client.loop_stop()
        client.disconnect() 
        print(f"[{now_str()}] INFO MQTT :: Client disconnected.")
        if bus:
            bus.close()
        sys.exit(0)

if __name__ == "__main__":
    main()

