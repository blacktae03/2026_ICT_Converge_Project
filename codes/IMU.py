import time
import math
from mpu6050 import mpu6050

class IMU:
    def __init__(self, address=0x68):
        """
        IMU 센서 초기화 함수
        :param address: I2C 주소 (기본값 0x68)
        """
        try:
            self.sensor = mpu6050(address)
            print(f"[IMU] MPU-6050 연결 성공 (주소: {hex(address)})")
        except Exception as e:
            print(f"[IMU] 연결 실패: {e}")
            self.sensor = None

        # [원리 1] 오차 보정용 변수 (Offset)
        # 센서는 가만히 있어도 0이 안 나옵니다. 초기값을 뺴줘야 정확합니다.
        self.gyro_x_offset = 0.0
        self.gyro_y_offset = 0.0
        self.gyro_z_offset = 0.0
        
        # [원리 2] 센서 퓨전용 변수 (상보 필터)
        # 가속도(진동에 취약)와 자이로(드리프트 발생)를 섞어서 정확한 각도를 만듭니다.
        self.roll = 0.0
        self.pitch = 0.0
        self.last_time = time.time()

    def calibrate_gyro(self, num_samples=100):
        """
        초기 자이로스코프 오차(Drift)를 계산하는 함수.
        기체를 평평한 곳에 가만히 두고 실행해야 함.
        """
        if self.sensor is None: return

        print("[IMU] 자이로 캘리브레이션 시작... (움직이지 마세요)")
        sum_gx, sum_gy, sum_gz = 0, 0, 0

        for _ in range(num_samples):
            gyro_data = self.sensor.get_gyro_data()
            sum_gx += gyro_data['x']
            sum_gy += gyro_data['y']
            sum_gz += gyro_data['z']
            time.sleep(0.01)

        self.gyro_x_offset = sum_gx / num_samples
        self.gyro_y_offset = sum_gy / num_samples
        self.gyro_z_offset = sum_gz / num_samples
        print(f"[IMU] 캘리브레이션 완료! Offset -> X:{self.gyro_x_offset:.2f}, Y:{self.gyro_y_offset:.2f}")

    def update(self):
        """
        현재 각도(Roll, Pitch)를 계산하여 업데이트하는 핵심 함수.
        메인 루프에서 계속 호출해줘야 함.
        """
        if self.sensor is None: return None

        # 1. 시간 간격(dt) 계산 (적분을 위해 필요)
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time

        # 2. Raw 데이터 읽기
        accel = self.sensor.get_accel_data()
        gyro = self.sensor.get_gyro_data()
        # 센서 자체가 읽을 수 있는 값은 이 2개가 전부임.
        # accel은 x, y, z축의 가속도임.
        # 무인기가 가만히 책상 위에 있다고 할 때는 중력가속도만 작용해서 accel['x'] = 0, accel['y'] = 0, accel['z'] = 1


        # 3. [원리 3] 가속도계로 각도 계산 (중력 가속도 이용)
        # 삼각함수(atan2)를 이용해 중력 벡터가 어디로 쏠렸는지 계산
        # Roll (X축 회전) / Pitch (Y축 회전)
        accel_roll = math.degrees(math.atan2(accel['y'], accel['z']))
        accel_pitch = math.degrees(math.atan2(-accel['x'], math.sqrt(accel['y']**2 + accel['z']**2)))
        # roll은 비행기가 얼마나 왼쪽 또는 오른쪽으로 기울었는가?
        # pitch는 비행기가 얼마나 위 또는 아래로 기울었는가?
        # accel_roll은 가속도 센서로 측정한 roll의 정도
        # accel_pitch는 가속도 센서로 측정한 pitch의 정도
        # yaw는 필요없는 이유 : 비행기가 어느 방향을 바라보고 가는지는 우리한테 중요한 게 아님. 그래도 필요하면 사용 가능

        # 4. [원리 4] 자이로스코프로 각도 계산 (적분)
        # (현재 각도) = (이전 각도) + (회전 속도 * 시간)
        # 오차(Offset)를 뺀 값을 사용해야 함
        gyro_rate_x = gyro['x'] - self.gyro_x_offset
        gyro_rate_y = gyro['y'] - self.gyro_y_offset
        # gyro_rate_x = 기체가 roll 방향으로 기울어지는 속도 계산.
        # gyro_rate_y = 기체가 pitch 방향으로 기울어지는 속도 계산.
        # offset으로 영점을 잡아서 그 영점과의 차이를 구해야함. offset은 calibrate_gyro 함수에서 정의함.


        # 5. [핵심] 상보 필터 (Complementary Filter) 적용
        # "자이로값 96% + 가속도값 4%" 비중으로 섞음
        # 자이로는 빠르지만 조금씩 틀어지고, 가속도는 정확하지만 진동에 흔들림.
        # 이 둘을 섞으면 빠르고 정확한 값을 얻을 수 있음.
        alpha = 0.96
        self.roll = alpha * (self.roll + gyro_rate_x * dt) + (1 - alpha) * accel_roll
        self.pitch = alpha * (self.pitch + gyro_rate_y * dt) + (1 - alpha) * accel_pitch
        # 만약 roll 방향으로 기체가 아무 변함이 없으면 gyro_rate_x는 0임.
        # 근데 이때 roll 방향으로 기울어져 있으면 그때는 가속도 센서 값으로 측정한 기울어진 정도를 각도로 읽으면 되고,
        # 만약 roll 방향으로 불안정하게 계속 떨리면, 가속도 센서에 노이즈가 발생, 이럴 때는 그냥 평균적으로 계속 측정되는 속도로 각도를 계산하면 됨.
        # 그 정도를 alpha로 나눠서 계산하게 한 것임. accel_roll에 0.04 밖에 할당이 안되는 이유는 불안정할 때 그 값을 거의 무시해야하기 때문임.
        # 그리고 이 0.04가 빛을 발할 때는, 저런 진동 같은 게 없을 때 계산 오차가 발생한 걸 가속도 값으로 잡을 수 있음.


        # 결과 반환 (Roll, Pitch, Yaw는 나침반이 없어서 자이로 적분값만 사용하거나 생략)
        return self.roll, self.pitch

    def get_raw_data(self):
        """디버깅용 Raw 데이터 반환"""
        return self.sensor.get_accel_data(), self.sensor.get_gyro_data()

# 이 파일만 단독으로 실행했을 때 테스트하는 코드
if __name__ == "__main__":
    imu = IMU()
    imu.calibrate_gyro()
    
    while True:
        roll, pitch = imu.update()
        print(f"Roll: {roll:.2f} | Pitch: {pitch:.2f}")
        time.sleep(0.05)