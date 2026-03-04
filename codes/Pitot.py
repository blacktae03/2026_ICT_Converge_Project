import math
import time

class Pitot:
    def __init__(self, adc_instance, channel=0, air_density=1.225):
        """
        :param adc_instance: 미리 생성된 ADC 클래스의 객체
        :param channel: 센서가 연결된 ADC 채널 (A0~A3)
        :param air_density: 공기 밀도 (기본값 1.225)
        """
        self.adc = adc_instance
        self.channel = channel
        self.rho = air_density
        
        # 영점(Offset) 초기값 - 캘리브레이션 전에는 2.5V로 가정
        self.offset_voltage = 2.5 
        print(f"[Pitot] MPXV7002DP 클래스 로드 완료 (채널: {channel})")

    def calibrate(self, num_samples=50):
        """
        바람이 없는 상태에서 영점(2.5V 근처)을 잡는 함수
        """
        print("[Pitot] 속도계 영점 조절 중... (바람을 막아주세요)")
        total_v = 0
        for _ in range(num_samples):
            total_v += self.adc.read_voltage(self.channel)
            time.sleep(0.02)
        
        self.offset_voltage = total_v / num_samples
        print(f"[Pitot] 캘리브레이션 완료! Offset Voltage: {self.offset_voltage:.4f}V")

    def get_airspeed(self):
        """
        현재 전압을 읽어 대기 속도(m/s)를 계산
        """
        current_v = self.adc.read_voltage(self.channel)
        
        # 1. 압력차(Pascal) 계산 
        # MPXV7002DP 공식: P = (Vout - Offset) / (Vsupply * 0.2) * 1000
        # 5V 공급 기준, (Vout - Offset) * 1000 이 대략적인 파스칼 값입니다.
        diff_pressure = (current_v - self.offset_voltage) * 1000.0
        
        # 2. 물리적 한계 처리 (압력이 음수면 속도는 0)
        if diff_pressure < 0:
            return 0.0
        
        # 3. 베르누이 공식 적용: v = sqrt(2 * P / rho)
        airspeed = math.sqrt((2 * diff_pressure) / self.rho)
        return airspeed

# 단독 테스트 코드
if __name__ == "__main__":
    from ADC import ADC # 같은 폴더의 ADC.py 가져오기
    
    my_adc = ADC(gain=1) # ADS1115 생성
    airspeed_sensor = Pitot(my_adc, channel=0)
    
    airspeed_sensor.calibrate()
    
    try:
        while True:
            speed = airspeed_sensor.get_airspeed()
            print(f"현재 대기 속도: {speed:.2f} m/s")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("종료")