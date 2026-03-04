# 설치 명령어: pip install adafruit-circuitpython-ads1x15
import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn

class ADC:
    def __init__(self, gain=1):
        """
        ADC 모듈 초기화
        :param gain: 증폭률 (1이면 약 +/- 4.096V 범위 측정)
        """
        try:
            # 1. I2C 통신 준비
            i2c = busio.I2C(board.SCL, board.SDA)
            # 2. ADS1115 객체 생성
            self.ads = ADS.ADS1115(i2c)
            self.ads.gain = gain
            print(f"[ADC] ADS1115 초기화 완료 (Gain: {gain})")
        except Exception as e:
            print(f"[ADC] 초기화 실패: {e}")
            self.ads = None

    def read_voltage(self, channel_index=0):
        """
        특정 채널의 전압(Voltage)을 읽어오는 함수
        :param channel_index: 0, 1, 2, 3 (A0 ~ A3)
        """
        if self.ads is None: return 0.0

        # 채널 선택 (예: AnalogIn(self.ads, ADS.P0))
        channels = [ADS.P0, ADS.P1, ADS.P2, ADS.P3]
        chan = AnalogIn(self.ads, channels[channel_index])
        
        # chan.voltage는 라이브러리가 자동으로 전압으로 변환해준 값임
        return chan.voltage

    def read_differential(self):
        """
        [원리] 차압 센서 전용 모드 (A0와 A1의 차이를 측정)
        피토튜브 센서는 두 구멍의 압력 차이를 측정하므로 이 모드가 더 정확할 수 있습니다.
        """
        chan = AnalogIn(self.ads, ADS.P0, ADS.P1)
        return chan.voltage

# 테스트용 코드
if __name__ == "__main__":
    my_adc = ADC()
    while True:
        voltage = my_adc.read_voltage(0) # A0 채널 읽기
        print(f"현재 측정 전압: {voltage:.4f} V")
        import time
        time.sleep(0.5)