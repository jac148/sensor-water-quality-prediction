import numpy as np  
from scipy.integrate import odeint  
import matplotlib.pyplot as plt
import time
import math
import random

class FishTank:
    def __init__(self, initial_DO, temperature, fish_count, plant_count, organic_matter, pump_rate):
        self.DO = initial_DO  # 초기 DO 값
        self.temperature = temperature  # 수온
        self.fish_count = fish_count  # 물고기 수
        self.plant_count = plant_count  # 식물 수
        self.organic_matter = organic_matter  # 유기물 양
        self.pump_rate = pump_rate  # 펌프가 켜졌을 때 산소 증가율

    # 시간당 DO 변화율을 계산하는 함수
    def oxygen_dynamics(self, DO, t, pump_on):
        # 물고기 산소 소비(시간당 산소 소비율)
        fish_oxygen_consumption = self.fish_count * self.fish_oxygen_rate()
        # 식물 산소 소비
        plant_oxygen_consumption = self.plant_count * self.plant_oxygen_rate()
        # 유기물 분해에 의한 산소 소비
        organic_oxygen_consumption = self.organic_matter_decomposition_rate()
        # 총 산소 소비
        total_oxygen_consumption = fish_oxygen_consumption + plant_oxygen_consumption + organic_oxygen_consumption
        
        # 펌프 켜져 있을 때 산소 공급량
        oxygen_addition = self.pump_rate if pump_on else 0
        
        dDOdt_10 = (oxygen_addition - total_oxygen_consumption)
                
        # 온도에 따른 산소 소비율 변화
        temperature_factor = (self.temperature - 10)/10
        if isinstance(temperature_factor, complex):
            temperature_factor = temperature_factor.real
    
        dDOdt_float = dDOdt_10 ** temperature_factor
        
        # 복소수인지 확인 후 실수로 변환
        if isinstance(dDOdt_float, complex):
            dDOdt_float = dDOdt_float.real
        
        dDOdt = round(dDOdt_float, 2)
        return dDOdt


    def fish_oxygen_rate(self):
        return 0.1

    def plant_oxygen_rate(self):
        return 0.05


    def organic_matter_decomposition_rate(self):
        return self.organic_matter * 0.01

class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.previous_error = 0
        self.integral = 0

    def update(self, measurement, dt):
        error = self.setpoint - measurement
        self.integral += error * self.Ki * dt
        derivative = (error - self.previous_error) / dt * self.Kd
        
        output = self.Kp * error + self.integral + derivative
        
        self.previous_error = error
        
        return output

# 초기 설정 값
initial_DO = 4.1
temperature = 18.2  #두개만 따로 나중에 불러오면 됨
fish_count = 90
plant_count = 5 
organic_matter = 1
pump_rate = 15

tank = FishTank(initial_DO, temperature, fish_count, plant_count, organic_matter, pump_rate)

pid = PIDController(Kp=0.3, Ki=0.05, Kd=0.1, setpoint=6.0)

# 시뮬레이션 15분
current_time = 0
time_step = 1/12

DO_values = []  # 시간에 따른 DO 값을 저장할 리스트
pump_states = []  # 시간에 따른 펌프 작동 여부를 저장할 리스트

i=1
while current_time < time_step * 24:  # n 시간 동안 시뮬레이션
    DO = tank.DO
    
    # 노이즈 추가 (예: -0.1 ~ 0.1 범위의 노이즈)
    noise = random.uniform(-0.5, 0.5)
    
    DO_values.append(DO)
    control_signal = pid.update(DO, time_step)
    pump_on = 1 if control_signal > 0 else 0
    pump_states.append(pump_on)
    dDOdt = tank.oxygen_dynamics(DO, current_time, pump_on)
    
    # DO 업데이트
    tank.DO += dDOdt * time_step + noise
    current_time += time_step
    
    # print(f"{i:.0f}step: {current_time * 60:.0f} min, DO: {tank.DO:.3f}, pump: {'ON' if pump_on else 'OFF'}, {control_signal:.3f}")
    time.sleep(0.2)  # 시뮬레이션 속도 조절을 위한 딜레이
    
    
    i+=1
    
print(pump_states.count(1))    


# 결과를 보간하여 펌프 상태를 일정 시간 유지
DO_times = np.arange(0, len(DO_values) * time_step, time_step)
pump_times = np.arange(0, len(pump_states) * time_step, time_step)

# print(len(DO_times))
# print(len(DO_values))

DO_values_interp = np.interp(np.arange(0, 24, time_step), DO_times, DO_values)
pump_states_interp = np.interp(np.arange(0, 24, time_step), pump_times, pump_states)

# 시각화
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

# 펌프 작동 여부
color = 'tab:red'
ax1.set_ylabel('on/off', color=color)
ax1.step(range(len(pump_states)), pump_states, color=color, where='post')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, axis='x')
ax1.set_yticks([0, 1])
ax1.set_title('Pump')

# DO 변화
color = 'tab:blue'
ax2.set_xlabel('step(time_step term)')
ax2.set_ylabel('DO', color=color)
ax2.plot(range(len(DO_values)), DO_values, color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.grid(True, axis='x')
ax2.set_title('DO')

plt.xticks(np.arange(1, len(DO_values)))
plt.tight_layout()
plt.show()
