import tensorflow as tf
import numpy as np
import random
import os

# 1. 랜덤 시드 고정 (이 3줄이 핵심입니다!)
# ------------------------------------------------
seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
# ------------------------------------------------

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# (이하 코드는 동일)
Xt = np.array([[1.0], [2.0], [3.0]], dtype=float)
Yt = np.array([1.0, 4.0, 2.0], dtype=float) # 산 모양 데이터

model = Sequential([
    tf.keras.Input(shape=(1,)), 
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss='mse')
model.fit(Xt, Yt, epochs=500, verbose=0)

# 결과 확인
x_range = np.linspace(0, 4, 100).reshape(-1, 1)
y_pred = model.predict(x_range)

plt.plot(x_range, y_pred)
plt.scatter(Xt, Yt, color='red')
plt.show()