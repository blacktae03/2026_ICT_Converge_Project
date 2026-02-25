import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2 # L2 규제 불러오기
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. 데이터 준비 (Feature를 1개로 축소!)
# ---------------------------------------------------------
# 입력: [1.0], [2.0] (1차원)
Xt = np.array([[1.0], [2.0], [3.0]], dtype=float)
Yt = np.array([1.0, 4.0, 2.0], dtype=float)


# 가상의 센서 데이터 (0V ~ 5V)

# -------------------------------------------------------
# [방법 1] 그냥 5로 나누기 (정태님의 직감!)
# -------------------------------------------------------
X_norm = Xt / 5.0

print(f"원래 값:\n{Xt}")
print(f"변환 값 (0~1):\n{X_norm}")

# 결과:
# 0.0  -> 0.0
# 2.5  -> 0.5
# 5.0  -> 1.0

# ---------------------------------------------------------
# 2. 다항식 전처리 (1개 -> 3개로 확장)
# ---------------------------------------------------------
# 1개의 입력 x가 들어오면 -> [1, x, x^2] 3개로 변환됨
# poly = PolynomialFeatures(degree=2)
# Xt_poly = poly.fit_transform(Xt)

# print("변환된 데이터 형태:", Xt_poly.shape) # (2, 3)이 나와야 함

# ---------------------------------------------------------
# 3. 모델 정의 (Input Shape 수정 중요!)
# ---------------------------------------------------------
model = Sequential([
    # 입력이 [1, x, x^2] 3개이므로 shape=(3,)
    tf.keras.Input(shape=(1,)), 
    
    # Hidden Layer (이 구조가 어떻게 곡선을 만드는지 보세요)
    Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(8, activation='relu', kernel_regularizer=l2(0.01)),
    
    # Output Layer
    Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss='mse')

# ---------------------------------------------------------
# 4. 학습
# ---------------------------------------------------------
print("학습 시작...")
model.fit(Xt, Yt, epochs=500, verbose=0)
print("학습 완료!")

# ---------------------------------------------------------
# 5. 그래프 그리기 (0.0 ~ 5.0까지 쭉 그려보기)
# ---------------------------------------------------------
# 테스트 데이터 생성 (0부터 5까지 100개의 점)
x_range = np.linspace(0, 5, 100).reshape(-1, 1) # (100, 1) 형태로 맞춤

# 테스트 데이터도 똑같이 전처리 (transform)
# x_range_poly = poly.transform(x_range)

# 예측
y_pred = model.predict(x_range)

# 시각화
plt.figure(figsize=(10, 6))

# AI가 예측한 함수 (파란 선)
plt.plot(x_range, y_pred, label='AI Prediction Model', color='blue', linewidth=2)

# 실제 학습시킨 데이터 (빨간 점)
plt.scatter(Xt, Yt, color='red', s=150, label='Training Data (Goal)', zorder=5)

# (3,3) 예측 지점 표시 (녹색 별)
test_val = np.array([[3.0]])
pred_val = model.predict(test_val)
plt.scatter(test_val, pred_val, color='green', marker='*', s=300, label=f'Input 3.0 -> Pred {pred_val[0][0]:.2f}', zorder=5)

plt.title('Simple 1D Input Regression')
plt.xlabel('Input Value (x)')
plt.ylabel('Output Value (y)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()