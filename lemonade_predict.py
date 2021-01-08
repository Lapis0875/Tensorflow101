import tensorflow as tf
import pandas as pd

# 1. 과거의 데이터를 준비합니다. (레모네이드 판매량 자료 : lemonade.csv)
data = pd.read_csv('lemonade.csv')
independent_variable = data[['온도']]
dependent_variable = data[['판매량']]

# 2. 모델의 구조를 만듭니다.
X = tf.keras.layers.Input([1])
Y = tf.keras.layers.Dense(1)(X)
model = tf.keras.models.Model(X, Y)

# 3, 주어진 데이터로 모델을 학습(fit) 합니다.
# epoch = 전체 데이터를 얼마나 반복하여 학습할 것인지 결정하는 수치
model.fit(independent_variable, dependent_variable, epochs=1000)

# 4. 모델을 이용합니다.
model.predict([[15]])
