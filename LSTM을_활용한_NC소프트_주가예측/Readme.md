
# LSTM을 활용한 NC소프트 주가예측
## 2020.09.13
하이퍼 파라미터
```
WINDOW_SIZE=20
BATCH_SIZE=32

model = Sequential([
    Conv1D(filters=32, kernel_size=5, padding='causal', activation = 'relu', input_shape=[WINDOW_SIZE, 1]),
    LSTM(16),
    Dense(16, activation='relu'),
    Dense(1),
    Lambda(lambda x: x*20),
])

loss = Huber()
adam = Adam(lr=0.0005)
model.compile(optimizer=adam, loss=loss, metrics=['mae'])
```
실제 값과 예측 값 차이가 너무 큼 

![image](https://user-images.githubusercontent.com/76654360/133051281-19366162-becd-4068-b58f-81935de410e8.png)
