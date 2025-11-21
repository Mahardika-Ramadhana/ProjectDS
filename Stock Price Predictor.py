import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# 1. Download data
company = 'INTC'
start = dt.datetime(2010, 1, 1)
end = dt.datetime(2023, 10, 1)
data = yf.download(company, start=start, end=end)
df = data[['Close']].copy()

# 2. Membuat label naik/turun/stabil
threshold = 0.001  # 0.1% toleransi stabil
df['Target'] = df['Close'].shift(-1) - df['Close']
df['Direction'] = df['Target'].apply(lambda x: 1 if x > threshold else (-1 if x < -threshold else 0))
df.dropna(inplace=True)

# 3. Normalisasi harga penutupan
scaler = MinMaxScaler()
df['Close'] = scaler.fit_transform(df[['Close']])

# 4. Membuat sequence data untuk LSTM
def create_sequences(data, labels, seq_length=10):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(labels[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 10
X, y = create_sequences(df['Close'].values, df['Direction'].values, seq_length)
y = to_categorical(y + 1, num_classes=3)  # -1,0,1 menjadi 0,1,2

# 5. Split data train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# 6. Model LSTM
model = Sequential([
    LSTM(64, input_shape=(seq_length, 1), return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(3, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 7. Training dengan epoch lebih banyak
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# 8. Evaluasi
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.2f}")

# 9. Prediksi satu hari ke depan
last_sequence = df['Close'].values[-seq_length:]
last_sequence = last_sequence.reshape((1, seq_length, 1))
pred = model.predict(last_sequence)
direction = np.argmax(pred) - 1  # 0: turun, 1: stabil, 2: naik -> -1,0,1

if direction == 1:
    print("Prediksi: Harga akan NAIK besok.")
elif direction == -1:
    print("Prediksi: Harga akan TURUN besok.")
else:
    print("Prediksi: Harga akan STABIL besok.")

# (Opsional) Plot akurasi training
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Training & Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()