import numpy as np  # Import library numpy untuk operasi numerik

# Fungsi aktivasi sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))  # Menghitung sigmoid dari x

# Turunan dari fungsi sigmoid
def sigmoid_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))  # Menghitung turunan sigmoid

# Kelas NeuralNetwork untuk membangun neural network sederhana
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Inisialisasi bobot dan bias layer pertama (input ke hidden)
        self.W1 = np.random.randn(input_size, hidden_size)  # Bobot input ke hidden
        self.b1 = np.zeros((1, hidden_size))                # Bias hidden layer
        # Inisialisasi bobot dan bias layer kedua (hidden ke output)
        self.W2 = np.random.randn(hidden_size, output_size) # Bobot hidden ke output
        self.b2 = np.zeros((1, output_size))                # Bias output layer

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1      # Linear kombinasi input dan bobot + bias (layer 1)
        self.a1 = sigmoid(self.z1)                  # Aktivasi sigmoid pada layer 1
        self.z2 = np.dot(self.a1, self.W2) + self.b2# Linear kombinasi hidden dan bobot + bias (layer 2)
        self.a2 = sigmoid(self.z2)                  # Aktivasi sigmoid pada layer 2 (output)
        return self.a2                              # Mengembalikan output akhir

    def backward(self, X, y, output, lr=0.1):
        m = X.shape[0]                              # Jumlah sampel data
        dz2 = (output - y) * sigmoid_deriv(self.z2) # Error pada output layer dikali turunan sigmoid
        dW2 = np.dot(self.a1.T, dz2) / m            # Gradien bobot layer 2
        db2 = np.sum(dz2, axis=0, keepdims=True) / m# Gradien bias layer 2

        dz1 = np.dot(dz2, self.W2.T) * sigmoid_deriv(self.z1) # Error pada hidden layer
        dW1 = np.dot(X.T, dz1) / m                            # Gradien bobot layer 1
        db1 = np.sum(dz1, axis=0, keepdims=True) / m          # Gradien bias layer 1

        # Update bobot dan bias dengan gradien descent
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

    def train(self, X, y, epochs=1000, lr=0.1):
        for i in range(epochs):                       # Loop sebanyak epoch
            output = self.forward(X)                  # Forward propagation
            self.backward(X, y, output, lr)           # Backpropagation dan update bobot
            if i % 100 == 0:                          # Setiap 100 epoch, print loss
                loss = np.mean((y - output) ** 2)     # Hitung mean squared error
                print(f"Epoch {i}, Loss: {loss:.4f}") # Tampilkan epoch dan loss

# Contoh penggunaan
if __name__ == "__main__":
    # Dataset XOR
    X = np.array([[0,0],[0,1],[1,0],[1,1]])          # Input data
    y = np.array([[0],[1],[1],[0]])                  # Target output

    nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1) # Inisialisasi neural network
    nn.train(X, y, epochs=1000, lr=0.1)              # Melatih neural network

    # Prediksi hasil
    preds = nn.forward(X)                            # Melakukan prediksi pada data X
    print("Predictions:")                            # Tampilkan hasil prediksi
    print(preds)