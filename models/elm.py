import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class ELM:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(sparse_output=False)  # Untuk mengubah label menjadi one-hot encoding
        self.hidden_weights = np.random.randn(input_size, hidden_size)
        self.output_weights = None

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        y_encoded = self.encoder.fit_transform(y.reshape(-1, 1))  # Mengubah y menjadi one-hot encoding
        H = np.tanh(np.dot(X_scaled, self.hidden_weights))
        self.output_weights = np.dot(np.linalg.pinv(H), y_encoded)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        H = np.tanh(np.dot(X_scaled, self.hidden_weights))
        predictions = np.dot(H, self.output_weights)
        return np.argmax(predictions, axis=1)  # Mengambil argmax untuk menentukan kelas prediksi
