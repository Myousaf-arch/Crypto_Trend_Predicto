import torch
import matplotlib.pyplot as plt
from dataset import get_crypto_data, create_sequences
from model import CryptoTransformer

print("Evaluation shuru ho rahi hai... Data aur Model load kar rahe hain.")

# 1. Data load karna
WINDOW_SIZE = 60
scaled_data, scaler = get_crypto_data(ticker="BTC-USD")
X, y = create_sequences(scaled_data, window_size=WINDOW_SIZE)

# 2. Tera Trained Model (Dimagh) load karna
device = torch.device("cpu")
model = CryptoTransformer(feature_size=1, d_model=64, nhead=4, num_layers=2)
model.load_state_dict(torch.load("crypto_transformer.pth", map_location=device, weights_only=True))
model.eval() # Test mode on (seekhna band)

# 3. Model se Predictions lena (bina autograd ke)
with torch.no_grad():
    predictions = model(X)

# 4. Data ko wapas Asal (Dollars $) mein badalna
predictions_actual = scaler.inverse_transform(predictions.numpy())
y_actual = scaler.inverse_transform(y.numpy())

# 5. Graph (Chart) Banana
plt.figure(figsize=(12, 6))
plt.plot(y_actual, label="Asal Price (Actual)", color="blue", alpha=0.6)
plt.plot(predictions_actual, label="Model ki Prediction (AI)", color="red", alpha=0.8, linestyle="dashed")

plt.title("Bitcoin (BTC) - Asal vs AI Prediction")
plt.xlabel("Din (Days)")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)

print("✅ Graph ban gaya hai! Nayi window mein open ho raha hai...")
plt.show()