from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from model import CryptoTransformer

app = FastAPI()

# Frontend (HTML/JS) se communication allow karne ke liye CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Trained Model Load Karna
device = torch.device("cpu")
model = CryptoTransformer(feature_size=1, d_model=64, nhead=4, num_layers=2)
model.load_state_dict(torch.load("crypto_transformer.pth", map_location=device, weights_only=True))
model.eval() # Model test/inference mode par set ho gaya

@app.get("/predict/{ticker}")
def predict_price(ticker: str = "BTC-USD"):
    try:
        # 2. Live market se pichle 60 din ka data uthana
        df = yf.download(ticker, period="60d", interval="1d")
        prices = df['Close'].values.reshape(-1, 1)
        
        # 3. Data Scaling
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_prices = scaler.fit_transform(prices)
        
        # 4. Shape set karna for model input [1, 60, 1]
        X_input = torch.tensor(scaled_prices, dtype=torch.float32).unsqueeze(0)
        
        # 5. Model se prediction lena (no_grad se speed fast hoti hai)
        with torch.no_grad():
            scaled_pred = model(X_input).item()
            
        # 6. Scaled value ko wapas actual dollars ($) mein badalna
        predicted_price = scaler.inverse_transform([[scaled_pred]])[0][0]
        current_price = prices[-1][0]
        
        trend = "Upar Jayega 🚀" if predicted_price > current_price else "Neeche Girega 📉"
        
        return {
            "coin": ticker.upper(),
            "aaj_ki_price": round(float(current_price), 2),
            "kal_ki_prediction": round(float(predicted_price), 2),
            "trend": trend
        }
    except Exception as e:
        return {"error": str(e)}