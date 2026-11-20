import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_crypto_data, create_sequences
from model import CryptoTransformer

# --- Training ki Settings ---
EPOCHS = 10          # Kitni dafa model poore data ko dekhega
BATCH_SIZE = 32      # Ek waqt mein kitne din ka data process karega
LEARNING_RATE = 0.001 # Model ke seekhne ki speed
WINDOW_SIZE = 60      # Pichle 60 din ka record

print("Data load ho raha hai... Thora wait karein.")

# 1. Data lana (dataset.py se)
scaled_data, scaler = get_crypto_data(ticker="BTC-USD")
X, y = create_sequences(scaled_data, window_size=WINDOW_SIZE)

# Data ko batches mein todna
dataset = torch.utils.data.TensorDataset(X, y)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 2. Dimagh (Model) ko jagana (model.py se)
model = CryptoTransformer(feature_size=1, d_model=64, nhead=4, num_layers=2)

# Ghalti (Loss) calculate karne aur usay theek karne (Optimizer) ka tareeqa
criterion = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 3. Asal Training Loop (Jahan model seekhta hai)
print("Training Shuru Ho Rahi Hai 🚀")
model.train()

for epoch in range(EPOCHS):
    total_loss = 0.0
    
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()                 # Purani calculations clear karo
        
        predictions = model(batch_X)          # Model se prediction lo
        loss = criterion(predictions, batch_y)# Asal price se compare karke ghalti nikalo
        
        loss.backward()                       # Ghalti se seekho (Backpropagation)
        optimizer.step()                      # Model ke dimagh ko update karo
        
        total_loss += loss.item()
    
    # Har epoch ke baad ghalti (loss) print karna
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss (Ghalti): {total_loss/len(dataloader):.6f}")

# 4. Pura seekhne ke baad Model ko Save karna
torch.save(model.state_dict(), "crypto_transformer.pth")
print("✅ Mubarak ho! Model 'crypto_transformer.pth' ke naam se save ho gaya hai!")