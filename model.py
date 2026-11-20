import torch
import torch.nn as nn

class CryptoTransformer(nn.Module):
    def __init__(self, feature_size=1, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(CryptoTransformer, self).__init__()
        
        # 1. Input embedding: 1 feature (price) ko 64 dimensions mein expand karna
        self.input_layer = nn.Linear(feature_size, d_model)
        
        # 2. Transformer Encoder: Yeh wo hissa hai jo past 60 dino par "Attention" dega
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Output layer: Aakhri prediction (1 single price value)
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x):
        # x ka aane wala shape: [Batch_Size, 60 (Window_Size), 1 (Feature)]
        out = self.input_layer(x)
        
        # Transformer ko data pass karna
        out = self.transformer_encoder(out)
        
        # Humein sirf aakhri din (last time-step) ka nateeja chahiye agla din predict karne ke liye
        out = out[:, -1, :]
        out = self.fc_out(out)
        
        return out

# --- TEST KARNE KE LIYE ---
if __name__ == "__main__":
    print("Testing Transformer Model...")
    # Dummy data banate hain check karne ke liye: Batch size 32, 60 days, 1 feature
    dummy_input = torch.rand(32, 60, 1)
    model = CryptoTransformer()
    output = model(dummy_input)
    print("Model test successful!")
    print("Output Shape:", output.shape) # Expected shape: [32, 1]