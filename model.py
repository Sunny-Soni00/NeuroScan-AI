import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """
    (Helper Block)
    AI ek baar mein nahi samajhta, isliye hum har step pe 2 baar process karte hain.
    Conv2d -> ReLU -> Conv2d -> ReLU
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False), # Image scan karo
            nn.BatchNorm2d(out_channels),  # Data ko stabilize karo
            nn.ReLU(inplace=True),         # Negative values hatao
            
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False), # Dobara scan karo
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        """
        in_channels=1 (Kyuki MRI Black & White hai)
        out_channels=1 (Kyuki Output bhi Black & White Mask hoga)
        """
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # Image size half karne ke liye

        # Feature sizes: 64 -> 128 -> 256 -> 512
        features = [64, 128, 256, 512]

        # === DOWN PART (Encoder) ===
        # Image choti hoti jayegi, par details badhti jayengi
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # === BOTTLENECK (Bottom Part) ===
        # Sabse compressed state (sabse deep features)
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # === UP PART (Decoder) ===
        # Image wapas badi hogi taaki mask bana sakein
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))

        # === FINAL OUTPUT ===
        # 1x1 convolution jo final result (Tumor yes/no) dega
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Downward Path (Compression)
        for down in self.downs:
            x = down(x)
            skip_connections.append(x) # Raste ki details save kar lo
            x = self.pool(x)

        # Bottom
        x = self.bottleneck(x)
        
        # Details wapas reverse order mein chahiye
        skip_connections = skip_connections[::-1]

        # Upward Path (Expansion)
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x) # Image badi karo
            skip_connection = skip_connections[idx//2] # Purani details add karo
            
            # Agar size mismatch ho (kabhi kabhi hota hai), toh resize karo
            if x.shape != skip_connection.shape:
                x = torch.nn.functional.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1) # Dono ko jodo
            x = self.ups[idx+1](concat_skip) # Process karo

        return self.final_conv(x)

# --- Test Block (Sirf shape check karne ke liye) ---
if __name__ == "__main__":
    x = torch.randn((3, 1, 256, 256)) # Fake image (3 images ka batch)
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    print(f"Input Shape: {x.shape}")
    print(f"Output Shape: {preds.shape}")
    print("✅ Model Architecture is Correct! (Ready for Step 3)")