import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Frame-level CNN
# -----------------------------
class TinyCNN(nn.Module):
    def __init__(self, in_channels=3, feat_dim=64):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.pool = nn.MaxPool2d(2, 2)

        # Input frame size = 160x160 → 40x40 after pooling
        self.fc = nn.Linear(32 * 40 * 40, feat_dim)

    def forward(self, x):
        # x: (B, 3, 160, 160)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x


# -----------------------------
# Video-level Violence Model
# -----------------------------
class CNN_GRU_Violence(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = TinyCNN(feat_dim=64)

        self.gru = nn.GRU(
            input_size=64,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(64 * 2, 2)  # NORMAL / VIOLENCE

    def forward(self, clips):
        # clips: (B, T, 3, 160, 160)
        B, T, C, H, W = clips.shape

        clips = clips.reshape(B * T, C, H, W)
        feats = self.cnn(clips)           # (B*T, 64)
        feats = feats.reshape(B, T, -1)   # (B, T, 64)

        out, _ = self.gru(feats)          # (B, T, 128)
        last = out[:, -1, :]              # last timestep

        last = self.dropout(last)
        logits = self.fc(last)            # (B, 2)

        return logits
