import cv2
import torch
from collections import deque
import os
from model import CNN_GRU_Violence

CLIP_LEN = 16
FRAME_SIZE = 160

device = torch.device("cpu")

# ✅ Robust model path (VERY IMPORTANT)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "violence_model.pth")

print("Loading model from:", MODEL_PATH)

model = CNN_GRU_Violence().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


def analyze_video(video_path):

    cap = cv2.VideoCapture(video_path)

    frame_buffer = deque(maxlen=CLIP_LEN)

    violence_detected = False
    max_prob = 0.0  # ✅ track highest probability

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (FRAME_SIZE, FRAME_SIZE))

        tensor = torch.from_numpy(resized).float() / 255.0
        tensor = tensor.permute(2, 0, 1)

        frame_buffer.append(tensor)

        if len(frame_buffer) == CLIP_LEN:

            clip = torch.stack(list(frame_buffer))
            clip = clip.unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(clip)
                probs = torch.softmax(logits, dim=1)
                violence_prob = probs[0, 1].item()

                # ✅ track max confidence
                max_prob = max(max_prob, violence_prob)

                # ✅ slightly higher threshold (reduces false positives)
                if violence_prob > 0.75:
                    violence_detected = True
                    break

    cap.release()

    return {
        "result": "VIOLENCE" if violence_detected else "NORMAL",
        "confidence": round(max_prob, 2)
    }