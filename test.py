# test_model_quick.py
import torch
from dataset import ViolenceVideoDataset
from model import CNN_GRU_Violence

model = CNN_GRU_Violence()
model.load_state_dict(torch.load("violence_model.pth", map_location="cpu"))
model.eval()

ds = ViolenceVideoDataset("data/violence/val")

for i in range(10):
    clip, label = ds[i]
    clip = clip.unsqueeze(0)
    with torch.no_grad():
        out = model(clip)
        prob = torch.softmax(out, dim=1)[0,1].item()
    print(f"GT={label}  ViolenceProb={prob:.3f}")
