import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class ViolenceVideoDataset(Dataset):
    def __init__(self, root_dir, clip_len=16, resize=160, shuffle=True):
        """
        root_dir: data/violence/train or data/violence/val
        """
        self.clip_len = clip_len
        self.resize = resize
        self.samples = []

        classes = ["normal", "violence"]
        for label, cls in enumerate(classes):
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir):
                continue

            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                    self.samples.append({
                        "path": os.path.join(cls_dir, fname),
                        "label": label
                    })

        if shuffle:
            random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def _load_clip(self, path):
        cap = cv2.VideoCapture(path)
        frames = []

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.resize, self.resize))
            frames.append(frame)

        cap.release()

        # Handle empty video
        if len(frames) == 0:
            frames = [
                np.zeros((self.resize, self.resize, 3), dtype=np.uint8)
                for _ in range(self.clip_len)
            ]

        # Uniform sampling to clip_len
        if len(frames) >= self.clip_len:
            step = len(frames) / self.clip_len
            chosen = [frames[int(i * step)] for i in range(self.clip_len)]
        else:
            chosen = frames.copy()
            while len(chosen) < self.clip_len:
                chosen.append(frames[-1])

        clip = np.stack(chosen).astype("float32") / 255.0   # (T, H, W, C)
        clip = torch.from_numpy(clip).permute(0, 3, 1, 2)  # (T, C, H, W)

        return clip

    def __getitem__(self, idx):
        sample = self.samples[idx]
        clip = self._load_clip(sample["path"])
        label = torch.tensor(sample["label"], dtype=torch.long)
        return clip, label
