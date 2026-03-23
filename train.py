import torch
from torch.utils.data import DataLoader
from dataset import ViolenceVideoDataset
from model import CNN_GRU_Violence
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

def main():
    # =====================
    # CONFIGURATION
    # =====================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS = 6            
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-5  #  lower LR to avoid bias

    print("Using device:", DEVICE)

    # =====================
    # LOAD DATA
    # =====================
    print(" Loading videos...")
    train_ds = ViolenceVideoDataset("data/violence/train")
    val_ds   = ViolenceVideoDataset("data/violence/val")
   

    print(f" Train samples: {len(train_ds)}")
    print(f" Val samples:   {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    # =====================
    # MODEL
    # =====================
    model = CNN_GRU_Violence().to(DEVICE)

    #  IMPORTANT: class weights (normal, violence)
    class_weights = torch.tensor([1.5, 1.0]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # =====================
    # TRAINING LOOP
    # =====================
    print(f" Starting training for {EPOCHS} epochs...\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")

        for clips, labels in pbar:
            clips = clips.to(DEVICE)    # [B, T, 3, 112, 112]
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(clips)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total

        print(f" Epoch {epoch}: Loss={epoch_loss:.4f}, Accuracy={epoch_acc:.4f}")

    # =====================
    # SAVE FINAL MODEL
    # =====================
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "violence_model.pth")

    print("\n Training complete!")
    print(" Final model saved as: violence_model.pth")

if __name__ == "__main__":
    main()
