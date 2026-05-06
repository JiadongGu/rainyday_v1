import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import sys

CLASSES = ["dew","fogsmog","frost","glaze","hail","lightning","rain","rainbow","rime","sandstorm","snow"]

DEVICE = torch.accelerator.current_accelerator().type

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    return model.to(DEVICE)

def run(data_root, epochs=25):
    print(DEVICE)
    full = datasets.ImageFolder(data_root, transform=transform)
    train_ds, val_ds = random_split(
        full, [0.75, 0.25],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    model = build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    losses = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = model(imgs)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}")

    torch.save(model.state_dict(), "weather_model.pt")
    print("Saved weather_model.pt")

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), losses, "b-o")
    plt.xlabel("epoch")
    plt.ylabel("training loss")
    plt.title("Training loss over epochs")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=120)
    print("Saved loss_curve.png")

    model.eval()
    n = len(CLASSES)
    confusion = torch.zeros(n, n, dtype=torch.long)
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds = model(imgs).argmax(dim=1)
            for t, p in zip(labels.cpu(), preds.cpu()):
                confusion[t, p] += 1

    tp = confusion.diag().float()
    fp = confusion.sum(dim=0).float() - tp
    fn = confusion.sum(dim=1).float() - tp

    accuracy = tp.sum() / confusion.sum()
    error_rate = 1 - accuracy
    precision_per_class = tp / (tp + fp).clamp(min=1)
    recall_per_class = tp / (tp + fn).clamp(min=1)
    precision = precision_per_class.mean()
    recall = recall_per_class.mean()

    print(f"\nAccuracy:    {accuracy:.4f}")
    print(f"Error rate:  {error_rate:.4f}")
    print(f"Precision:   {precision:.4f}  (macro avg)")
    print(f"Sensitivity: {recall:.4f}  (macro avg)")

    print("\nPer-class breakdown:")
    print(f"{'class':12s}  {'precision':>10s}  {'recall':>10s}")
    for i, cls in enumerate(CLASSES):
        print(f"{cls:12s}  {precision_per_class[i].item():>10.4f}  {recall_per_class[i].item():>10.4f}")

def predict(image_path, checkpoint="weather_model.pt"):
    model = build_model()
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    model.eval()
    img = Image.open(image_path)
    x = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    for i in probs.argsort(descending=True):
        print(f" {CLASSES[i]:12s} {probs[i]:.3f}")

if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2])
    elif sys.argv[1] == "predict":
        predict(sys.argv[2])