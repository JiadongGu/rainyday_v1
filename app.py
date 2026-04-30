import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from PIL import Image
import sys

CLASSES = ["dew","fogsmog","frost","glaze","hail","lightning","rain","rainbow","rime","sandstorm","snow"]

DEVICE = torch.accelerator.current_accelerator().type 

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]) #imagenet average RGB values
])

def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features,len(CLASSES)) #change from 1000 imagenet classes to 11
    return model.to(DEVICE)

def train(data_root, epochs=10):
    dataset = datasets.ImageFolder(data_root, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle = True, num_workers=2)
    model = build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = model(imgs)
            loss = loss_fn(logits,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print("Epoch " + str(epoch+1) + "/" + str(epochs) + " loss = " + str(total_loss/len(loader)))
        torch.save(model.state_dict(), "weather_model.pt")
        print("Saved, weather_model.pt")

def predict(image_path, checkpoint="weather_model.pt"):
    model = build_model()
    model.load_state_dict(torch.load(checkpoint, map_location = DEVICE))
    model.eval()
    img = Image.open(image_path)

    x = transform(img).unsqueeze(0).to(DEVICE) #batch dimension

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    for i in probs.argsort(descending=True):
        print(f" {CLASSES[i]:12s} {probs[i]:.3f})")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
    elif sys.argv[1] == "train":
        train(sys.argv[2])
    elif sys.argv[1] == "predict":
        predict(sys.argv[2])