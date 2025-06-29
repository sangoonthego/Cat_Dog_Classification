import torch
import torch.nn as nn
from torch.optim import Adam
from utils.dataset_loader import get_data_transform
from models.resnet_model import get_resnet18

# tang toc do train neu dung GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, test_loader = get_data_transform("data/train", "data/test", batch_size=8)

model = get_resnet18().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.fc.parameters(), lr=0.001)

epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}, Accuracy: {acc:.4f}")

torch.save(model.state_dict(), "cat_dog_model.pth")

