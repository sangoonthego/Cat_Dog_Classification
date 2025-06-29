import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from models.resnet_model import get_resnet18
from utils.dataset_loader import get_data_transform
from sklearn.metrics import classification_report, confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_, test_loader = get_data_transform("data/train", "data/test", batch_size=8)

model = get_resnet18().to(device)
model.load_state_dict(torch.load("cat_dog_model.pth", map_location=device))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(preds.cpu().numpy())

# print classification report
print("Classification Report:\n")
print(classification_report(all_labels, all_preds, target_name=["cat", "dog"]))

# print confusion matrix
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, xticklabels=["cat", "dog"], yticklabels=["cat", "dog"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()