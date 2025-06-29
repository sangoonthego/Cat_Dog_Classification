import os 
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from models.resnet_model import get_resnet18
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

predict_dir = "test_images"
model_path = "cat_dog_model.pth"

model = get_resnet18().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        prob = F.softmax(output, dim=1)
        pred = output.argmax(dim=1).item()
        label = "cat" if pred == 0 else "dog"

    return label, prob.squeeze().cpu().numpy()

print("Predict Image: ", predict_dir)
for filename in os.listdir(predict_dir):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(predict_dir, filename)
        label, prob = predict_image(path)
        print(f"{filename}: {label}")
        print(f"Prob - cat: {prob[0]:.2f}, dog: {prob[1]:.2f}")

        image = Image.open(path).convert("RGB")
        plt.imshow(image)
        plt.title(f"{filename} -> Predicted: {label}")
        plt.axis("off")
        plt.show()
