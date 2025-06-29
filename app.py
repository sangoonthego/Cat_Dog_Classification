import gradio as gr
import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from models.resnet_model import get_resnet18

model_path = "cat_dog_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_resnet18().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

tranform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

def predict_image_gradio(image):
    img = image.convert("RGB")
    input_tensor = tranform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        prob = F.softmax(output, dim=1)
        pred = output.argmax(dim=1).item()
        label = "cat" if pred == 0 else "dog"
        confidence = prob.squeeze().cpu().numpy()

    return f"{label} \ncat: {confidence[0]:.2f}, dog: {confidence[1]:.2f}"

gr.Interface(
    fn=predict_image_gradio,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Cat and Dog Classification",
    description="Download Dog and Cat images to classify by ResNet18 model"
).launch()