import torch
from torchvision import transforms
from PIL import Image
from model import create_model
from config import Config

def load_image(img_path):
    transform = transforms.Compose([
        transforms.Resize(Config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(img_path).convert("RGB")
    return transform(image).unsqueeze(0)

def predict_image(img_path):
    model = create_model()
    model.load_state_dict(torch.load(Config.MODEL_DIR + "/final_model.pth"))
    model.eval()

    image = load_image(img_path).to(Config.DEVICE)
    with torch.no_grad():
        output = model(image)
        prediction = torch.sigmoid(output).item()
        return "Real Image" if prediction > 0.5 else "Fake Image"

# Example usage:
print(predict_image(r"C:\Users\pytorch\Desktop\deepfake_detection\archive\test\0\9643521L.png"))
