import gradio as gr
import torch
from PIL import Image
from torchvision import transforms
from src.model import create_model
from src.config import Config

# Function to load and preprocess the image
def load_image(img_path):
    image = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(Config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Function to predict the class of an image
def predict_image(img_path):
    model = create_model()
    model.load_state_dict(torch.load(Config.MODEL_DIR + "/final_model.pth"))
    model.eval()

    image = load_image(img_path).to(Config.DEVICE)
    with torch.no_grad():
        output = model(image)
        prediction = torch.sigmoid(output).item()
        return "Real Image" if prediction > 0.5 else "Fake Image"

# Gradio Interface
def inference_gradio(image_path):
    result = predict_image(image_path)
    return result

# Create Gradio Interface
interface = gr.Interface(
    fn=inference_gradio,
    inputs=gr.Image(type="filepath", label="Upload an Image"),
    outputs=gr.Textbox(label="Prediction"),
    title="Deepfake Detection",
    description="Drag and drop an image of a knee osteoarthritis scan to predict if it's real or fake."
)

# Launch Gradio App
if __name__ == "__main__":
    interface.launch()
