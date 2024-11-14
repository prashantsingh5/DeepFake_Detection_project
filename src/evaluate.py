import torch
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import load_data
from model import create_model
from config import Config

def evaluate_model():
    _, val_loader = load_data()
    model = create_model()
    model.load_state_dict(torch.load(Config.MODEL_DIR + "/final_model.pth"))
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
            outputs = model(inputs)
            preds = torch.round(torch.sigmoid(outputs.view(-1)))

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=Config.CLASSES, yticklabels=Config.CLASSES)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("results/confusion_matrix.png")
    plt.show()

    # Classification Report
    print(classification_report(y_true, y_pred, target_names=Config.CLASSES))

if __name__ == "__main__":
    evaluate_model()
