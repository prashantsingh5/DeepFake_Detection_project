import os
import torch  # Add this import to use torch.cuda

class Config:
    DATA_DIR = "C:/Users/pytorch/Desktop/Deepfake_Detection_Project/data"  # Update this to your dataset path
    MODEL_DIR = "models"
    CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 1e-4
    CLASSES = ["real", "fake"]
    TRAIN_TEST_SPLIT = 0.2
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Set device based on GPU availability
