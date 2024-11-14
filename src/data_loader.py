import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from src.config import Config

def load_data():
    transform = transforms.Compose([
        transforms.Resize(Config.IMG_SIZE),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(Config.DATA_DIR, transform=transform)
    train_size = int((1 - Config.TRAIN_TEST_SPLIT) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    return train_loader, val_loader
