import torch
import torch.nn as nn
import torch.optim as optim
from model import create_model
from data_loader import load_data
from config import Config

def train_model():
    train_loader, val_loader = load_data()
    model = create_model()

    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    best_val_loss = float("inf")
    for epoch in range(Config.EPOCHS):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(Config.DEVICE), labels.float().to(Config.DEVICE)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs.view(-1), labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{Config.EPOCHS}], Loss: {epoch_loss:.4f}")

        # Validation phase
        val_loss = validate_model(model, val_loader, criterion)

        # Save best model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), Config.CHECKPOINT_DIR + "/best_model.pth")

    # Save final model
    torch.save(model.state_dict(), Config.MODEL_DIR + "/final_model.pth")

def validate_model(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(Config.DEVICE), labels.float().to(Config.DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs.view(-1), labels)
            val_loss += loss.item() * inputs.size(0)

    val_loss /= len(val_loader.dataset)
    print(f"Validation Loss: {val_loss:.4f}")
    return val_loss

if __name__ == "__main__":
    train_model()
