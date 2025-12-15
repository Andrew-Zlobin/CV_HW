import torch
import torch.nn as nn
import torch.optim as optim


def accuracy(outputs, targets):
    preds = torch.argmax(outputs, dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)


def train(
    model
    , train
    , val
    , lr = 0.001
    , loss_fn = nn.CrossEntropyLoss()
    , num_epochs = 25
    , device = "cuda"
):
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr
    )
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
    
        for images, labels in train:
            images = images.to(device)
            labels = labels.to(device)
    
            optimizer.zero_grad()
    
            outputs = model(images)
            loss = loss_fn(outputs, labels)
    
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
            running_acc += accuracy(outputs, labels)
    
        epoch_train_loss = running_loss / len(train)
        epoch_train_acc = running_acc / len(train)
    
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)
    
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
    
        with torch.no_grad():
            for images, labels in val:
                images = images.to(device)
                labels = labels.to(device)
    
                outputs = model(images)
                loss = loss_fn(outputs, labels)
    
                val_loss += loss.item()
                val_acc += accuracy(outputs, labels)
    
        epoch_val_loss = val_loss / len(val)
        epoch_val_acc = val_acc / len(val)
    
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
    
        print(
            f"Epoch [{epoch+1}/{num_epochs}] | "
            f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}"
        )
    return train_losses, val_losses, train_accs, val_accs
