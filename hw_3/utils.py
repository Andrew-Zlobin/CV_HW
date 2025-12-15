import matplotlib.pyplot as plt
import seaborn as sns
import torch

def plot_training_history(history):

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0, 0].plot(history['val_loss'], label='Val Loss', marker='o')
    axes[0, 0].set_title('Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

# acc
    axes[0, 1].plot(history['train_acc'], label='Train acc', marker='o')
    axes[0, 1].plot(history['val_acc'], label='Val acc', marker='o')
    axes[0, 1].set_title('Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Dice Coefficient
    axes[1, 0].plot(history['train_dice'], label='Train Dice', marker='o')
    axes[1, 0].plot(history['val_dice'], label='Val Dice', marker='o')
    axes[1, 0].set_title('Dice Coefficient', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Dice')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # IoU Score
    axes[1, 1].plot(history['train_iou'], label='Train IoU', marker='o')
    axes[1, 1].plot(history['val_iou'], label='Val IoU', marker='o')
    axes[1, 1].set_title('IoU Score', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('IoU')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Learning Rate
    # axes[1, 1].plot(history['lr'], label='Learning Rate', marker='o', color='green')
    # axes[1, 1].set_title('Learning Rate', fontsize=14, fontweight='bold')
    # axes[1, 1].set_xlabel('Epoch')
    # axes[1, 1].set_ylabel('LR')
    # axes[1, 1].set_yscale('log')
    # axes[1, 1].legend()
    # axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('История обучения U-Net', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    

    print(f"Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"Val Loss: {history['val_loss'][-1]:.4f}")
    print(f"Train Dice: {history['train_dice'][-1]:.4f}")
    print(f"Val Dice: {history['val_dice'][-1]:.4f}")
    print(f"Train IoU: {history['train_iou'][-1]:.4f}")
    print(f"Val IoU: {history['val_iou'][-1]:.4f}")
    print(f"Val Accuracy: {history['val_acc'][-1]:.4f}")


@torch.no_grad()
def visualize_predictions(model, dataset, device, n=5):
    model.eval()
    plt.figure(figsize=(12, 4*n))

    for i in range(n):
        x, y = dataset[i]
        # print(y)
        x = x.unsqueeze(0).to(device)

        logits = model(x)
        pred = (torch.sigmoid(logits) > 0.5).float()
        
        img = x.cpu()[0].permute(1,2,0)
        
        min_img = img.min()
        max_img = img.max()
        if max_img - min_img > 0:
            img = (img - min_img) / (max_img - min_img)
        else:
            img = np.zeros_like(img)
        # gt = y[1]
        gt = y
        pr = pred.cpu()[0,0]

        plt.subplot(n, 3, 3*i + 1)
        plt.imshow(img)
        plt.title("Input")
        plt.axis("off")

        plt.subplot(n, 3, 3*i + 2)
        plt.imshow(pr, cmap="gray")
        plt.title("Prediction")
        plt.axis("off")

        plt.subplot(n, 3, 3*i + 3)
        plt.imshow(gt, cmap="gray")
        plt.title("GT")
        plt.axis("off")

    plt.show()

