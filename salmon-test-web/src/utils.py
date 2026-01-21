"""
Utility functions for training visualization and helpers
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from torchvision.utils import make_grid


class UnNormalize:
    """Reverse normalization transform for visualization"""
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def __call__(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Normalized tensor, shape [B,C,H,W] or [C,H,W]

        Returns
        -------
        torch.Tensor
            Unnormalized tensor
        """
        return x * self.std + self.mean


def show_batch_unnorm(loader, mean, std, n=8, title="preview"):
    """
    Display a batch of images after reversing normalization

    Parameters
    ----------
    loader : torch.utils.data.DataLoader
        DataLoader to get images from
    mean : list
        Mean values used in normalization
    std : list
        Std values used in normalization
    n : int, optional
        Number of images to display. Default 8
    title : str, optional
        Title for the plot. Default "preview"
    """
    unnorm = UnNormalize(mean, std)
    imgs, *rest = next(iter(loader))
    imgs = imgs[:n]
    imgs = torch.clamp(unnorm(imgs), 0, 1)
    grid = make_grid(imgs, nrow=n, padding=2)
    grid = grid.permute(1, 2, 0).cpu().numpy()

    plt.figure(figsize=(1.6 * n, 2))
    plt.imshow(grid)
    plt.axis("off")
    plt.title(title)
    plt.show()


def plot_training_history(train_loss, val_loss, train_acc, val_acc, save_path=None):
    """
    Plot training and validation loss/accuracy curves

    Parameters
    ----------
    train_loss : list
        Training loss history
    val_loss : list
        Validation loss history
    train_acc : list
        Training accuracy history
    val_acc : list
        Validation accuracy history
    save_path : str, optional
        If provided, save plot to this path
    """
    if len(train_loss) == 0:
        print("No training data to plot")
        return

    epochs = np.arange(1, len(train_loss) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot Loss
    ax1.plot(epochs, train_loss, 'b-o', label='Train Loss', linewidth=2, markersize=6)
    ax1.plot(epochs, val_loss, 'r-s', label='Val Loss', linewidth=2, markersize=6)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([1, len(epochs)])

    # Plot Accuracy
    ax2.plot(epochs, train_acc, 'b-o', label='Train Accuracy', linewidth=2, markersize=6)
    ax2.plot(epochs, val_acc, 'r-s', label='Val Accuracy', linewidth=2, markersize=6)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([1, len(epochs)])
    ax2.set_ylim([0, 1])

    plt.tight_layout()

    if save_path:
        # Ensure parent directory exists
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()

    # Print summary statistics
    print("\n=== Training Summary ===")
    print(f"Total Epochs: {len(train_loss)}")
    print(f"Best Train Loss: {min(train_loss):.4f} (Epoch {np.argmin(train_loss) + 1})")
    print(f"Best Val Loss: {min(val_loss):.4f} (Epoch {np.argmin(val_loss) + 1})")
    print(f"Best Train Acc: {max(train_acc):.4f} (Epoch {np.argmax(train_acc) + 1})")
    print(f"Best Val Acc: {max(val_acc):.4f} (Epoch {np.argmax(val_acc) + 1})")


def save_checkpoint(model, optimizer, epoch, val_acc, path):
    """
    Save model checkpoint

    Parameters
    ----------
    model : torch.nn.Module
        Model to save
    optimizer : torch.optim.Optimizer
        Optimizer state to save
    epoch : int
        Current epoch number
    val_acc : float
        Validation accuracy
    path : str
        Path to save checkpoint
    """
    # Ensure parent directory exists
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "val_acc": val_acc
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path} (epoch {epoch}, val_acc {val_acc:.4f})")


def load_checkpoint(model, optimizer, path, device):
    """
    Load model checkpoint

    Parameters
    ----------
    model : torch.nn.Module
        Model to load weights into
    optimizer : torch.optim.Optimizer
        Optimizer to load state into
    path : str
        Path to checkpoint file
    device : torch.device
        Device to map model to

    Returns
    -------
    tuple(int, float)
        epoch, val_acc from checkpoint
    """
    checkpoint = torch.load(path, map_location=device)

    # Support both old and new key names
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        # Backward compatibility with old checkpoints
        model.load_state_dict(checkpoint["model"])
        if optimizer is not None and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])

    epoch = checkpoint.get("epoch", 0)
    val_acc = checkpoint.get("val_acc", 0.0)
    print(f"Checkpoint loaded from {path} (epoch {epoch}, val_acc {val_acc:.4f})")
    return epoch, val_acc


def get_device():
    """
    Get available device (CUDA or CPU) and print info

    Returns
    -------
    torch.device
        Device to use for training
    """
    print("=" * 50)
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    print("=" * 50)
    return device
