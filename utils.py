import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torchvision import transforms


def check_directory(directory_path):
    """Check if a directory exists and if it contains any files/folders."""
    if not os.path.exists(directory_path):
        print("FAIL: The directory DOES NOT EXIST.")
        return False

    if not os.path.isdir(directory_path):
        print("FAIL: The path exists but IS NOT A DIRECTORY.")
        return False

    contents = os.listdir(directory_path)
    if not contents:
        print("WARNING: The directory exists but is EMPTY (0 items).")
        return False
    else:
        first_level_items = glob.glob(os.path.join(directory_path, '*'))
        print(f"SUCCESS: The directory exists and contains {len(contents)} items.")
        print("\tExample items (first level):")
        for item in first_level_items[:5]:
            print(f"\t- {os.path.basename(item)}")
        return True
    

def plot_random_batch_images(loader, classes, num_images_to_show=9, figsize=(10, 10)):
    """Plot a random batch of images from a DataLoader."""
    images, labels = next(iter(loader))
    images = images[:num_images_to_show]
    labels = labels[:num_images_to_show]

    num_to_plot = len(images)
    fig = plt.figure(figsize=figsize)
    n_cols = int(np.ceil(np.sqrt(num_to_plot)))
    n_rows = int(np.ceil(num_to_plot / n_cols))

    print(f"Displaying {num_to_plot} random images...")

    for i in range(num_to_plot):
        ax = fig.add_subplot(n_rows, n_cols, i + 1, xticks=[], yticks=[])
        img = images[i].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        ax.set_title(f"Label: {classes[labels[i]]}")

    plt.tight_layout()
    plt.show()


def plot_history(history, titles):
    """Plot training and validation accuracy/loss."""
    plt.figure(figsize=(12, 5))

    # Accuracy subplot
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label=f'{titles[0]} Train')
    plt.plot(history['val_accuracy'], label=f'{titles[0]} Val')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Loss subplot
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label=f'{titles[0]} Train')
    plt.plot(history['val_loss'], label=f'{titles[0]} Val')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()


def display_and_predict(model, img_path, classes, device, target_size=(224, 224)):
    """Load an image, display it, predict its class using the trained PyTorch model."""
    if not os.path.exists(img_path):
        print(f"ERROR: Image path not found: {img_path}")
        return

    img = Image.open(img_path).convert('RGB')
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(f"Input Image: {os.path.basename(img_path)}")
    plt.axis('off')
    plt.show()

    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor()
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        preds = model(img_tensor)
        pred_idx = torch.argmax(preds, dim=1).item()
        confidence = torch.softmax(preds, dim=1)[0][pred_idx].item() * 100

    print(f"Predicted dish: {classes[pred_idx]}")
    print(f"Confidence: {confidence:.2f}%")