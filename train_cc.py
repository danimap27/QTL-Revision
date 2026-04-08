# train_cc.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns
import time
import os
import argparse
import numpy as np
import datetime
import random
import json
import csv
try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Predefined datasets
PREDEFINED_DATASETS = ['hymenoptera', 'brain_tumor', 'cats_dogs', 'solar_dust']

def resolve_dataset(dataset_name):
    """Resolve dataset path and number of classes based on dataset name."""
    if dataset_name in PREDEFINED_DATASETS:
        base_path = os.path.join('Resultados', 'datasets')
        dataset_path = os.path.join(base_path, dataset_name)
        
        # Predefined class counts
        class_counts = {
            'hymenoptera': 2,
            'brain_tumor': 4, 
            'cats_dogs': 2,
            'solar_dust': 2
        }
        
        return dataset_path, class_counts[dataset_name]
    else:
        # Custom dataset - try to infer number of classes
        if os.path.exists(dataset_name):
            train_path = os.path.join(dataset_name, 'train')
            if os.path.exists(train_path):
                n_classes = len([d for d in os.listdir(train_path) 
                               if os.path.isdir(os.path.join(train_path, d))])
                return dataset_name, n_classes
        raise ValueError(f"Dataset {dataset_name} not found")

def train_classical(dataset_file="hymenoptera", classical_model="resnet18", epochs=5, id="null", batch_size=16, learning_rate=1e-4, seed=42, output_dir=None):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("============================================================")
    print("Classical Transfer Learning")
    print("============================================================")
    print(f"Dataset: {dataset_file}")
    print(f"Model: {classical_model}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"ID: {id}")
    print("============================================================")
    print(f"Using device: {device}")
    
    print("Step 1/7: Loading and preparing datasets...")
    # Define image transformations for training/validation and test
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    transform_test = transform_train
    
    def _resolve_dataset_dir(name: str):
        script_root = os.path.dirname(os.path.abspath(__file__))
        cwd = os.getcwd()
        candidates = [
            os.path.join(script_root, 'Resultados', 'datasets', name),
            os.path.join(script_root, 'datasets', name),
            os.path.join(cwd, 'Resultados', 'datasets', name),
            os.path.join(cwd, 'datasets', name),
            os.path.join(script_root, 'user_datasets', name),
            os.path.join(cwd, 'user_datasets', name)
        ]
        for c in candidates:
            if os.path.isdir(os.path.join(c, 'train')) and os.path.isdir(os.path.join(c, 'test')):
                return c
        raise FileNotFoundError(f"Dataset '{name}' was not found. Checked paths: " + ' | '.join(candidates))
    data_dir = _resolve_dataset_dir(dataset_file)
    print(f"Using dataset path: {data_dir}")
    print(f"Train path: {os.path.join(data_dir, 'train')}")
    print(f"Test path: {os.path.join(data_dir, 'test')}")
    
    full_train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform_train)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=transform_test)
    
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_classes = len(full_train_dataset.classes)

    print("Step 2/7: Defining classical base model...")
    if classical_model.lower() == "resnet18":
        model = models.resnet18(pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif classical_model.lower() == "resnet34":
        model = models.resnet34(pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif classical_model.lower() == "mobilenetv2":
        model = models.mobilenet_v2(pretrained=True)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif classical_model.lower() == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=True)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif classical_model.lower() == "regnet_x_400mf":
        model = models.regnet_x_400mf(pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    else:
        raise ValueError("Unsupported classical model. Use 'resnet18', 'resnet34', 'mobilenetv2', 'efficientnet_b0', or 'regnet_x_400mf'.")
    
    model = model.to(device)
    
    print("Step 3/7: Setting up training components...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_hist = []
    acc_hist = []

    def evaluate(loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        return correct / total if total > 0 else 0.0
    
    print("Step 4/7: Starting training...")
    if CODECARBON_AVAILABLE:
        tracker = EmissionsTracker(
            project_name=f"QTL_{id}",
            output_dir=output_dir or "results/energy",
            measure_power_secs=15,
            tracking_mode="process",
            log_level="warning"
        )
        tracker.start()
    start_train = time.time()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        val_acc = evaluate(val_loader)
        loss_hist.append(running_loss / len(train_loader))
        acc_hist.append(val_acc)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.4f}, Val Acc: {val_acc:.4f}")
    train_time = time.time() - start_train
    
    final_val_acc = evaluate(val_loader)
    print(f"Final Validation Accuracy: {final_val_acc:.4f}")
    
    print("Step 5/7: Evaluating model on test set...")
    start_test = time.time()
    test_acc = evaluate(test_loader)
    test_time = time.time() - start_test
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Training Time: {train_time:.2f} s, Test Evaluation Time: {test_time:.2f} s")
    
    print("Step 6/7: Saving model...")
    os.makedirs("model_saved", exist_ok=True)
    model_path = os.path.join("model_saved", f"CC_{id}_{classical_model}_{dataset_file}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    print("Step 7/7: Saving comprehensive metrics...")
    metrics_dir = os.path.join("static","metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    plt.figure()
    plt.plot(range(1,epochs+1), loss_hist)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.savefig(os.path.join(metrics_dir,f"{id}_loss.png"))
    plt.close()

    plt.figure()
    plt.plot(range(1,epochs+1), acc_hist)
    plt.xlabel("Epoch")
    plt.ylabel("Val Accuracy")
    plt.title("Accuracy vs Epoch")
    plt.savefig(os.path.join(metrics_dir,f"{id}_acc.png"))
    plt.close()

    # Get predictions and probabilities for ROC curves
    y_true, y_pred, y_scores = [], [], []
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            y_true.extend(y.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_scores.extend(probabilities.cpu().numpy())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)
    num_classes = len(full_train_dataset.classes)

    # Enhanced Loss vs Epoch plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs+1), loss_hist, marker='o', label='Training Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(metrics_dir, f"{id}_loss.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Enhanced Accuracy vs Epoch plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs+1), [a*100 for a in acc_hist], marker='o', color='green', label='Validation Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Validation Accuracy vs Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(metrics_dir, f"{id}_acc.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Enhanced Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=full_train_dataset.classes, yticklabels=full_train_dataset.classes)
    plt.title("Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(metrics_dir, f"{id}_confmat.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # ROC Curves
    if num_classes == 2:
        # Binary classification
        fpr, tpr, _ = roc_curve(y_true, y_scores[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(os.path.join(metrics_dir, f"{id}_roc.png"), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        # Multi-class classification
        y_true_bin = label_binarize(y_true, classes=range(num_classes))
        
        plt.figure(figsize=(10, 8))
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'ROC curve class {full_train_dataset.classes[i]} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Multi-class')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(os.path.join(metrics_dir, f"{id}_roc.png"), dpi=300, bbox_inches='tight')
        plt.close()

    # Print comprehensive metrics
    print('Val Loss per epoch:', [float(f"{v:.4f}") for v in loss_hist])
    print('Val Acc  per epoch:', [float(f"{v:.4f}") for v in acc_hist])
    print(f"Final Test Accuracy: {test_acc:.4f}")
    print(f"Model saved as: CC_{id}_{classical_model}_{dataset_file}.pth")
    print(f"Plots saved in: {metrics_dir}")

    # CodeCarbon stop
    energy_kwh = 0.0
    co2_kg = 0.0
    if CODECARBON_AVAILABLE:
        emissions = tracker.stop()
        energy_kwh = tracker._total_energy.kWh if hasattr(tracker, '_total_energy') else 0.0
        co2_kg = emissions if emissions else 0.0

    # Save structured CSV
    csv_dir = output_dir or "results/seeds"
    os.makedirs(csv_dir, exist_ok=True)

    approach = "classical"

    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    try:
        if num_classes == 2:
            auc_roc = roc_auc_score(y_true, y_scores[:, 1] if isinstance(y_scores, np.ndarray) and y_scores.ndim > 1 else y_scores)
        else:
            auc_roc = roc_auc_score(y_true, y_scores, multi_class='ovr', average='weighted')
    except:
        auc_roc = 0.0

    csv_filename = f"{approach}_{classical_model}_{dataset_file}_seed{seed}.csv"
    csv_path = os.path.join(csv_dir, csv_filename)

    row = {
        'approach': approach,
        'backbone': classical_model,
        'dataset': dataset_file,
        'seed': seed,
        'n_qubits': 'NA',
        'quantum_depth': 'NA',
        'test_accuracy': test_acc,
        'precision_weighted': precision,
        'recall_weighted': recall,
        'f1_weighted': f1,
        'auc_roc_weighted': auc_roc,
        'train_time_s': train_time,
        'test_time_s': test_time,
        'energy_kwh': energy_kwh,
        'co2_kg': co2_kg,
        'epochs_actual': len(loss_hist),
        'loss_history': json.dumps(loss_hist),
        'val_acc_history': json.dumps(acc_hist)
    }

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)
    print(f"Results saved to: {csv_path}")

    return test_acc, train_time, test_time

# CLI Interface
def main():
    p = argparse.ArgumentParser(description="Train a classical transfer learning model")
    p.add_argument('--dataset', default='hymenoptera', help='Dataset name')
    p.add_argument('--model', default='resnet18', choices=['resnet18','resnet34','mobilenetv2','efficientnet_b0','regnet_x_400mf'], help='Classical model architecture')
    p.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    p.add_argument('--batch-size', type=int, default=16, help='Batch size')
    p.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    p.add_argument('--id', default=None, help='Run identifier (auto if not provided)')
    p.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    p.add_argument('--output-dir', type=str, default=None, help='Output directory for results CSV')
    args = p.parse_args()

    # Generate ID if not provided
    if args.id is None:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.id = f"{args.model}_{timestamp}"

    print("=" * 60)
    print("Classical Transfer Learning")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"ID: {args.id}")
    print("=" * 60)

    try:
        test_acc, train_time, test_time = train_classical(
            dataset_file=args.dataset,
            classical_model=args.model,
            epochs=args.epochs,
            id=args.id,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            seed=args.seed,
            output_dir=args.output_dir
        )
        
        print("=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print(f"Final Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"Training Time: {train_time:.2f} seconds")
        print(f"Testing Time: {test_time:.2f} seconds")
        print("=" * 60)
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
