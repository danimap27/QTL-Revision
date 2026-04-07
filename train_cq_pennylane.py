import warnings
warnings.filterwarnings("ignore")

import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns

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

# Transforms
GLOBAL_TRANSFORMS = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Quantum layer factory
def create_quantum_layer(n_qubits, n_outputs, depth):
    dev = qml.device('default.qubit', wires=n_qubits)
    @qml.qnode(dev, interface='torch')
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(w)) for w in range(n_outputs)]
    weight_shapes = {'weights': (depth, n_qubits)}
    return qml.qnn.TorchLayer(circuit, weight_shapes)

class HybridModel(nn.Module):
    def __init__(self, backbone, feat_dim, n_qubits, n_outputs, depth=3):
        super().__init__()
        for p in backbone.parameters(): p.requires_grad = False
        self.backbone = backbone
        self.fc = nn.Linear(feat_dim, n_qubits)
        self.qlayer = create_quantum_layer(n_qubits, n_outputs, depth)

    def forward(self, x):
        x = self.backbone(x)
        if x.dim() > 2:
            x = torch.flatten(x, 1)
        x = self.fc(x)
        x = torch.tanh(x) * (np.pi / 2)
        if x.dim() == 1:
            return self.qlayer(x)
        # Evaluate sample-by-sample to avoid the QNode treating batch entries as wires
        outs = [self.qlayer(xi) for xi in x]
        return torch.stack(outs, dim=0)

def train_quantum_hybrid_pennylane(dataset_file, classical_model, n_qubits, epochs, id, batch_size=32, learning_rate=1e-3, early_stop_patience=10, quantum_depth=3, gamma=0.9):
    print("============================================================")
    print("PennyLane Quantum Transfer Learning")
    print("============================================================")
    print(f"Dataset: {dataset_file}")
    print(f"Model: {classical_model}")
    print(f"Qubits: {n_qubits}")
    print(f"Depth: {quantum_depth}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"ID: {id}")
    print("============================================================")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Robust dataset path resolution
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
    
    print("Step 1/7: Loading and preparing datasets...")
    data_dir = _resolve_dataset_dir(dataset_file)
    print(f"Using dataset path: {data_dir}")
    print(f"Train path: {os.path.join(data_dir, 'train')}")
    print(f"Test path: {os.path.join(data_dir, 'test')}")
    
    transforms_cfg = GLOBAL_TRANSFORMS

    # Load datasets
    full_train = datasets.ImageFolder(os.path.join(data_dir, 'train'), transforms_cfg['train'])
    test_ds = datasets.ImageFolder(os.path.join(data_dir, 'test'),  transforms_cfg['val'])
    train_size = int(0.8 *len(full_train))
    val_size = len(full_train) - train_size
    train_ds, val_ds = random_split(full_train, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    num_classes = len(full_train.classes)

    print("Step 2/7: Defining classical base model...")
    # Build backbone
    if classical_model.lower() == 'resnet18':
        backbone = models.resnet18(pretrained=True)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
    elif classical_model.lower() == 'resnet34':
        backbone = models.resnet34(pretrained=True)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
    elif classical_model.lower() == 'mobilenetv2':
        backbone = models.mobilenet_v2(pretrained=True)
        backbone.classifier = nn.Identity()
        feat_dim = 1280
    elif classical_model.lower() == 'efficientnet_b0':
        backbone = models.efficientnet_b0(pretrained=True)
        feat_dim = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
    elif classical_model.lower() == 'regnet_x_400mf':
        backbone = models.regnet_x_400mf(pretrained=True)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
    else:
        raise ValueError("Unsupported model. Use 'resnet18', 'resnet34', 'mobilenetv2', 'efficientnet_b0', or 'regnet_x_400mf'.")
    backbone = backbone.to(device)

    print("Step 3/7: Creating quantum circuit and hybrid model...")
    # Instantiate hybrid model
    model = HybridModel(backbone, feat_dim, n_qubits, num_classes, quantum_depth).to(device)
    
    print("Step 4/7: Setting up training components...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    print("Step 5/7: Starting training...")
    # Training loop with early stopping
    best_loss = float('inf')
    patience = 0
    train_losses, val_losses, val_accs = [], [], []
    start_time = time.time()
    for epoch in range(1, epochs+1):
        print(f"Epoch {epoch}/{epochs}...")
        model.train()
        running = 0
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out,y)
            loss.backward()
            optimizer.step()
            running += loss.item()*x.size(0)
        train_loss = running/len(train_loader.dataset)
        train_losses.append(train_loss)
        # Validation
        model.eval()
        vl, correct=0,0
        total=0
        with torch.no_grad():
            for x,y in val_loader:
                x,y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out,y)
                vl+=loss.item()*x.size(0)
                _,preds = torch.max(out,1)
                correct += (preds==y).sum().item()
                total+=y.size(0)
        val_loss = vl/total
        acc = correct/total
        val_losses.append(val_loss)
        val_accs.append(acc)
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {acc:.4f}")
        if val_loss < best_loss: 
            best_loss, patience = val_loss, 0
        else: 
            patience += 1                       
        if patience >= early_stop_patience: 
            break

    train_time = time.time() - start_time
    
    print("Step 6/7: Evaluating model on test set...")
    # Test evaluation
    model.eval()
    tl, tc, tot = 0,0,0
    start_time_test = time.time()
    with torch.no_grad():
        for x,y in test_loader:
            x,y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out,y)
            tl+=loss.item()*x.size(0)
            _,pred = torch.max(out,1)
            tc += (pred==y).sum().item()
            tot+=y.size(0)
    test_acc = tc/tot
    test_time = time.time() - start_time_test
    print(f"Test Acc: {test_acc:.4f}")
    print(f"Training time: {train_time:.2f}s, Testing time: {test_time:.2f}s")
    
    print("Step 7/7: Saving model and comprehensive metrics...")
    # Save model
    os.makedirs('model_saved', exist_ok=True)
    fn = f"PL_{id}_{classical_model}_{dataset_file}.pth"
    torch.save(model.state_dict(), os.path.join('model_saved', fn))

    # Get predictions and probabilities for ROC curves
    model.eval()
    y_true, y_pred, y_scores = [], [], []
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

    # Save metrics plots
    mdir = os.path.join('static','metrics')
    os.makedirs(mdir, exist_ok=True)
    epochs_range = list(range(1,len(train_losses)+1))
    
    # Loss vs Epoch
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs_range, val_losses, label='Val Loss', marker='s')
    plt.title('Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(mdir, f"{id}_loss.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Accuracy vs Epoch
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, [a*100 for a in val_accs], label='Val Accuracy', marker='o', color='green')
    plt.title('Validation Accuracy vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(mdir, f"{id}_acc.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=full_train.classes, yticklabels=full_train.classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(mdir, f"{id}_confmat.png"), dpi=300, bbox_inches='tight')
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
        plt.savefig(os.path.join(mdir, f"{id}_roc.png"), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        # Multi-class classification
        y_true_bin = label_binarize(y_true, classes=range(num_classes))
        
        plt.figure(figsize=(10, 8))
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'ROC curve class {full_train.classes[i]} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Multi-class')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(os.path.join(mdir, f"{id}_roc.png"), dpi=300, bbox_inches='tight')
        plt.close()

    # Print metrics
    print('Val Loss per epoch:', [float(f"{v:.4f}") for v in val_losses])
    print('Val Acc  per epoch:', [float(f"{v:.4f}") for v in val_accs])
    print(f"Final Test Accuracy: {test_acc:.4f}")
    print(f"Model saved as: {fn}")
    print(f"Plots saved in: {mdir}")

    return test_acc, train_time, test_time

# CLI Interface
def main():
    p = argparse.ArgumentParser(description="Train a PennyLane hybrid model")
    p.add_argument('--dataset', default='hymenoptera', help='Dataset name')
    p.add_argument('--model', default='resnet18', choices=['resnet18','resnet34','mobilenetv2','efficientnet_b0','regnet_x_400mf'], help='Classical model architecture')
    p.add_argument('--n-qubits', type=int, default=4, help='Number of qubits')
    p.add_argument('--depth', type=int, default=3, help='Quantum circuit depth')
    p.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    p.add_argument('--batch-size', type=int, default=16, help='Batch size')
    p.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    p.add_argument('--gamma', type=float, default=0.9, help='Learning rate scheduler gamma')
    p.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    p.add_argument('--id', default=None, help='Run identifier (auto if not provided)')
    args = p.parse_args()

    # Generate ID if not provided
    if args.id is None:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.id = f"{args.model}_{args.n_qubits}q_depth{args.depth}_{timestamp}"

    print("=" * 60)
    print("PennyLane Quantum Transfer Learning")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Qubits: {args.n_qubits}")
    print(f"Depth: {args.depth}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"ID: {args.id}")
    print("=" * 60)

    try:
        test_acc, train_time, test_time = train_quantum_hybrid_pennylane(
            dataset_file=args.dataset,
            classical_model=args.model,
            n_qubits=args.n_qubits,
            epochs=args.epochs,
            id=args.id,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            early_stop_patience=args.patience,
            quantum_depth=args.depth,
            gamma=args.gamma
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
