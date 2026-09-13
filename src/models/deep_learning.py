import gc
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# ==========================================
# 1. Enhanced EEGNet Architecture
# ==========================================
class EnhancedEEGNet(nn.Module):
    def __init__(self, channels=64, time_samples=641, num_classes=2, F1=16, D=2, F2=32, kernel_length=64, dropout_rate=0.35):
        super(EnhancedEEGNet, self).__init__()
        # Block 1: Temporal & Spatial Filtering
        self.conv1 = nn.Conv2d(1, F1, kernel_size=(1, kernel_length), padding=(0, kernel_length // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwise = nn.Conv2d(F1, F1 * D, kernel_size=(channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.drop1 = nn.Dropout(dropout_rate)
        
        # Block 2: Separable Convolutions
        self.separable_depthwise = nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, 16), padding=(0, 8), groups=F1 * D, bias=False)
        self.separable_pointwise = nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.drop2 = nn.Dropout(dropout_rate)
        
        with torch.no_grad():
            dummy = torch.zeros(1, 1, channels, time_samples)
            x = self.drop1(self.pool1(F.elu(self.bn2(self.depthwise(self.bn1(self.conv1(dummy)))))))
            x = self.drop2(self.pool2(F.elu(self.bn3(self.separable_pointwise(self.separable_depthwise(x))))))
            flattened_dim = x.view(1, -1).size(1)
            
        self.fc = nn.Linear(flattened_dim, num_classes)

    def forward(self, x):
        x = self.drop1(self.pool1(F.elu(self.bn2(self.depthwise(self.bn1(self.conv1(x)))))))
        x = self.drop2(self.pool2(F.elu(self.bn3(self.separable_pointwise(self.separable_depthwise(x))))))
        x = x.view(x.size(0), -1)
        return self.fc(x)

def augment_batch(batch_x, noise_level=0.02, shift_max=10):
    """Subtle jitter and time shifting to prevent single-subject overfitting."""
    noise = torch.randn_like(batch_x) * noise_level
    shift = np.random.randint(-shift_max, shift_max)
    return torch.roll(batch_x + noise, shifts=shift, dims=-1)

# ==========================================
# 2. Execution Pipeline
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on device: {device}")

    # 1. Load and Standardize Dataset
    X_raw = np.load('data/processed/X.npy').astype(np.float32)
    y = np.load('data/processed/y.npy').astype(np.int64)

    if np.min(y) > 0:
        y = y - np.min(y)

    mean = np.mean(X_raw, axis=-1, keepdims=True)
    std = np.std(X_raw, axis=-1, keepdims=True) + 1e-8
    X_raw = (X_raw - mean) / std
    X_raw = np.expand_dims(X_raw, axis=1)

    # 2. Split: Subject 1 (Target) vs Global Population
    subject_epochs = 45 
    X_sub1 = X_raw[:subject_epochs]
    y_sub1 = y[:subject_epochs]
    
    X_global = torch.from_numpy(X_raw[subject_epochs:])
    y_global = torch.from_numpy(y[subject_epochs:])

    # 3. Phase 1: Deep Global Pretraining (45 Epochs)
    print("\n--- Phase 1: Deep Pretraining Global Model ---")
    global_model = EnhancedEEGNet(channels=64, time_samples=641, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(global_model.parameters(), lr=0.0025, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=45)
    
    global_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_global, y_global), batch_size=64, shuffle=True)
    
    global_model.train()
    for epoch in range(1, 46):
        running_loss = 0.0
        for bx, by in global_loader:
            bx, by = bx.to(device), by.to(device)
            bx = augment_batch(bx)
            optimizer.zero_grad()
            loss = criterion(global_model(bx), by)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        if epoch % 15 == 0 or epoch == 45:
            print(f"   Global Epoch {epoch:02d}/45 | Loss: {running_loss/len(global_loader):.4f}")

   # 4. Phase 2: Subject-Specific Calibrated Fine-Tuning
    print("\n--- Phase 2: Subject-Specific Calibrated Fine-Tuning ---")
    
    # Freeze ONLY the temporal filter. 
    # WE UNFREEZE the depthwise spatial filter so it adapts to Subject 1's skull!
    for param in global_model.conv1.parameters():
        param.requires_grad = False
    # (Removed the global_model.depthwise parameter freeze here)

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    transfer_accuracies = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_sub1, y_sub1), 1):
        import copy
        fold_model = copy.deepcopy(global_model)
        
        # Lower learning rate slightly, but increase weight decay to prevent overfitting
        fold_optimizer = optim.AdamW(filter(lambda p: p.requires_grad, fold_model.parameters()), lr=0.0005, weight_decay=0.03)
        fold_scheduler = optim.lr_scheduler.CosineAnnealingLR(fold_optimizer, T_max=30)
        
        X_tr = torch.from_numpy(X_sub1[train_idx])
        y_tr = torch.from_numpy(y_sub1[train_idx])
        X_te = torch.from_numpy(X_sub1[test_idx])
        y_te = torch.from_numpy(y_sub1[test_idx])
        
        # MICRO-BATCHING: Dropped batch_size to 4 for more gradient updates
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_tr, y_tr), batch_size=4, shuffle=True)
        
        fold_model.train()
        for epoch in range(1, 31): # Bumped to 30 epochs for the micro-batches
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                bx = augment_batch(bx, noise_level=0.015, shift_max=5)
                fold_optimizer.zero_grad()
                loss = criterion(fold_model(bx), by)
                loss.backward()
                fold_optimizer.step()
            fold_scheduler.step()
                
        fold_model.eval()
        with torch.no_grad():
            preds = torch.argmax(fold_model(X_te.to(device)), dim=1).cpu().numpy()
            acc = accuracy_score(y_te.numpy(), preds)
            transfer_accuracies.append(acc)
            print(f"==> Fold {fold} Fine-Tuned Accuracy: {acc * 100:.2f}%")

    mean_acc = np.mean(transfer_accuracies)
    std_acc = np.std(transfer_accuracies)
    print(f"\n==========================================")
    print(f"Overall Subject 1 Fine-Tuned Accuracy: {mean_acc * 100:.2f}% (+/- {std_acc * 100:.2f}%)")
    print(f"==========================================")