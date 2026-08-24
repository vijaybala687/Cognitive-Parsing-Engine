import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

class SimpleEEGCNN(nn.Module):
    """
    A lightweight CNN that self-adjusts its matrix dimensions.
    Expects input shape: (batch_size, 1, channels, time_samples)
    """
    def __init__(self, channels=64, time_samples=640, num_classes=2):
        super(SimpleEEGCNN, self).__init__()
        # Temporal convolution to learn frequency bands
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(1, 64), padding=(0, 32))
        self.batchnorm1 = nn.BatchNorm2d(16)
        
        # Spatial convolution to learn electrode combinations
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(channels, 1))
        self.batchnorm2 = nn.BatchNorm2d(32)
        
        self.pooling = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout = nn.Dropout(0.25)
        
        # --- DYNAMIC MATRIX SIZING ---
        with torch.no_grad():
            dummy = torch.randn(1, 1, channels, time_samples)
            x = F.elu(self.batchnorm1(self.conv1(dummy)))
            x = self.pooling(F.elu(self.batchnorm2(self.conv2(x))))
            flat_size = x.view(1, -1).size(1)
            
        self.fc1 = nn.Linear(flat_size, num_classes) 

    def forward(self, x):
        x = F.elu(self.batchnorm1(self.conv1(x)))
        x = self.dropout(self.pooling(F.elu(self.batchnorm2(self.conv2(x)))))
        x = x.view(x.size(0), -1) 
        x = self.fc1(x)
        return x

import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on device: {device}")

    # 1. Load data as float32 to halve memory footprint
    print("Loading raw EEG epochs...")
    X_raw = np.load('data/processed/X.npy').astype(np.float32)
    y = np.load('data/processed/y.npy').astype(np.int64)

    # Shift labels to 0-indexed [0, 1]
    if np.min(y) > 0:
        y = y - np.min(y)

    # Add channel dimension: (N, 1, Channels, Time_Samples)
    X_raw = np.expand_dims(X_raw, axis=1)
    epochs, _, channels, time_samples = X_raw.shape

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accuracies = []

    print("\n--- Starting Memory-Optimized PyTorch CNN Cross Validation ---")
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_raw, y), 1):
        print(f"\nTraining Fold {fold}...")

        # Zero-copy conversion with from_numpy
        X_train = torch.from_numpy(X_raw[train_idx])
        y_train = torch.from_numpy(y[train_idx])
        X_test = torch.from_numpy(X_raw[test_idx])
        y_test = torch.from_numpy(y[test_idx])

        model = SimpleEEGCNN(channels=channels, time_samples=time_samples).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        # Training
        model.train()
        for epoch in range(15):
            running_loss = 0.0
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            if (epoch + 1) % 5 == 0:
                print(f"   Epoch {epoch+1}/15 - Loss: {running_loss/len(dataloader):.4f}")

        # Batch evaluation
        model.eval()
        test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

        fold_preds = []
        fold_targets = []
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                fold_preds.extend(preds)
                fold_targets.extend(batch_y.numpy())

        acc = accuracy_score(fold_targets, fold_preds)
        fold_accuracies.append(acc)
        print(f"==> Fold {fold} Accuracy: {acc:.4f}")

        # Clean up memory immediately
        del model, dataset, dataloader, test_dataset, test_loader, X_train, y_train, X_test, y_test
        torch.cuda.empty_cache()
        gc.collect()

    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    print(f"\nOverall CNN Cross-Validation Accuracy: {mean_acc:.4f} (+/- {std_acc:.4f})")