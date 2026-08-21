import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleEEGCNN(nn.Module):
    """
    A lightweight CNN baseline for Motor Imagery classification.
    Expects input shape: (batch_size, 1, channels, time_samples)
    """
    def __init__(self, num_classes=2):
        super(SimpleEEGCNN, self).__init__()
        # Temporal convolution to learn frequency bands
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(1, 64), padding=(0, 32))
        self.batchnorm1 = nn.BatchNorm2d(16)
        
        # Spatial convolution to learn electrode combinations
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(64, 1))
        self.batchnorm2 = nn.BatchNorm2d(32)
        
        self.pooling = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout = nn.Dropout(0.25)
        
        # Adjust the linear input dimension based on your specific epoch length
        self.fc1 = nn.Linear(480, num_classes)
    def forward(self, x):
        x = F.elu(self.batchnorm1(self.conv1(x)))
        x = self.dropout(self.pooling(F.elu(self.batchnorm2(self.conv2(x)))))
        x = x.view(x.size(0), -1) 
        x = self.fc1(x)
        return x


import torch.optim as optim

def train_cnn(model: nn.Module, X_train: torch.Tensor, y_train: torch.Tensor, epochs: int = 50, batch_size: int = 32, lr: float = 0.001):
    """
    Trains the simple EEG CNN model.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(dataloader):.4f}")
            
    return model



if __name__ == "__main__":
    # Create a dummy tensor representing 16 epochs, 1 color channel, 64 electrodes, 60 time samples
    dummy_eeg_data = torch.randn(16, 1, 64, 60)
    dummy_labels = torch.randint(0, 2, (16,))
    
    print("Testing CNN Forward Pass...")
    model = SimpleEEGCNN(num_classes=2)
    output = model(dummy_eeg_data)
    print(f"Output shape: {output.shape} (Expected: [16, 2])")
    
    print("\nTesting Training Loop...")
    trained_model = train_cnn(model, dummy_eeg_data, dummy_labels, epochs=10)