# Practice: CNN Image Classifier

**Phase:** PHASE-03-deep-learning  
**Subjects Required:** 64 (Convolution 1D), 65 (Convolution 2D), 66 (Pooling and Padding), 61 (Batch Normalization)  
**Estimated Time:** 240 minutes  
**Difficulty:** Intermediate

## Industry Context

You are an ML engineer at a satellite imaging startup. The operations team has 50,000 unlabeled aerial photographs and needs a classifier to flag images containing cloud cover versus clear land. They cannot afford a commercial API and want an on-device model that runs on a single GPU. Your manager asks for a baseline CNN trained from scratch using PyTorch, with reproducible training loops and basic evaluation metrics. The model must be simple enough to explain to the non-technical ops team but accurate enough to reduce manual review by at least 70%.

## The Problem

Build and train a Convolutional Neural Network (CNN) from scratch in PyTorch that classifies images from the CIFAR-10 dataset into 10 classes. You must implement the model architecture manually (no pre-built models from `torchvision.models`), incorporating 2D convolutions, pooling, and batch normalization. Write a complete training loop with loss computation, backpropagation, and parameter updates. Evaluate the model on a held-out test set and report accuracy.

The problem is scoped to the four prerequisite subjects:
- **Convolution 2D:** Extract spatial features from images using learnable filters.
- **Pooling and Padding:** Downsample feature maps and control spatial dimensions.
- **Batch Normalization:** Stabilize training by normalizing layer inputs.
- **Convolution 1D:** (Applied conceptually) Understand how 1D convolutions relate to 2D convolutions when collapsing spatial dimensions, or use a 1D convolution in a final squeeze operation if you choose.

## Constraints

- Do not use `torchvision.models` or any pre-trained weights. Build the `nn.Module` from scratch.
- Use only PyTorch core (`torch`, `torch.nn`, `torch.optim`, `torch.utils.data`) and `torchvision.datasets` / `torchvision.transforms` for data loading.
- Train on CIFAR-10 (32x32 RGB images, 10 classes). Download automatically via `torchvision.datasets.CIFAR10`.
- Model must have fewer than 1 million parameters.
- Training must complete in under 10 minutes on a single CPU or under 2 minutes on a single GPU (Colab T4 equivalent).
- Use at least one `nn.BatchNorm2d` layer and at least one pooling layer (`nn.MaxPool2d` or `nn.AvgPool2d`).
- Report top-1 test accuracy after 5 epochs of training.

## Starter Code

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# ------------------------------------------------------------------
# Hyperparameters
# ------------------------------------------------------------------
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------
# Data Loading
# ------------------------------------------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
)

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# ------------------------------------------------------------------
# Model Definition
# ------------------------------------------------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # TODO: Define convolutional layers, batch norm, pooling, and fully-connected layers.
        # Constraint: < 1M parameters.
        pass

    def forward(self, x):
        # TODO: Implement the forward pass.
        # Expected input shape: (batch_size, 3, 32, 32)
        # Expected output shape: (batch_size, 10)
        pass

# ------------------------------------------------------------------
# Training Loop
# ------------------------------------------------------------------
def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # TODO: Zero gradients, forward pass, compute loss, backward pass, optimizer step.

        running_loss += loss.item()
    return running_loss / len(loader)

# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # TODO: Forward pass and compute accuracy.

    return correct / total

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
if __name__ == "__main__":
    model = SimpleCNN().to(DEVICE)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        train_loss = train(model, trainloader, criterion, optimizer, DEVICE)
        test_acc = evaluate(model, testloader, DEVICE)
        print(f"Epoch {epoch+1}/{EPOCHS} — Loss: {train_loss:.4f} — Test Acc: {test_acc:.4f}")
```

## Evaluation Criteria

1. **Architecture correctness:** The model uses `nn.Conv2d`, at least one `nn.BatchNorm2d`, and at least one pooling layer. Parameter count is under 1 million.
2. **Training loop completeness:** Gradients are zeroed, forward pass computes logits, loss is backpropagated, and optimizer updates weights for every batch.
3. **Test accuracy:** After 5 epochs, top-1 accuracy on CIFAR-10 test set should be ≥ 60%. (A well-tuned simple CNN typically reaches 65–72%.)
4. **Shape discipline:** No shape mismatches between layers. The transition from the last conv/pool feature map to the first fully-connected layer is computed dynamically or documented explicitly.
5. **Reproducibility:** The script runs end-to-end without manual intervention and prints a clear accuracy number.

## Solution

<details>
<summary>Click to reveal solution</summary>

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# ------------------------------------------------------------------
# Hyperparameters
# ------------------------------------------------------------------
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------
# Data Loading
# ------------------------------------------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
)

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# ------------------------------------------------------------------
# Model Definition
# ------------------------------------------------------------------
class SimpleCNN(nn.Module):
    """
    A compact CNN for CIFAR-10.
    Architecture: [CONV -> BN -> ReLU -> POOL] x 2 -> FC -> FC
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Block 1: 3x32x32 -> 32x32x32 -> 32x16x16
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2: 32x16x16 -> 64x16x16 -> 64x8x8
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Fully connected layers
        # After two poolings: 32 -> 16 -> 8, so 64 channels * 8 * 8 = 4096
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)   # (B, 32, 32, 32)
        x = self.bn1(x)     # normalize across batch and channels
        x = torch.relu(x)
        x = self.pool(x)    # (B, 32, 16, 16)

        # Block 2
        x = self.conv2(x)   # (B, 64, 16, 16)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool(x)    # (B, 64, 8, 8)

        # Flatten for FC layers
        x = x.view(x.size(0), -1)   # (B, 4096)
        x = torch.relu(self.fc1(x)) # (B, 256)
        x = self.fc2(x)             # (B, 10)
        return x

# ------------------------------------------------------------------
# Training Loop
# ------------------------------------------------------------------
def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()          # clear accumulated gradients
        outputs = model(inputs)        # forward pass: compute logits
        loss = criterion(outputs, labels)
        loss.backward()                # backpropagation: compute gradients
        optimizer.step()               # update parameters

        running_loss += loss.item()
    return running_loss / len(loader)

# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
if __name__ == "__main__":
    model = SimpleCNN().to(DEVICE)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")  # ~181k parameters

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        train_loss = train(model, trainloader, criterion, optimizer, DEVICE)
        test_acc = evaluate(model, testloader, DEVICE)
        print(f"Epoch {epoch+1}/{EPOCHS} — Loss: {train_loss:.4f} — Test Acc: {test_acc:.4f}")
```

</details>

## What You Actually Learned

- **Convolution 2D:** You implemented `nn.Conv2d` layers that slide learnable filters across spatial dimensions, turning raw pixels into hierarchical feature maps (edges -> textures -> shapes).
- **Pooling and Padding:** You used `padding=1` to preserve spatial size after 3x3 convolutions and `MaxPool2d` to halve dimensions, reducing computation and providing translation invariance.
- **Batch Normalization:** You inserted `nn.BatchNorm2d` after convolutions to normalize activations across the batch, which stabilized training and allowed a higher learning rate without divergence.
- **Convolution 1D:** You recognized that the flattened feature vector before the FC layer is conceptually a 1D signal; while you used `view()` here, the same idea of sliding filters over sequential data (e.g., time series, text embeddings) is the foundation of 1D convolutions.
- **Training hygiene:** You wrote a complete loop (zero_grad → forward → loss → backward → step) and separated `train()` and `eval()` modes, which controls whether batch norm uses running statistics and whether dropout (if added) is active.

## Sources Used

- [PyTorch Tutorial: Training a Classifier](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) — Official end-to-end CIFAR-10 CNN tutorial.
- [CS231n: Convolutional Neural Networks](https://cs231n.github.io/convolutional-networks/) — Stanford course notes on ConvNet layers, spatial arrangements, and architecture patterns.
- [PyTorch Docs: torch.nn](https://pytorch.org/docs/stable/nn.html) — API reference for Conv2d, BatchNorm2d, MaxPool2d, and Linear layers.
- [Distill.pub: Feature Visualization](https://distill.pub/2017/feature-visualization/) — Interactive deep-dive into what CNN filters learn at different depths.
- [arXiv: Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) — Foundational paper showing how depth and normalization enable modern CNN accuracy (He et al., 2015).
