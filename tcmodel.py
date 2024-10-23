import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Step 1: Define hyperparameters
batch_size = 64
learning_rate = 0.001
epochs = 20

# Step 2: Define transformations for the dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Step 3: Load MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Step 4: Define the neural network model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # Input layer (784 nodes), Hidden layer (128 nodes)
        self.fc2 = nn.Linear(128, 64)       # Hidden layer (128 nodes), Hidden layer (64 nodes)
        self.fc3 = nn.Linear(64, 10)        # Hidden layer (64 nodes), Output layer (10 classes)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the 28x28 images into a single vector (28*28 = 784)
        x = F.relu(self.fc1(x))   # Apply ReLU activation to hidden layers
        x = F.relu(self.fc2(x))
        x = self.fc3(x)           # Output layer (no activation, will use Softmax in loss)
        return x

# Step 5: Instantiate the model, define the loss and optimizer
model = SimpleNN()
criterion = nn.CrossEntropyLoss()  # Suitable for classification tasks
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Step 6: Train the model
def train(model, train_loader, criterion, optimizer, epochs):
    model.train()  # Set the model to training mode
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Step 7: Evaluate the model on test data
def evaluate(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # Turn off gradients for evaluation
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on test images: {accuracy:.2f}%')

# Run the training and evaluation
train(model, train_loader, criterion, optimizer, epochs)
evaluate(model, test_loader)
