from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
from medmnist import PathMNIST
from PIL import Image

class MyCustomCNN(nn.Module):
    def __init__(self):
        super(MyCustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 9) 
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


root = './PATHMNIST'
train_dataset = PathMNIST(root=root, split='train', download=True,transform=transform)
test_dataset = PathMNIST(root=root, split='test', download = True,transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = MyCustomCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, criterion, optimizer, test_loader, num_epochs=50):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            if labels.dim() > 1:
                labels = torch.argmax(labels, dim=1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

        if (epoch + 1) in [10, 20, 30, 40, 50]:
            test_accuracy = evaluate_model(model, test_loader)
            print(f"Test Accuracy after {epoch + 1} epochs: {test_accuracy:.2f}%")

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return (correct / total) * 100

train_model(model, train_loader, criterion, optimizer, test_loader, num_epochs=50)

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')  
    image = transform(image)
    image = image.unsqueeze(0)  
    return image

def predict(model, image):
    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        return predicted.item()

image_path = 'C:/Users/vasud/AppData/Roaming/Microsoft/Windows/Start Menu/Programs/Python 3.12/pathmnist_img.jpg'  
image = load_image(image_path)

predicted_label = predict(model, image)
print(f'Predicted Digit: {predicted_label}')


plt.imshow(np.transpose(image.squeeze(0).numpy(), (1, 2, 0)))  
plt.axis('off')
plt.title(f'Predicted Digit: {predicted_label}')
plt.show()
