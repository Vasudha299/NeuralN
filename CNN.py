import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class MyCustumCNN(nn.Module):
    def __init__(self):
        super(MyCustumCNN, self).__init__()
        self.Conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.Conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)  # Adjusted the input size
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.Conv1(x)))
        x = self.pool(self.relu(self.Conv2(x)))
        x = x.view(-1, 64 * 5 * 5)  
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = MyCustumCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=1000, shuffle=False)


def train_model(model, trainloader, criterion, optimizer, testloader, num_epochs=50):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}")
        
        if (epoch + 1) in [10, 20, 30, 40, 50]:
            test_accuracy = evaluate_model(model, testloader)
            print(f"Test Accuracy after {epoch + 1} epochs: {test_accuracy:.2f}%")


def evaluate_model(model, testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return (correct / total) * 100


train_model(model, trainloader, criterion, optimizer, testloader, num_epochs=50)


def load_image(image_path):
    image = Image.open(image_path).convert('L')
    image = transform(image)
    image = image.unsqueeze(0)
    return image

def predict(model, image):
    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        return predicted.item()


image_path = 'C:/Users/vasud/AppData/Roaming/Microsoft/Windows/Start Menu/Programs/Python 3.12/cnnimg.png'  
image = load_image(image_path)


predicted_label = predict(model, image)
print(f'Predicted Digit: {predicted_label}')


plt.imshow(np.squeeze(image.numpy()), cmap='gray')
plt.axis('off')
plt.title(f'Predicted Digit: {predicted_label}')
plt.show()










 



            


        









