
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class MultiMNISTDataset(Dataset):
    def __init__(self, data_dir, transform=None, is_multi=False):
        self.images, self.labels = self.load_mnist_data(data_dir, transform, is_multi)
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, index):
        return self.images[index], self.labels[index]
    
    @staticmethod
    def get_multi_labels(label_str, max_length):
        # Initialize output vector with space for 11 classes per position
        output_vector = np.zeros(max_length * 11)
        
        if label_str == '0':
            # For empty/zero case, set the 11th position (no digit) to 1 for all positions
            for i in range(max_length):
                output_vector[i * 11 + 10] = 1
            return output_vector
            
        # Fill in the actual digits
        for i in range(max_length):
            if i < len(label_str):
                # Set the corresponding digit position to 1
                output_vector[i * 11 + int(label_str[i])] = 1
            else:
                # Set the no-digit indicator (11th position) to 1
                output_vector[i * 11 + 10] = 1
                
        return output_vector
    
    @staticmethod
    def load_mnist_data(data_dir, transform=None, is_multi=False):
        max_length = 0
        for label_str in os.listdir(data_dir):
            max_length = max(max_length, len(label_str))
            
        images = []
        labels = []
        
        for label_str in os.listdir(data_dir):
            folder_path = os.path.join(data_dir, label_str)
            
            if is_multi:
                label = MultiMNISTDataset.get_multi_labels(label_str, max_length)
            else:
                label = len(label_str) if label_str != '0' else 0
                
            for image_file in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_file)
                image = Image.open(image_path).convert('L')
                if transform:
                    image = transform(image)
                images.append(image)
                labels.append(label)
                
        return torch.stack(images), torch.tensor(labels, dtype=torch.float32)

    @staticmethod
    def get_dataloader(data_dir, batch_size=32, shuffle=True, transform=None, is_multi=False):
        dataset = MultiMNISTDataset(data_dir, transform, is_multi)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    @staticmethod
    def get_dimensions(loader):
        for image, _ in loader:
            return image.shape[1], (image.shape[2], image.shape[3])
    
    @staticmethod
    def get_max_classes(data_dir):
        max_length = 0
        for label_str in os.listdir(data_dir):
            max_length = max(max_length, len(label_str))
        return max_length

class MultiLabelCNN(nn.Module):
    def __init__(self, layer_config=[], input_channels=1, input_size=(128,128), max_classes=3):
        super().__init__()
        
        self.max_classes = max_classes
        self.layers = nn.ModuleList()
        self.activations = []
        
        layer_in = input_channels
        current_size = input_size
        
        # Build convolutional layers
        for config in layer_config:
            if config['type'] == 'conv2d':
                layer_out = config['out_channels']
                self.layers.append(
                    nn.Conv2d(layer_in, layer_out, 
                             kernel_size=config['kernel_size'],
                             stride=config['stride'],
                             padding=config['padding'])
                )
                self.activations.append(self.get_activation(config['activation']))
                
                current_size = self.calculate_conv_output_size(
                    current_size, config['kernel_size'],
                    config['stride'], config['padding']
                )
                layer_in = layer_out
                
            elif config['type'] == 'pool':
                self.layers.append(
                    nn.MaxPool2d(kernel_size=config['kernel_size'], stride=2)
                    if config['pool_type'] == 'max'
                    else nn.AvgPool2d(kernel_size=config['kernel_size'], stride=2)
                )
                current_size = self.calculate_pool_output_size(
                    current_size, config['kernel_size'], stride=2
                )
        
        # Calculate flattened size
        flattened_size = layer_in * current_size[0] * current_size[1]
        
        # Fully connected layers with dropout
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(flattened_size, 512)
        self.fc2 = nn.Linear(512, max_classes * 11)  # 11 classes per position (0-9 + no digit)
        
    def get_activation(self, activation):
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }
        return activations.get(activation, nn.ReLU())
        
    def calculate_conv_output_size(self, input_size, kernel_size, stride, padding):
        h_in, w_in = input_size
        h_out = ((h_in - kernel_size + 2 * padding) // stride) + 1
        w_out = ((w_in - kernel_size + 2 * padding) // stride) + 1
        return (h_out, w_out)
        
    def calculate_pool_output_size(self, input_size, kernel_size, stride):
        h_in, w_in = input_size
        h_out = ((h_in - kernel_size) // stride) + 1
        w_out = ((w_in - kernel_size) // stride) + 1
        return (h_out, w_out)
        
    def forward(self, x):
        # Convolutional layers
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.activations):
                x = self.activations[i](x)
        
        # Flatten and fully connected layers
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        # Reshape and apply softmax for each digit position
        batch_size = x.size(0)
        x = x.view(batch_size, self.max_classes, 11)
        x = F.softmax(x, dim=2)
        x = x.view(batch_size, -1)
        
        return x
    
    @staticmethod
    def calculate_accuracy(outputs, labels, max_digits):
        batch_size = outputs.size(0)
        outputs = outputs.view(batch_size, max_digits, 11)
        labels = labels.view(batch_size, max_digits, 11)
        
        predicted_classes = torch.argmax(outputs, dim=2)
        true_classes = torch.argmax(labels, dim=2)
        
        correct_predictions = (predicted_classes == true_classes).sum().item()
        total_positions = batch_size * max_digits
        
        return 100 * correct_predictions / total_positions
        
    def train_model(self, train_loader, val_loader, optimizer, loss_function, epochs=5):
        best_accuracy = 0
        
        for epoch in range(epochs):
            self.train()
            total_loss = 0.0
            
            for i, (inputs, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            # Validation
            self.eval()
            total_accuracy = 0.0
            val_loss = 0.0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = self(inputs)
                    val_loss += loss_function(outputs, labels).item()
                    accuracy = self.calculate_accuracy(outputs, labels, self.max_classes)
                    total_accuracy += accuracy
                    
            avg_accuracy = total_accuracy / len(val_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            print(f"Epoch {epoch + 1}/{epochs}:")
            print(f"Training Loss: {total_loss/len(train_loader):.4f}")
            print(f"Validation Loss: {avg_val_loss:.4f}")
            print(f"Validation Accuracy: {avg_accuracy:.2f}%")
            
            if avg_accuracy > best_accuracy:
                best_accuracy = avg_accuracy
                self.save('best_model.pth')
                
    def test_model(self, test_loader):
        self.eval()
        total_accuracy = 0.0
        test_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self(inputs)
                accuracy = self.calculate_accuracy(outputs, labels, self.max_classes)
                total_accuracy += accuracy
                
        avg_accuracy = total_accuracy / len(test_loader)
        print(f"Test Accuracy: {avg_accuracy:.2f}%")
        return avg_accuracy
        
    def save(self, path):
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")
        
    def load(self, path):
        self.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")

# Configuration and training setup
layer_config = [
    {'type': 'conv2d', 'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'activation': 'relu'},
    {'type': 'pool', 'kernel_size': 2, 'pool_type': 'max'},
    {'type': 'conv2d', 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'activation': 'relu'},
    {'type': 'pool', 'kernel_size': 2, 'pool_type': 'max'},
    {'type': 'conv2d', 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'activation': 'relu'},
    {'type': 'pool', 'kernel_size': 2, 'pool_type': 'max'},
]

# Data loading
train_loader = MultiMNISTDataset.get_dataloader('../../data/external/double_mnist/train', batch_size=32, shuffle=True, transform=transform, is_multi=True)

val_loader = MultiMNISTDataset.get_dataloader('../../data/external/double_mnist/val', batch_size=32, shuffle=True, transform=transform, is_multi=True)

test_loader = MultiMNISTDataset.get_dataloader('../../data/external/double_mnist/test', batch_size=32, shuffle=True, transform=transform, is_multi=True)

max_classes = MultiMNISTDataset.get_max_classes('../../data/external/double_mnist/train')
input_channels, input_size = MultiMNISTDataset.get_dimensions(train_loader)

# Model initialization and training
model = MultiLabelCNN(layer_config=layer_config, input_channels=input_channels, input_size=input_size, max_classes=max_classes)

optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_function = nn.CrossEntropyLoss()  # Binary Cross Entropy Loss for multi-label classification

# Train the model
model.train_model(train_loader, val_loader, optimizer, loss_function, epochs=10)

# Test the model
model.test_model(test_loader)
