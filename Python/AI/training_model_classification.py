import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
import os

# Chargement du modèle ResNet pré-entraîné
resnet = models.resnet18(pretrained=True)

# Modification de la dernière couche pour correspondre au nombre de classes
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, 2)  # 2 classes : lit et background

# Définition de la fonction de perte et de l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet.parameters(), lr=0.001, momentum=0.9)

# Définition des transformations pour les données d'entraînement
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Chargement des données d'entraînement et de validation
train_data = datasets.ImageFolder(root='classification/Train_5x/', transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True, num_workers=2)

# Définition des transformations pour les données de validation
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Chargement des données de validation
val_data = datasets.ImageFolder(root='classification/Valid_5x/', transform=val_transform)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=4, shuffle=False, num_workers=2)


# Entraînement du modèle avec validation
best_accuracy = 0.0
for epoch in range(30):  # Nombre d'époques
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()

        outputs = resnet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:    # Imprimer toutes les 100 mini-batchs
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
    
    # Validation du modèle à la fin de chaque époque
    correct = 0
    total = 0
    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            outputs = resnet(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print('Accuracy on validation set: %d %%' % (100 * accuracy))

    # Enregistrement du modèle si l'accuracy de validation est la meilleure
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(resnet.state_dict(), 'best_model.pth')

print('Finished Training')



