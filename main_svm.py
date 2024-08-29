import torch
from models.cnn import CNNModel
from models.svm import SVMClassifier
from utils.data import load_data
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Class labels
class_labels = ['bacterial leaf blight','bacterial leaf streak','bacterial panicle blight', 'blast', 'brown spot', 'dead heart', 'downy mildew','hispa', 'normal','tungro']

# Load Data
print("Loading Data...")
train_loader, test_loader = load_data('dataset1/train_images', 'dataset1/test_images')
print("Data Loaded Successfully!")

# Hyperparameters
epochs = 10
learning_rate = 0.001

# Build CNN model
print("Building CNN model...")
cnn_model = CNNModel().to(device)
optimizer = torch.optim.Adam(cnn_model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# Lists to store losses and accuracies
train_losses = []
train_accuracies = []

# Training Loop
print("Starting Training...")
for epoch in range(epochs):
    cnn_model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = cnn_model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct / total

    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)

    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy * 100:.2f}%')

print("Training Complete!")

# Plot Loss and Accuracy on the same plot
plt.figure(figsize=(10, 5))
epochs_range = range(1, epochs + 1)

# Plot Loss
plt.plot(epochs_range, train_losses, marker='o', color='blue', label='Loss')

# Plot Accuracy
plt.plot(epochs_range, train_accuracies, marker='o', color='green', label='Accuracy')

plt.title('Training Loss and Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Loss/Accuracy')
plt.legend(loc='best')
plt.grid(True)
plt.show()

# Testing Phase - Feature Extraction using CNN
print("Extracting features using CNN...")
cnn_model.eval()
test_features, test_labels = [], []
with torch.no_grad():
    for images, targets in test_loader:
        images, targets = images.to(device), targets.to(device)
        outputs = cnn_model(images)
        test_features.append(outputs.cpu().detach().numpy())
        test_labels.append(targets.cpu().numpy())
test_features = np.vstack(test_features)
test_labels = np.hstack(test_labels)
print("Feature extraction complete!")

# Train and Test SVM
print("Training SVM model...")
svm_model = SVMClassifier()
svm_model.fit(test_features, test_labels)
print("SVM training complete!")

# Evaluate SVM model
print("Evaluating the SVM model...")
predictions = svm_model.predict(test_features)

# Debugging: Print the shape of predictions
print(f'Predictions shape: {predictions.shape}')

# Calculate accuracy
accuracy = accuracy_score(test_labels, predictions)
print(f'Final Test Accuracy: {accuracy * 100:.2f}%')

# Classification Report
print("Classification report SVM on 100*100 image inputs:")
report = classification_report(test_labels, predictions, target_names=class_labels)
print(report)

# Confusion Matrix
cm = confusion_matrix(test_labels, predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix SVM on 100*100 image inputs')
plt.show()

# Save the models
print("Saving models...")
model_path = 'models/hybrid_cnn_svm.pth'
torch.save({
    'cnn_state_dict': cnn_model.state_dict(),
    'svm_model': svm_model.svm_model,
    'svm_scaler': svm_model.scaler
}, model_path)
print(f'Models saved to {model_path}')
