# Metal Casting Defect Detection

## Project Description
This project implements a Convolutional Neural Network (CNN) to automatically detect defects in metal casting products. The model analyzes images of casting components and classifies them as either defective or non-defective with high accuracy. This automated inspection system can significantly improve quality control processes in manufacturing environments by reducing manual inspection time and ensuring consistent evaluation criteria.

## Installation Instructions

### Prerequisites
- Python 3.6+
- PyTorch
- torchvision
- matplotlib
- numpy
- scikit-learn
- Google Colab (recommended for GPU acceleration)

### Dataset
The dataset used in this project can be downloaded from [Kaggle](https://www.kaggle.com/koheimuramatsu/real-life-industrial-dataset-of-casting-product/data).

### Setup
1. Clone this repository or download the project files
2. Upload the dataset to your Google Drive with the following structure:
   ```
   casting_data/
     casting_data/
       train/
         def_front/  # Defective casting images for training
         ok_front/   # Non-defective casting images for training
       test/
         def_front/  # Defective casting images for testing
         ok_front/   # Non-defective casting images for testing
   ```
3. Mount your Google Drive in Colab:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

## Usage Instructions

### Training the Model
1. Set the correct paths to your dataset:
   ```python
   DATASET_PATH = '/content/drive/MyDrive/casting_data/casting_data'
   TRAIN_PATH = f'{DATASET_PATH}/train'
   TEST_PATH = f'{DATASET_PATH}/test'
   ```

2. Create data loaders:
   ```python
   train_dataset = CastingDataset(
       defect_dir=f'{TRAIN_PATH}/def_front',
       ok_dir=f'{TRAIN_PATH}/ok_front',
       transform=train_transforms)

   test_dataset = CastingDataset(
       defect_dir=f'{TEST_PATH}/def_front',
       ok_dir=f'{TEST_PATH}/ok_front',
       transform=test_transforms)

   train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
   test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
   ```

3. Initialize and train the model:
   ```python
   model = CastingDefectCNN()
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   model.to(device)
   
   criterion = nn.CrossEntropyLoss()
   optimizer = optim.Adam(model.parameters(), lr=0.001)
   
   train_model(model, train_loader, criterion, optimizer, device, num_epochs=10)
   ```

### Evaluating the Model
```python
test_loss, test_accuracy, report, cm = evaluate_model(model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.2f}%")
print("\nClassification Report:")
print(report)
```

## Output Images

The model produces the following outputs:

1. **Classification Report**: Displays precision, recall, and F1-score for both OK and Defective classes, with an overall accuracy of 90.35%.

2. **Confusion Matrix**: Shows the distribution of predictions:
   - 262 OK samples correctly identified
   - 0 OK samples incorrectly classified as Defective
   - 69 Defective samples incorrectly classified as OK
   - 384 Defective samples correctly identified

3. **Sample Visualizations**: The code can generate visualization of model predictions on test images, showing both correct and incorrect classifications.

## Technologies Used
- **PyTorch**: Deep learning framework for model implementation
- **torchvision**: For image transformations and data augmentation
- **matplotlib**: For visualization of results and confusion matrix
- **scikit-learn**: For evaluation metrics and classification reports
- **Google Colab**: Cloud-based platform for training with GPU acceleration

## Model Architecture
```python
class CastingDefectCNN(nn.Module):
    def __init__(self):
        super(CastingDefectCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        
        # Calculate the size of flattened features
        self.fc_input_features = 128 * 14 * 14
        
        self.fc1 = nn.Linear(self.fc_input_features, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 2)  # 2 classes: defective and OK
```
![Architecture](https://res.cloudinary.com/dqhyudo4x/image/upload/v1742299276/Archetecture_c1jvxz.jpg)

## Results
- **Test Accuracy**: 90.35%
- **OK Class**: Precision: 0.79, Recall: 1.00, F1-score: 0.88
- **Defective Class**: Precision: 1.00, Recall: 0.85, F1-score: 0.92
![Result](https://res.cloudinary.com/dqhyudo4x/image/upload/v1742299275/Accuracy_zfh2vk.jpg)

- **Confusion Matrix**:
![Confusion Matrix](https://res.cloudinary.com/dqhyudo4x/image/upload/v1742299275/Confusion_onf5nr.jpg)

- **Sample Visualizations**:
![Sample Visualizations](https://res.cloudinary.com/dqhyudo4x/image/upload/v1742299276/test_pzlmqg.jpg)

## Future Improvements
1. **Model Optimization**: 
   - Experiment with different CNN architectures
   - Implement more extensive data augmentation techniques
   - Apply transfer learning with pre-trained models

2. **Feature Enhancement**:
   - Develop a defect localization system that highlights the defective regions
   - Implement a multi-class classifier to identify specific defect types
   - Create a real-time detection system for production line integration

3. **User Interface**:
   - Develop a web-based dashboard for uploading and analyzing casting images
   - Create visualization tools to better interpret model predictions
   - Implement a feedback mechanism for continuous model improvement

4. **Deployment**:
   - Package the model for production deployment
   - Create APIs for integration with manufacturing systems
   - Optimize for edge computing devices for on-site deployment

## Author Information
Pranjal Sharma 
GitHub: https://github.com/Pranjal-sharma-SDE

