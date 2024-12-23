# Malaria Detection Using Deep Learning

## Project Overview
This project implements a Convolutional Neural Network (CNN) using Keras to classify cell images as infected or uninfected by malaria. The goal is to assist in the early detection and treatment of malaria through automated image classification.

## Features
- **Deep Learning Architecture**: Uses Keras Sequential API to build a CNN with multiple layers, including convolutional, pooling, and dropout layers.
- **Batch Normalization**: Improves training stability and accelerates convergence.
- **Customizable Model**: Designed for easy experimentation with hyperparameters and layers.
- **VGG16**: A pre-trained deep learning model fine-tuned for this task.
- **Machine Learning**:
-   ***Support Vector Machine (SVM)**: A classical machine learning approach applied to extracted features.
-   **K-Nearest Neighbors (KNN)**: A distance-based classifier tested on the dataset.

## Installation
To run this project, ensure you have the following dependencies installed:

```bash
pip install tensorflow keras matplotlib numpy
```

## Dataset
The project requires a dataset of cell images, categorized into infected and uninfected classes. Ensure the dataset is organized as follows:

```
./Malaria Cells
  /single_prediction
  /train
    /Parasitized
    /uninfected
  /test
    /Parasitized
    /uninfected
./Test_Images
```


## Usage
1. Clone the repository:
   ```bash
   git clone <https://github.com/salehbeda41/Disease-Prediction-Model-master.git>
   cd Disease-Prediction-Model-master
   ```

2. Ensure the dataset is in place.

3. Run the notebook:
   ```bash
   jupyter notebook malaria-pred.ipynb
   ```

4. Follow the steps in the notebook to train and evaluate the model.

## CNN Model Architecture
The CNN model consists of:
- Multiple convolutional layers with ReLU activation
- Max pooling layers
- Dropout layers for regularization
- Batch normalization layers
- Fully connected dense layers

## Results
### CNN Model
- **Training Accuracy**: 94.16%
- **Validation Accuracy**: 95.88%
- **Test Accuracy**: 95.93%
- **Final Loss (Validation)**: 0.1159
- **Final Loss (Test)**: 0.1092

### VGG16
- **Training Accuracy**: 87.99%
- **Validation Accuracy**: 91.30%
- **Final Validation Loss**: 0.2142

### Support Vector Machine (SVM)
- **Accuracy**: 49.07%

### K-Nearest Neighbors (KNN)
- **Accuracy**: 50.26%

## Contributing
Contributions are welcome! Feel free to fork this repository and submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements
- Keras for providing an easy-to-use deep learning API
- [Kaggle.com] for the malaria cell images dataset

---
Add or replace placeholders (e.g., `<https://github.com/salehbeda41/Disease-Prediction-Model-master.git>`, dataset source) with specific information.
