Malaria Cell Classification Project
===================================

This project develops a convolutional neural network (CNN) to classify malaria cell images as either "Parasitised" or "Uninfected." The goal is to create a reliable model for aiding malaria diagnosis using image data. 

Contents
--------
- Project Overview
- Model Architecture
- Data Preprocessing
- Training and Evaluation
- Results
- Requirements

Project Overview
----------------
Malaria diagnosis can be significantly enhanced through automation. This project uses image data of cell samples to train a CNN that can identify whether a cell is infected with malaria or not. The key objectives include:

- Achieving high classification accuracy.
- Addressing challenges like class imbalance and overfitting.
- Using advanced techniques like data augmentation and callbacks to optimize the model.

Model Architecture
-------------------
The CNN architecture comprises:

1. *Convolutional Layers*:
   - Three convolutional layers with increasing filters (64, 128, 256).
   - Each layer uses ReLU activation and 3x3 kernels.

2. *Pooling and Batch Normalization*:
   - MaxPooling layers reduce spatial dimensions.
   - Batch normalization stabilizes training.

3. *Fully Connected Layers*:
   - Two dense layers (256 and 128 units) with ReLU activation.
   - Dropout layers (50% rate) to prevent overfitting.

4. *Output Layer*:
   - A single dense layer with sigmoid activation for binary classification.

5. *Compilation*:
   - Optimizer: Adam
   - Loss function: Binary crossentropy
   - Metrics: Accuracy

Data Preprocessing
-------------------
*Training Data*:
- Augmented with:
  - Rotations, zooms, flips, shifts, and rescaling.

*Test Data*:
- Preprocessed with rescaling to normalize pixel values.

*Class Imbalance*:
- Handled by defining custom class weights (higher weight for underrepresented class).

Training and Evaluation
------------------------
The training process includes:

1. *Callbacks*:
   - EarlyStopping: Stops training if validation loss doesn't improve.
   - ReduceLROnPlateau: Reduces learning rate when validation loss plateaus.

2. *Training*:
   - Epochs: 20
   - Batch size: 32
   - Steps per epoch: Determined dynamically based on training data size.

3. *Validation*:
   - Validation data: Separate dataset from testing set.
   - Metrics: Monitored loss and accuracy.

Results
-------
- Test accuracy: *96.2%*
- Loss and accuracy plots confirm model convergence.
- Robust performance with low overfitting risk.

Requirements
------------
To run this project, ensure the following dependencies are installed:

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib

Run the project:

.. code-block:: bash

    python malaria-pred.ipynb

Contact
-------
For any issues or contributions, please contact:
- *Email*: [salehbeda41@gmail.com]
- *GitHub*: [https://github.com/salehbeda41]

License
-------
This project is licensed under the MIT License.