# Human Activity Recognition using Sensor Data

## Project Description
This project aims to classify human activities using sensor data collected from smartphones. The dataset (UCI HAR) consists of 561 features extracted from accelerometer and gyroscope readings. The goal is to build a robust neural network model to accurately distinguish between six different activities:
- Walking
- Walking Upstairs
- Walking Downstairs
- Sitting
- Standing
- Laying

## Model Architecture
The solution employs a fully connected neural network (Deep Learning) implemented in **TensorFlow/Keras**.

### Layers Structure
1.  **Input Layer**: 561-dimensional input feature vector.
2.  **Hidden Layer 1**: 128 units, `ReLU` activation.
3.  **Hidden Layer 2**: 64 units, `ReLU` activation.
4.  **Output Layer**: 6 units, `Linear` activation (Logits) for multi-class classification.

### Hyperparameters
-   **Optimizer**: Adam
-   **Learning Rate**: 0.0006
-   **Loss Function**: Sparse Categorical Crossentropy (from logits)
-   **Batch Size**: 64
-   **Epochs**: 100 (with Early Stopping patience=10)

## Performance Analysis
The model was evaluated on a held-out test set (15% of the data, ~1,545 samples) after training.

### Test Results
-   **Test Accuracy**: **98.45%** 🚀
-   **Test Loss**: (Low validation loss observed)

### Key Insights
-   **Dynamic Activities**: The model achieves near-perfect classification for dynamic activities like *Walking*, *Walking Upstairs*, and *Walking Downstairs*.
-   **Static Activities**: There is a slight challenge in distinguishing between *Sitting* and *Standing*. This is a known characteristic of the dataset, as the sensor patterns for these two stationary postures are extremely similar.
-   **Bias vs Variance**: The learning curves (Training vs Validation) show a stable convergence, indicating low bias and low variance. The model generalizes well to unseen data without significant overfitting.

## Technologies Used
-   **Python 3.11**
-   **TensorFlow / Keras 2.x**
-   **NumPy** (Data manipulation)
-   **Pandas** (Data loading)
-   **Matplotlib** (Visualization)
