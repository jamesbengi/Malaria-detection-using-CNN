# Malaria Classification Model

## Overview
This project implements a machine learning model to classify cell images as either malaria-infected or uninfected. The goal is to assist in the rapid and accurate detection of malaria, improving diagnostics and potentially saving lives.

---

## Dataset
The model uses the publicly available **Malaria Cell Images Dataset** from the National Institutes of Health (NIH). The dataset contains:
- **Parasitized:** Images of cells infected with malaria.
- **Uninfected:** Images of healthy cells.

**Dataset Source:** [Malaria Dataset - NIH](https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria)

**Dataset Size:**
- Approximately 27,558 images (split evenly between the two classes).
- Image resolution: 128x128 pixels (grayscale or RGB).

---

## Model Architecture
The architecture used in this project is designed to leverage the strength of **residual learning**. This allows the implementation of deep convolutional layers without the vanishing gradient problem by introducing **skip connections**.

### Key Features of the Architecture:
1. **Residual Block:**
   - Defined as:
     \[ y = F(x, \{Wi\}) + x \]
     where:
     - \(x\) and \(y\) are the input and output vectors of the layers considered.
     - \(F(x, \{Wi\})\) represents the residual mapping to be learned.

2. **Custom ResNet Configuration:**
   - Lightweight with only **10 layers** compared to the original ResNet with up to 152 layers.

3. **Downsampling:**
   - Uses **strided convolutions** instead of max pooling to enhance feature learning during downsampling.

4. **Residual Blocks:**
   - Three blocks with increasing filter sizes:
     - Block 1: 64 filters.
     - Block 2: 126 filters.
     - Block 3: 256 filters.

5. **Global Average Pooling (GAP):**
   - Reduces the number of parameters after the residual blocks.

6. **Dense Layers:**
   - A dense layer with 512 units and ReLU activation processes the GAP output.
   - Dropout (rate: 0.5) is applied to prevent overfitting.
   - A final dense layer with a sigmoid activation outputs a single value for binary classification.

7. **Projection Shortcut:**
   - When spatial dimensions change due to strided convolutions, a **1×1 convolution** is applied to match dimensions.
   - Skip connections perform element-wise addition of the batch normalization layer and the shortcut projection.

---

## Training
### Steps:
1. **Preprocessing:**
   - Resized all images to 128x128 pixels.
   - Normalized pixel values to the range [0, 1].
   - Split the dataset into training, validation, and testing sets (e.g., 70/15/15).

2. **Data Augmentation:**
   - Applied techniques like rotation, flipping, and zooming to increase dataset diversity.

3. **Model Compilation:**
   - Loss Function: Binary Crossentropy.
   - Optimizer: Adam (learning rate = 0.001).
   - Metrics: Accuracy.

4. **Training:**
   - Epochs: 100.
   - Batch Size: 32.

## Results
- **Training Accuracy:** 98%
- **Validation Accuracy:** 96%
- **Test Accuracy:** 96%
- **Precision, Recall, F1-Score:** Computed to evaluate model robustness.
## Acknowledgments
- NIH for the dataset.
- Open-source libraries and frameworks used in this project.

---

## Contact
For any inquiries or contributions, please contact:
**James Bengi Wanjala**
- Email: [jamesbengi21@gmail.com](mailto:jamesbengi21@gmail.com)

