# Fashion-MNIST Image Classification with Custom CNN

### Project Summary
This project implements a custom Convolutional Neural Network (CNN) from scratch using PyTorch to classify the **Fashion-MNIST** dataset. The dataset consists of 70,000 28x28 grayscale images of clothing items across 10 distinct classes.

The pipeline includes:
- Data loading and normalization
- Train/validation split
- CNN architecture design
- Training using Adam optimizer
- Test evaluation with accuracy, precision, recall, F1-score
- Confusion matrix analysis

This project demonstrates end-to-end deep learning workflow.

---
## Preprocessing

- Images converted to tensors using `transforms.ToTensor()` (scales pixels to [0,1]).
- Normalization applied using `transforms.Normalize((0.5,), (0.5,))`.
- Training data split into:
  - 80% training
  - 20% validation
- Batch size: 64
- Training loader shuffled; validation and test loaders not shuffled.

---
### Model Architecture
The architecture follows a modular design for feature extraction and classification:
* **Feature Extraction Block:** * Three convolutional layers with increasing filter counts (32, 64, 128).
    * Each layer utilizes a $3 \times 3$ kernel and ReLU activation.
    * Max-pooling ($2 \times 2$) follows each convolution to reduce spatial dimensions from $28 \times 28$ to $3 \times 3$.
* **Classification Block:**
    * A fully connected layer with 256 neurons.
    * **Dropout (0.5)** to mitigate overfitting during training.
    * A final output layer with 10 neurons corresponding to the clothing classes.

* **Loss Function:** CrossEntropyLoss  
* **Optimizer:** Adam  
* **Learning Rate:** 0.001  
* **Epochs:** 10  
* **Device:** CUDA (GPU)

---

### Model Choice Justification
* **CNN for Spatial Hierarchy:** Unlike standard MLP models, CNNs are specifically chosen here to capture the spatial dependencies and textures inherent in clothing items.
* **Depth and Reduction:** Three pooling stages are used to effectively compress the feature map size while increasing the depth of learned features.
* **Regularization:** Dropout is implemented in the dense layers because custom models trained on small grayscale images are highly susceptible to memorizing training data rather than generalizing.

--- 
### Results & Key Findings
* **Accuracy:** 0.9140  
* **Precision (weighted):** 0.9139  
* **Recall (weighted):** 0.9140  
* **F1-score (weighted):** 0.9133  
* **Observations:** According to the confusion matrix, the model is highly accurate for "Trouser" and "Ankle Boot" classes but occasionally confuses "Shirt" with "T-shirt/top" and "Coat".

---
## Technical Takeaways

- Clean modular training + evaluation functions
- Proper weighted metrics reporting
- GPU acceleration
- Regularization via Dropout
- Reproducible training configuration
