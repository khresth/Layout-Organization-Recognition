# Layout-Organization-Recognition HumanAI RenAIssance Test I
A model based on convolutional neural network with a regression head for optically recognizing the layout of each data source. It detects where in each page main text is present, and disregards other embellishments. 

The dataset consists of PDFs of early modern printed sources, with the first 3 pages of each source used for training and evaluation. The goal is to predict a bounding box around the main text region on each page, ignoring headers, marginalia, or decorative elements.

## Approach
### Model Choice
I chose a **Convolutional Neural Network (CNN)** with a regression head to predict the bounding box coordinates (x_min, y_min, x_max, y_max) of the main text region. The CNN architecture includes multiple Conv2D layers (32, 64, 128, 256 filters) with MaxPooling for spatial feature extraction, followed by Dense layers with dropout for regularization, and a final Dense layer with sigmoid activation to output normalized coordinates.

- A CNN is well-suited for spatial feature extraction, which is critical for identifying text regions in images. The regression head simplifies the task to predicting a single box, aligning with the focus on the main text.
- While convolutional-recurrent or transformer architectures were options, Recurrent layers are better for sequential data (e.g., text recognition), and transformers are computationally expensive and more suited for tasks requiring global context. Given the small dataset (18 images, augmented to 36) and the simplicity of the layout (one main text block per page), a CNN was the most practical and effective choice.
- Self-supervised learning was not used due to the small dataset and the specific nature of the task, which benefits from supervised training with Tesseract-based annotations.

### Data Preparation
- **PDF to Images**: Converted the first 3 pages of each PDF to images using `pdf2image`, resulting in 18 images. Images were preprocessed by converting to grayscale, resizing to 224x224, and normalizing pixel values to [0, 1].
- **Ground-Truth Annotations**: Used Tesseract OCR to detect word-level bounding boxes, merging them into a single box for the main text region. This approach improved over an initial contour-based method, which often misidentified headers as the main text. Tesseract was configured with the Spanish language model (`lang='spa'`) to better handle early modern Spanish texts.
- **Data Augmentation**: Applied random rotation and brightness adjustments to double the dataset size to 36 images, improving generalization on the small dataset.

### Model Training
- **Architecture**: The CNN includes Conv2D layers with increasing filters (32, 64, 128, 256), MaxPooling for downsampling, and Dense layers with dropout (0.5) to prevent overfitting. The output layer predicts 4 normalized coordinates.
- **Custom Loss**: Implemented a custom loss function that combines Mean Squared Error (MSE) with a size penalty to favor larger boxes, as main text regions are typically large in early modern printed sources.
- **Training**: Trained the model for 50 epochs with a batch size of 8, using early stopping (patience=5) to prevent overfitting. The Adam optimizer was used with the custom loss and Mean Absolute Error (MAE) as an additional metric.

### Evaluation Metrics
- **Intersection over Union (IoU)**: Measures the overlap between predicted and true bounding boxes, averaged across the validation set. A higher IoU indicates better alignment (target > 0.5 for "good" detection).
- **Precision**: Fraction of predicted boxes with IoU > 0.5, indicating the proportion of correct detections.
- **Recall**: Fraction of true boxes detected with IoU > 0.5 (simplified as equal to precision due to the one-box assumption).
- These metrics are standard for object detection tasks and directly assess the model’s ability to localize the main text region.

## Results
- **Performance**: The model achieved a high IoU (0.8214) on validation samples, indicating strong alignment between predicted and true bounding boxes. Precision and recall are also high Precision (IoU > 0.5): 1.0000 Recall (IoU > 0.5): 1.0000 as most predictions exceed the IoU > 0.5 threshold.

- **Visualization**: The visualization below shows a validation sample with the true (green) and predicted (red) bounding boxes. The close alignment confirms the model’s effectiveness in detecting the main text region while ignoring marginalia.

![image](https://github.com/user-attachments/assets/d3dd60bd-c765-4118-9ec2-312fa1d490cc)


## Limitations
- The model assumes a single main text block per page, which may not handle complex layouts (e.g., columns or multiple text regions). A more advanced architecture like Faster R-CNN could address this but was not necessary for the dataset.
- The dataset is small (18 images, augmented to 36), which limits generalization. Tesseract annotations, while effective, may miss some text regions in noisy scans. Manual annotations could further improve accuracy.
- The model was trained on early modern Spanish texts, and performance may vary on other languages or printing styles.



