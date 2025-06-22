# Product Label Print Quality Classifier

This project contains a Machine Learning model developed using TensorFlow and Keras to classify the print quality of product labels as either "GOOD_PRINT_QUALITY" or "BAD_PRINT_QUALITY". This is a component of a larger "Smart Product Labeling and Traceability System" developed for an Intel Unnati industrial training project.

## Overview

The primary goal of this ML model is to automate the visual inspection of product labels, identifying those with defects such as smudges, faint print, or minor tears.

## Dataset

The model was trained on a custom dataset of label images, manually categorized into 'good_labels' and 'bad_labels'. The dataset was structured as follows:
dataset/
train/
good_labels/ # .jpg or .png images of good quality labels
bad_labels/ # .jpg or .png images of bad quality labels
validation/
good_labels/
bad_labels/
test/
good_labels/
bad_labels/

Initially, the model was trained with approximately:
*   Training set: 15 images per class (30 total)
*   Validation set: 5 images per class (10 total)
*   Test set: 5 images per class (10 total)

**Note:** The image dataset itself is not included in this repository due to its size. Please prepare your own dataset following the structure above.

## Methodology

The model employs a transfer learning approach:
1.  **Data Preprocessing & Augmentation:** Images are resized to 224x224. Data augmentation techniques applied to the training set include random flips (horizontal & vertical), rotations, zoom, contrast adjustments, and brightness adjustments.
2.  **Base Model:** A pre-trained MobileNetV2 model (trained on ImageNet) is used as the convolutional base, with its layers initially frozen.
3.  **Classification Head:** A custom classification head is added on top of MobileNetV2, consisting of:
    *   GlobalAveragePooling2D layer
    *   Dropout layer (rate 0.2)
    *   Dense output layer with a single unit and sigmoid activation for binary classification.
4.  **Training:** The model was compiled with the Adam optimizer (learning rate 0.001) and BinaryCrossentropy loss.

## Current Results (Baseline - Frozen Base Model)

The initial training run (with MobileNetV2 base frozen) yielded the following performance:
*   **Peak Validation Accuracy:** ~90%
*   **Test Accuracy (after 20 epochs):** 80.00%
*   **Test Loss:** ~0.6046

*(You can add a screenshot of your training plots here if you like. To do that, save the plot image, add it to your repo, and use Markdown image syntax: `![Training Plots](path/to/your/plot_image.png)`)*

## Setup & How to Run

1.  This project is primarily developed using **Google Colab**.
2.  Ensure a **GPU runtime** is selected in Colab for efficient training.
3.  **Clone this repository** or download the `label_quality_classification.ipynb` notebook.
4.  **Mount your Google Drive** in Colab.
5.  **Update the `base_drive_path` variable** in the notebook to point to the location where your `dataset` folder (structured as described above) resides on your Google Drive.
6.  The necessary Python libraries are listed in `requirements.txt` (primarily TensorFlow and Matplotlib, which are standard in Colab).

## Saved Model

The initially trained model (frozen base, 20 epochs) is saved in the `saved_models/` directory in the Keras v3 native format:
*   `saved_models/label_quality_classifier_initial_frozen.keras`

This model achieved 80% accuracy on the test set.

## Next Steps

*   Evaluate results from re-training with `ModelCheckpoint` to capture the best performing weights.
*   Explore fine-tuning the MobileNetV2 base model by unfreezing some layers and training with a lower learning rate.
*   Potentially expand the dataset for improved robustness and accuracy.
*   Integrate the trained model into the main Python-based "Smart Product Labeling and Traceability System".

## Author

*   Shahazad_Abdulla - ([Link to your GitHub Profile if different from repo owner])
