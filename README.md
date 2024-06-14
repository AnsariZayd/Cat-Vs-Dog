# Kaggle Cat and Dog Image Classification

## Overview
This project demonstrates a basic image classification pipeline using convolutional neural networks (CNNs) to distinguish between images of cats and dogs. The dataset used is from Kaggle and includes training and testing sets. The project includes steps for data downloading, preprocessing, model building, training, and prediction.

## Setup and Execution

### Prerequisites
Ensure you have the following libraries installed:
- Python 3.x
- NumPy
- pandas
- OpenCV
- Matplotlib
- TensorFlow
- urllib
- zipfile
- tarfile

### Data Download and Preparation
1. **Data Import**: The initial part of the script downloads the dataset from a specified URL and extracts it into the `/kaggle/input` directory.
2. **Directory Setup**: The script sets up the required directory structure and symbolic links to emulate the Kaggle notebook environment.

### Code Execution
1. **Run the Import Cell**: Ensure to run the initial cell to download and extract the dataset to the correct location.
2. **Data Preprocessing**: The images are preprocessed by resizing and normalizing them. This is done using the `image_preprocessing` function.
3. **Load Images**: The `load_images` function loads the images from specified directories and labels them appropriately.

### Model Training
1. **Model Architecture**: The CNN model is defined with three convolutional layers, followed by max pooling layers, a flattening layer, and two dense layers.
2. **Compilation**: The model is compiled using the Adam optimizer and binary cross-entropy loss.
3. **Training**: The model is trained for 10 epochs with the training and testing datasets.

### Prediction
The script includes a sample prediction step where an image is processed and predicted to be either a cat or a dog using the trained model.

## Usage

### Step-by-Step Instructions
1. **Download Data**: Run the initial data download and extraction cell.
2. **Preprocess and Load Data**: Ensure the images are preprocessed and loaded correctly by running the subsequent cells.
3. **Build and Train Model**: Execute the cells that build and train the model.
4. **Predict**: Use the provided prediction code to classify a new image.

### Example Commands
```python
# Run the data import and extraction
# Make sure the data sources are correctly imported into /kaggle/input

# Display an example image
img = cv2.imread("/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.1.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.show()

# Preprocess an example image
img_path = "/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.1.jpg"
img = image_preprocessing(img_path)
plt.imshow(img)
plt.show()

# Train the model
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Predict a new image
Image = "/content/dog4.jpeg"
Image = image_preprocessing(Image)
prediction = model.predict(np.expand_dims(Image, axis=0))
predicted_label = "dog" if prediction[0][0] > 0.5 else "cat"
print(f"The model predicts this image is a {predicted_label}.")
