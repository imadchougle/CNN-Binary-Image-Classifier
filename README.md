# CNN-Binary-Image-Classifier

This code demonstrates a Convolutional Neural Network (CNN) image classifier to differentiate between two iconic personalities, Narendra Modi and Sachin Tendulkar, based on their images. 

The goal is to create a binary classifier that can identify whether an input image contains Narendra Modi or Sachin Tendulkar.


**Requirements:**

TensorFlow

OpenCV

Matplotlib

Numpy


**Instructions:**

Organize the dataset with images of Narendra Modi and Sachin Tendulkar in separate subdirectories under the 'data' directory.


**Generating Dataset:**

  The dataset is created using google images of the two famous celebrites over 312 images were downloaded of each and placed in their respective directory for data cleaning and preprocessing

  a. Cleaning consists of removing the vector files and images below 10kb.
  
  b. The code removes any non-image files from the dataset to ensure the dataset contains only valid image files with extensions jpeg, jpg, bmp, png.


  **Data Preprocessing:**

  a. The code uses TensorFlow's image_dataset_from_directory to create a dataset object from the 'data' directory. Images are categorized to values between 0 and 1 by dividing the pixel values by 255. 
  
  b. The dataset is splitted into train and test ratio, the train part is 80% and the testing part is 20% of whole dataset.

  **Model Architecture:**

Convolutional layers are the core building blocks of a CNN. These layers are responsible for extracting features from the input images. In this model, there are three convolutional layers:
The first convolutional layer has 16 filters, each with a size of (3, 3). The activation function used is ReLU, which introduces non-linearity to the model. The input shape for this layer is (256, 256, 3), representing the height, width, and color channels of the input image.The second convolutional layer also has 32 filters with a size of (3, 3) and a ReLU activation function.
The third convolutional layer has 16 filters with a size of (3, 3) and a ReLU activation function.
After each convolutional layer, a MaxPooling layer is applied to downsample the feature maps.
After the last MaxPooling layer, the feature maps are flattened into a one-dimensional vector, preparing the data for the fully connected (dense) layers.
The dense layer helps in learning complex patterns and representations from the extracted features.
The final layer of the model is a single output neuron with a sigmoid activation function. Since this is a binary classification problem, a sigmoid activation function is used. The output of the sigmoid function is a probability value between 0 and 1, representing the probability that the input image belongs to Narendra Modi.

1. The model takes an input image of size (256, 256, 3) and applies three convolutional layers with MaxPooling to extract features. 

2. After flattening the features, the model uses a dense layer with 256 neurons to learn complex patterns. 

3. The output layer with a sigmoid activation function predicts the probability of the input image being Narendra Modi. 

4. The model is trained to optimize its parameters using the binary cross-entropy loss and the Adam optimizer to make accurate predictions.

5. The model is compiled with the 'adam' optimizer and binary cross-entropy loss, suitable for binary classification tasks. 

6. The model is trained on the training data for 10 epochs.

**New Image Prediction:**

A test image of Narendra Modi or Sachin Tendulkar is resized to the model's input shape (256x256x3) and passed through the model for prediction. The output is a probability value between 0 and 1. A probability close to 0 indicates a prediction for Sachin Tendulkar, while a probability close to 1 indicates a prediction for Narendra Modi. Threshold value is added. if greater value is greater than 0.5 than Narendra Modi or else Sachin Tendulkar
