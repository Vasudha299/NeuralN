# CONVOLUTIONAL NEURAL NETWORK 

This is my first Repository. 



# Hi, I'm Vasudha! 👋


## 🚀 About Me
I'm a passionate Computer Science student with a keen interest in  Machine Learning. I love exploring new technologies and applying my knowledge to solve real-world problems.



## 🔗 Links

[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/vasudha29) 




## Installation
Install Dependencies

torch torchvision,pip,
numpy,
pandas,
matplotlib,
scikit-learn,PIL, torchvision.transforms, torch,torch.utils.data and medmnist.


    
## I. Design of a small CNN using Pytorch for PATHMNIST datasets. Trained it on the training set for 50 epochs and reporting the accuracy on the test set after 10, 20, 30, 40, and 50 epochs.

### 1. **Setup the Environment**
   - **Install Dependencies**: Make sure you have Python and PyTorch installed. You’ll also need additional libraries like `torchvision` for handling image data, and `matplotlib` for visualization. Create a virtual environment to manage dependencies and install the required packages.

### 2. **Prepare the Dataset**
   - **Download the PathMNIST Dataset**: PathMNIST is a dataset of handwritten digits. Download and extract the dataset. This dataset can be loaded using PyTorch’s `torchvision` library.
   - **Transform the Data**: Convert images to tensors and normalize them. Normalization helps in speeding up the training process and achieving better performance.

### 3. **Define the CNN Architecture**
   - **Create the Model Class**: Define a CNN model by creating a subclass of `torch.nn.Module`. Specify the layers in the model, including convolutional layers, activation functions (like ReLU), pooling layers, and fully connected layers.
   - **Forward Pass**: Implement the forward method to specify how data passes through the network. This involves applying convolutional layers, activation functions, pooling, and flattening the data before passing it through fully connected layers.

### 4. **Set Up Training Components**
   - **Loss Function**: Choose a loss function to measure the error between the predicted output and the actual labels. For classification tasks, `CrossEntropyLoss` is commonly used.
   - **Optimizer**: Select an optimizer to update the weights of the network during training. Popular choices include Stochastic Gradient Descent (SGD) or Adam.
   - **Data Loaders**: Use `torch.utils.data.DataLoader` to create iterators for the training and validation datasets. This helps in batching and shuffling data during training.

### 5. **Train the Model**
   - **Training Loop**: Iterate over the training dataset for a fixed number of epochs. For each epoch, perform the following steps:
     - **Forward Pass**: Pass the input images through the network to get predictions.
     - **Compute Loss**: Calculate the loss using the predicted outputs and the actual labels.
     - **Backward Pass**: Compute the gradients by backpropagating the loss through the network.
     - **Update Weights**: Update the network weights using the optimizer.
   - **Validation**: After training for each epoch, evaluate the model on a validation set to monitor its performance and adjust hyperparameters if needed.

### 6. **Evaluate the Model**
   - **Test the Model**: After training, evaluate the model on the test set to determine its accuracy. This helps in assessing how well the model generalizes to unseen data.
   - **Metrics**: Calculate and report metrics like accuracy, precision, recall, and F1-score to understand the model’s performance.

### 7. **Save and Load the Model**
   - **Save the Model**: Save the trained model’s weights to a file. This allows you to reuse the model later without retraining.
   - **Load the Model**: Load the saved model weights to make predictions or further fine-tune the model.



## II. A small CNN using Pytorch for MNIST handwritten digit image classification. Trained it on the training set for 50 epochs and reporting the accuracy on the test set after 10, 20, 30, 40, and 50 epochs.

### 1. **Set Up Your Environment**

1. **Install Necessary Software**: Make sure you have Python and PyTorch installed. PyTorch is a popular deep learning library that provides tools to build and train neural networks. You can install it via `pip`, and it’s also recommended to use a virtual environment to manage dependencies.

2. **Create a Virtual Environment**: A virtual environment helps keep your project dependencies isolated from other projects. You can create one and activate it using commands like `python -m venv venv` followed by `source venv/bin/activate` on Unix-like systems or `venv\Scripts\activate` on Windows.

### 2. **Prepare the Dataset**

1. **Download the MNIST Dataset**: MNIST is a dataset of handwritten digits, where each image is 28x28 pixels. PyTorch provides a convenient way to download and load this dataset using its `torchvision` library. 

2. **Transform the Data**: To make the data suitable for training, you need to transform it. Typically, you convert the images to tensors and normalize their pixel values. This helps the neural network to learn more effectively.

3. **Organize Data**: Divide the dataset into training and testing sets. The training set is used to train the model, while the test set is used to evaluate its performance.

### 3. **Define the CNN Architecture**

1. **Design the Model**: Build the architecture of your CNN. This usually involves several types of layers:
   - **Convolutional Layers**: These layers apply filters to the input images to capture patterns like edges.
   - **Pooling Layers**: These reduce the spatial dimensions of the images, making the model less sensitive to slight translations of the input.
   - **Fully Connected Layers**: These layers make predictions based on the features extracted by the convolutional and pooling layers.

2. **Implement the Model**: In PyTorch, you create a class that defines the layers and how they connect to each other. This class will include methods for the forward pass, which dictates how data moves through the network.

### 4. **Set Up Training Components**

1. **Choose a Loss Function**: This is a measure of how well your model's predictions match the actual labels. For classification problems like MNIST, the Cross Entropy Loss is commonly used.

2. **Select an Optimizer**: The optimizer adjusts the model's weights based on the computed loss to minimize errors. Common choices are Stochastic Gradient Descent (SGD) or Adam.

### 5. **Train the Model**

1. **Training Process**: The training involves feeding the training images into the network, calculating the loss, and updating the model’s weights to minimize this loss. This process is repeated over multiple iterations (epochs) until the model performs well.

2. **Monitoring Progress**: During training, you track the loss to see how well the model is learning. Lower loss values generally indicate that the model is learning and improving.

### 6. **Evaluate the Model**

1. **Test the Model**: After training, evaluate the model’s performance using the test set. This helps assess how well the model generalizes to new, unseen data.

2. **Check Accuracy**: Measure how often the model correctly predicts the digits in the test set. This accuracy metric gives you an idea of how well the model is performing overall.


## III. Creating an Image using Numpy and editing it
### 1. **Install Required Libraries**

Before you start, make sure you have NumPy and an image processing library like Pillow (PIL) installed. You can install these using pip if you haven't already.

```bash
pip install numpy pillow
```

### 2. **Create an Image with NumPy**

1. **Initialize a NumPy Array**: An image can be represented as a NumPy array where each pixel is an element in the array. For a grayscale image, each pixel's intensity can be an integer value from 0 (black) to 255 (white). For a color image, each pixel is usually represented by three values (Red, Green, Blue).

   ```python
   import numpy as np

   # Create a blank grayscale image of size 100x100 pixels
   height, width = 100, 100
   image_array = np.zeros((height, width), dtype=np.uint8)
   ```

   ```python
   # Create a blank color image of size 100x100 pixels
   # Each pixel has three values (R, G, B)
   image_array_color = np.zeros((height, width, 3), dtype=np.uint8)
   ```

2. **Modify the Image**: You can set pixel values to create shapes, patterns, or any desired content.

   ```python
   # Set a 50x50 square in the center to white (255)
   image_array[25:75, 25:75] = 255
   ```

   ```python
   # Create a red square in the center of a color image
   image_array_color[25:75, 25:75] = [255, 0, 0]  # RGB for red
   ```

### 3. **Save the Image**

1. **Convert the NumPy Array to an Image**: Use Pillow to convert the NumPy array into an image file format like PNG or JPEG.

   ```python
   from PIL import Image

   # Convert grayscale NumPy array to an image
   image = Image.fromarray(image_array)
   image.save('grayscale_image.png')
   ```

   ```python
   # Convert color NumPy array to an image
   image_color = Image.fromarray(image_array_color)
   image_color.save('color_image.png')
   ```

### 4. **Edit the Image**

1. **Load the Image**: If you need to edit an existing image, load it into a NumPy array.

   ```python
   image = Image.open('color_image.png')
   image_array_color = np.array(image)
   ```

2. **Perform Editing Operations**: Modify the NumPy array to edit the image. For example, you can change colors, add shapes, or apply filters.

   ```python
   # Change the center region to green
   image_array_color[25:75, 25:75] = [0, 255, 0]  # RGB for green
   ```

3. **Save the Edited Image**: After making changes, save the modified image back to disk.

   ```python
   image_edited = Image.fromarray(image_array_color)
   image_edited.save('edited_color_image.png')
   ```

## IV. Reading an Image from the local systerm and making changes and playing with it (editing the colors, dimensions)
### 1. **Read the Image from Your Local System**

1. **Open the Image File**: Use an image processing library like Pillow to open the image file stored on your computer. This library can read various image formats such as PNG, JPEG, and GIF.

2. **Convert to NumPy Array**: Once the image is opened, convert it into a NumPy array. This array will represent the image as a matrix of pixel values, where each pixel's color and intensity are encoded in the array.

### 2. **Make Changes to the Image**

1. **Edit Colors**: Modify the pixel values in the NumPy array to change the colors of specific regions or the entire image. For instance, you can adjust the red, green, and blue values to change the colors or apply filters.

2. **Change Dimensions**: Alter the size of the image by resizing it. This involves adjusting the dimensions of the NumPy array to either enlarge or shrink the image. You can use interpolation methods to handle resizing.

3. **Apply Additional Edits**: You might also want to add shapes, text, or other graphical elements to the image. This can be done by manipulating the pixel values directly or using image processing functions provided by the library.

### 3. **Save the Edited Image**

1. **Convert Back to Image Format**: After making the desired changes to the NumPy array, convert it back to an image format using the same image processing library.

2. **Save the Image**: Save the edited image to your local system with a new file name or overwrite the existing file if you prefer.


