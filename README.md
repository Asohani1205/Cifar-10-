# Cifar-10
We import the relevant libraries for the code, including TensorFlow and scikit-learn, and 
load the CIFAR-10 dataset. Using the keras.datasets.cifar10.load_data() function, which 
delivers the training and testing sets as Numpy arrays, we also load the CIFAR-10 dataset.
o Data pre-processing: We must perform data pre-processing prior to training the model. In 
this stage, we divide the pixel values by 255 to normalise them to the range [0, 1]. Since this 
function performs a binary classification task, we also convert the labels to binary (0 or 1).
o Determining the architecture of the deep learning classifier model: This step involves 
defining the deep learning classifier model. We employ a sequential model that consists of 
two fully connected layers, a flatten layer, and convolutional and max pooling layers. For 
binary classification, the output layer employs the sigmoid activation function.
o The model is then compiled using the compile() function after being defined. For binary 
classification, we employ the Adam optimizer and binary cross-entropy loss function. 
Additionally, we define the metrics for training that we wish to monitor, in this case merely 
accuracy.
o Model training: Using the CIFAR-10 training set, we train the deep learning classifier model at 
this point. With a batch size of 64, we train the model for 10 epochs using the fit() function. 
20% of the training set was also put aside as a validation set.
o Feature extraction from the validation set: Following model training, we use one of the fully 
connected layers to extract features from the validation set. The original model's input is 
used to construct a new model, which produces the dense layer's output. The features from 
the validation and test sets are then extracted using the predict() function.
o Training an SVM with the extracted features: After extracting the features, we train an SVM 
with the linear kernel on the validation set. The SVM object is created using the SVC() 
function, and it is trained using the features taken from the validation set using the fit() 
function.
o Using the features that were derived from the same fully connected layer that we used for 
the validation set, we finally evaluate the SVM model on the test set. To calculate the 
accuracy, precision, recall, and F1 score of the SVM model on the test set, we utilise the 
metrics functions in scikit-learn
