# Tensorflow DNN on MNIST Data

## Background
The objective of this project is to learn the fundamental mechanisms of a machine learning application through a simple use-case of classifying handwritten numbers.

Various techniques of Machine Learning will be attempted on the dataset. This repo tests a **Deep Neural Network** method via TF's `DNNClassifier` API to add (1) hidden layer of nodes and train with the AdamOptimizer algorithm.

## Model Results
See **[Evaluations](Evaluations.md)**

## Get Started
```
python app.py
```

## Steps
- Download and prime the input training data set of 60,000 images and 10,000 labels.
- Define the `DNNClassifier` API object with basic configuations (feature cols, learning rate, hidden layer size, and the optimizer algorithm).
- Let the Classifier API `.fit()` the TF model on the input data.
  - The network will utilize the hidden layer to find better fits than a linear stochastic method could.
- The Classifier API will evaluate the accuracy of the predictions on the test dataset.
- Lastly, the Classifier API will apply the trained model on a sampling of the validation dataset and print results in emoji.

## Technologies
- [Tensorflow](https://www.tensorflow.org/)
- [MNIST](http://yann.lecun.com/exdb/mnist/) Handwritten Numbers Data

