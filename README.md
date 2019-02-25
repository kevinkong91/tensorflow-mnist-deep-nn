# Tensorflow DNN on MNIST Data

## Background
The objective of this project is to learn the fundamental mechanisms of a machine learning application through a simple use-case of classifying handwritten numbers.

Various techniques of Machine Learning will be attempted on the dataset. This repo tests the **Adam** optimizer algorithm, a **Deep Neural Network** method via TF's `DNNClassifier` API.

## Adam Optimization Algorithm
The Adam optimization algorithm, short for *"adaptive moment"* estimation, updates the network weights iteratively based on training data. It is a good replacement for stochastic gradient descent.

Stochastic gradient descent maintains a static alpha (learning rate) for all weight updates. In Adam, by contrast, a learning rate is maintained for *each* network weight(parameter) and individually calibrated as learning progresses. This enables a more fine-tuned and accurate calibration.

### Similar Optimizations
- **Adaptive Gradient Algorithm** (AdaGrad) maintains a per-parameter learning rate that increases the learning rate for sparse parameters and vice-versa, improving performance on problems with sparse data (NLP and Computer Vision)
- **Root Mean Square Propagation** (RMSProp) also maintains a per-parameter learning rates that are adapted based on the average of recent magnitudes of the gradients for the weight (first moment, or how quickly it's changing). RMSProp therefore excels on online and non-stationary problems (noisy).

### Why Adam Optimizer?
1. Adam offers the best properties of both AdaGrad and RMSProp algorithms.
2. Adam is also relatively easy to configure. Default configs are adequate for most problems.

In RMSProp, the parameter learning rates are calibrated based on the moving average of the gradient and the squared gradient (average first moment), while the Adam also accounts for the **average decay rate** (second moments) of those moving averages (uncentered variance).

In short, **Adam achieves good results fast**:

![Comparison of Adam v. Alternatives](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2017/05/Comparison-of-Adam-to-Other-Optimization-Algorithms-Training-a-Multilayer-Perceptron.png "Adam v. Alternatives")

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

## Further Reading
- [Adam Optimization Algorithm](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/)
- [Tensorflow](https://www.tensorflow.org/)
- [MNIST](http://yann.lecun.com/exdb/mnist/) Handwritten Numbers Data

