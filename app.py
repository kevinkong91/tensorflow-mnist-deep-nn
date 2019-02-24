import numpy as np
import emoji
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.logging.set_verbosity(tf.logging.ERROR)

# Configs
# Input / network configs
image_size = 28
labels_size = 10
hidden_size = 1024

# Processing configs
batch_size = 100
steps = 1000

# Read in the MNIST dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)


def parse_features_labels_from_dataset(dataset, slice_depth=None):
    features = dataset.images if slice_depth is None else dataset.images[:slice_depth]
    labels_set = dataset.labels if slice_depth is None else dataset.labels[:slice_depth]
    labels = labels_set.astype(np.int32)
    return features, labels


# Define the Estimator
feature_columns = [tf.contrib.layers.real_valued_column(
    "", dimension=image_size*image_size)]
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[hidden_size],
                                            n_classes=labels_size,
                                            optimizer=tf.train.AdamOptimizer())

# Fit the model
features, labels = parse_features_labels_from_dataset(mnist.train)
classifier.fit(x=features, y=labels, batch_size=batch_size, steps=steps)

# Evaluate the model on the test data
features, labels = parse_features_labels_from_dataset(mnist.test)
test_accuracy = classifier.evaluate(x=features, y=labels, steps=1)["accuracy"]

print("\nTest accuracy: %g %%" % (test_accuracy*100))

# Predict the new examples and compare with the underlying values
features, labels = parse_features_labels_from_dataset(mnist.validation, 10)

predictions = list(classifier.predict(x=features))
actual_labels = list(labels)

print("\nPredicted labels: %s" % predictions)
print("Actual values:    %s" % actual_labels)

# For convenient evaluation, print emojis for each result pair
print("Correct?:         ")

emoji_success = ':thumbs_up:'
emoji_failure = ':new_moon:'

for i in range(len(predictions)):
    emoji_alias = emoji_success if predictions[i] == actual_labels[i] else emoji_failure
    emoji_status = emoji.emojize("%d. %s" % (i + 1, emoji_alias))
    print(emoji_status)
