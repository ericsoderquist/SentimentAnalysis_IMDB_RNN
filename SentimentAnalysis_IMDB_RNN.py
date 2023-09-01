import os
import re
import tarfile
import random
import numpy as np
import requests
from collections import Counter
from contextlib import closing
from multiprocessing import Pool

"""
This function gets the downloads folder from the system. 
It expands the user's home directory, then appends "Downloads" to it to get the downloads path.
"""
def get_downloads_folder():
    user_home = os.path.expanduser("~")
    downloads_folder = os.path.join(user_home, "Downloads")
    return downloads_folder

"""
This function is used to download and extract the data from the given URL into the given destination. 
The URL should be valid and the destination should be an existing directory. 
The function will check if the dataset is already present in the destination and if so, 
it will not download it again. If not, then it will download the dataset from the URL and then extract it into the destination.
"""
def download_and_extract_data(url, destination):
    # Check if the dataset is already downloaded and extracted
    if os.path.exists(os.path.join(destination, "aclImdb")):
        print("Dataset already exists in the specified directory.")
        return
    file_name = url.split("/")[-1]
    file_path = os.path.join(destination, file_name)
    print("Downloading dataset from {}...".format(url))
    with closing(requests.get(url, stream=True, verify=False)) as response:
        total_length = int(response.headers.get("content-length"))
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
    print("Dataset downloaded successfully.")
    print("Extracting dataset...")
    with tarfile.open(file_path, "r:gz") as tar:
        for member in tar.getmembers():
            if "aclImdb" in member.name:
                member.name = os.path.relpath(member.name, "aclImdb")  # Remove the 'aclImdb' prefix from the path
                tar.extract(member, path=destination)
    print("Dataset extracted successfully.")

"""
This code defines a class 'Neuron' which models a single neuron in a neural network. 
It is initialized with weights and a bias and has three methods: activation_function(), forward(), 
and update_weights(). The activation_function() determines how a neuron responds to its inputs, while forward() 
calculates the weighted sum of its inputs and passes them through the activation_function(). Finally, 
update_weights() adjusts the neuron's weights and bias according to the delta and learning rate.
"""
class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def activation_function(self, x):
        return max(0, x)  # ReLU activation function

    def forward(self, inputs):
        weighted_sum = sum(x * w for x, w in zip(inputs, self.weights)) + self.bias
        return self.activation_function(weighted_sum)

    def update_weights(self, inputs, delta, learning_rate):
        new_weights = []
        for x, w in zip(inputs, self.weights):
            new_weights.append(w - learning_rate * delta * x)
        self.weights = new_weights
        self.bias -= learning_rate * delta

"""This is a class that implements a recurrent neural network (RNN) algorithm. 
The class takes in parameters such as the input size, hidden layer size, output size, and learning rate. 
The class then initializes weights and biases for the hidden layers and the output layer. 
The class has methods such as sigmoid, softmax, forward, and backward. 
The sigmoid and softmax methods are used to calculate the activation functions of the RNN. 
The forward method receives input and calculates the output values through the weights and biases of the neural network. 
The backward method calculates the errors between the output and target values and adjusts the weights and biases accordingly."""
class RecurrentNeuralNetwork:
    def __init__(self, input_size, hidden_layer_size, output_size, learning_rate):
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.weights_ih = np.random.randn(hidden_layer_size, input_size) * 0.01
        self.weights_hh = np.random.randn(hidden_layer_size, hidden_layer_size) * 0.01
        self.weights_ho = np.random.randn(output_size, hidden_layer_size) * 0.01

        self.biases_h = np.zeros((hidden_layer_size, 1))
        self.biases_o = np.zeros((output_size, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x, axis=0)

    def forward(self, inputs):
        hidden_layer_outputs = []
        hidden = np.zeros((self.hidden_layer_size, 1))
        for i in range(len(inputs)):
            input_t = inputs[i].reshape(-1, 1)
            hidden = np.tanh(np.dot(self.weights_ih, input_t) + np.dot(self.weights_hh, hidden) + self.biases_h)
            hidden_layer_outputs.append(hidden)
        output = self.softmax(np.dot(self.weights_ho, hidden) + self.biases_o)
        return output, hidden_layer_outputs

    def backward(self, inputs, hidden_layer_outputs, output, target):
        output_error = output - target
        delta_weights_ho = np.dot(output_error, hidden_layer_outputs[-1].T)
        delta_biases_o = output_error

        delta_hidden = np.dot(self.weights_ho.T, output_error)
        for t in reversed(range(len(inputs))):
            input_t = inputs[t].reshape(-1, 1)
            hidden_derivative = (1 - hidden_layer_outputs[t] ** 2)
            delta_hidden_t = delta_hidden * hidden_derivative

            if t > 0:
                delta_weights_hh = np.dot(delta_hidden_t, hidden_layer_outputs[t - 1].T)
            else:
                delta_weights_hh = np.zeros_like(self.weights_hh)

            delta_weights_ih = np.dot(delta_hidden_t, input_t.T)
            delta_biases_h = delta_hidden_t

            delta_hidden = np.dot(self.weights_hh, delta_hidden_t)

            self.weights_ih -= self.learning_rate * delta_weights_ih
            self.weights_hh -= self.learning_rate * delta_weights_hh
            self.biases_h -= self.learning_rate * delta_biases_h

        self.weights_ho -= self.learning_rate * delta_weights_ho
        self.biases_o -= self.learning_rate * delta_biases_o

"""The preprocess_data function takes two parameters, data and max_features which indicates the maximum number of features for the embedding.
First, the function cleans the text by removing any HTML tags and punctuation and by making all words lowercase. 
It then counts the words and bigrams in the text to determine the most frequent words or bigrams. 
Next, the most common words and bigrams are collected and stored in a word_to_index dictionary. 
Finally, a feature vector is created by assigning a value of 1 to any of the most frequent words/bigrams found in the text. 
The feature vector is then stored along with the corresponding label in a list. 
This list is what is returned by the function."""
def preprocess_data(data, max_features):
    def clean_text(text):
        text = re.sub('<.*?>', ' ', text)
        text = re.sub('[^a-zA-Z]', ' ', text)
        text = text.lower()
        words = text.split()
        return words

    def get_bigrams(words):
        return [f"{w1}_{w2}" for w1, w2 in zip(words[:-1], words[1:])]

    word_counts = Counter()
    for text, _ in data:
        words = clean_text(text)
        word_counts.update(words)
        bigrams = get_bigrams(words)
        word_counts.update(bigrams)

    most_common_words = [word for word, _ in word_counts.most_common(max_features)]
    word_to_index = {word: i for i, word in enumerate(most_common_words)}

    processed_data = []
    for text, label in data:
        words = clean_text(text)
        bigrams = get_bigrams(words)
        tokens = words + bigrams
        indices = [word_to_index.get(token, -1) for token in tokens]
        indices = [i for i in indices if i >= 0]
        feature_vector = [0] * max_features
        for index in indices:
            feature_vector[index] = 1
        processed_data.append((feature_vector, label))
    return processed_data

"""The function "load_data" loads movie reviews from a given data directory. 
It takes two parameters - the data directory and a subset size (optional). 
It then extracts positive and negative reviews from the train and test subdirectories and 
returns a tuple containing a list of tuples (text and label) for the train and test data."""
def load_data(data_dir, subset_size=None):
    def load_reviews_from_dir(directory, label):
        reviews = []
        for filename in os.listdir(directory):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                text = file.read()
                reviews.append((text, label))
            if subset_size is not None and len(reviews) >= subset_size:
                break # Stop loading reviews if we've reached the subset size
            return reviews

    train_pos_dir = os.path.join(data_dir, 'train', 'pos')
    train_neg_dir = os.path.join(data_dir, 'train', 'neg')
    test_pos_dir = os.path.join(data_dir, 'test', 'pos')
    test_neg_dir = os.path.join(data_dir, 'test', 'neg')

    train_pos = load_reviews_from_dir(train_pos_dir, 1)
    train_neg = load_reviews_from_dir(train_neg_dir, 0)
    test_pos = load_reviews_from_dir(test_pos_dir, 1)
    test_neg = load_reviews_from_dir(test_neg_dir, 0)

    train_data = train_pos + train_neg
    test_data = test_pos + test_neg
    return train_data, test_data

"""This function is used to train the specified network using the given training data. 
The parameters to this function are the network, training data, learning rate, total epochs and batch size. 
The network is trained by shuffling the data and iterating over the batches of data. 
For each batch, the input data and target data is split and then processed in a separate process using the pool.map() function. 
The learning rate is applied to the network after each batch is processed. 
The function prints out the status of the epochs that have been completed."""
def train_network(network, training_data, learning_rate, epochs, batch_size):
    num_batches = len(training_data) // batch_size
    pool = Pool()
    for epoch in range(epochs):
        random.shuffle(training_data)
        for batch_idx in range(num_batches):
            batch = training_data[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            inputs = []
            targets = []
            for input_data, target_data in batch:
                inputs.append(input_data)
                targets.append([1 if i == target_data else 0 for i in range(2)])
            pool.map(process_batch, [(network, input_data, target_data, learning_rate) for input_data, target_data in zip(inputs, targets)])

        print(f"Epoch {epoch + 1} completed")

"""
This function takes in four parameters, a network, inputs, target and a learning rate. 
It then applies the backpropagation algorithm using the given parameters to update the weights of the network. 
"""
def process_batch(args):
    network, inputs, target, learning_rate = args
    network.backpropagation(inputs, target, learning_rate)

"""The test_network function takes in a network and a test data set and returns the accuracy of the network on the given data set. 
The function iterates over the test data and passes the inputs in the network. 
It uses the network's forward function to get the output and compares the prediction with the label. 
If the prediction and label match, the function increments the correct counter. 
Finally, it calculates the accuracy by dividing the correct count by the total number of test data."""
def test_network(network, test_data):
    correct = 0
    for inputs, label in test_data:
        inputs = [np.array(inputs)]  # Wrap the input in a list
        outputs, _ = network.forward(inputs)
        prediction = np.argmax(outputs)
        if prediction == label:
            correct += 1
    return correct / len(test_data)

"""This function applies a grid search to find the best hyperparameters for a recurrent neural network classifier. 
It takes a data directory containing the train and test data, a maximum number of features, and a list of hyperparameter tuples as inputs. 
It then preprocesses the data, creates a neural network model with the specified parameters and trains it. 
Finally, it tests the model on the test data and returns the best parameters and the best accuracy score."""
def grid_search(data_dir, max_features, hyperparameters):
    train_data, test_data = load_data(data_dir)
    train_data = preprocess_data(train_data, max_features)
    test_data = preprocess_data(test_data, max_features)
    best_accuracy = 0
    best_params = None

    for params in hyperparameters:
        print(f"Testing hyperparameters: {params}")
        hidden_layers, learning_rate, epochs, batch_size = params
        network = RecurrentNeuralNetwork(input_size=max_features, hidden_layer_size=hidden_layers[0], output_size=2, learning_rate=learning_rate)
        train_network(network, train_data, learning_rate, epochs, batch_size)
        accuracy = test_network(network, test_data)
        print(f"Accuracy: {accuracy}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params

    return best_params, best_accuracy

"""This code performs grid search to find the best hyperparameter combination for a given dataset. 
The dataset used is the ACLIMDB dataset which is a dataset of movie reviews. 
The code downloads and extracts the dataset, preprocesses it, and searches over a range of hyperparameters to find the best combination. 
The results of the search are printed out along with the best accuracy."""
if __name__ == "__main__":
    data_dir = os.path.join(os.path.expanduser("~"), "Downloads", "aclImdb")
    dataset_url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    download_and_extract_data(dataset_url, data_dir)
    max_features = 5000 
    # Load a smaller subset of the data
    train_data, test_data = load_data(data_dir, subset_size=500)

    train_data = preprocess_data(train_data, max_features)
    test_data = preprocess_data(test_data, max_features)

    """The below hyperparameters are a collection of varying configurations for a deep neural network. 
    Each hyperparameter contains four components: the size of the layers in the model, the learning rate, the number of epochs, and the batch size. 
    The different combinations of layer sizes range from a single layer of 32 nodes to multiple layers of 64 and 128 nodes. The learning rate is set to either 0.1 or 0.01. 
    The number of epochs is fixed to 10 and the batch size is 32."""
    hyperparameters = [    ([32], 0.1, 10, 32),
        ([64], 0.1, 10, 32),
        ([128], 0.1, 10, 32),
        ([32, 16], 0.1, 10, 32),
        ([64, 32], 0.1, 10, 32),
        ([128, 64], 0.1, 10, 32),
        ([32, 16], 0.01, 10, 32),
        ([64, 32], 0.01, 10, 32),
        ([128, 64], 0.01, 10, 32)
    ]

    best_params, best_accuracy = grid_search(data_dir, max_features, hyperparameters)

    print("Best hyperparameters:", best_params)
    print("Best accuracy:", best_accuracy)
