import os
import re
import tarfile
import random
import numpy as np
import requests
from collections import Counter
from contextlib import closing
from multiprocessing import Pool

def get_downloads_folder():
    """
    Returns the path to the user's Downloads folder.

    Returns:
    str: The path to the user's Downloads folder.
    """
    user_home = os.path.expanduser("~")
    downloads_folder = os.path.join(user_home, "Downloads")
    return downloads_folder

def download_and_extract_data(url, destination):
    """
    Downloads and extracts the IMDB dataset from the specified URL to the specified destination directory.

    Args:
        url (str): The URL of the IMDB dataset.
        destination (str): The directory where the dataset will be downloaded and extracted.

    Returns:
        None
    """
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

class Neuron:
    """
    A class representing a single neuron in a neural network.

    Attributes:
    - weights (list): A list of weights for each input to the neuron.
    - bias (float): A bias term added to the weighted sum of inputs.
    """

    def __init__(self, weights, bias):
        """
        Initializes a new instance of the Neuron class.

        Args:
        - weights (list): A list of weights for each input to the neuron.
        - bias (float): A bias term added to the weighted sum of inputs.
        """
        self.weights = weights
        self.bias = bias

    def activation_function(self, x):
        """
        Applies the ReLU activation function to the given input.

        Args:
        - x (float): The input to the activation function.

        Returns:
        - The result of applying the ReLU function to the input.
        """
        return max(0, x)  # ReLU activation function

    def forward(self, inputs):
        """
        Computes the output of the neuron for the given inputs.

        Args:
        - inputs (list): A list of input values to the neuron.

        Returns:
        - The output of the neuron for the given inputs.
        """
        weighted_sum = sum(x * w for x, w in zip(inputs, self.weights)) + self.bias
        return self.activation_function(weighted_sum)

    def update_weights(self, inputs, delta, learning_rate):
        """
        Updates the weights and bias of the neuron based on the given error signal.

        Args:
        - inputs (list): A list of input values to the neuron.
        - delta (float): The error signal for the neuron.
        - learning_rate (float): The learning rate used to update the weights.

        Returns:
        - None
        """
        new_weights = []
        for x, w in zip(inputs, self.weights):
            new_weights.append(w - learning_rate * delta * x)
        self.weights = new_weights
        self.bias -= learning_rate * delta

class RecurrentNeuralNetwork:
    """
    A class representing a Recurrent Neural Network (RNN) for sentiment analysis of IMDB movie reviews.

    Attributes:
    -----------
    input_size : int
        The size of the input layer.
    hidden_layer_size : int
        The size of the hidden layer.
    output_size : int
        The size of the output layer.
    learning_rate : float
        The learning rate of the RNN.
    weights_ih : numpy.ndarray
        The weights between the input and hidden layers.
    weights_hh : numpy.ndarray
        The weights between the hidden layers.
    weights_ho : numpy.ndarray
        The weights between the hidden and output layers.
    biases_h : numpy.ndarray
        The biases of the hidden layer.
    biases_o : numpy.ndarray
        The biases of the output layer.

    Methods:
    --------
    sigmoid(x: numpy.ndarray) -> numpy.ndarray:
        Returns the sigmoid function applied to the input.
    sigmoid_derivative(x: numpy.ndarray) -> numpy.ndarray:
        Returns the derivative of the sigmoid function applied to the input.
    softmax(x: numpy.ndarray) -> numpy.ndarray:
        Returns the softmax function applied to the input.
    forward(inputs: List[numpy.ndarray]) -> Tuple[numpy.ndarray, List[numpy.ndarray]]:
        Performs forward propagation on the RNN and returns the output and hidden layer outputs.
    backward(inputs: List[numpy.ndarray], hidden_layer_outputs: List[numpy.ndarray], output: numpy.ndarray, target: numpy.ndarray) -> None:
        Performs backward propagation on the RNN and updates the weights and biases.
    """
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
        """
        Returns the sigmoid function applied to the input.

        Parameters:
        -----------
        x : numpy.ndarray
            The input to the sigmoid function.

        Returns:
        --------
        numpy.ndarray:
            The output of the sigmoid function.
        """
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """
        Returns the derivative of the sigmoid function applied to the input.

        Parameters:
        -----------
        x : numpy.ndarray
            The input to the sigmoid function.

        Returns:
        --------
        numpy.ndarray:
            The derivative of the sigmoid function.
        """
        return x * (1 - x)

    def softmax(self, x):
        """
        Returns the softmax function applied to the input.

        Parameters:
        -----------
        x : numpy.ndarray
            The input to the softmax function.

        Returns:
        --------
        numpy.ndarray:
            The output of the softmax function.
        """
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x, axis=0)

    def forward(self, inputs):
        """
        Performs forward propagation on the RNN and returns the output and hidden layer outputs.

        Parameters:
        -----------
        inputs : List[numpy.ndarray]
            The inputs to the RNN.

        Returns:
        --------
        Tuple[numpy.ndarray, List[numpy.ndarray]]:
            The output of the RNN and the hidden layer outputs.
        """
        hidden_layer_outputs = []
        hidden = np.zeros((self.hidden_layer_size, 1))
        for i in range(len(inputs)):
            input_t = inputs[i].reshape(-1, 1)
            hidden = np.tanh(np.dot(self.weights_ih, input_t) + np.dot(self.weights_hh, hidden) + self.biases_h)
            hidden_layer_outputs.append(hidden)
        output = self.softmax(np.dot(self.weights_ho, hidden) + self.biases_o)
        return output, hidden_layer_outputs

    def backward(self, inputs, hidden_layer_outputs, output, target):
        """
        Performs backward propagation on the RNN and updates the weights and biases.

        Parameters:
        -----------
        inputs : List[numpy.ndarray]
            The inputs to the RNN.
        hidden_layer_outputs : List[numpy.ndarray]
            The outputs of the hidden layer.
        output : numpy.ndarray
            The output of the RNN.
        target : numpy.ndarray
            The target output of the RNN.

        Returns:
        --------
        None
        """
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

from typing import List, Tuple
from collections import Counter
import re

def preprocess_data(data: List[Tuple[str, int]], max_features: int) -> List[Tuple[List[int], int]]:
    """
    Preprocesses the given data for sentiment analysis using a bag-of-words approach with bigrams.

    Args:
        data: A list of tuples, where each tuple contains a string representing a review and an integer representing its sentiment label (0 for negative, 1 for positive).
        max_features: An integer representing the maximum number of features to use in the bag-of-words representation.

    Returns:
        A list of tuples, where each tuple contains a list of integers representing the feature vector for a review and its corresponding sentiment label.
    """
    def clean_text(text: str) -> List[str]:
        """
        Cleans the given text by removing HTML tags, non-alphabetic characters, and converting to lowercase.

        Args:
            text: A string representing the text to clean.

        Returns:
            A list of strings representing the cleaned words.
        """
        text = re.sub('<.*?>', ' ', text)
        text = re.sub('[^a-zA-Z]', ' ', text)
        text = text.lower()
        words = text.split()
        return words

    def get_bigrams(words: List[str]) -> List[str]:
        """
        Returns a list of bigrams for the given list of words.

        Args:
            words: A list of strings representing the words to generate bigrams from.

        Returns:
            A list of strings representing the bigrams.
        """
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

def load_data(data_dir, subset_size=None):
    """
    Load the IMDB dataset from the given directory.

    Args:
        data_dir (str): The directory containing the IMDB dataset.
        subset_size (int, optional): The maximum number of reviews to load from each subset.

    Returns:
        tuple: A tuple containing two lists of tuples. The first list contains the training data, and the second list
        contains the test data. Each tuple in the lists contains a review text and a label (0 for negative, 1 for positive).
    """
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

def train_network(network, training_data, learning_rate, epochs, batch_size):
    """
    Trains a neural network using the given training data.

    Args:
        network (object): The neural network to train.
        training_data (list): A list of tuples containing input and target data.
        learning_rate (float): The learning rate to use during training.
        epochs (int): The number of epochs to train for.
        batch_size (int): The batch size to use during training.

    Returns:
        None
    """
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

def process_batch(args):
    """
    Trains the network using backpropagation on a batch of inputs and targets.

    Args:
        args (tuple): A tuple containing the network, inputs, target, and learning rate.

    Returns:
        None
    """
    network, inputs, target, learning_rate = args
    network.backpropagation(inputs, target, learning_rate)

def test_network(network, test_data):
    """
    Test the given network on the provided test data and return the accuracy.

    Args:
        network (RNN): The RNN network to be tested.
        test_data (list): A list of tuples containing the inputs and labels for the test data.

    Returns:
        float: The accuracy of the network on the test data.
    """
    correct = 0
    for inputs, label in test_data:
        inputs = [np.array(inputs)]  # Wrap the input in a list
        outputs, _ = network.forward(inputs)
        prediction = np.argmax(outputs)
        if prediction == label:
            correct += 1
    return correct / len(test_data)

def grid_search(data_dir, max_features, hyperparameters):
    """
    Perform a grid search over the given hyperparameters to find the best set of hyperparameters for a recurrent neural network
    trained on sentiment analysis data.

    Args:
    - data_dir (str): The directory containing the sentiment analysis data.
    - max_features (int): The maximum number of features to use when preprocessing the data.
    - hyperparameters (list): A list of tuples, where each tuple contains the hyperparameters to test. The hyperparameters
    are (hidden_layers, learning_rate, epochs, batch_size), where hidden_layers is a list of integers representing the
    number of neurons in each hidden layer, learning_rate is a float representing the learning rate of the network,
    epochs is an integer representing the number of epochs to train the network for, and batch_size is an integer
    representing the batch size to use during training.

    Returns:
    - best_params (tuple): The set of hyperparameters that achieved the highest accuracy on the test data.
    - best_accuracy (float): The accuracy achieved by the network using the best set of hyperparameters.
    """
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
