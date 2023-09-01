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
------------------------------------------------------------------------
Author: Eric Soderquist
Date: 2023-09-01
Purpose: 
    This script is designed to perform sentiment analysis on the IMDB dataset. 
    It incorporates basic Python libraries for data manipulation and extraction, 
    including NumPy for numerical operations and 'requests' for data retrieval. 
    The script also employs Python's multiprocessing library to parallelize certain tasks 
    for computational efficiency.

Key Contributions:
    - Extraction and preprocessing of the IMDB dataset.
    - Implementation of a sentiment analysis model.
    - Utilization of multiprocessing for optimized data processing.
    - Comprehensive evaluation metrics including accuracy, precision, recall, F1-score, 
      and AUC-ROC for model evaluation.

Dependencies:
    - NumPy for numerical operations.
    - 'requests' for fetching data.
    - 'tarfile' and 'os' for file and directory manipulation.
    - 're' for regular expressions.
    - 'random' for random number generation.
    - 'multiprocessing' for parallel processing.
------------------------------------------------------------------------
"""

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
    Downloads and extracts a dataset from a given URL to a specified destination directory.

    Args:
        url (str): The URL of the dataset to download.
        destination (str): The directory to save the downloaded and extracted dataset.

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
    - weights (list): A list of weights for each input.
    - bias (float): A bias value added to the weighted sum of inputs.

    Methods:
    - activation_function(x): Returns the output of the ReLU activation function for a given input x.
    - forward(inputs): Computes the output of the neuron for a given set of inputs.
    - update_weights(inputs, delta, learning_rate): Updates the weights and bias of the neuron based on the given inputs, error delta, and learning rate.
    """
    def __init__(self, weights, bias):
        """
        Initializes a new instance of the Neuron class.

        Parameters:
        - weights (list): A list of weights for each input.
        - bias (float): A bias value added to the weighted sum of inputs.
        """
        self.weights = weights
        self.bias = bias

    def activation_function(self, x):
        """
        Returns the output of the ReLU activation function for a given input x.

        Parameters:
        - x (float): The input value.

        Returns:
        - The output of the ReLU activation function for the given input x.
        """
        return max(0, x)  # ReLU activation function

    def forward(self, inputs):
        """
        Computes the output of the neuron for a given set of inputs.

        Parameters:
        - inputs (list): A list of input values.

        Returns:
        - The output of the neuron for the given inputs.
        """
        weighted_sum = sum(x * w for x, w in zip(inputs, self.weights)) + self.bias
        return self.activation_function(weighted_sum)

    def update_weights(self, inputs, delta, learning_rate):
        """
        Updates the weights and bias of the neuron based on the given inputs, error delta, and learning rate.

        Parameters:
        - inputs (list): A list of input values.
        - delta (float): The error delta for the neuron.
        - learning_rate (float): The learning rate for the neuron.

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
    A class representing a recurrent neural network.

    Attributes:
    - input_size (int): The number of input nodes.
    - hidden_layer_size (int): The number of nodes in the hidden layer.
    - output_size (int): The number of output nodes.
    - learning_rate (float): The learning rate used in backpropagation.

    Methods:
    - sigmoid(x): Returns the sigmoid function applied to x.
    - sigmoid_derivative(x): Returns the derivative of the sigmoid function applied to x.
    - softmax(x): Returns the softmax function applied to x.
    - forward(inputs): Performs a forward pass through the network and returns the output and hidden layer outputs.
    - backpropagation(inputs, hidden_layer_outputs, output, target): Performs a backpropagation pass through the network and updates the weights and biases.
    """
    def __init__(self, input_size, hidden_layer_size, output_size, learning_rate):
        """
        Initializes the weights, biases, and other attributes of the network.

        Args:
        - input_size (int): The number of input nodes.
        - hidden_layer_size (int): The number of nodes in the hidden layer.
        - output_size (int): The number of output nodes.
        - learning_rate (float): The learning rate used in backpropagation.
        """
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
        Returns the sigmoid function applied to x.

        Args:
        - x (numpy.ndarray): The input to the sigmoid function.

        Returns:
        - numpy.ndarray: The output of the sigmoid function.
        """
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """
        Returns the derivative of the sigmoid function applied to x.

        Args:
        - x (numpy.ndarray): The input to the sigmoid function.

        Returns:
        - numpy.ndarray: The derivative of the sigmoid function.
        """
        return x * (1 - x)

    def softmax(self, x):
        """
        Returns the softmax function applied to x.

        Args:
        - x (numpy.ndarray): The input to the softmax function.

        Returns:
        - numpy.ndarray: The output of the softmax function.
        """
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x, axis=0)

    def forward(self, inputs):
        """
        Performs a forward pass through the network and returns the output and hidden layer outputs.

        Args:
        - inputs (list): A list of numpy arrays representing the input sequence.

        Returns:
        - tuple: A tuple containing the output and a list of numpy arrays representing the hidden layer outputs.
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
        Performs a backpropagation pass through the network and updates the weights and biases.

        Args:
        - inputs (list): A list of numpy arrays representing the input sequence.
        - hidden_layer_outputs (list): A list of numpy arrays representing the hidden layer outputs.
        - output (numpy.ndarray): The output of the network.
        - target (numpy.ndarray): The target output.

        Returns:
        - None
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

def preprocess_data(data, max_features):
    """
    Preprocesses the given data by cleaning the text, extracting bigrams, and converting the text into feature vectors.

    Args:
        data (List[Tuple[str, int]]): A list of tuples where each tuple contains a string of text and its corresponding label.
        max_features (int): The maximum number of features to use in the feature vectors.

    Returns:
        List[Tuple[List[int], int]]: A list of tuples where each tuple contains a feature vector and its corresponding label.
    """
    def clean_text(text):
        """
        Cleans the given text by removing HTML tags, non-alphabetic characters, and converting to lowercase.

        Args:
            text (str): The text to clean.

        Returns:
            List[str]: A list of cleaned words.
        """
        text = re.sub('<.*?>', ' ', text)
        text = re.sub('[^a-zA-Z]', ' ', text)
        text = text.lower()
        words = text.split()
        return words

    def get_bigrams(words):
        """
        Extracts bigrams from the given list of words.

        Args:
            words (List[str]): The list of words to extract bigrams from.

        Returns:
            List[str]: A list of bigrams.
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
    Load movie review data from a directory.

    Args:
        data_dir (str): The directory containing the movie review data.
        subset_size (int, optional): The maximum number of reviews to load. If None, all reviews are loaded.

    Returns:
        tuple: A tuple containing two lists of tuples. The first list contains training data, and the second list contains test data. Each tuple in the lists contains a movie review text and a label (0 for negative, 1 for positive).
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
        network (NeuralNetwork): The neural network to train.
        training_data (list): A list of tuples containing the input and target data for each training example.
        learning_rate (float): The learning rate to use during training.
        epochs (int): The number of epochs to train for.
        batch_size (int): The size of each training batch.

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
    Trains the neural network using backpropagation algorithm on a batch of inputs and targets.

    Args:
        args (tuple): A tuple containing the neural network, inputs, targets, and learning rate.

    Returns:
        None
    """
    network, inputs, target, learning_rate = args
    network.backpropagation(inputs, target, learning_rate)


from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, roc_auc_score

def test_network(network, test_data):
    '''
    Tests the given neural network on the provided test data and returns various performance metrics.
    Performance metrics include: Accuracy, Precision, Recall, F1-Score, and AUC (Area Under the Curve).

    Args:
        network (NeuralNetwork): The neural network to be tested.
        test_data (list): A list of tuples containing the inputs and labels for each test case.

    Returns:
        dict: A dictionary containing various performance metrics.
    '''
    correct = 0
    true_labels = []
    predicted_labels = []
    predicted_scores = []

    for inputs, label in test_data:
        inputs = [np.array(inputs)]  # Wrap the input in a list
        outputs, _ = network.forward(inputs)
        prediction = np.argmax(outputs)
        predicted_scores.append(outputs[1])  # Assuming the second output corresponds to the 'positive' class
        true_labels.append(label)
        predicted_labels.append(prediction)
        if prediction == label:
            correct += 1

    # Calculate additional performance metrics
    accuracy = correct / len(test_data)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    fpr, tpr, _ = roc_curve(true_labels, predicted_scores)
    auc = roc_auc_score(true_labels, predicted_scores)

    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'FPR': fpr,
        'TPR': tpr,
        'AUC': auc
    }

    return metrics



def grid_search(data_dir, max_features, hyperparameters):
    """
    Perform a grid search to find the best hyperparameters for a recurrent neural network.

    Args:
        data_dir (str): The directory containing the training and testing data.
        max_features (int): The maximum number of features to use for the data preprocessing step.
        hyperparameters (list): A list of hyperparameters to test. Each hyperparameter is a tuple containing the
                                number of hidden layers, learning rate, number of epochs, and batch size.

    Returns:
        tuple: A tuple containing the best hyperparameters and the corresponding accuracy.
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
        metrics = test_network(network, test_data)
        accuracy = metrics['Accuracy']
        print(f"Accuracy: {accuracy}")
    print(f"Precision: {metrics['Precision']})")
    print(f"Recall: {metrics['Recall']}")
    print(f"F1-Score: {metrics['F1-Score']}")
    print(f"AUC: {metrics['AUC']})")
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
