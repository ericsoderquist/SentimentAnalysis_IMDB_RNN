"""
Sentiment Analysis of IMDB reviews using Hand-crafted RNN.
Author: Eric Soderquist
"""

# Standard Library Imports
import os
import re
import tarfile
import random
from collections import Counter
from contextlib import closing
from multiprocessing import Pool

# Third-Party Imports
import numpy as np
import requests

class SentimentAnalysisRNN:
    """
    A class for performing sentiment analysis on IMDB reviews using a hand-crafted RNN.
    """
    def __init__(self):
        """
        Initializes the SentimentAnalysisRNN object.
        """
        self.vocab_size = 10000  # The number of most frequent words to keep in the vocabulary
        self.embedding_size = 300  # The size of the word embeddings
        self.hidden_size = 128  # The size of the hidden state of the RNN
        self.learning_rate = 0.001  # The learning rate for the optimizer
        self.batch_size = 32  # The batch size for training
        self.num_epochs = 10  # The number of epochs to train for
        self.train_data = None  # The training data
        self.test_data = None  # The test data
        self.word_to_index = None  # A dictionary mapping words to their indices in the vocabulary
        self.index_to_word = None  # A dictionary mapping indices in the vocabulary to words
        self.model = None  # The RNN model

    def get_downloads_folder(self) -> str:
        """
        Get the downloads folder path from the system.
        The function expands the user's home directory and appends "Downloads" to obtain the path.

        Returns:
            str: The path to the downloads folder.
        """
        user_home = os.path.expanduser('~')
        downloads_folder = os.path.join(user_home, 'Downloads')
        return downloads_folder

    def download_and_extract_data(self, url: str, destination: str) -> None:
        """
        Downloads and extracts the IMDB dataset from the specified URL to the specified destination.

        Args:
            url (str): The URL of the dataset.
            destination (str): The destination directory to download and extract the dataset to.

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

        print("Extracting dataset to {}...".format(destination))

        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(destination)

        os.remove(file_path)

    def preprocess_data(self, data_dir: str) -> None:
        """
        Preprocesses the IMDB dataset by tokenizing the reviews and creating the vocabulary.

        Args:
            data_dir (str): The directory containing the IMDB dataset.

        Returns:
            None
        """
        # Load the reviews and labels from the dataset
        train_reviews, train_labels = self.load_data(os.path.join(data_dir, "train"))
        test_reviews, test_labels = self.load_data(os.path.join(data_dir, "test"))

        # Tokenize the reviews
        train_reviews = [self.tokenize(review) for review in train_reviews]
        test_reviews = [self.tokenize(review) for review in test_reviews]

        # Create the vocabulary
        self.word_to_index, self.index_to_word = self.create_vocabulary(train_reviews)

        # Convert the reviews to sequences of indices
        train_reviews = [self.convert_to_indices(review) for review in train_reviews]
        test_reviews = [self.convert_to_indices(review) for review in test_reviews]

        # Pad the sequences to a fixed length
        train_reviews = self.pad_sequences(train_reviews)
        test_reviews = self.pad_sequences(test_reviews)

        # Convert the labels to numpy arrays
        train_labels = np.array(train_labels)
        test_labels = np.array(test_labels)

        # Save the preprocessed data
        self.train_data = (train_reviews, train_labels)
        self.test_data = (test_reviews, test_labels)

    def load_data(self, data_dir: str) -> tuple:
        """
        Loads the reviews and labels from the IMDB dataset.

        Args:
            data_dir (str): The directory containing the IMDB dataset.

        Returns:
            tuple: A tuple containing the reviews and labels.
        """
        reviews = []
        labels = []

        for label in ["pos", "neg"]:
            label_dir = os.path.join(data_dir, label)

            for file_name in os.listdir(label_dir):
                file_path = os.path.join(label_dir, file_name)

                with open(file_path, "r", encoding="utf-8") as f:
                    review = f.read()

                reviews.append(review)
                labels.append(1 if label == "pos" else 0)

        return reviews, labels

    def tokenize(self, review: str) -> list:
        """
        Tokenizes a review by splitting it into words and removing punctuation and other non-alphabetic characters.

        Args:
            review (str): The review to tokenize.

        Returns:
            list: A list of tokens.
        """
        review = review.lower()
        review = re.sub(r"[^a-z ]", "", review)
        tokens = review.split()
        return tokens

    def create_vocabulary(self, reviews: list) -> tuple:
        """
        Creates a vocabulary from the reviews.

        Args:
            reviews (list): A list of tokenized reviews.

        Returns:
            tuple: A tuple containing the word-to-index and index-to-word dictionaries.
        """
        word_counts = Counter()

        for review in reviews:
            word_counts.update(review)

        most_common_words = word_counts.most_common(self.vocab_size - 2)
        word_to_index = {"<PAD>": 0, "<UNK>": 1}

        for i, (word, count) in enumerate(most_common_words, start=2):
            word_to_index[word] = i

        index_to_word = {i: word for word, i in word_to_index.items()}

        return word_to_index, index_to_word

    def convert_to_indices(self, review: list) -> list:
        """
        Converts a review from a list of tokens to a list of indices.

        Args:
            review (list): A list of tokens.

        Returns:
            list: A list of indices.
        """
        indices = []

        for token in review:
            if token in self.word_to_index:
                indices.append(self.word_to_index[token])
            else:
                indices.append(self.word_to_index["<UNK>"])

        return indices

    def pad_sequences(self, sequences: list) -> np.ndarray:
        """
        Pads a list of sequences to a fixed length.

        Args:
            sequences (list): A list of sequences.

        Returns:
            np.ndarray: An array of padded sequences.
        """
        max_length = max(len(sequence) for sequence in sequences)
        padded_sequences = np.zeros((len(sequences), max_length), dtype=np.int32)

        for i, sequence in enumerate(sequences):
            padded_sequences[i, :len(sequence)] = sequence

        return padded_sequences

    def build_model(self) -> None:
        """
        Builds the RNN model.

        Returns:
            None
        """
        self.model = RNN(self.vocab_size, self.embedding_size, self.hidden_size)

    def train(self) -> None:
        """
        Trains the RNN model.

        Returns:
            None
        """
        # Get the training data
        train_reviews, train_labels = self.train_data

        # Shuffle the training data
        indices = np.arange(len(train_reviews))
        np.random.shuffle(indices)
        train_reviews = train_reviews[indices]
        train_labels = train_labels[indices]

        # Split the training data into batches
        num_batches = len(train_reviews) // self.batch_size
        train_reviews = np.array_split(train_reviews[:num_batches * self.batch_size], num_batches)
        train_labels = np.array_split(train_labels[:num_batches * self.batch_size], num_batches)

        # Train the model
        optimizer = Adam(lr=self.learning_rate)
        loss_fn = BinaryCrossentropy(from_logits=True)

        for epoch in range(self.num_epochs):
            epoch_loss = 0

            for batch_reviews, batch_labels in zip(train_reviews, train_labels):
                with tf.GradientTape() as tape:
                    logits = self.model(batch_reviews)
                    loss = loss_fn(batch_labels, logits)

                gradients = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

                epoch_loss += loss.numpy()

            epoch_loss /= num_batches

            print("Epoch {} loss: {:.4f}".format(epoch + 1, epoch_loss))

    def evaluate(self) -> None:
        """
        Evaluates the RNN model on the test data.

        Returns:
            None
        """
        # Get the test data
        test_reviews, test_labels = self.test_data

        # Evaluate the model
        logits = self.model(test_reviews)
        predictions = tf.round(tf.sigmoid(logits))
        accuracy = np.mean(predictions.numpy() == test_labels)

        print("Test accuracy: {:.2f}%".format(accuracy * 100))

class RNN(tf.keras.Model):
    """
    A class for a simple RNN model.
    """
    def __init__(self, vocab_size: int, embedding_size: int, hidden_size: int):
        """
        Initializes the RNN object.

        Args:
            vocab_size (int): The size of the vocabulary.
            embedding_size (int): The size of the word embeddings.
            hidden_size (int): The size of the hidden state of the RNN.
        """
        super(RNN, self).__init__()

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.rnn = tf.keras.layers.SimpleRNN(hidden_size)
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Performs a forward pass through the RNN model.

        Args:
            inputs (tf.Tensor): The input tensor.

        Returns:
            tf.Tensor: The output tensor.
        """
        x = self.embedding(inputs)
        x = self.rnn(x)
        x = self.dense(x)
        return x

if __name__ == "__main__":
    # Create the SentimentAnalysisRNN object
    rnn = SentimentAnalysisRNN()

    # Download and extract the IMDB dataset
    url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    destination = rnn.get_downloads_folder()
    rnn.download_and_extract_data(url, destination)

    # Preprocess the data
    data_dir = os.path.join(destination, "aclImdb")
    rnn.preprocess_data(data_dir)

    # Build the model
    rnn.build_model()

    # Train the model
    rnn.train()

    # Evaluate the model
    rnn.evaluate()
