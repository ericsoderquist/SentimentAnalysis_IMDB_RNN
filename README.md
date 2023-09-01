# Sentiment Analysis on IMDB Reviews using Hand-crafted RNN
## University of Illinois Urbana-Champaign
### Author: Eric Soderquist
---
## Table of Contents
1. [Introduction](#introduction)
2. [Motivation](#motivation)
3. [Prerequisites](#prerequisites)
4. [File Structure](#file-structure)
5. [Methods Overview](#methods-overview)
6. [Usage](#usage)
7. [Theoretical Background](#theoretical-background)
8. [Performance Metrics](#performance-metrics)
9. [Contributing](#contributing)
10. [License](#license)
11. [Acknowledgements](#acknowledgements)
## Introduction

This project aims to perform sentiment analysis on IMDB movie reviews. 
Specifically, it employs a hand-crafted Recurrent Neural Network (RNN) for this task. 
Sentiment Analysis is an important subfield in Natural Language Processing (NLP) 
that classifies the polarity of a given text.

## Motivation

With the advent of online platforms, user reviews have become a critical 
component in decision-making processes for consumers. Analyzing these reviews 
at scale can offer invaluable insights into public opinion. This project aims to 
automate the process of sentiment classification for IMDB reviews.

## Prerequisites

- Python 3.x
- NumPy
- requests

## File Structure

- `SentimentAnalysis_IMDB_RNN.py`: Main Python script containing the model and utility functions.

## Methods Overview

The project uses a hand-crafted Recurrent Neural Network (RNN) for classifying sentiments. 
Here is a simplified example of how the RNN layer is implemented in the code:
```python
def rnn_layer(self, input_data, weights, biases):
    # Initialize hidden state
    hidden_state = np.zeros((input_data.shape[0], self.hidden_dim))

    # RNN computation
    for t in range(input_data.shape[1]):
        hidden_state = np.tanh(np.dot(input_data[:, t, :], weights) + np.dot(hidden_state, biases))
    return hidden_state
```
The RNN model captures the sequential dependencies in the review texts, making it well-suited for the task.

## Usage

1. Clone the repository: 
```bash
git clone <repository_url>
```
2. Navigate to the project directory: 
```bash
cd <project_directory>
```
3. Install the required packages: 
```bash
pip install -r requirements.txt
```
4. Run the main Python script: 
```bash
python SentimentAnalysis_IMDB_RNN.py
```
5. Optional: For advanced usage, refer to the code comments and documentation.

## Theoretical Background

Recurrent Neural Networks (RNNs) are specialized for sequence-based tasks. The core idea behind an RNN is to maintain a hidden state \( h_t \) that captures the information of all the previous steps in the sequence. The hidden state at time \( t \) is computed as:

\[ h_t = \sigma(W_{hh} h_{t-1} + W_{xh} x_t) \]

Where \( \sigma \) is the activation function, \( W_{hh} \) and \( W_{xh} \) are weight matrices, \( h_{t-1} \) is the previous hidden state, and \( x_t \) is the input at time \( t \).

This characteristic makes RNNs ideal for sentiment analysis on textual data.

## Performance Metrics

The model's performance is evaluated using the following metrics:
- Accuracy
- Precision
- Recall
- F1 Score

## Contributing

Contributions are welcome. Please submit a pull request for any enhancements or bug fixes.

## License
MIT License
## Acknowledgements
Special thanks to the University of Illinois Urbana-Champaign for the academic environment that made this project possible.
## Known Issues and Limitations

- The model might not perform well on extremely long or short reviews due to the limitations of vanilla RNNs.
- The current implementation does not include any pre-trained embeddings, which might affect the model's performance.

## References and Further Reading

- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Sentiment Analysis: An Overview](https://arxiv.org/abs/1912.01612)

## Visual Aids

- Architecture Diagram: The architecture for the RNN model can be conceptually summarized as follows:
```
Input Layer (Word Embeddings) ---> RNN Layer ---> Output Layer (Sentiment Score)
```
- Example Outputs: The chart represents hypothetical sentiment scores that the model might output for a set of IMDB reviews.
![Example Sentiment Scores for IMDB Reviews](/mnt/data/example_sentiment_scores.png)
Note: The example output chart is based on hypothetical data and should be validated with actual code outputs.
