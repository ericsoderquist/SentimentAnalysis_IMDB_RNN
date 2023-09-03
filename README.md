# Sentiment Analysis on IMDB Reviews using RNNs

## Author
[Eric Soderquist](mailto:eys3@illinois.edu)

## Introduction
This repository contains a Python implementation of a Recurrent Neural Network (RNN) model for sentiment analysis on the IMDB dataset. Sentiment analysis is a subfield of semantic analysis that focuses on the task of identifying subjective information from text data. Understanding the sentiments expressed in texts like reviews, tweets, or comments can be pivotal for businesses, policymakers, and individuals alike. The model utilizes different configurations of hyperparameters to identify the best set for maximizing classification accuracy.

<a id="theoretical-background"></a>
<details>
<summary><strong>Theoretical Background</strong></summary>

### Recurrent Neural Networks (RNNs)
Recurrent Neural Networks (RNNs) are a class of artificial neural networks designed for sequence-based data. Unlike traditional feedforward neural networks, RNNs possess recurrent connections that loop back within the network. This unique architecture allows the network to maintain a 'state' or 'memory' across the sequence, which is invaluable for tasks such as natural language processing, time-series prediction, and, notably, semantic and sentiment analysis.

The Recurrent Neural Network (RNN) model used in this project consists of the following layers:

- **Input Layer**: Takes tokenized sequences as input.
- **Hidden Layers**: Contains RNN units with activation functions to capture sequential dependencies.
- **Output Layer**: Uses a sigmoid activation function to output a sentiment score between 0 and 1.

#### The Basic Recurrent Unit
The fundamental equation that governs the behavior of a basic recurrent unit is:

$h_t = \sigma(W_x x_t + W_h h_{t-1} + b)$

Where:
- $h_t$: Hidden state at time $t$
- $x_t$: Input at time $t$
- $h_{t-1}$: Hidden state at time $t-1$
- $W_x$, $W_h$: Weight matrices
- $b$: Bias vector
- $\sigma$: Activation function (commonly tanh or ReLU)

#### Challenges with Basic RNNs
While RNNs are powerful, they suffer from issues like the vanishing and exploding gradient problems. These issues limit the network's ability to learn long-range dependencies, making them less effective for complex tasks.

### Long Short-Term Memory (LSTM) Units
Long Short-Term Memory (LSTM) units are a type of recurrent neural network architecture designed to remember information for extended periods. It was introduced to combat the vanishing gradient problem that plagued traditional RNNs. An LSTM unit is composed of a cell, an input gate, an output gate, and a forget gate. The cell is responsible for "remembering" values over arbitrary time intervals, while the gates regulate the flow of information into and out of the cell.

The governing equations for an LSTM unit are as follows:

$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$

$\tilde{C}_t$

$= tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$

$C_t = f_t * C_{t-1} + i_t * \tilde{C}_t$

$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$

$h_t = o_t * tanh(C_t)$

Where:
- $f_t, i_t, o_t$: Forget, input, and output gates at time $t$
- $C_t$: Cell state at time $t$
- $\tilde{C}_t$: Candidate cell state at time $t$
- $h_t$: Hidden state at time $t$

### Gated Recurrent Units (GRU)
Gated Recurrent Units (GRU) are a variation of LSTM units, designed to be more computationally efficient. They combine the forget and input gates into a single "update gate" and also merge the cell state and hidden state, resulting in a simpler and more streamlined architecture.

The governing equations for a GRU unit are as follows:

$z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)$

$r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)$

$\tilde{C}_t$  

$= tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$

$h_t = (1 - z_t) * h_{t-1} + z_t * \tilde{h}_t$

Where:
- $z_t$: Update gate at time $t$
- $r_t$: Reset gate at time $t$
- $\tilde{h}_t$: Candidate hidden state at time $t$
- $h_t$: Hidden state at time $t$

### Importance of Semantic Analysis
Semantic analysis refers to the study of meaning in language. In the context of machine learning and natural language processing, semantic analysis is pivotal for understanding the nuances and context behind a piece of text. This is particularly important in tasks like sentiment analysis, where the objective is not just to understand the syntactic structure but also to capture the underlying sentiment or opinion. By employing RNNs and their advanced variants like LSTMs and GRUs, we can build models that understand the temporal dependencies in text data, thereby capturing the semantic essence more effectively.

</details>


### Recurrent Neural Networks (RNNs)
RNNs are designed for sequence-based data. Unlike traditional feedforward networks, RNNs have connections that loop back within the network, allowing information to persist.

#### The Basic Recurrent Unit
The fundamental unit of an RNN is defined by the equation:

$h_t = \sigma(W_x x_t + W_h h_{t-1} + b)$

#### Long Short-Term Memory (LSTM) Units
LSTMs are an improvement over basic RNNs and are defined by the following equations:

$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$

$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$

$\tilde{C}_t$

$= \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$

$C_t = f_t * C_{t-1} + i_t * \tilde{C}_t$

$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$

$h_t = o_t * \tanh(C_t)$

</details>

## Table of Contents
- [Introduction](#introduction)
- [Theoretical Background](#theoretical-background)
- [Requirements](#requirements)
- [Usage](#usage)
- [Quick Start](#quick-start)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Results](#results)
- [Visualization](#visualization)
- [Future Work](#future-work)
- [License](#license)
- [Contact](#contact)

## Requirements
- Python 3.x
- NumPy
- Matplotlib
- Seaborn
- Requests
- tarfile

## Usage
1. Clone the repository.
2. Navigate to the directory where the code is stored.
3. Run the `SentimentAnalysis_IMDB_RNN.py` script.

## Quick Start
## Clone the repository
```bash
git clone https://github.com/ericsoderquist/SentimentAnalysis_IMDB_RNN.git
```
## Navigate to the repository
```bash
cd SentimentAnalysis_IMDB_RNN
```
## Run the script
```bash
python SentimentAnalysis_IMDB_RNN.py
```

## Hyperparameter Tuning
The model undergoes a series of tests with different hyperparameters to determine the optimal set for achieving the highest accuracy. The hyperparameters include:
- Layer configurations (e.g., [32], [32, 16])
- Learning rate (e.g., 0.1, 0.01)
- Number of epochs (fixed at 10)
- Batch size (fixed at 32)

## Results
The best-performing model was achieved with the following hyperparameters:
- Layer configuration: [32, 16]
- Learning rate: 0.1
- Epochs: 10
- Batch size: 32
- Accuracy: 1.0

## Performance Metrics

The model's performance is evaluated using the following metrics:

- **Accuracy**: Measures the proportion of correctly classified samples.
- **Precision**: Measures the proportion of true positive samples among the samples predicted as positive.
- **Recall**: Measures the proportion of true positive samples among all actual positive samples.
- **F1-Score**: Harmonic mean of Precision and Recall, providing a balanced view of the model's performance.
- **AUC**: Area Under the Receiver Operating Characteristic Curve, measuring the model's ability to distinguish between classes.

## Visualization
![Hyperparameter Testing: Accuracy by Layer Configuration and Learning Rate](/visualization.jpg)

## Future Work
- Integrate more complex architectures such as LSTM and GRU.
- Experiment with different optimization algorithms.
- Perform feature engineering to improve model performance.

## License
This project is licensed under the MIT License - see the [LICENSE](/LICENSE) file for details.

## Contact
For any questions or contributions, please feel free to contact me. - [Eric Soderquist](mailto:eys3@illinois.edu)

## References
- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

### Social Impact and Applications
The advancements in semantic and sentiment analysis through RNNs and their advanced architectures have profound implications beyond academia and industry. Notably, they hold the potential to create meaningful change in the lives of marginalized or specialized groups.

#### Disability Aids for Neurodivergent Individuals
One such application is in the development of assistive technologies for neurodivergent individuals. Semantic analysis can be instrumental in creating more intuitive and responsive communication aids, facilitating a better understanding of nuanced human emotions and intentions. This can greatly enhance the quality of life for individuals who may experience challenges in social interaction and communication.

#### Translation Aids for English Second-Language Learners
Similarly, these technologies can significantly benefit people who are learning English as a second language. Accurate sentiment and semantic analysis can help in developing advanced translation aids that capture not just the literal meaning of sentences but also the emotional nuances and cultural context, making the language-learning journey more enriching and effective.

These applications underscore the broader societal impact of advancements in this field, driving home the importance of continued research and development.
