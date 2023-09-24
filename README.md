
# ELMo & News Classification

The best model can be accessed through the following link:
https://iiitaphyd-my.sharepoint.com/:f:/g/personal/swetha_vipparla_students_iiit_ac_in/EkX4AnkFITtGpX6kacEvA2gByvqkSZy9U7NUMKAZXkmXEQ?e=85ibWM

## ELMo Model Pretraining

The entire code is split into 4 files, which have different purposes.

- `preprocessing.py`
  Contains the code for preprocessing the data. The unicode characters are normalised, repeating characters are removed, and the important punctuations are retained.

- `nnlm_dataset.py`
    Contains the code for creating the dataset for the NNLM model. The dataset is created using the `torch.utils.data.Dataset` class. It takes the n-grams of the sentences and the corresponding next word as contexts and targets accordingly. It also creates the vocabulary for the dataset using the training set.

- `nnlm_model.py`
    Contains the code for the NNLM model. The model is a simple feed-forward neural network with 2 hidden layers. The model is trained using the `torch.nn` module. The activation function used is `ReLU` and the loss function used is `CrossEntropyLoss`.

- `nnlm_train_test.py`
    Contains the code for training and testing the model. The model is trained for the optimal number of epochs using early stopping technique. The hyperparameters are tuned using random search. The model is tested on the test set and the perplexity is calculated.
    The entire sweep can be viewed on the wandb console.

### Execution of Code

Make sure the following dependencies are installed:

- nltk
- numpy
- torch
- wandb
- gensim

The code can be run by executing the following command:

```bash
python nnlm_train_test.py
```

### Best Model

The best model after training is obtained by evaluation on the validation set. The model is stored in the `best_nnlm_model.pth` file. This model can be restored by running the following command:

```python
model = NNLM(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout)
model.load_state_dict(torch.load('best_nnlm_model.pth'))
```

## Task 2: RNN Language Model

The entire code is split into 4 files, which have different purposes.

- `preprocessing.py`
  Contains the code for preprocessing the data. The unicode characters are normalised, repeating characters are removed, and the important punctuations are retained.
  In addition, `sos` and `eos` tokens are added to the sentences.

- `lstm_dataset.py`
    Contains the code for creating the dataset for the LSTM model. The dataset is created using the `torch.utils.data.Dataset` class. It takes the sentences as contexts and the corresponding next word as targets accordingly. It also creates the vocabulary for the dataset using the training set.

- `lstm_model.py`
    Contains the code for the LSTM model. The model takes the input of the previous word and the hidden state of the previous time step and outputs the next word and the hidden state of the current time step. The model is trained using the `torch.nn` module. The activation function used is `ReLU` and the loss function used is `CrossEntropyLoss`.

- `lstm_train_test.py`
    Contains the code for training and testing the model. The model is trained for the optimal number of epochs using early stopping technique. It is then tested on the test set and the perplexity is calculated.

### Execution of Code

Make sure the following dependencies are installed:

- nltk
- numpy
- torch
- gensim

The code can be run by executing the following command:

```bash
python lstm_train_test.py
```

### Best Model

The best model after training is obtained by evaluation on the validation set. The model is stored in the `best_lstm_model.pth` file. This model can be restored by running the following command:

```python
model = mdl.LSTM_Model(embedding_matrix, vocab_size)
model.load_state_dict(torch.load('best_lstm_model.pth'))
```
