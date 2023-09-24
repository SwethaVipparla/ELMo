
# ELMo & News Classification

The best elmo model can be accessed on my [OneDrive](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/swetha_vipparla_students_iiit_ac_in/EbT8rpo4EldOp7D8josGFBABtlhU8zROk_aIlJOHaO8EjQ?e=9dpF4a).

The entire code is split into 5 files, which have different purposes.

- `preprocessing.py`
  Contains the code for preprocessing the data. The unicode characters are normalised, repeating characters are removed, and the important punctuations are retained for both `train.csv` and `test.csv`.

- `dataset.py`
    Contains the code for creating the dataset for the ELMo and Downstream task models. The dataset is created using the `torch.utils.data.Dataset` class. It adds the `sos` and `eos` tokens to the sentences and creates the vocabulary for the dataset using the training set. The indexed sentence, its reverse, and the corresponding classification labels are returned.  

- `elmo_model.py`
    Contains the code for the ELMo model. The ELMo class has 2 LSTMs, forward and backwards, for bidirectional language modelling. Either the forward or backward layer is executed based on the argument in the forward function.

- `downstream_model.py`
    Contains the code for the downstream classification task model. This class obtains the hidden and cell states from the forward and the backward LSTMs in the ELMo model. These are then concatenated and each concatenated vector is multiplied with delta values. The products are summed and fed to a linear layer that has a softmax activation layer.

- `train_test.py`
    Contains the code for pre-training the elmo and training and testing the downstream model. Both the models are trained for the optimal number of epochs using early stopping technique. Multiple delta values are tested for analysing the model performance.
    The entire sweep can be viewed on the wandb console.

### Execution of Code

Make sure the following dependencies are installed:

- nltk
- numpy
- torch
- gensim
- tqdm
- sklearn

The code can be run by executing the following command:

```bash
python train_test.py
```

### Best Model

The best model after pre training the ELMo is obtained by evaluation on the validation set. The model is stored in the `best_elmo_model.pth` file. This model can be restored by running the following command:

```python
model = ELMO(embedding_matrix, hidden_dim, num_layers, dropout, 'best_elmo_model.pth')
```
