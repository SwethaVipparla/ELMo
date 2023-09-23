# %%
import torch
import preprocessing as pp
import elmo_dataset as ds
import elmo_model as mdl

# %%
train_set = list(pp.df['Description'])
pre_train_set = train_set[:8000]
pre_val_set = train_set[8000:10000]

# %%
vocab = ds.ELMO_Dataset.create_vocab(train_set)
word_to_ix = {word: idx for idx, word in enumerate(vocab)}
print(len(word_to_ix))

# %%
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    batch = sorted(batch, key=lambda x: x.shape[0], reverse=True)
    padded_batch = pad_sequence(batch, batch_first=True, padding_value=word_to_ix['<pad>'])
    lengths = torch.LongTensor([len(x) for x in batch])

    input_tensor = padded_batch[:, :-1]
    target_truth = padded_batch[:, 1:]

    return input_tensor, target_truth, lengths - 1

# %%
from torch.utils.data import DataLoader

pre_train_dataset = ds.ELMO_Dataset(pre_train_set, word_to_ix)
pre_val_dataset = ds.ELMO_Dataset(pre_val_set, word_to_ix)

batch_size = 16

pre_train_loader = DataLoader(pre_train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True, num_workers=4)
pre_val_loader = DataLoader(pre_val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True, num_workers=4)

# %%
import gensim
import gensim.downloader

# glove_vectors = gensim.downloader.load('glove-wiki-gigaword-50')

# %%
import numpy as np

def embedding_matrix(word_to_ix, glove_vectors):
    embedding_dim = glove_vectors.vector_size
    embedding_matrix = np.zeros((len(word_to_ix), embedding_dim))

    average_vector = np.mean(glove_vectors.vectors, axis=0)

    special_token_embeddings = {
        '<pad>': np.zeros(embedding_dim),
        '<unk>': average_vector,
        '<sos>': np.random.randn(embedding_dim),
        '<eos>': np.random.randn(embedding_dim)
    }

    for word, i in word_to_ix.items():
        if word in glove_vectors:
            embedding_vector = glove_vectors[word]
        else:
            embedding_vector = special_token_embeddings.get(word, average_vector)
        
        embedding_matrix[i] = embedding_vector

    print(embedding_matrix.shape)

    return torch.FloatTensor(embedding_matrix)

# %%
# model = mdl.ELMO(embedding_matrix, 300, 2, 0.5)

# %%
from tqdm import tqdm

def run_epoch(model, data_loader, loss_fn, epoch, optimizer=None):
    if optimizer:
        model.train()
    else:
        model.eval()

    total_loss = 0
    p_bar = tqdm(data_loader)

    for (input_tensor, target_truth, lengths) in p_bar:
        input_tensor = input_tensor.cuda()
        target_truth = target_truth.cuda()

        output = model(input_tensor, lengths)
        output = output.reshape(-1, output.shape[2])

        loss = loss_fn(output, target_truth.reshape(-1))
        total_loss += loss.item()

        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        mean_loss = total_loss / len(data_loader)

        p_bar.set_description(f'{"T" if optimizer else "V"} Loss: {mean_loss:.4f}, count: {epoch}')

    return mean_loss

# %%
import torch.nn as nn

def train(optimizer, embedding_dimension, hidden_dimension, dropout_rate, learning_rate):
    num_epochs = 100

    glove_vectors = gensim.downloader.load(f'glove-wiki-gigaword-{embedding_dimension}')
    loss_fn = nn.CrossEntropyLoss()

    model = mdl.ELMO(embedding_matrix(word_to_ix, glove_vectors), hidden_dimension, 2, dropout_rate)
    optimizer = getattr(torch.optim, optimizer)(model.parameters(), lr=learning_rate)

    model.cuda()

    best_val_loss = float('inf')

    all_val_loss = []
    all_train_loss = []

    for epoch in range(num_epochs):
        train_loss = run_epoch(model, pre_train_loader, loss_fn, epoch+1, optimizer)
        all_train_loss.append(train_loss)

        with torch.no_grad():
            val_loss = run_epoch(model, pre_val_loader, loss_fn, epoch+1)
            all_val_loss.append(val_loss)

        print('Epoch: {}, Train Loss: {:.4f}, Val Loss: {:.4f}'.format(epoch+1, train_loss, val_loss))

        if val_loss < best_val_loss:
            counter = 0
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_lstm_model.pth')
        else:
            counter += 1
            if counter == 3:
                break
    
    return train_loss, val_loss

# %%
# train_loss, val_loss = train('Adam', 50, 300, 0.5, 0.001)

# %%
#wandb config
param_dist = {
    'method': 'random',
    'name': 'elmo',
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'optimizer': {'values': ['Adam', 'RMSprop']},
        'embedding_dimension': {'values': [50, 100, 200]},
        'hidden_dimension': {'values': [100, 300, 500]},
        'dropout_rate': {'values': [0.0, 0.2, 0.4]},
        'learning_rate': {'values': [0.001, 0.01, 0.1]}
    }
}

# %%
import wandb 

def random_search(config=None):
    with wandb.init(config=config):
        cfg = wandb.config
        
        train_loss, val_loss = train(cfg.optimizer, cfg.embedding_dimension, cfg.hidden_dimension, cfg.dropout_rate, cfg.learning_rate)
        wandb.log({'train_loss': train_loss, 'val_loss': val_loss})

# %%
sweep_id = wandb.sweep(param_dist, project='elmo')
wandb.agent(sweep_id, random_search, count=20)

# %%



