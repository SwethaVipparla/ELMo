# %%
import torch
import preprocessing as pp
import dataset as ds
import elmo_model as em
import downstream_model as dm


# %%
PRE_TRAIN_SET = 8000
PRE_VAL_SET = 2000
DOWNSTREAM_TRAIN_SET = 80000
DOWNSTREAM_VAL_SET = 20000

# %%
train_set = list(pp.train_df['Description'])
train_labels = list(pp.train_df['Class Index'])

pre_train_set = train_set[:PRE_TRAIN_SET]
pre_val_set = train_set[PRE_TRAIN_SET:PRE_TRAIN_SET+PRE_VAL_SET]

# %%
vocab = ds.Records.create_vocab(train_set)
word_to_ix = {word: idx for idx, word in enumerate(vocab)}
print(len(word_to_ix))

# %%
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    forward = pad_sequence([x[0] for x in batch], batch_first=True, padding_value=word_to_ix['<pad>'])
    backward = pad_sequence([x[1] for x in batch], batch_first=True, padding_value=word_to_ix['<pad>'])
    
    x_forward = torch.LongTensor(forward[:, :-1])  
    x_backward = torch.LongTensor(backward[:, :-1])
    
    y_forward = torch.LongTensor(forward[:, 1:])
    y_backward = torch.LongTensor(backward[:, 1:])
    
    lengths = torch.LongTensor([len(x[0]) for x in batch])
    labels = torch.LongTensor([x[2] - 1 for x in batch])

    return x_forward, x_backward, y_forward, y_backward, lengths - 1, labels

# %%
from torch.utils.data import DataLoader

pre_train_dataset = ds.Records(pre_train_set, word_to_ix, train_labels)
pre_val_dataset = ds.Records(pre_val_set, word_to_ix, train_labels)

batch_size = 32

pre_train_loader = DataLoader(pre_train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True, num_workers=2)
pre_val_loader = DataLoader(pre_val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True, num_workers=2)

# %%
import gensim
import gensim.downloader

glove_vectors = gensim.downloader.load('glove-wiki-gigaword-100')

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
from tqdm import tqdm

def run_epoch(model, data_loader, criterion, epoch, forward, optimizer=None):
    if optimizer:
        model.train()
    else:
        model.eval()
    
    total_loss = 0
    p_bar = tqdm(data_loader)

    for x_forward, x_backward, y_forward, y_backward, lengths, labels in p_bar:
        if forward:
            input_tensor = x_forward.cuda()
            target_tensor = y_forward.cuda()
        else:
            input_tensor = x_backward.cuda()
            target_tensor = y_backward.cuda()

        output, _, _, _ = model(input_tensor, lengths, forward)
        loss = criterion(output.view(-1, output.shape[2]), target_tensor.view(-1))

        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        mean_loss = total_loss / len(data_loader)
        
        p_bar.set_description(f'{"T" if optimizer else "V"} Loss: {mean_loss:.4f}, count: {epoch}')
        
    return mean_loss

# %%
import torch.nn as nn


def train(optimizer, embedding_dimension, hidden_dimension, dropout_rate, learning_rate):
    num_epochs = 50

    glove_vectors = gensim.downloader.load(f'glove-wiki-gigaword-{embedding_dimension}')
    loss_fn = nn.CrossEntropyLoss(ignore_index=word_to_ix['<pad>'])

    model = em.ELMO(embedding_matrix(word_to_ix, glove_vectors), hidden_dimension, 1, dropout_rate)

    optimizer = getattr(torch.optim, optimizer)(model.parameters(), lr=learning_rate)

    model.cuda()

    best_val_loss = float('inf')

    all_val_loss = []
    all_train_loss = []

    for epoch in range(num_epochs):

        forward_train_loss = run_epoch(model, pre_train_loader, loss_fn, epoch+1, 1, optimizer)
        backward_train_loss = run_epoch(model, pre_train_loader, loss_fn, epoch+1, 0, optimizer)

        with torch.no_grad():
            forward_val_loss = run_epoch(model, pre_val_loader, loss_fn, epoch+1, 1)
            backward_val_loss = run_epoch(model, pre_val_loader, loss_fn, epoch+1, 0)

        print('Epoch: {}, F Train Loss: {:.4f}, F Val Loss: {:.4f}'.format(epoch+1, forward_train_loss, forward_val_loss))
        print('Epoch: {}, B Train Loss: {:.4f}, B Val Loss: {:.4f}'.format(epoch+1, backward_train_loss, backward_val_loss))

        average_train_loss = (forward_train_loss + backward_train_loss) / 2
        average_val_loss = (forward_val_loss + backward_val_loss) / 2

        all_train_loss.append(average_train_loss)
        all_val_loss.append(average_val_loss)
        
        print('Average Train Loss: {:.4f}, Average Val Loss: {:.4f}'.format(average_train_loss, average_val_loss))
        print('-------------------------------------------------------')

        if average_val_loss < best_val_loss:
            counter = 0
            best_val_loss = average_val_loss
            torch.save(model.state_dict(), 'best_elmo_model.pth')
        else:
            counter += 1
            if counter == 3:
                break

    return all_train_loss, all_val_loss

# %%
all_emlo_train_loss, all_elmo_val_loss = train('Adam', 100, 100, 0.2, 0.001)

# %%
downstream_train_set = train_set[PRE_TRAIN_SET+PRE_VAL_SET:PRE_TRAIN_SET+PRE_VAL_SET+DOWNSTREAM_TRAIN_SET]
downstream_train_labels = train_labels[PRE_TRAIN_SET+PRE_VAL_SET:PRE_TRAIN_SET+PRE_VAL_SET+DOWNSTREAM_TRAIN_SET]

downstream_val_set = train_set[PRE_TRAIN_SET+PRE_VAL_SET+DOWNSTREAM_TRAIN_SET:PRE_TRAIN_SET+PRE_VAL_SET+DOWNSTREAM_TRAIN_SET+DOWNSTREAM_VAL_SET]
downstream_val_labels = train_labels[PRE_TRAIN_SET+PRE_VAL_SET+DOWNSTREAM_TRAIN_SET:PRE_TRAIN_SET+PRE_VAL_SET+DOWNSTREAM_TRAIN_SET+DOWNSTREAM_VAL_SET]

# %%
downstream_train_dataset = ds.Records(downstream_train_set, word_to_ix, downstream_train_labels, )
downstream_val_dataset = ds.Records(downstream_val_set, word_to_ix, downstream_val_labels)

downstream_train_loader = DataLoader(downstream_train_dataset, batch_size=32, shuffle=True, pin_memory=True, num_workers=2, collate_fn=collate_fn)
downstream_val_loader = DataLoader(downstream_val_dataset, batch_size=32, shuffle=False, pin_memory=True, num_workers=2, collate_fn=collate_fn)

# %%
def run_downstream_epoch(model, data_loader, loss_fn, epoch, optimizer=None):
    if optimizer:
        model.train()
    else:
        model.eval()

    total_loss = 0

    p_bar = tqdm(data_loader)

    for (x_forward, x_backward, y_forward, y_backward, lengths, labels) in p_bar:
        x_forward = x_forward.cuda()
        x_backward = x_backward.cuda()

        labels = labels.cuda()

        output = model(x_forward, x_backward, lengths)

        output = output.reshape(-1, output.shape[-1])

        loss = loss_fn(output, labels)
        total_loss += loss.item()

        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        mean_loss = total_loss / len(data_loader)

        p_bar.set_description(f'{"T" if optimizer else "V"} Loss: {mean_loss:.4f}, count: {epoch}')

    return mean_loss

# %%
def train_downstream_classifier(optimizer, downstream_model):
    num_epochs = 100

    loss_fn = nn.CrossEntropyLoss(ignore_index=word_to_ix['<pad>'])

    optimizer = getattr(torch.optim, optimizer)(downstream_model.parameters(), lr=0.001)

    best_val_loss = float('inf')

    all_val_loss = []
    all_train_loss = []

    for epoch in range(num_epochs):
        train_loss= run_downstream_epoch(downstream_model, downstream_train_loader, loss_fn, epoch+1, optimizer)
        all_train_loss.append(train_loss)
 
        with torch.no_grad():
            val_loss= run_downstream_epoch(downstream_model, downstream_val_loader, loss_fn, epoch+1)
            all_val_loss.append(val_loss)

        print('Epoch: {}, Train Loss: {:.4f}, Val Loss: {:.4f}'.format(epoch+1, train_loss, val_loss))

        if val_loss < best_val_loss:
            counter = 0
            best_val_loss = val_loss
            torch.save(downstream_model.state_dict(), 'best_downstream_model.pth')
        else:
            counter += 1
            if counter == 3:
                break
    
    downstream_model.load_state_dict(torch.load('best_downstream_model.pth'))
    return all_train_loss, all_val_loss, downstream_model

# %%
test_set = list(pp.test_df['Description'])
test_labels = list(pp.test_df['Class Index'])

test_dataset = ds.Records(test_set, word_to_ix, test_labels)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=True, num_workers=2, collate_fn=collate_fn)

# %%
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
    
def test_downstream_classifier(downstream_model):
    loss_fn = nn.CrossEntropyLoss(ignore_index=word_to_ix['<pad>'])
    downstream_model.eval()
    
    pred_labels = []
    true_labels = []
    total_loss = []

    with torch.no_grad():
        for x_forward, x_backward, lengths, labels in test_loader:
            x_forward = x_forward.cuda()
            x_backward = x_backward.cuda()
            labels = labels.cuda()

            output = downstream_model(x_forward, x_backward, lengths)

            output = output.reshape(-1, output.shape[-1])

            loss = loss_fn(output, labels)
            total_loss.append(loss.item())

            _, predicted = torch.max(output, 1)
            pred_labels.extend(predicted.tolist())
            true_labels.extend(labels.tolist())

    mean_loss = np.mean(total_loss)
    
    print('Test Loss: {:.4f}'.format(mean_loss))
    
    print(classification_report(true_labels, pred_labels))
    cm = confusion_matrix(true_labels, pred_labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    

# %%
num_classes = 4
delta_values = [None, [0, 0, 1], [1, 2, 3], [1, 1, 1], [1, 0, 0]]

for i in range(len(delta_values)):
    if delta_values[i] is not None:
        delta_values[i] = np.array(delta_values[i]).reshape(1, -1)
        delta_values[i] = torch.FloatTensor(delta_values[i]).cuda()
        
for delta in delta_values:
    downstream_model = dm.Downstream_Model(embedding_matrix(word_to_ix, glove_vectors), 100, 2, 4, 0.2)
    downstream_model = downstream_model.cuda()
    if delta is not None:
        downstream_model.delta = nn.Parameter(delta, requires_grad=False)
    all_downstream_train_loss, all_downstream_val_loss, downstream_model = train_downstream_classifier('Adam', downstream_model)
    test_downstream_classifier(downstream_model)
    print(delta)
    plt.show()
    print('-------------------------------------------------------')
    

# %%
import pickle

results_dict = {
    'elmo_train_loss': all_emlo_train_loss,
    'elmo_val_loss': all_elmo_val_loss,
    'downstream_train_loss': all_downstream_train_loss,
    'downstream_val_loss': all_downstream_val_loss,
}

with open('results.pkl', 'wb') as pickle_file:
  pickle.dump(results_dict, pickle_file)


