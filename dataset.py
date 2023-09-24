import torch 
from torch.utils.data import Dataset

class Records(Dataset):
    def __init__(self, data, word_to_ix, labels):
        self.data = [['<sos>'] + sentence + ['<eos>'] for sentence in data]
        self.word_to_ix = word_to_ix
        self.indexed_data = [self.index_sentence(sentence) for sentence in self.data]
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
            return torch.tensor(self.indexed_data[idx]), torch.tensor(self.indexed_data[idx][::-1]), torch.tensor(self.labels[idx])

    def index_sentence(self, sentence):
        indexed_sentence = [self.word_to_ix.get(word, self.word_to_ix['<unk>']) for word in sentence]
        return indexed_sentence

    @staticmethod
    def create_vocab(data):
        vocab = {word for sentence in data for word in sentence}
        special_tokens = ['<sos>', '<eos>', '<unk>', '<pad>']
        vocab = vocab.union(set(special_tokens)) 
        return vocab