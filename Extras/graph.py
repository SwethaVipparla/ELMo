import pickle

with open('results.pkl', 'rb') as pickle_file:
    results_dict = pickle.load(pickle_file)
    
all_emlo_train_loss = results_dict['elmo_train_loss']
all_elmo_val_loss = results_dict['elmo_val_loss']
all_downstream_train_loss = results_dict['downstream_train_loss']
all_downstream_val_loss = results_dict['downstream_val_loss']
all_downstream_train_accuracy = results_dict['downstream_train_accuracy']
all_downstream_val_accuracy = results_dict['downstream_val_accuracy']

import numpy as np
def get_graph(train_loss, val_loss):
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set()
    x = np.arange(1, len(train_loss) + 1)
    plt.title('Training/Validation Loss vs Epoch (ELMo Pretraining))')
    plt.figure(figsize=(12, 6))
    plt.plot(x, train_loss, label='Training Loss')
    plt.plot(x, val_loss, label='Validation Loss')
    plt.xticks(x)
    plt.xlabel('Epoch')
    plt.ylabel('Cross Entropy Loss')
    plt.legend()
    plt.show()
    
get_graph(all_downstream_train_loss, all_downstream_val_loss)