import torch
import numpy as np
import random

TEST_RATIO = 0.2
CATEGORY2ID = {'Causal' : 0, 'Follow' : 1, 'Concurrency' : 2}

def load_data(file, word2idx):
    data_list = []

    fopen = open(file, 'r')
    for line in fopen.readlines():
        contents = line.split('\t')
        relType = contents[0].strip()
        source = contents[1].strip()
        target = contents[2].strip()

        relTypeId = CATEGORY2ID[relType]

        x1 = [word2idx[w] for w in source.split(' ')]
        x2 = [word2idx[w] for w in target.split(' ')]

        data_list.append([[x1, x2], relTypeId])
    test_num = int(len(data_list)*TEST_RATIO)
    random.shuffle(data_list)
    train_list = data_list[:-test_num]
    test_list = data_list[-test_num:]
    train_x = [x_y[0] for x_y in train_list]
    train_x_tensor = torch.from_numpy(np.array(train_x))
    train_y = [x_y[1] for x_y in train_list]
    train_y_tensor = torch.from_numpy(np.array(train_y))
    test_x = [x_y[0] for x_y in test_list]
    test_x_tensor = torch.from_numpy(np.array(test_x))
    test_y = [x_y[1] for x_y in test_list]
    test_y_tensor = torch.from_numpy(np.array(test_y))
    test_y_tensor = torch.unsqueeze(test_y_tensor, 1)
    return train_x_tensor, train_y_tensor, test_x_tensor, test_y_tensor


    #x = np.array(x)
    #y = np.array(y, np.int64)
    #x_tensor = torch.from_numpy(x)
    #y_tensor = torch.from_numpy(y)
    #y_tensor = torch.unsqueeze(y_tensor, 1)

    #return x_tensor, y_tensor

def adjust_learning_rate(optimizer, decay_rate= 0.5):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate
