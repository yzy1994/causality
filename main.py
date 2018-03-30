import pickle
from model import *
from Util import *
import numpy as np
import torch.utils.data as Data
import torch.nn.functional as F
import torch
from sklearn.metrics import classification_report, confusion_matrix

CATEGORY2ID = {'Causal' : 0, 'Follow' : 1, 'Concurrency' : 2}
TARGET_SET = ['Causal', 'Follow', 'Concurrency']

INPUT_DIR = './data/casual_input'
VOCAB_DIR = './data/vocab.pkl'
BATCH_SIZE = 64
EPOCH_NUM = 40
EMBEDDING_DIM = 60
HIDDEN_DIM = 70
USE_GRU = False
#GRU cell or LSTM cell
DROPOUT_RATE = 0.2


if __name__ == '__main__':
    datalist = pickle.load(open(VOCAB_DIR, 'r'))
    word2idx = datalist[0]
    vector_list = datalist[1]

    w_embedding = torch.from_numpy(np.array(vector_list))

    x_tensor, y_tensor, x_test_tensor, y_test_tensor = load_data(INPUT_DIR, word2idx)
    train_dataset = Data.TensorDataset(data_tensor=x_tensor, target_tensor= y_tensor)

    loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model = SiameseNetwork(embedding_dims=EMBEDDING_DIM, hidden_dims=HIDDEN_DIM,
                           pretrained_embedding=w_embedding, word_nums=len(w_embedding),
                           use_gru=USE_GRU, dropout_rate=DROPOUT_RATE)
    citerion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)


    for epoch in range(EPOCH_NUM):
        for step, (batch_x, batch_y) in enumerate(loader):
            #train
            x1_x2 = torch.chunk(batch_x, 2, 1)
            x1 = x1_x2[0]
            x2 = x1_x2[1]
            x1 = torch.squeeze(x1, 1)
            x2 = torch.squeeze(x2, 1)
            x1, x2, y = Variable(x1), Variable(x2), Variable(batch_y)
            output = model(x1, x2)
            loss = citerion(output, y)
            #print loss.data[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch%5 == 0:
            adjust_learning_rate(optimizer)
            x1_x2 = torch.chunk(x_test_tensor, 2, 1)
            x1 = torch.squeeze(x1_x2[0])
            x2 = torch.squeeze(x1_x2[1])
            x1, x2, y = Variable(x1), Variable(x2), Variable(y_test_tensor)
            output = model(x1, x2)
            y = torch.squeeze(y, 1)
            loss = citerion(output, y)

            _, pred = torch.max(output, 1)
            result_list = []

            print '------------------eval-------------------'
            pred_numpy = pred.data.numpy()
            y_numpy = y.data.numpy()
            print classification_report(y_numpy, pred_numpy, target_names=TARGET_SET, digits=4)
            print confusion_matrix(y_numpy, pred_numpy)
            # print 'Category\tPrecision\tRecall\tF-measure'
            # for cate in CATEGORY2ID:
            #     cateid = int(CATEGORY2ID[cate])
            #     mask = y.eq(cateid)
            #     mask_pred = torch.masked_select(pred, mask)
            #     num = len(mask_pred)
            #     TP = (mask_pred==cateid).sum().data[0]
            #     FN = len(mask_pred) - TP
            #     mask = y.ne(cateid)
            #     mask_pred = torch.masked_select(pred, mask)
            #     FP = (mask_pred==cateid).sum().data[0]
            #     p = (float)(TP)/(TP + FP)
            #     r = (float)(TP)/(TP + FN)
            #     f = (float)(2*p*r)/(p+r)
            #     result_list.append([num, p, r, f])
            #     print '%s\t%.4f\t%.4f\t%.4f' % (cate, p, r, f)
            # p_all = 0
            # r_all = 0
            # for result in result_list:
            #     p_all += result[0] * result[1]
            #     r_all += result[0] * result[2]
            # p_all = p_all/len(y)
            # r_all = r_all/len(y)
            # print 'all\t%.4f\t%.4f\t%.4f' % (p_all, r_all, 2*p_all*r_all/(p_all+r_all))
            # print '------------------------------------------'

torch.save(model, './model.pkl')
