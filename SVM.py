from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import torch
import pickle
import numpy as np
from torch.autograd import Variable
from Util import adjust_xe

VOCAB_DIR = './data/vocab.pkl'
MODEL_PATH = './model.pkl'
EVENT_INPUT_PATH = './data/event_input'
CATEGORY_DICT = {'statement':0, 'operation':1, 'emergency':2,'perception':3,
                 'stateChange':4, 'movement':5, 'action':6}

model = torch.load(MODEL_PATH)
datalist = pickle.load(open(VOCAB_DIR, 'r'))
word2idx = datalist[0]

knn = KNeighborsClassifier()
svm = SVC(kernel='linear', C=0.4)

fopen = open(EVENT_INPUT_PATH, 'r')
x_list = []
y_list = []
xe_list = []

for line in fopen.readlines():
    contents = line.split('\t')
    category = contents[0].strip()
    event_content = contents[1].strip()
    x_e = contents[2].strip()
    x = [word2idx[w] for w in event_content.split(' ')]
    x_e = [int(w) for w in x_e.split(' ')]
    x_e = adjust_xe(x_e)

    y = CATEGORY_DICT[category]
    x_list.append(x)
    xe_list.append(x_e)
    y_list.append(y)

event_x = np.array(x_list)
event_y = np.array(y_list)
xe_list = np.array(xe_list)
event_x = torch.from_numpy(event_x)
event_x = Variable(event_x)
event_xe = torch.from_numpy(xe_list)
event_xe = torch.unsqueeze(event_xe, 1)
event_xe = Variable(event_xe)


event_x = model.generate_event_embedding(event_x, event_xe)
event_x = event_x.data.numpy()

x_train, x_test, y_train, y_test = train_test_split(
    event_x, event_y, test_size=0.2)

knn.fit(x_train, y_train)
svm.fit(x_train, y_train)
target_names = []
for target_name in CATEGORY_DICT.keys():
    target_names.append(target_name)

pred = knn.predict(x_test)

print 'KNN Classification Report'
print metrics.classification_report(y_test, pred, target_names=target_names, digits=4)

pred = svm.predict(x_test)
print 'SVM Classification Report'
print metrics.classification_report(y_test, pred, target_names=target_names, digits=4)