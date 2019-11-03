from hyperopt import hp, tpe, fmin
from train import dataloader
from model import Net
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch
import numpy as np
from utils import bw_filter
import os
from random import shuffle

MAX_EVALS = 30
WIN_SIZE = 150
PATH = './'
RATIO_OF_SAMPLE = 5

# batch size is fixed to 9 sequence (3 * 3)
# input hyper-parameters and output model loss on dataset
def objective(hyperpara):
    if not (hyperpara['kernel_neuron1'] and hyperpara['kernel_neuron2'] and
            hyperpara['class_neuron1'] and hyperpara['class_neuron2'] and hyperpara['class_neuron3'] and
            hyperpara['epoch'] > 30):
        return 1
    training_path = PATH + './training/'
    model = Net(kernel_size=15, kernel_neuron=[hyperpara['kernel_neuron1'], hyperpara['kernel_neuron2']],
                acti_func='relu', dp_rate=hyperpara['dp_rate'],
                class_neuron=[hyperpara['class_neuron1'], hyperpara['class_neuron2'], hyperpara['class_neuron3']]
                ).double().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    batch_size = 3    # number of files trained in each time
    ep = hyperpara['epoch']
    for epoch in range(ep):
        filelist = os.listdir(training_path)
        file_list = []
        for item in filelist:
            if not item.startswith('.'):
                file_list.append(item)
        shuffle(os.listdir(training_path))
        batch_number = int(np.floor(len(file_list) / batch_size))
        for batch in range(batch_number):
            inputs = list()
            labels = list()
            for file_count in range(batch_size):
                train_seq = file_list[batch * batch_size + file_count]
                train_seq = training_path + train_seq
                seq, l = dataloader(train_seq)
                for s in range(len(seq)):
                    inputs.append(seq[s])
                    labels.append(l[s])

            for iter in range(len(labels)):
                for item in range(5):
                    # filter
                    inputs[iter][item] = bw_filter(inputs[iter][item])

            inputs = torch.from_numpy(np.array(inputs)).to(device)
            labels = torch.from_numpy(np.array(labels)).to(device)
            optimizer.zero_grad()

            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()

            optimizer.step()
        if epoch % 30 == 0:
            print('epoch:', epoch, float(loss))
    correct = 0
    for i, item in enumerate(testing_bucket):
        for k in range(5):
            item[k] = bw_filter(item[k])
        input = torch.from_numpy(np.array([item])).to(device)
        predict = model(input)
        if predict[0][0] >= predict[0][1]:
            predict = 0
        else:
            predict = 1
        label = int(testing_label[i])
        if predict == label:
            correct += 1

    result = correct / (i + 1)
    print('accuracy:', result)
    return -1 * result


data_path = './'
space = {
         'kernel_neuron1': hp.randint('kernel_neuron1', 200),
         'kernel_neuron2': hp.randint('kernel_neuron2', 200),
         'class_neuron1': hp.randint('class_neuron1', 200),
         'class_neuron2': hp.randint('class_neuron2', 200),
         'class_neuron3': hp.randint('class_neuron3', 200),
         'dp_rate': hp.uniform('dp_rate', 0.0, 1.0),
         'epoch': hp.randint('epoch', 500)
}

testing_path = PATH + './testing/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
testing_bucket = []
testing_label = []
filelist = os.listdir(testing_path)
file_list = []
for item in filelist:
    if not item.startswith('.'):
        file_list.append(item)
for file in file_list:
    file = testing_path + file
    seq, l = dataloader(file)
    for s in range(len(seq)):
        testing_bucket.append(seq[s])
        testing_label.append(l[s])

best = fmin(fn=objective, space=space, algo=tpe.suggest,max_evals=MAX_EVALS)
print(best)
