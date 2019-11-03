import json
import numpy as np
import os
import random
import csv
from model import Net
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from utils import bw_filter
from random import shuffle

WIN_SIZE = 150
SEQ_NUM = 3
PATH = './'

# put data into two bucket in terms of label, only consider one transient activity
def dataloader(path):
    with open(path, 'r') as f:
        data = json.load(f)

    result = []
    label = []
    length = len(data['ax']) - WIN_SIZE - 1

    for i in range(SEQ_NUM):
        seq = []
        seed = random.randint(0, length)
        start = seed
        end = seed + WIN_SIZE

        seq.append(data['ax'][start:end])
        seq.append(data['ay'][start:end])
        seq.append(data['az'][start:end])

        seq.append(data['a'][start:end])

        g = data['g'][start:end]
        seq.append(g - np.mean(g))

        seq.append(data['m'][start:end])

        label.append(data['label'])
        result.append(seq)
    return result, label


def preprocessing(source, target ='./training/'):
    if not os.path.exists(target):
        os.mkdir(target)

    file_list = os.listdir(source)

    for i, file_name in enumerate(file_list):
        if file_name.startswith('.'):
            continue
        raw = []
        result = {'ax': [], 'ay': [], 'az': [], 'a': [], 'gx': [], 'gy': [], 'gz': [], 'g': [], 'm': [], 'label': 0}
        with open(source+file_name, 'r') as f:
            reader = csv.reader(f)
            for item in reader:
                raw.append(item)
        for num, item in enumerate(raw):
            if num == 0:
                temp = list(item[-1])
                while(' ' in temp):
                    temp.remove(' ')
                while ('(' in temp):
                    temp.remove('(')
                while (')' in temp):
                    temp.remove(')')
                result['label'] = int(temp[0])
            else:
                result['gx'].append(float(item[1]))
                result['gy'].append(float(item[2]))
                result['gz'].append(float(item[3]))
                result['g'].append(float(item[3]))

                result['ax'].append(float(item[9]))
                result['ay'].append(float(item[10]))
                result['az'].append(float(item[11]))
                result['a'].append(float(item[8]))

                result['m'].append(float(item[17]))

        save = target + str(i) + '.json'

        with open(save, 'w') as f:
            json.dump(result, f)


def trainModel(n1, n2, dp, c1, c2, c3, ep):
    training_path = PATH + './training/'
    model = Net(kernel_size=15, kernel_neuron=[n1, n2], acti_func='relu', dp_rate=dp,
                class_neuron=[c1, c2, c3]).double().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    batch_size = 3  # number of files trained in each time
    for epoch in range(ep):
        file_list = os.listdir(training_path)
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
    print('training done!')

    torch.save(model, './model/model.pth')

if __name__ == "__main__":

    c1 = 70
    c2 = 150
    c3 = 42
    dp = 0.35
    ep = 300
    n1 = 64
    n2 = 172

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

    trainModel(n1, n2, dp, c1, c2, c3, ep)


