from model import Net
import torch
import json
from utils import bw_filter
import matplotlib.pyplot as plt
import numpy as np


WIN_SIZE = 150

model_path = './model/model.pth'
def HAR_predict(filepath):
    model = torch.load(model_path, map_location='cpu')

    with open(filepath, 'r') as f:
        data = json.load(f)

    data_length = len(data['ax'])
    segNum = int(data_length / WIN_SIZE)
    result = []

    for i in range(segNum):
        seq = []
        start = i * WIN_SIZE
        end = (i + 1) * WIN_SIZE

        seq.append(data['ax'][start:end])
        seq.append(data['ay'][start:end])
        seq.append(data['az'][start:end])

        seq.append(data['a'][start:end])

        g = data['g'][start:end]
        seq.append(g - np.mean(g))

        seq.append(data['m'][start:end])

        for k in range(5):
            seq[k] = bw_filter(seq[k])

        input = torch.from_numpy(np.array([seq])).to(device)
        predict = model(input)
        if predict[0][0] >= predict[0][1]:
            predict = 0
        else:
            predict = 1

        result.append(predict)
    return result

if __name__ == '__main__':
    filepath = './testing_samples/test.json'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = [0,3,6,9,12,15]
    gt = [0,0,0,0,1,1] # ground truth of test1 is [1,1,1,0,0,0]

    prediction = HAR_predict(filepath)

    fig, axs = plt.subplots(2, 1, constrained_layout=True)
    axs[0].plot(x, gt, 'r--')
    axs[0].set_title('ground truth')
    axs[0].set_xlabel('t(s)')
    axs[0].set_ylabel('texting while walking or not')

    axs[1].plot(x, prediction, 'r--')
    axs[1].set_xlabel('t (s)')
    axs[1].set_title('prediction')
    axs[1].set_ylabel('texting while walking or not')

    plt.show()
