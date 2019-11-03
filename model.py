import torch.nn as nn
import torch.nn.functional as f
import torch
import numpy as np
import utils

WIN_LEN = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Net(nn.Module):
    def __init__(self, kernel_size=5, kernel_neuron=[10, 10], acti_func='relu',dp_rate=0,
                 class_neuron=[10, 10, 10], freq=50):
        super(Net, self).__init__()
        self.freq = freq
        self.acti_func = acti_func
        self.dp_rate = dp_rate
        self.kernel_size = kernel_size
        self._padding = int((self.kernel_size-1)/2)
        self.conv1 = nn.Conv1d(in_channels=6, out_channels=kernel_neuron[0], kernel_size=self.kernel_size,
                               padding=self._padding)
        self.conv2 = nn.Conv1d(in_channels=kernel_neuron[0], out_channels=kernel_neuron[1],
                               kernel_size=self.kernel_size, padding=self._padding)
        fcInput = 6
        extra = 8

        fcInput = fcInput * kernel_neuron[-1] + extra

        self.fc1 = nn.Linear(fcInput, class_neuron[0])
        self.fc2 = nn.Linear(class_neuron[0], class_neuron[1])
        self.fc3 = nn.Linear(class_neuron[1], class_neuron[2])
        self.fc4 = nn.Linear(class_neuron[2], 2)


    def forward(self, x):
        extra_features = self.extraFea(x)
        extra = torch.from_numpy(np.array([extra_features]).squeeze())
        if np.array(extra).ndim == 1:
            extra = torch.from_numpy(np.array([extra.numpy()]))
        extra = extra.to(device)

        x = f.max_pool1d(f.relu(self.conv1(x)), kernel_size=5)
        x = f.max_pool1d(f.relu(self.conv2(x)), kernel_size=5)
        x = x.view(-1, self.num_flat_features(x))
        x = torch.cat((x, extra), 1)

        x = f.relu(self.fc1(x))
        x = f.dropout(x, p=self.dp_rate)
        x = f.relu(self.fc2(x))
        x = f.dropout(x, p=self.dp_rate)
        x = f.relu(self.fc3(x))
        x = f.dropout(x, p=self.dp_rate)
        x = f.relu(self.fc4(x))

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_feas = 1
        for s in size:
            num_feas *= s
        return num_feas

    def extraFea(self, x):
        output = []
        input = x
        for x in input:
            result = []
            x = abs(x[3].cpu().numpy())
            result.append(utils.mean_abs_dev(x))
            result.append(utils.average(x))
            result.append(utils.fre_skewness(x))
            result.append(utils.fre_kurtosis(x))
            result.append(utils.energy(x))
            result.append(utils.entropy(x))
            #temp = utils.ar_coef(x)
            #for i in temp:
            #    result.append(i)
            x1 = x[0: int(len(x) / 2)]
            x2 = x[int(len(x) / 2):]
            result.append(utils.correlation(x1, x2))
            result.append(utils.fswa(x))
            output.append([result])
        return output

