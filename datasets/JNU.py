import os
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from .SequenceDatasets import dataset
from .sequence_aug import *
from tqdm import tqdm

signal_size = 3072



#Three working conditions

dataname= {0:["ib600_2.csv","n600_3_2.csv","ob600_2.csv","tb600_2.csv"],
           1:["ib800_2.csv","n800_3_2.csv","ob800_2.csv","tb800_2.csv"],
           2:["ib1000_2.csv","n1000_3_2.csv","ob1000_2.csv","tb1000_2.csv"]}

label = [i for i in range(0,4)]


#generate Training Dataset and Testing Dataset
def get_files(root, N):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    '''
    
    data = []
    lab =[]
    for k in range(len(N)):
        for i in tqdm(range(len(dataname[N[k]]))):
            path1 = os.path.join('/tmp',root,dataname[N[k]][i])
            data1, lab1 = data_load(path1,label=label[i])
            data += data1
            lab +=lab1


    return [data, lab]

# def data_load(filename,label):
#     '''
#     This function is mainly used to generate test data and training data.
#     filename:Data location
#     '''
#     fl = np.loadtxt(filename)
#     fl = fl.reshape(-1,1)
#     data=[]
#     lab=[]
#     start,end=0,signal_size
#     while end<=fl.shape[0]:
#         data.append(fl[start:end])
#         lab.append(label)
#         start +=signal_size
#         end +=signal_size
#
#     return data, lab


def data_load(filename, label, num_samples=1000, signal_size=3072):
    '''
    This function is mainly used to generate test data and training data with overlapping sampling.
    filename: Data location
    label: Label for the data
    num_samples: Number of samples to generate (default 1000)
    signal_size: Size of each signal sample (default 3072)
    '''
    fl = np.loadtxt(filename)
    fl = fl.reshape(-1, 1)

    data = []
    lab = []

    total_data_length = fl.shape[0]

    # If data is too short to generate even one sample
    if total_data_length < signal_size:
        return data, lab  # Return empty lists

    if num_samples <= 1:
        # Just take the first signal_size samples
        data.append(fl[:signal_size])
        lab.append(label)
        return data, lab

    # Calculate step size to generate approximately num_samples
    # Total span needed: (num_samples - 1) * step + signal_size <= total_data_length
    # So step <= (total_data_length - signal_size) / (num_samples - 1)
    step = int((total_data_length - signal_size) / (num_samples - 1))  # Downward rounding

    if step <= 0:
        # If step is 0 or negative, set it to 1 to generate as many samples as possible
        step = 1

    start = 0
    count = 0

    while count < num_samples and start + signal_size <= total_data_length:
        end = start + signal_size

        # Extract the current sample
        current_signal = fl[start:end]

        # Only add if we have enough data (length equals signal_size)
        if current_signal.shape[0] == signal_size:
            data.append(current_signal)
            lab.append(label)
            count += 1

        # Update start position for next iteration
        start += step

    return data, lab
#--------------------------------------------------------------------------------------------------------------------
class JNU(object):
    num_classes = 4
    inputchannel = 1

    def __init__(self, data_dir, transfer_task, normlizetype="0-1"):
        self.data_dir = data_dir
        self.source_N = transfer_task[0]
        self.target_N = transfer_task[1]
        self.normlizetype = normlizetype
        self.data_transforms = {
            'train': Compose([
                Reshape(),
                Normalize(self.normlizetype),
                # RandomAddGaussian(),
                # RandomScale(),
                # RandomStretch(),
                # RandomCrop(),
                Retype(),
                # Scale(1)
            ]),
            'val': Compose([
                Reshape(),
                Normalize(self.normlizetype),
                Retype(),
                # Scale(1)
            ])
        }

    def data_split(self, transfer_learning=True):
        if transfer_learning:
            # get source train and val
            list_data = get_files(self.data_dir, self.source_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split(data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"])
            source_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            source_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])

            # get target train and val
            list_data = get_files(self.data_dir, self.target_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split(data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"])
            target_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            target_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])
            return source_train, source_val, target_train, target_val
        else:
            #get source train and val
            list_data = get_files(self.data_dir, self.source_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split(data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"])
            source_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            source_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])

            # get target train and val
            list_data = get_files(self.data_dir, self.target_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            target_val = dataset(list_data=data_pd, transform=self.data_transforms['val'])
            return source_train, source_val, target_val



