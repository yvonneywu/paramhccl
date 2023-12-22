import numpy as np 
import torch
# print(torch.__version__)
# print(np.__version__)
import os 

data_path = './data/ECG/'
train_dataset = torch.load(os.path.join(data_path, "test.pt"))
X_train = train_dataset["samples"]
print(X_train.shape)