import torch
import os
import math
class dataset(): #TODO: decide on inheritance
    def __init__(self, files_to_ignore=["dataset.py", "description.txt","documentation.txt", "__pycache__"], labels = "profile.txt", classification = 0, dir = "../datasets/Hydraulic"):
        self.class_num = classification
        self.dir = dir
        files = os.listdir(dir)
        files.remove(labels)
        files.sort()
        files =  [f for f in files if f not in files_to_ignore]
        D = []
        L = []
        for f in files:
            try:
                with open(os.path.join(dir, f), "r") as source:
                    print(f"loading file: {f}")
                    content = source.readlines()
                    content = [line.split("\t") for line in content]
                    content = torch.tensor([[float(c) for c in line] for line in content], dtype=torch.float32)
                    D.append(content)
            except:
                continue
        with open(os.path.join(dir, labels), "r") as source:
                content = source.readlines()
                content = [line.split("\t") for line in content]
                content = torch.tensor([[float(c) for c in line] for line in content], dtype=torch.float32)
                L = content
        self.data = torch.cat(tuple(D), dim=1), L #NOTE: Labels (L) has 5 different classification tasks, parameter classification would determine which one to return
        
        #NOTE: needs to rewrite the __getitem_() function

    def __getitem__(self, idx):
        return self.data[0][idx], self.data[1][idx][self.class_num]
    
    def __len__(self):
        return self.data[0].shape[0]

    def set_classification_task(classification: int = 0):
        self.class_num = classification
    
    def split(self, ratio:int = 0.5):
        n = self.data[0].shape[0] * ratio
        r = torch.randperm(self.data[0].shape[0])
        train = subdataset(self.data[0][r[:int(math.floor(n))]], self.data[1][r[:int(math.floor(n))]], self.class_num)
        test =  subdataset(self.data[0][r[int(math.floor(n)) + 1:]], self.data[1][r[int(math.floor(n)) + 1:]], self.class_num)
        return train,test
    
    def to_type(self, data_type): #changes label data types
        self.data = self.data[0], torch.tensor(self.data[1], dtype=data_type) 
    
    def fit_classification(self):
        #self.to_type(torch.int64)
        values = []
        for i in range(2205):
            val = self.data[1][i][self.class_num]
            if val not in values:
                values.append(int(val))
            #self.data[1][i][self.class_num] = torch.tensor(values.index(val), dtype=torch.int64)
        #print(values)
        for i in range(self.data[1].shape[0]):
            self.data[1][i][self.class_num] = torch.tensor(values.index(int(self.data[1][i][self.class_num])), dtype=torch.int64)
    
    
class subdataset(dataset):
    def __init__(self, data, labels, class_num):
        self.data = data, labels
        self.class_num = class_num

class gen_dataset(dataset):
    def __init__(self, data_file = "data.csv" , labels = "labels.csv", classification = 0, dir = "../datasets/TCGA-PANCAN-HiSeq-801x20531"):
        self.class_num = 0
        content = []
        with open(os.path.join(dir,data_file)) as f:
            lines = f.readlines()
            content = [line.split(",")[1:] for line in lines[1:]]
            content = [[float(c) for c in line] for line in content]
        with open(os.path.join(dir,labels)) as f:
            lines = f.readlines()
            labels = [line.split(",")[1:] for line in lines[1:]]
        values = []
        for i in range(len(labels)):
            val = labels[i][0]
            if val not in values:
                values.append(val)
            labels[i][0] = values.index(val)

        self.data = torch.tensor(content, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)



    
if "main" in __name__:
    #d = dataset("../datasets/Hydraulic")
    d = gen_dataset()
    a,b = d[0]
    print(d.data[0].shape)
    print(d.data[1].reshape(801))
