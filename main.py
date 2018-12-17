import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import dataset
from prune import *
from heapq import nsmallest
from operator import itemgetter
import numpy as np

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
		    nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(8))
        
        self.classifier = nn.Sequential(nn.Linear(256, 10))
        

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

class FilterPrunner:
    def __init__(self, model):
        self.model = model
        self.reset()
        
    def reset(self):
        self.filter_ranks = {}
        
    def forward(self, x):
        self.activations = []
        self.gradients = []
        self.grad_index = 0
        self.activation_to_layer = {}
        
        activation_index = 0
        for layer, (name, module) in enumerate(self.model.features._modules.items()):
            x = module(x)
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                x.register_hook(self.compute_rank)
                self.activations.append(x)
                self.activation_to_layer[activation_index] = layer
                activation_index += 1
                
        return self.model.classifier(x.view(x.size(0), -1))
    
    def compute_rank(self, grad):
        activation_index = len(self.activations) - self.grad_index - 1
        activation = self.activations[activation_index]
        
        values = torch.sum((activation * grad), dim=0).\
            sum(dim=1).sum(dim=1).data
        values /= (activation.size(0) * activation.size(2) * activation.size(3))
        
        if activation_index not in self.filter_ranks:
            self.filter_ranks[activation_index] = \
                torch.FloatTensor(activation.size(1)).zero_().cuda()
        
        self.filter_ranks[activation_index] += values
        self.grad_index += 1
        
    def lowest_ranking_filters(self, num):
        data = []
        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                data.append((self.activation_to_layer[i], j, self.filter_ranks[i][j]))
                
        return nsmallest(num, data, itemgetter(2))
    
    def normalize_ranks_per_layer(self):
        for i in self.filter_ranks:
            v = torch.abs(self.filter_ranks[i]).cpu()
            v = v / np.sqrt(torch.sum(v * v))
            self.filter_ranks[i] = v
        
    def get_prunning_plan(self, num_filters_to_prune):
        filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune)
        
        filters_to_prune_per_layer = {}
        for (l, f, _) in filters_to_prune:
            if l not in filters_to_prune_per_layer:
                filters_to_prune_per_layer[l] = []
            filters_to_prune_per_layer[l].append(f)
        
        for l in filters_to_prune_per_layer:
            filters_to_prune_per_layer[l] = sorted(filters_to_prune_per_layer[l])
            for i in range(len(filters_to_prune_per_layer[l])):
                filters_to_prune_per_layer[l][i] = filters_to_prune_per_layer[l][i] - i 
        
        filters_to_prune = []
        for l in filters_to_prune_per_layer:
            for i in filters_to_prune_per_layer[l]:
                filters_to_prune.append((l, i))
        
        return filters_to_prune
    
class PrunningFineTuner_CNN:
    def __init__(self, model):
        self.train_data_loader = dataset.trainloader()
        self.test_data_loader = dataset.testloader()
        
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.prunner = FilterPrunner(self.model)
        self.model.train()
        
    def test(self):
        self.model.eval()
        correct = 0
        total = 0
        
        for i, (batch, label) in enumerate(self.test_data_loader):
            batch = batch.cuda()
            output = self.model(Variable(batch))
            pred = output.data.max(1)[1]
            correct += pred.cpu().eq(label).sum()
            total += label.size(0)
            
        print("Accuracy :", float(correct) / total)
        
        self.model.train()
    
    def train(self, optimizer = None, epoches = 10):
        if optimizer is None:
            optimizer = optim.SGD(self.model.parameters(), lr=0.0001, momentum=0.9)
        
        for i in range(epoches):
            print("Epoch: ", i)
            self.train_epoch(optimizer)
            self.test()
        print("Finished fine tuning.")
    
    def train_epoch(self, optimizer = None, rank_filters = False):
        for batch, label in self.train_data_loader:
            self.train_batch(optimizer, batch.cuda(), label.cuda(), rank_filters)
    
    def train_batch(self, optimizer, batch, label, rank_filters):
        self.model.zero_grad()
        input = Variable(batch)
        
        if rank_filters:
            output = self.prunner.forward(input)
            self.criterion(output, Variable(label)).backward()
        else:
            self.criterion(self.model(input), Variable(label)).backward()
            optimizer.step()
    
    def get_candidates_to_prune(self, num_filters_to_prune):
        self.prunner.reset()
        self.train_epoch(rank_filters = True)
        self.prunner.normalize_ranks_per_layer()
        return self.prunner.get_prunning_plan(num_filters_to_prune)
    
    def total_num_filters(self):
        filters = 0
        for name, module in self.model.features._modules.items():
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                filters = filters + module.out_channels
        return filters
    
    def prune(self):
        self.test()
        self.model.train()
        
        for param in self.model.parameters():
            param.requires_grad = True
            
        number_of_filters = self.total_num_filters()
        num_filters_to_prune_per_iteration = 32
        iterations = int(float(number_of_filters) / num_filters_to_prune_per_iteration)
        
        iterations = int(iterations * 9.0 / 10)
        print("Number of prunning iterations to reduce 90% filters", iterations)
        
        Layers_Prunned = []
        Acc = []
        for epoch_no in range(iterations):
            print("Ranking filters.. ")
            prune_targets = self.get_candidates_to_prune(num_filters_to_prune_per_iteration)
            layers_prunned = {}
            for layer_index, filter_index in prune_targets:
                if layer_index not in layers_prunned:
                    layers_prunned[layer_index] = 0
                layers_prunned[layer_index] = layers_prunned[layer_index] + 1
            
            print("Layers that will be prunned", layers_prunned)
            print("Prunning filters.. ")
            
            model = self.model.cpu()
            for layer_index, filter_index in prune_targets:
                
                model = prune_conv_layer(model, layer_index, filter_index)
                
            self.model = model.cuda()
            
            message = str(100*float(self.total_num_filters()) / number_of_filters) + "%"
            print("Filters prunned", str(message))
            self.test()
            print("Fine tuning to recover from prunning iteration.")
            optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
            self.train(optimizer, epoches = 25)
            torch.save(model, "model_prunned_"+str(epoch_no))
            
            self.model.eval()
            correct = 0
            total = 0
            for i, (batch, label) in enumerate(self.test_data_loader):
                batch = batch.cuda()
                output = self.model(Variable(batch))
                pred = output.data.max(1)[1]
                correct += pred.cpu().eq(label).sum()
                total += label.size(0)
            acc = float(correct) / total
            Acc.append(acc)
            Layers_Prunned.append(layers_prunned)
            
        print("Finished. Going to fine tune the model a bit more")
        #self.train(optimizer, epoches = 15)
        torch.save(model, "model_prunned")
        return Acc, Layers_Prunned
        

"""
model = CNN().cuda()
fine_tuner = PrunningFineTuner_CNN(model)

opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
fine_tuner.train(optimizer = opt, epoches = 10)
opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
fine_tuner.train(optimizer = opt, epoches = 15)
opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
fine_tuner.train(optimizer = opt, epoches = 15)

torch.save(model, "model")
"""

model = torch.load("model").cuda()
fine_tuner = PrunningFineTuner_CNN(model)
Acc, Layers_Prunned = fine_tuner.prune()

"""
net = torch.load('model_prunned')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
testloader = dataset.testloader()


import time
time_start=time.time()
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %f %%' % (
    100 * correct / total))


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2f %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
time_end=time.time()
print('Time cost:',time_end-time_start, "s")
"""