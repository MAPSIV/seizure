## 2 CLASSES

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve
from torchsummary import summary

import pandas as pd
import os

print("DOS CLASES\n")
print('Check GPU runtime type... ')
use_cuda = torch.cuda.is_available()
if not use_cuda:
  device = "cpu"
  print('Change Runtype Type in top menu for GPU acceleration')
else:
  device = "cuda"
  print('OK!')

#BALANCEJATS

feat_list = '/export/home/mapiv02/seizure/datos_red/windows_balanced/'
gt_path = '/export/home/mapiv02/seizure/datos_red/metadatos_balanced/'

save_model_path = '/export/home/mapiv02/seizure/datos_red/Mejores3/'

windows_dict = {}

tags_dict = {}

for filename1, filename2 in zip(sorted(os.listdir(feat_list)), sorted(os.listdir(gt_path))):
  numero = int(filename1.split('_')[0][3:])
  print(numero)
  print(filename1)
  print(filename2)
  metrics_windows = np.load(feat_list + '/' + filename1)["windows"]
  targets = np.load(gt_path + '/' + filename2)['tag']
  windows = []
  targets_list = []

  for window,tag in zip(metrics_windows, targets):
    if tag == 0 or tag == 1:
      #metrics = window.transpose()
      windows.append(window)
      targets_list.append(tag)
  
  print(len(windows), windows[0].shape)
  print(len(targets_list), targets_list[0].shape)

  windows_dict[numero] = windows
  tags_dict[numero] = targets_list


#NO BALANCEJATS

feat_list_originals = '/export/home/mapiv02/seizure/datos_red/windows/'
gt_path_originals = '/export/home/mapiv02/seizure/datos_red/metadatos/'


windows_dict_originals = {}
tags_dict_originals = {}


for filename1, filename2 in zip(sorted(os.listdir(feat_list_originals)), sorted(os.listdir(gt_path_originals))):
  numero = int(filename1.split('_')[0][3:])
  print("Originals")
  print(numero)
  print(filename1)
  print(filename2)

  metrics_windows_originals = np.load(feat_list_originals + '/' + filename1)["windows"]
  targets_originals =  pd.read_parquet(gt_path_originals + '/' + filename2)["tag"].values 
  windows_originals = []
  targets_list_originals = []

  for window,tag in zip(metrics_windows_originals, targets_originals):
    if tag == 0 or tag == 1:
      metrics = window.transpose()
      windows_originals.append(metrics)
      targets_list_originals.append(tag)
  
  print(len(windows_originals), windows_originals[0].shape)
  print(len(targets_list_originals), targets_list_originals[0].shape)
  
  windows_dict_originals[numero] = windows_originals
  tags_dict_originals[numero] = targets_list_originals


 
train_kwargs = {'batch_size': 1024}
test_kwargs  = {'batch_size': 1024}

if use_cuda:
    cuda_kwargs = {'num_workers': 1,
                   'pin_memory': True,
                   'shuffle': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.channels = 21
        self.L = 128
        
        self.backbone = nn.Sequential(
          nn.AvgPool2d((self.channels,1)), 

          nn.Conv1d(1, 16, kernel_size = 3, stride=1, padding = 1),
          nn.ReLU(),
          nn.MaxPool1d(kernel_size=2, stride=2),

          nn.Conv1d(16, 32, kernel_size = 3, stride=1, padding = 1),
          nn.ReLU(),
          nn.MaxPool1d(kernel_size=2, stride=2),

          nn.Conv1d(32, 64, kernel_size = 3, stride=1, padding = 1), #padding per no perdre 2 entrades de larray i que es mantinguin
          nn.ReLU(),
          nn.MaxPool1d(kernel_size=2, stride=2),

          nn.Flatten(),
          
          nn.Linear(1024,self.L),
          nn.Linear(self.L,2)

          )
        

    def forward(self, x):
        features = self.backbone(x)

        return features

#summary(Net(),(21,128),device="cpu")
def train(model, device, train_loader_windows, train_loader_targets, optimizer, epoch, scheduler=None):
    model.train()
    loss_values = []
    min_loss = 1000000000
    for (batch_idx, sample_batched), (_, sample_batched2) in zip(enumerate(train_loader_windows), enumerate(train_loader_targets)):
        data = sample_batched
        target = sample_batched2
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(output, target)
        
        if loss < min_loss:
            min_loss = loss.detach().cpu()
            torch.save(model.state_dict(), os.path.join(save_model_path, 'model_fold_best_{}.pth'.format(fold)))

        loss_values.append(loss.detach().cpu())
        loss.backward()
        optimizer.step()
        
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tMin Loss:{:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader_windows.dataset),
                100. * batch_idx / len(train_loader_windows), loss.item(), min_loss))
        if scheduler is not None:
            scheduler.step()
    return loss_values

"""
def test(model, device, test_loader_windows, test_loader_targets):
    model.eval()
    test_loss = 0
    correct = 0
    accuracy = 0
    precision = 0
    recall = 0
    f1 = 0
    roc = 0
    loops = 0
    with torch.no_grad():
        for (batch_idx, sample_batched), (_, sample_batched2) in zip(enumerate(test_loader_windows), enumerate(test_loader_targets)):
            data = sample_batched
            target = sample_batched2
            data, target = data.to(device), target.to(device)
            output = model(data)

            criterion = torch.nn.CrossEntropyLoss()
            test_loss = criterion(output, target)

            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

            accuracy += accuracy_score(target.cpu(), pred.cpu())
            precision += precision_score(target.cpu(), pred.cpu(), zero_division = 1)
            recall += recall_score(target.cpu(), pred.cpu(), zero_division = 1)
            f1 += f1_score(target.cpu(), pred.cpu(), zero_division = 1)
            if loops == 0:
              matriu = confusion_matrix(target.cpu(), pred.cpu(), labels=[0, 1])
              #print("MATRIU", matriu)
            else:
              m = confusion_matrix(target.cpu(), pred.cpu(), labels=[0, 1])
              #print("M", m)
              matriu[0][0] += m[0][0]
              matriu[0][1] += m[0][1]
              matriu[1][0] += m[1][0]
              matriu[1][1] += m[1][1]
            #roc += roc_curve(target.cpu(), pred.cpu())
            loops += 1

    test_loss /= (len(test_loader_windows.dataset)/test_loader_windows.batch_size)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader_windows.dataset),
        100. * correct / len(test_loader_windows.dataset)))
    print('Accuracy:', round(accuracy/loops,3))
    print('Precision:', round(precision/loops,3))
    print('Recall:', round(recall/loops,3))
    print('F1:', round(f1/loops,3))
    print('ROC:', round(roc/loops,3))
    matriu_confusio = pd.DataFrame(matriu)
    matriu_confusio.index.name = 'True'
    print(matriu_confusio)
"""
def test(model, device, test_loader_windows, test_loader_targets):
    model.eval()
    test_loss = 0
    correct = 0
    accuracy = 0
    precision = 0
    recall = 0
    f1 = 0
    roc = 0
    loops = 0
    with torch.no_grad():
        for (batch_idx, sample_batched), (_, sample_batched2) in zip(enumerate(test_loader_windows), enumerate(test_loader_targets)):
          data = sample_batched
          target = sample_batched2
          data, target = data.to(device), target.to(device)
          output = model(data)

          criterion = torch.nn.CrossEntropyLoss()
          test_loss = criterion(output, target)

          pred = output.data.max(1, keepdim=True)[1]

          if loops == 0:
            T_preds = list(np.squeeze(pred.cpu().numpy()))
            T_targets = list(target.cpu().numpy())
          else:
            T_preds += list(np.squeeze(pred.cpu().numpy()))
            T_targets += list(target.cpu().numpy())

          loops += 1
          correct += pred.eq(target.view_as(pred)).sum().item()
        
        print(np.unique(T_preds))
        print(np.unique(T_targets))

        accuracy = accuracy_score(T_targets, T_preds)
        precision = precision_score(T_targets, T_preds,zero_division=1)
        recall = recall_score(T_targets, T_preds,zero_division=1)
        f1 = f1_score(T_targets, T_preds,zero_division=1)

        matriu = confusion_matrix(T_targets, T_preds, labels=[0,1])


    test_loss /= (len(test_loader_windows.dataset)/test_loader_windows.batch_size)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader_windows.dataset),
        100. * correct / len(test_loader_windows.dataset)))
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1:', f1)

    matriu_confusio = pd.DataFrame(matriu)
    matriu_confusio.index.name = 'True'
    print(matriu_confusio)

import copy
for fold in range(1,25):
  print('\n')
  print("TEST PACIENT",fold)
  print("UNIQUE VALUES", np.unique(tags_dict[fold]))


  copia_windows_dict = copy.deepcopy(windows_dict)
  copia_windows_dict.pop(fold)

  dataset_train = np.concatenate(list(copia_windows_dict.values()))
  dataset_test = windows_dict_originals[fold]

  copia_tags_dict = copy.deepcopy(tags_dict)
  copia_tags_dict.pop(fold)

  target_train = np.concatenate(list(copia_tags_dict.values())) 
  target_test = tags_dict_originals[fold]

  train_loader_windows = DataLoader(dataset_train, batch_size=2048, drop_last=True)
  train_loader_targets = DataLoader(target_train, batch_size=2048, drop_last=True)

  test_loader_windows = DataLoader(dataset_test, batch_size=2048, drop_last=True)
  test_loader_targets = DataLoader(target_test, batch_size=2048, drop_last=True)

  model = Net().to(device)


  optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.1)

  epochs = 20
  scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.00001, max_lr=0.001)

  log_interval = 16 # how many batches to wait before logging training status

  loss_history = []
  for epoch in range(1, epochs + 1):
      loss_values = train(model, device, train_loader_windows, train_loader_targets , optimizer, epoch, scheduler)
      loss_history += loss_values
      
  torch.save(model.state_dict(), os.path.join(save_model_path, 'model_fold_{}.pth'.format(fold)))


  test(model, device, test_loader_windows, test_loader_targets)
  
  model = Net().to(device)
  model.load_state_dict(torch.load(os.path.join(save_model_path, 'model_fold_best_{}.pth'.format(fold)), map_location=torch.device('cpu')))
  model.eval()
  
  test(model, device, test_loader_windows, test_loader_targets)

  
  #plt.figure()
  #plt.plot(np.arange(len(loss_history)), loss_history)
  #plt.title(f'Training loss pacient {fold} fold\n')
  #plt.savefig(f'/export/home/mapiv02/seizure/graficos_red_2/patient_{fold}_fold.png')
