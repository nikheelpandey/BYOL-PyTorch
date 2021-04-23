#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import numpy as np
from math import pi, cos 


import torch
import torchvision
import torch.nn as nn
from logger import Logger
from torch import allclose
from datetime import datetime
import torch.nn.functional as tf 
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.testing import assert_allclose
from torchvision import datasets, transforms
from tqdm import tqdm

import kornia
from kornia import augmentation as K
import kornia.augmentation.functional as F
import kornia.augmentation.random_generator as rg
from torchvision.transforms import functional as tvF


# In[2]:


uid = 'byol'
dataset_name = 'stl10'
data_dir = 'dataset'
ckpt_dir = "./ckpt/"+str(datetime.now().strftime('%m%d%H%M%S'))
log_dir = "runs/"+str(datetime.now().strftime('%m%d%H%M%S'))

if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    


# In[3]:


# transformations

_MEAN =  [0.5, 0.5, 0.5]
_STD  =  [0.2, 0.2, 0.2]



class InitalTransformation():
    def __init__(self):
        self.transform = T.Compose([
            T.ToTensor(),
            transforms.Normalize(_MEAN,_STD),
        ])

    def __call__(self, x):
        x = self.transform(x)
        return  x


def gpu_transformer(image_size,s=.2):
        
    train_transform = nn.Sequential(

                kornia.augmentation.RandomResizedCrop(image_size,scale=(0.5,1.0)),
                kornia.augmentation.RandomHorizontalFlip(p=0.5),
                kornia.augmentation.ColorJitter(0.8*s,0.8*s,0.8*s,0.2*s,p=0.3),
                kornia.augmentation.RandomGrayscale(p=0.05),)

    test_transform = nn.Sequential(  
                kornia.augmentation.RandomResizedCrop(image_size,scale=(0.5,1.0)),
                kornia.augmentation.RandomHorizontalFlip(p=0.5),
                kornia.augmentation.ColorJitter(0.8*s,0.8*s,0.8*s,0.2*s,p=0.3),
                kornia.augmentation.RandomGrayscale(p=0.05),)

    return train_transform , test_transform
                



# In[4]:


def get_train_test_dataloaders(dataset = "stl10", data_dir="./dataset", batch_size = 64,num_workers = 4, download=True): 
    
    train_loader = torch.utils.data.DataLoader(
        dataset = torchvision.datasets.STL10(data_dir, split="train", transform=InitalTransformation(), download=download),
        shuffle=True,
        batch_size= batch_size,
        num_workers = num_workers
    )
    

    test_loader = torch.utils.data.DataLoader(
        dataset = torchvision.datasets.STL10(data_dir, split="test", transform=InitalTransformation(), download=download),
        shuffle=True,
        batch_size= batch_size,
        num_workers = num_workers
        )
    return train_loader, test_loader


# In[5]:


import copy
from torch import nn
import torchvision.models as models

def loss_fn(q1,q2, z1t,z2t):
    
    l1 = - tf.cosine_similarity(q1, z1t.detach(), dim=-1).mean()
    l2 = - tf.cosine_similarity(q2, z2t.detach(), dim=-1).mean()
    
    return (l1+l2)/2


class MLPHead(nn.Module):
    def __init__(self, in_dim, hidden_size=4096, projection_size=256):
        super(MLPHead, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)
    


class BYOL(nn.Module):
    def __init__(self, backbone=None,base_target_ema=0.996,**kwargs):
        super().__init__()
        self.base_ema = base_target_ema
        
        if backbone is None:
            backbone = models.resnet50(pretrained=False)
            backbone.output_dim = backbone.fc.in_features
            backbone.fc = torch.nn.Identity()

#         encoder = torch.nn.Sequential(*list(backbone.children())[:-1])
        projector = MLPHead(in_dim=backbone.output_dim)
        
        self.online_encoder = nn.Sequential(
            backbone,
            projector)
        
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.online_predictor = MLPHead(in_dim=256,hidden_size=1024, projection_size=256)
        
            

    @torch.no_grad()
    def update_moving_average(self, global_step, max_steps):
        
        tau = 1- ((1 - self.base_ema)* (cos(pi*global_step/max_steps)+1)/2) 
        
        for online, target in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target.data = tau * target.data + (1 - tau) * online.data     
    
    def forward(self,x1,x2):
        
        z1 = self.online_encoder(x1)
        z2 = self.online_encoder(x2)
        
        q1 = self.online_predictor(z1)
        q2 = self.online_predictor(z2)
        
        with torch.no_grad():
            z1_t = self.target_encoder(x1)
            z2_t = self.target_encoder(x2)
       
        loss = loss_fn(q1, q2, z1_t, z2_t)
        
        return loss


# In[6]:


if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    device = torch.device("cuda")
    # torch.cuda.set_device(device_id)
else:
    dtype = torch.FloatTensor
    device = torch.device("cpu")
    
print(device)


# In[7]:


weight_decay = 1.5e-6
warmup_epochs =  10
warmup_lr = 0
momentum = 0.9
lr =  0.002
final_lr =  0
epochs = 25
stop_at_epoch = 100
batch_size = 64
knn_monitor = False
knn_interval = 5
knn_k = 200
image_size = (92,92)


# In[8]:


train_loader, test_loader = get_train_test_dataloaders(batch_size=batch_size)
train_transform,test_transform = gpu_transformer(image_size)


# In[ ]:


from lr_scheduler import LR_Scheduler
from lars import LARS

loss_ls = []
acc_ls = []

model = BYOL().to(device)


optimizer = LARS(model.named_modules(), lr=lr, momentum=momentum, weight_decay=weight_decay)

        
scheduler = LR_Scheduler(
	    optimizer, warmup_epochs, warmup_lr*batch_size/8,

	    epochs, lr*batch_size/8, final_lr*batch_size/8, 
	    len(train_loader),
	    constant_predictor_lr=True 
	    )


min_loss = np.inf 
accuracy = 0

# start training 
logger = Logger(log_dir=log_dir, tensorboard=True, matplotlib=True)
global_progress = tqdm(range(0, epochs), desc=f'Training')
data_dict = {"loss": 100}

for epoch in global_progress:
    model.train()   
    local_progress = tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}')
    
    for idx, (image, label) in enumerate(local_progress):
        image = image.to(device)
        aug_image = train_transform(image)
 
        model.zero_grad()
        loss = model.forward(image.to(device, non_blocking=True), aug_image.to(device, non_blocking=True))

        loss_scaler = loss.item()
        data_dict['loss'] = loss_scaler
        loss_ls.append(loss_scaler)
        loss.backward()
        
        optimizer.step()
        model.update_moving_average(epoch, epochs)
        
        scheduler.step()
        
        data_dict.update({'lr': scheduler.get_last_lr()})
        local_progress.set_postfix(data_dict)
        logger.update_scalers(data_dict)
    
    current_loss = data_dict['loss']
    
    global_progress.set_postfix(data_dict)
    logger.update_scalers(data_dict)
    
    model_path = os.path.join(ckpt_dir, f"{uid}_{datetime.now().strftime('%m%d%H%M%S')}.pth")

    if min_loss > current_loss:
        min_loss = current_loss
        
        torch.save({
        'epoch':epoch+1,
        'state_dict': model.state_dict() }, model_path)
        print(f'Model saved at: {model_path}')


# In[ ]:





# In[ ]:




