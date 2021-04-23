import copy
import torch 
import numpy as np
import torch.nn as nn
from math import pi, cos 
import torch.nn.functional as tf 
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
