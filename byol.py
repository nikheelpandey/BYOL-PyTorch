import os 
import torch 
import torchvision
import torch.nn.functional as F
from torch.utils.data.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader

class BYOL(object):
    def __init__(self, 
                online_network,
                target_network,
                optimizer,
                device,
                predictor):

        self.online_network = None
        self.target_network = None
        self.optimizer  = None
        self.device     = None
        self.predictor  = None
    

    def train(self):
        pass