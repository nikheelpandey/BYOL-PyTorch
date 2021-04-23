import torch 
import torch.nn as nn
import torchvision.models as models 


class ProjectionHead(nn.Module):
    
    def __init__(self,in_channel,hidden_size, projection_size):
        super(ProjectionHead,self).__init__()

        # Linear>BN>ReLU>Linear
        self.net = nn.Sequential(
            nn.Linear(in_channel, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size)
        )

    def forward(self,x):
        return self.net(x)


class BaseResNet(nn.Module):
    def __init__(self,*args, **kwargs):
        super(BaseResNet,self).__init__()
        if kwargs['name']== 'resnet18':
            resnet = models.resnet18(pretrained=False)
        if kwargs['name']== 'resnet34':
            resnet = models.resnet34(pretrained=False)
        if kwargs['name']== 'resnet50':
            resnet = models.resnet50(pretrained=False)
        
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.projection = ProjectionHead(in_channel=resnet.fc.in_features, **kwargs['projection'])
    
    def forward(self,x):
        h =self.encoder(x)
        h = h.reshape(h.shape[0],h.shape[1])
        return self.projection(h)
