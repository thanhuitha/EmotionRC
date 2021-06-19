import torch
import torch.nn as nn
from torchvision import models
import cv2


class Shuffle_self_curve(nn.Module):
    def __init__(self, num_classes = 7, drop_rate = 0):
        super(Shuffle_self_curve, self).__init__()
        self.drop_rate = drop_rate
        resnet  = models.shufflenet_v2_x1_0(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1],
                                      nn.AdaptiveAvgPool2d((1,1))) # after avgpool 512x1

        fc_in_dim = list(resnet.children())[-1].in_features # original fc layer's in dimention 512
   
        self.fc = nn.Linear(fc_in_dim, num_classes) # new fc layer 512x7
        self.alpha = nn.Sequential(nn.Linear(fc_in_dim, 1),nn.Sigmoid())

    def forward(self, x):
        x = self.features(x)
        
        if self.drop_rate > 0:
            x =  nn.Dropout(self.drop_rate)(x)
        x = x.view(x.size(0), -1)
        
        attention_weights = self.alpha(x)
        out = attention_weights * self.fc(x)
        return attention_weights, out

