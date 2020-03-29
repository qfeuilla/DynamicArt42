import torch
import torchvision.models as models
import torch.nn as nn

class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = list(models.vgg16(pretrained = True).features)[:23]
        # features 3，8，15，22: relu1_2,relu2_2,relu3_3,relu4_3
        self.features = nn.ModuleList(features).eval() 
        
    def forward(self, x):
        results = []
        for ii,model in enumerate(self.features):
            x = model(x)
            if ii in {3,8,15,22}:
                results.append(x)
        return (results)