import torch
from torch import nn
import torch.nn.functional as F

# input img -> hidden dim -> mean, std -> reparametrization trick -> decoder -> output img
class SuperResolution(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=9, padding='same'
        )
        
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=16, kernel_size=1, padding='same'
        )

        self.conv3 = nn.Conv2d(
            in_channels=16, out_channels=1, kernel_size=5, padding='same'
        )
        
        # defining relu
        self.relu = nn.ReLU() 
    
    def forward(self,x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        return x
    
if __name__=='__main__':
    batch_size = 1
    x=torch.randn(batch_size, 1, 272, 160)
    model=SuperResolution()
    y = model(x)
    print(x.shape)