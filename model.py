import torch.nn as nn
from torch.nn import Module  
import torch



class Layer(nn.Module):
    def __init__(self,hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size , hidden_size)
        self.ln = nn.LayerNorm(hidden_size)
        self.leakRL = nn.LeakyReLU(0.1)

    def forward(self , x):
        x = self.fc(x)
        x = self.ln(x)
        x = self.leakRL(x)
        return x 

class Feather_MLP(Module):
    def __init__(self , xyz , hidden_size ,feathers_size, layer_num):
        super().__init__()
        self.fc1 = nn.Linear(xyz , hidden_size , bias=False)
        self.ln1 = nn.LayerNorm(hidden_size)

        self.layers = nn.ModuleList(
            [Layer(hidden_size) for _ in range(layer_num)])

        self.out = nn.Linear(hidden_size , feathers_size)

        self.leakRL = nn.LeakyReLU(0.1)

    def forward(self, x ):
        B , N , _ = x.shape

        #1 layer
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.leakRL(x)

        for layer in self.layers:
            x = layer(x)
        #out
        x = self.out(x)
        x= self.leakRL(x)

        return x # (B,  N , feather_size)

  
class Sign_language_PointNet(Module):
    def __init__(self , mlp , hidden_size, layer_num ,out_size):
        super().__init__()   
        
        self.feather_mlp = mlp
        feather_size = mlp.out.out_features
        
        self.fc1 = nn.Linear(feather_size ,hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)

        self.layers = nn.ModuleList(
            [Layer(hidden_size) for _ in range(layer_num)])
        
        
        self.out = nn.Linear(hidden_size ,out_size)

        self.leakRL = nn.LeakyReLU(0.1)    

    
    def forward(self,x):
        x = self.feather_mlp(x)

        x = torch.max(x , dim=1)[0] #(B,feathers)
        
        #layer  1 
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.leakRL(x)

        for layer in self.layers:
            x = layer(x)
        

        x = self.out(x)

        return x 