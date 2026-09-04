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

class FeatureMLP(Module):
    def __init__(self , xyz , hidden_size ,feature_size, layer_num):
        super().__init__()
        self.fc1 = nn.Linear(xyz , hidden_size , bias=False)
        self.ln1 = nn.LayerNorm(hidden_size)

        self.layers = nn.ModuleList(
            [Layer(hidden_size) for _ in range(layer_num)])

        self.out = nn.Linear(hidden_size , feature_size)

        self.leakRL = nn.LeakyReLU(0.1)

    def forward(self, x ):
        B , N , _ = x.shape

        
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.leakRL(x)

        for layer in self.layers:
            x = layer(x)
        
        x = self.out(x)
        x= self.leakRL(x)

        return x 

  
class SignlanguagePointNet(Module):
    def __init__(self , mlp , hidden_size, layer_num ,out_size):
        super().__init__()   
        
        self.feature_mlp = mlp
        feature_size = mlp.out.out_features
        
        self.fc1 = nn.Linear(feature_size ,hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)

        self.layers = nn.ModuleList(
            [Layer(hidden_size) for _ in range(layer_num)])
        
        
        self.out = nn.Linear(hidden_size ,out_size)

        self.leakRL = nn.LeakyReLU(0.1)    

    
    def forward(self,x):
        x = self.feature_mlp(x)

        x = torch.max(x , dim=1)[0] 
        
         
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.leakRL(x)

        for layer in self.layers:
            x = layer(x)
        

        x = self.out(x)

        return x 