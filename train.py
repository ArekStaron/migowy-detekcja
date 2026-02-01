from torch.utils.data import TensorDataset ,DataLoader ,random_split
import torch
import torch.nn as nn
from model import Feather_MLP , Sign_language_PointNet
import optuna

data = torch.load("tensordata.pt")

x = data["X"]
y = data["Y"]


tensor_dataset = TensorDataset(x,y)

train_len = int(0.8 * len(tensor_dataset))
valid_len= len(tensor_dataset) - train_len 

train_dataset , valid_dataset = random_split(tensor_dataset , [train_len , valid_len])

traning_loader = DataLoader( train_dataset , batch_size= 32, shuffle=True)
valid_loader = DataLoader(valid_dataset , batch_size=32 , shuffle=False)

label_map = data["label_map"]
out_size =len(label_map)


#Get Hiperparamiters
def train_eval(model,lr, train_data, valid_data , epoch, device = "cuda"):
    
    optimizer = torch.optim.Adam(model.parameters() , lr)
    criterion = nn.CrossEntropyLoss()


    train_losses, valid_losses = [] , []
    
    
    for _ in range(epoch):
        model.to(device)
        model.train()
        
        running_loss = 0.0
        for x , y in train_data:
            optimizer.zero_grad()
            x , y = x.to(device) , y.to(device)
            output = model(x)
            loss = criterion(output , y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * y.size(0)
        train_loss = running_loss/ len(train_data.dataset)
        train_losses.append(train_loss)

        model.eval()
        with torch.no_grad():
            running_loss = 0.0
            for x , y in valid_data:
                x , y = x.to(device) , y.to(device)
                output = model(x)
                loss = criterion(output , y)
                
                running_loss += loss.item() * y.size(0)
            valid_loss = running_loss/ len(valid_data.dataset)
            valid_losses.append(valid_loss)
    
    return train_losses , valid_losses




def objective(trail):

    mlp_hidden_size = trail.suggest_int("mlp_hidden_size" ,32,128)
    feather_size = trail.suggest_int("feather_size" ,32,128)
    hidden_size = trail.suggest_int("hidden_size" , 32, 128)
    lr = trail.suggest_float("lr" , 1e-4, 1e-2 , log=True)  
    layer_num_mlp =  trail.suggest_int("layer_num_mlp" , 2, 5)
    layer_num = trail.suggest_int("layer_num", 2 , 5)
    
    
    mlp = Feather_MLP(
                xyz =3,
                hidden_size= mlp_hidden_size ,
                layer_num= layer_num_mlp ,
                feathers_size= feather_size 
                        )
    
    model = Sign_language_PointNet(
                mlp = mlp ,
                hidden_size= hidden_size,
                layer_num=layer_num ,
                out_size = out_size
                )
    
    _ , valid_loss = train_eval(model , lr , traning_loader , valid_loader , epoch=5)
    

    return valid_loss[-1]

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=40)

p = study.best_params 
hidden_size_mlp = p.get("mlp_hidden_size") 
feather_size = p.get("feather_size") 
hidden_size = p.get("hidden_size") 
lr = p.get("lr") 
layer_num_mlp = p.get("layer_num_mlp")
layer_num = p.get("layer_num")

feather_model = Feather_MLP(xyz=3 , 
                            hidden_size=hidden_size_mlp, 
                            feathers_size=feather_size,
                            layer_num= layer_num_mlp
                              )
model = Sign_language_PointNet(mlp = feather_model ,
                                hidden_size=hidden_size, 
                                out_size=out_size,
                                layer_num=layer_num  
                                  )

train_eval(model,lr,traning_loader , valid_loader , epoch=10)

torch.save(model.state_dict() , "model.pt")

