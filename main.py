from pathlib import Path
import shutil
import os
import mediapipe as mp
import cv2 
import torch
class dataset:
    def __init__(self , *data_dir :tuple ,num_hands=1,):
        assert num_hands in (1,2) , f"num_hands must be 1 or 2 can't be {num_hands} "
        
        self.num_hands = num_hands
        self.data_dir = data_dir 

        self.mp_hands =  mp.solutions.hands
        self.hands_model_static  = self.mp_hands.Hands(static_image_mode=True, max_num_hands= self.num_hands , min_detection_confidence=0.1)

        if len(self.data_dir) > 1:
            self.data_dir = self.merge() # Path
        else:
            self.data_dir = Path(self.data_dir[0])

        labels = sorted(os.listdir(self.data_dir))
        
        self.label_map = { k : v for v ,k in enumerate(labels)}

    def merge(self , out_dir ="merged_data"):
        
        
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True)

        for d in self.data_dir:
            for cls in os.listdir(d):
                src = Path(d) / cls
                if not src.is_dir(): continue
                dst = out_dir / cls
                dst.mkdir(exist_ok=True)
                for f in os.listdir(src):
                    shutil.copy(src / f, dst / f)
        return out_dir
    
    def data2tensor(self, save =True):
        data = []
        errors = []

        count = 0

        for class_name in os.listdir(self.data_dir):
            class_dir = os.path.join(self.data_dir , class_name)


            for image_file in os.listdir(class_dir):
                count +=1
                image_path = os.path.join(class_dir , image_file)
                image  =cv2.imread(image_path)
                image =  cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image , (128 , 128))
                results = self.hands_model_static.process(image)
                
                if results.multi_hand_landmarks:
                    hand_landmarks_1 = results.multi_hand_landmarks[0]
                    
                    points1 = torch.tensor([[p.x , p.y , p.z] for p in hand_landmarks_1.landmark])
                    points1 = self.normalize_points(points1)
                    if self.num_hands ==2:
                        if len(results.multi_hand_landmarks) >1:
                                hand_landmarks_2 = results.multi_hand_landmarks[1]
                                points2 = torch.tensor([[p.x , p.y , p.z] for p in hand_landmarks_2.landmark])
                                points2 = self.normalize_points(points2)
                        else:
                            points2=  torch.zeros(21,3)
                        
                        points = torch.cat([points1,points2] , dim=0)
                    else:
                        points = points1


                    data.append((points ,class_name))
                else:
                    errors.append(image_path)
                
                if count%1000 == 0:
                    print(count , len(errors))

        if save:
            torch.save(data , "tensordata" )
        X = torch.stack([t[0] for t in data])
        Y = torch.stack([torch.tensor(self.label_map[t[1]]) for t in data])

        return X ,Y
    @staticmethod
    def normalize_points(points):
        
        points = points - points[0]
        return points

#----------------------------------------------------------
#test
sample_set = r"sample_set"

data =  dataset((sample_set))
x , y = data.data2tensor(save=False)

print(x.shape , y.shape) # test

#-------------------------------------------------------------
#making dataloader
from torch.utils.data import TensorDataset ,DataLoader ,random_split

data = dataset("data_2")
x ,y = data.data2tensor(save=False)

train_len = int(0.8 * len(data))
valid_len= len(dataset) - train_len 

tensor_dataset = TensorDataset(x,y)

train_dataset , valid_dataset = random_split(tensor_dataset , [train_len , valid_len])

traning_loader = DataLoader( train_dataset , batch_size= 32, shuffle=True)
valid_loader = DataLoader(valid_dataset , batch_size=32 , shuffle=False)

label_map = data.label_map
out_size =len(label_map)

#--------------------------------------------------------------------
# Model
import torch.nn as nn
from torch.nn import Module  

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
    def __init__(self , mlp , hidden_size, layer_num , hand_num=1 ):
        super().__init__()
        assert hand_num in (1,2) , f"num hand must be 1 or 2 not {hand_num}"
        self.hand_num = hand_num
        
        self.out_size = 21 if hand_num==1 else 42
        
        self.feather_mlp = mlp
        feather_size = mlp.out.out_features
        
        self.fc1 = nn.Linear(feather_size ,hidden_size )
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

    def detect(self):
        
        mp_hands = mp.solutions.hands
        hands_model = mp_hands.Hands(static_image_mode=False, max_num_hands=self.hand_num , min_detection_confidence=0.8)

        cap = cv2.VideoCapture(0 , cv2.CAP_DSHOW)

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            
            frame = cv2.flip(frame,1)
            frame_rgb= cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)
            frame_resize = cv2.resize(frame_rgb , (128,128))
            results  = hands_model.process(frame_resize)

            if results.multi_hand_landmarks:
                hand_landmarks_1 = results.multi_hand_landmarks[0]
            
                points1 = torch.tensor([[p.x , p.y , p.z] for p in hand_landmarks_1.landmark])
                if self.hand_num ==2:
                    if len(results.multi_hand_landmarks) >1:
                            hand_landmarks_2 = results.multi_hand_landmarks[1]
                            points2 = torch.tensor([[p.x , p.y , p.z] for p in hand_landmarks_2.landmark])
                    
                    else:
                        points2=  torch.zeros(21,3)
                    
                    points = torch.cat([points1,points2] , dim=0)
                else:
                    points = points1

                
                with torch.no_grad():
                    points = points.unsqueeze(0)
                    logits = self(points)
                    preds = torch.softmax(logits , dim=1)
                    conf , idx = torch.max(preds, dim=1)
                
                label = idx_to_label[idx.item()]
                
                pred_text = f"It is {label} with {conf.item():.2f} accuracy"


                cv2.putText(frame ,pred_text , (10,40) , cv2.FONT_HERSHEY_SIMPLEX,1 , (0,255,0),2 )    

            cv2.imshow("kamerka" , frame)
            
            if (cv2.waitKey(1) & 0xFF) in (ord('q') ,27):
                break
            
    
        cap.release()
        cv2.destroyAllWindows()

import json
with open("best_params.json" , 'r') as f:
    params = json.load(f)

mlp_hidden_size = params["mlp_hidden_size"]
feather_size    = params["feather_size"]
hidden_size     = params["hidden_size"]
lr              = params["lr"]


feather_model = Feather_MLP(3,mlp_hidden_size , feather_size)
model = Sign_language_PointNet(feather_model , hidden_size,hand_num=2)

model.load_state_dict(torch.load("model.pt"))
model.to("cpu")
model.eval()
model.detect()
