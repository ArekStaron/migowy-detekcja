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
        self.data_dir = data_dir # (data1 ,data2) , (data1)

        self.mp_hands =  mp.solutions.hands
        self.hands_model_static  = self.mp_hands.Hands(static_image_mode=True, max_num_hands= self.num_hands , min_detection_confidence=0.1)

        if len(self.data_dir) > 1:
            self.data_dir = self.merge() # Path
        else:
            self.data_dir = Path(self.data_dir[0])

        labels = os.listdir(self.data_dir)
        
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
                    
                    if self.num_hands ==2:
                        if len(results.multi_hand_landmarks) >1:
                                hand_landmarks_2 = results.multi_hand_landmarks[1]
                                points2 = torch.tensor([[p.x , p.y , p.z] for p in hand_landmarks_2.landmark])
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

#----------------------------------------------------------

sample_set = r"Sign-Language-Digits-Dataset-master\Examples"

dataset()

