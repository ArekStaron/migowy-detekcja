import mediapipe as mp
import cv2 
import torch
from model import Sign_language_PointNet , Feather_MLP

def detect(model, idx_to_label, hand_num):
        
        mp_hands = mp.solutions.hands
        hands_model = mp_hands.Hands(static_image_mode=False, max_num_hands= hand_num , min_detection_confidence=0.8)

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
                if hand_num ==2:
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
                    logits = model(points)
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
mlp = Feather_MLP()
model = Sign_language_PointNet()


