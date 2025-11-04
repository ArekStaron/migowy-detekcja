Sign Language Recognition Model (PyTorch)
This project detects sign language gestures in real time.

You can easily switch datasets depending on which sign language you want to train — the model should adapt without major changes.

If you need to merge multiple datasets, uncomment the first and second code cells.
If your CPU is powerful enough, you can also uncomment the fourth cell to enable multithreaded preprocessing for faster data loading.

If you switch to “Sign-Language-Digits-Dataset-master” or any other dataset that uses only one hand, make sure to:

set max_num_hands = 1 in hands_model,

remove that part"

else:
    points2 = torch.zeros(21, 3)
"

in both the model and dataset code.

data_2 and Sign-Language-Digits-Dataset-master ARE NOT COMPATIBLE!!!!
