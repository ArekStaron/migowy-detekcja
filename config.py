from dataclasses import dataclass



@dataclass
class ModelConfig:
    
    xyz: int 
    
    mlp_hidden_size: int

    mlp_layers: int

    feather_size: int 
    
    hidden_size: int

    layer_num: int

    out_size: int
     

NUM_HANDS = 2 
DATA_DIR = "data_2"
BATCH_SIZE = 64


