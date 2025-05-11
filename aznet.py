#EE6892 Reinforcement Learning 
#Chinese Chess (Xiangqi) Environment Neural Network Module


import torch

# Neural network architecture
# This code is generated for loading the neural network for Chinese Chess enviroement 
# The model takes a board state as input and predicts the policy and value
# Policy decides on the move and value predicts the likeliness of position to from win state 



class AZNet(torch.nn.Module):

    # We will have 2 dimensional convolutional layer 
    # Source + implementations : https://github.com/NeymarL/ChineseChess-AlphaZero

    # Network Architecture
    # We have to have 15 channels 
    # In Xiangqi there are 6 tyes of piece x 2 players which makes the first 12 channels
    # We locate and then put = 1 where the pieces are located thus making the a 10 * 9 binary plane
    # We have 1 channel for current player (if its all 1's then that means its RED turn otherwise its blacks turn
    # This channel helps us to represent certain rules such as checks or draws or even repetition
    # We also added 1 more channel to possibly add other rules/features in future development

    # We also apply 3x3 convolution with padding which sets our output dimension to 128
    # Then, we define resiudual blocks which consist of 5 residual block architecture
    # In each block we have 2 convolutional layers  + batch norm after that to normalize the output values

        
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(15, 128, 3, padding=1)
        self.res_blocks = torch.nn.Sequential(*[
            torch.nn.Sequential(
                torch.nn.Conv2d(128,128,3,padding=1), torch.nn.BatchNorm2d(128), torch.nn.ReLU(),
                torch.nn.Conv2d(128,128,3,padding=1), torch.nn.BatchNorm2d(128)
            ) for _ in range(5)
        ])

        # Policy Head Architecture
        # One 1x1 convolutional layer to decrease our hidden states then we flatten the architecture [32,10,9] to tensor size 2880 ( 32 * 9 * 10 ) 
        # Which corresponds to fully connected 8100 logit units which corresponds the legall moves (8100)
        
        self.p_conv = torch.nn.Conv2d(128, 32, 1)
        self.p_fc   = torch.nn.Linear(32 * 10 * 9, 8100)

        # Value Head
        # As we did in the policy head we first shrink the hidden states and flatten to 2880
        self.v_conv = torch.nn.Conv2d(128, 32, 1)
        self.v_fc1  = torch.nn.Linear(32 * 10 * 9, 128)
        self.v_fc2  = torch.nn.Linear(128, 1)

        
    def forward(self, x):
            
        # Initial convolution
        x = torch.nn.functional.relu(self.conv1(x))
            
        # Residual blocks
        for block in self.res_blocks:
            x = torch.nn.functional.relu(block(x) + x)
                
        # Policy head
        p = torch.nn.functional.relu(self.p_conv(x)).flatten(1)
        policy = self.p_fc(p)
            
        # Value head
        v = torch.nn.functional.relu(self.v_conv(x)).flatten(1)
        v = torch.nn.functional.relu(self.v_fc1(v))
        value = torch.tanh(self.v_fc2(v)).squeeze(-1)
            
        return policy, value

# Load the model
def load_model(model_path="aznet_chinese_chess.pth"):
    model = AZNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model