import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from enum import IntEnum

class Action(IntEnum):
    BUY = 0
    SELL = 1
    HOLD = 2

# You need to include any network definitions
class TraderNetwork(nn.Module):
    def __init__(self, row_size):
        super(TraderNetwork, self).__init__()

        self.size = row_size

        self.fc1 = nn.Linear(row_size, row_size*2)
        self.fc2 = nn.Linear(row_size*2, row_size)
        self.fc3 = nn.Linear(row_size, 3)

    def forward(self, t):
        # Have the tensor "flow" through the network
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = F.relu(self.fc3(t))

        return t

class Agent:
    def __init__(self, row_size):
        """
        Write your custom initialization sequence here.
        This can include loading models from file.
        """
        self.tn = TraderNetwork(row_size).double()        
        self.tn.load_state_dict(torch.load("./example_model.pt"))        
        self.tn.eval()

    def step(self, row):
        """
        Make a decision to be executed @ the open of the next timestep. 

        row is a numpy array with the same format as the training data

        Return a tuple (Action, fraction). Fraction means different 
        things for different actions...
        
        Action.BUY:  represents fraction of cash to spend on purchase 
        Action.SELL: represents fraction of owned shares to sell 
        Action.HOLD: value ignored.

        See the code below on how to return
        """

        t = torch.tensor(row)
        choice = torch.argmax(self.tn(t).squeeze(0)).item()

        # The plan was to never have to use constants...
        # Yeah, we're assuming consistency in buy=0, sell=1, and hold=2
        if choice == 0:
            return (Action.BUY, 1)
        elif choice == 1:
            return (Action.SELL, 1)

        return (Action.HOLD, 0)