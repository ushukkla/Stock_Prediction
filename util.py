from enum import IntEnum
import numpy as np
import gym
from gym import spaces
import pandas as pd

class Action(IntEnum):
    BUY = 0
    SELL = 1
    HOLD = 2

class DeepStockTraderEnv(gym.Env):
  """
  Custom Environment that follows gym interface
  This environment enables agents to make a decision at every timestep in
  a historical stock environment.

  The reward function is defined by how much money the bot made in a particular 
  timestep. (This is 0 in cases where no shares are held)
  """

  metadata={ 'render.modes': ['console'] }

  def __init__(self, pd_data):
    super(DeepStockTraderEnv, self).__init__()

    self.data = pd_data.values
    self.columns_map = {c.lower(): i for i, c in enumerate(pd_data.columns)}

    self.row_size = len(self.columns_map)

    min_val = np.min(self.data)
    low = np.array([min_val for i in range(self.row_size)])

    max_val = np.max(self.data)
    high = np.array([max_val for i in range(self.row_size)])

    self.observation_space = spaces.Box(low=low, 
                                            high=high, 
                                            shape=(self.row_size,), 
                                            dtype=np.float64)

    self.action_space = spaces.Discrete(3)

    # Variables that track the bot's current state
    self.n_shares = 0 # num of shares currently held
    self.cash = 1000  # starting cash
    self.timestep = 0 # cur index of row/timestep in dataset
    self.n_buys = 0   # num of buys
    self.n_sells = 0  # num of sells
    self.n_holds = 0  # num of holds
    self.account_vals = [] # list tracking the account performance over time

  def reset(self):
    self.n_shares = 0 
    self.cash = 1000
    self.timestep = 1 # + 1 since we return the first observation
    self.n_buys = 0
    self.n_sells = 0
    self.n_holds = 0
    self.account_vals = []

    return np.copy(self.data[0])

  def total(self, timestep=-1, open=True):
    return self.cash + self.n_shares * self.data[timestep, self.columns_map["open" if open else "close"]]

  def step(self, action):

    # ********************** EXECUTE ACTION **********************
    open_j = self.columns_map["open"]
    close_j = self.columns_map["close"]
    if action == Action.BUY:
        self.n_shares += self.cash / self.data[self.timestep, open_j]
        self.cash = 0
        self.n_buys += 1
    elif action == Action.SELL:
        self.cash += self.n_shares * self.data[self.timestep, open_j]
        self.n_shares = 0
        self.n_sells += 1
    elif action == Action.HOLD:
        self.n_holds += 1
    else:
        raise ValueError(f"Illegal Action value: {action}")

    self.account_vals.append(self.total(self.timestep))
    # ************************************************************

    # IMPORTANT 
    # We define reward to be (total account value at close) - (total account value at open)
    # Basically your reward is the amount gained over the course of the day 
    reward = self.total(self.timestep, open=False) - self.total(self.timestep)
    done = self.timestep+1 == len(self.data)-1
    info = {
        "n_buys": self.n_buys,
        "n_sells": self.n_sells,
        "n_holds": self.n_holds,
        "cash": self.cash,
        "n_shares": self.n_shares
    }

    self.timestep += 1

    return np.copy(self.data[self.timestep]), reward, done, info

  def render(self, mode='console'):
    if mode != 'console':
        raise NotImplementedError()
    
    print(f"------------Step {self.timestep}------------")
    print(f'total:   \t{self.total(self.timestep)}')
    print(f'cash:    \t{self.cash}')
    print(f'n_shares:\t{self.n_shares}')
    print(f'n_buys:  \t{self.n_buys}')
    print(f'n_sells:\t{self.n_sells}')
    print(f'n_holds:\t{self.n_holds}')

class Evaluation:
    def __init__(self, test_data, starting_cash, agent):
        self.cash = starting_cash
        self.n_shares = 0
        self.test_data = test_data
        self.except_msg = ""

        self.agent = agent

        self.n_buys = 0
        self.n_sells = 0
        self.n_holds = -1 # Always start out with a hold (so need to cancel the default +1). 
        self.account_values = []

        self.evaluate()

    def evaluate(self):
        a = Action.HOLD
        frac = 0        
        for timestep, row in self.test_data.iterrows():            
            # Catch illegal frac value
            if frac > 1:
                raise ValueError(f"You set frac to a value greater than 1 on timestep {timestep} of the test dataset")
            
            if a == Action.BUY:
                self.n_shares += frac*self.cash / row.open
                self.cash -= frac * self.cash
                self.n_buys += 1
            elif a == Action.SELL:
                self.cash += frac * self.n_shares * row.open
                self.n_shares -= frac * self.n_shares
                self.n_sells += 1
            elif a == Action.HOLD:
                self.n_holds += 1
            else:
                raise ValueError(f"Somehow you returned an illegal action (or not an action at all) on timestep {timestep} of the test dataset. Please fix and try again")

            self.account_values.append(self.total(timestep))

            a, frac = self.agent.step(row.values)

    def total(self, timestep=-1):
        return self.cash + self.n_shares * self.test_data.iloc[timestep].close