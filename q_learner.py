import numpy as np
from matplotlib import pyplot as plt
from SoccerGameEnv import SoccerGameEnv
import os

# 0 -> N | 1 -> S | 2 -> E | 3 -> W | 4 -> STICK (don't move)
NORTH = 0
SOUTH = 1
EAST = 2
WEST = 3
STICK = 4

class TabluarQLearner:
   def __init__(self,number_actions,epsilon,alpha,gamma):
      self.Q_old = {} # t-1
      self.Q = {}     # t
      self.num_actions = number_actions
      self.gamma = gamma
      self.epsilon = epsilon
      self.alpha = alpha
      self.error_vector = []
      self.update_rate = 1
      self.simulation_iteration = 0
   def prepare_episode(self):
      # Decay alpha to 0.001
      self.alpha*=0.9999
      if self.alpha<0.001:
         self.alpha = 0.001
      self.epsilon*=0.9995
      if self.epsilon<0.001:
         self.epsilon=0.001
   def select_action(self,state):
      if state not in self.Q.keys():
         self.Q[state] = np.random.random([self.num_actions])
      if np.random.random(1) > self.epsilon:
         action = np.argmax(self.Q[state])
      else:
         action = np.random.randint(0,high=self.num_actions,size=None,dtype=int)
      return action
   def update_values(self,state,action,reward,state_prime,action_prime):
      self.simulation_iteration+=1
      if state not in self.Q.keys():
         self.Q[state] = np.random.random([self.num_actions])
      if state_prime not in self.Q.keys():
         self.Q[state_prime] = np.random.random([self.num_actions])
      self.Q_old[state] = self.Q[state].copy()
      self.Q[state][action] = self.Q[state][action]+self.alpha*(reward+self.gamma*self.Q[state_prime][action_prime]-self.Q[state][action])
      error = np.linalg.norm(self.Q[state][SOUTH] - self.Q_old[state][SOUTH])
      if self.simulation_iteration%self.update_rate == 0:
         self.error_vector.append(error)

if __name__ == '__main__':
   # Instantiate the game:
   game = SoccerGameEnv(verbose=False,hard_code_init=True)
   num_steps_allowed = 1000
   num_episodes = 250000
   start_seed = 11
   number_actions = 5
   a = TabluarQLearner(number_actions,epsilon=0.99,alpha=0.1,gamma=0.9)
   b = TabluarQLearner(number_actions,epsilon=0.99,alpha=0.1,gamma=0.9)
   a_wins = []
   for _ in range(num_episodes):
      # Reset the state
      state = game.reset(start_seed)
      # Select actions
      a_action = a.select_action(state)
      b_action = b.select_action(state)
      for _ in range(num_steps_allowed):
         # Take actions
         state_prime, rewardA, rewardB, terminal = game.step(a_action,b_action)
         # Update internal states
         a_action_prime = a.select_action(state_prime)
         b_action_prime = b.select_action(state_prime)
         a.update_values(state,a_action,rewardA,state_prime,a_action_prime)
         b.update_values(state,b_action,rewardB,state_prime,b_action_prime)
         state = state_prime
         a_action = a_action_prime
         b_action = b_action_prime
         #game.render()
         if terminal:
            # Game ended, break
            #print("Game over!")
            # Build some meta data
            if rewardA == 100:
               a_wins.append(1)
            elif rewardB == 100:
               a_wins.append(0)
            break
      # Increment the seed:
      start_seed+=1
      # Update for next step
      a.prepare_episode()
      b.prepare_episode()
   print("A win percentage over {0} episodes:  {1}".format(num_episodes,np.round(100*sum(a_wins)/len(a_wins),2)))
   # Save to File
   file_name = "data/q_learner_data.txt"
   if os.path.exists(file_name):
      os.remove(file_name)
   a_file = open(file_name, "w")
   output = np.reshape(a.error_vector,(-1,len(a.error_vector)))
   for value in output:
      np.savetxt(a_file,value)
   a_file.close()
   print("Q-Learning Training Complete")