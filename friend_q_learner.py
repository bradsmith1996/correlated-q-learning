import numpy as np
import cvxpy as cp
from matplotlib import pyplot as plt
from SoccerGameEnv import SoccerGameEnv
import os

# 0 -> N | 1 -> S | 2 -> E | 3 -> W | 4 -> STICK (don't move)
NORTH = 0
SOUTH = 1
EAST = 2
WEST = 3
STICK = 4

# Two options for initialization
initializer = np.zeros
#initializer = np.random.random

class CorrelatedQLearner:
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
      self.alpha*=0.9995
      if self.alpha<0.001:
         self.alpha = 0.001
      self.epsilon*=0.9995
      if self.epsilon<0.001:
         self.epsilon=0.001

   def select_action(self,state):
      if (np.random.random(1) > self.epsilon) and (state in self.Q.keys()):
         action = np.argmax(np.max(self.Q[state], axis = 0))
      else:
         action = np.random.randint(0,high=self.num_actions,size=None,dtype=int)
      return action

   def update_values(self,state,action_i, action_mi,reward,state_prime):
      self.simulation_iteration+=1
      if state not in self.Q.keys():
         self.Q[state] = initializer([self.num_actions,self.num_actions])
      if state_prime not in self.Q.keys():
         self.Q[state_prime] = initializer([self.num_actions,self.num_actions])
      self.Q_old[state] = self.Q[state].copy()
      Vj = self.Q[state].max() # Friend Q this is just the max of the entire matrix
      self.Q[state][action_mi][action_i] = (1-self.alpha)*self.Q[state][action_mi][action_i]+self.alpha*((1-self.gamma)*reward+self.gamma*Vj)
      error = np.linalg.norm(self.Q[state][STICK][SOUTH] - self.Q_old[state][STICK][SOUTH])
      if self.simulation_iteration%self.update_rate == 0:
         self.error_vector.append(error)

if __name__ == '__main__':
   # Instantiate the game:
   game = SoccerGameEnv(verbose=False,hard_code_init=True)
   debug = False
   num_steps_allowed = 100
   num_episodes = 40000
   start_seed = 11 # Random start seed
   number_actions = 5
   a = CorrelatedQLearner(number_actions,epsilon=0.99,alpha=0.1,gamma=0.9)
   b = CorrelatedQLearner(number_actions,epsilon=0.99,alpha=0.1,gamma=0.9)
   a_wins = []
   for ep in range(num_episodes):
      # Reset the state
      state = game.reset(start_seed)
      if debug:
         print("The Start Board         :")
         game.render()
      # Select actions
      a_action = a.select_action(state)
      b_action = b.select_action(state)
      for t in range(num_steps_allowed):
         # Simulate the actions of a and b, observe action profile and rewards
         state_prime, rewardA, rewardB, terminal = game.step(a_action,b_action)
         if state_prime not in a.Q.keys():
            a.Q[state_prime] = initializer([number_actions,number_actions])
         if state_prime not in b.Q.keys():
            b.Q[state_prime] = initializer([number_actions,number_actions])
         # Update state-action values:
         a.update_values(state,a_action,b_action,rewardA,state_prime)
         b.update_values(state,b_action,a_action,rewardB,state_prime)
         # Update internal states
         a_action = a.select_action(state_prime)
         b_action = b.select_action(state_prime)
         # Make updates for the next loop:
         state = state_prime
         if terminal:
            # Game ended, break
            if debug:
               print("Information For Episode : {0}".format(ep))
               print("Number of Iterations    : {0}".format(t))
               print("The Final Board         :")
               game.render()
               print("Final Reward for A      : {0}".format(rewardA))
               print("Final Reward for B      : {0}".format(rewardB))
               print("\n\n")
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
   file_name = "data/friend_q_learner.txt"
   if os.path.exists(file_name):
      os.remove(file_name)
   a_file = open(file_name, "w")
   output = np.reshape(a.error_vector,(-1,len(a.error_vector)))
   for value in output:
      np.savetxt(a_file,value)
   a_file.close()
   # Plot Data:
   print("Friend Q-Learning Training Complete")