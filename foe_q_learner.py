import numpy as np
import cvxpy as cp
from cvxopt import matrix, solvers
from matplotlib import pyplot as plt
from SoccerGameEnv import SoccerGameEnv
import os

# 0 -> N | 1 -> S | 2 -> E | 3 -> W | 4 -> STICK (don't move)
NORTH = 0
SOUTH = 1
EAST = 2
WEST = 3
STICK = 4


solvers.options['show_progress'] = False

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
      self.policy = {}

   def prepare_episode(self):
      # Decay alpha to 0.001
      self.alpha*=0.9999
      if self.alpha<0.001:
         self.alpha = 0.001
      self.epsilon*=0.999
      if self.epsilon<0.001:
         self.epsilon=0.001

   def select_action(self,state,suggested_action=0):
      if np.random.random(1) > self.epsilon:
         action = suggested_action
      else:
         action = np.random.randint(0,high=self.num_actions,size=None,dtype=int)
      return action

   def update_values(self,state,action_i, action_mi,reward,state_prime,Vj):
      self.simulation_iteration+=1
      if state not in self.Q.keys():
         self.Q[state] = np.zeros([number_actions,number_actions])
      if state_prime not in self.Q.keys():
         self.Q[state_prime] = np.zeros([number_actions,number_actions])
      self.Q_old[state] = self.Q[state].copy()
      self.Q[state][action_mi][action_i] = (1-self.alpha)*self.Q[state][action_mi][action_i]+self.alpha*((1-self.gamma)*reward+self.gamma*Vj)
      error = np.linalg.norm(self.Q[state][action_mi][action_i] - self.Q_old[state][action_mi][action_i])
      error = np.linalg.norm(self.Q[state][STICK][SOUTH] - self.Q_old[state][STICK][SOUTH])
      if self.simulation_iteration%self.update_rate == 0:
         #if error < 0.000001 and len(self.error_vector):
         if 0:
            self.error_vector.append(self.error_vector[-1])
         else:
            self.error_vector.append(error)

if __name__ == '__main__':
   # Instantiate the game:
   game = SoccerGameEnv(verbose=False,hard_code_init=False)
   debug = False
   num_steps_allowed = 100
   num_episodes = 800000
   num_iterations = 1000000
   start_seed = 11 # Random start seed
   number_actions = 5
   a = CorrelatedQLearner(number_actions,epsilon=0.99,alpha=0.1,gamma=0.9)
   b = CorrelatedQLearner(number_actions,epsilon=0.99,alpha=0.1,gamma=0.9)
   a_wins = []
   for ep in range(num_episodes):
      #print(ep)
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
         # Set up c vector:
         c = np.zeros(6) # value, north, south, east, west, stick
         c[0] = 1
         # There are 25 variables:
         xA = cp.Variable(6) # Probabilities can't be less than 0
         xB = cp.Variable(6) # Probabilities can't be less than 0
         # Initialize the A matrix:
         A = np.ones([12,6]) # 44 linear inequalities, 26 variables
         B = np.ones([12,6])
         A[5][0] = 0.0
         A[6][:]*= -1.0
         A[6][0] = 0.0

         # All probabilities must b greater than or equal to 0
         A[7][:] = np.array([0,-1,0,0,0,0],dtype=float)
         A[8][:] = np.array([0,0,-1,0,0,0],dtype=float)
         A[9][:] = np.array([0,0,0,-1,0,0],dtype=float)
         A[10][:] = np.array([0,0,0,0,-1,0],dtype=float)
         A[11][:] = np.array([0,0,0,0,0,-1],dtype=float)

         B = A.copy()
         # Make specific Q-table overrides:
         for i in range(number_actions):
            for j in range(number_actions):
               A[i][j+1] = float(-1.0*a.Q[state_prime][i][j])
               B[i][j+1] = float(-1.0*b.Q[state_prime][i][j])

         AA = matrix(A,tc='d')
         BB = matrix(B,tc='d')
         c = matrix([-1,0,0,0,0,0],tc='d')
         b_vec = matrix([0,0,0,0,0,1,-1,0,0,0,0,0],tc='d')
         solA=solvers.lp(c,AA,b_vec,verbose=False)
         solB=solvers.lp(c,BB,b_vec,verbose=False)

         valueA = solA['x'][0]
         valueB = solB['x'][0]
         # Sample probabilities for actions:
         probA = np.array(solA['x'][1:])
         probB = np.array(solB['x'][1:])
         probA[probA<0] = 0.0
         probB[probB<0] = 0.0
         a.policy[state_prime] = probA
         b.policy[state_prime] = probB
         # Normalize:
         probA/=sum(probA)
         probB/=sum(probB)
         # Take weighted sample:
         a_suggested_action = np.random.choice(len(probA),1, p=probA.flatten())
         b_suggested_action = np.random.choice(len(probB),1, p=probB.flatten())
         # Update state-action values:
         if terminal:
            valueA = rewardA
            valueB = rewardB
         a.update_values(state,a_action,b_action,rewardA,state_prime,valueA)
         b.update_values(state,b_action,a_action,rewardB,state_prime,valueB)
         # Update internal states
         a_action = a.select_action(state_prime, a_suggested_action[0])
         b_action = b.select_action(state_prime, b_suggested_action[0])
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
               #print("A wins!")
            elif rewardB == 100:
               a_wins.append(0)
               #print("B wins!")
            break
      if num_iterations <= a.simulation_iteration:
         break
      else:
         print(a.simulation_iteration)
      # Increment the seed:
      start_seed+=1
      # Update for next step
      a.prepare_episode()
      b.prepare_episode()
   # print("A Policy at the beginning")
   # print(a.Q['020102'])
   # print(a.policy['020102'])
   # print("B Policy at the beginning")
   # print(b.Q['020102'])
   # print(b.policy['020102'])
   # print("Policy A obvious goal")
   # print(a.Q['010201'])
   # print(a.policy['010201'])
   # print("Policy B obvious goal")
   # print(b.Q['010202'])
   # print(b.policy['010202'])
   # print("Policy B obvious goal (A perspective")
   # print(a.Q['010202'])
   # print(a.policy['010202'])
   # print("Policy A obvious goal (B perspective")
   # print(b.Q['010201'])
   # print(b.policy['010201'])
   # print("Policy A potential own goal")
   # print(a.Q['020102'])
   # print(a.policy['020102'])
   # #print("Policy B potential own goal")
   # #print(a.Q['010202'])
   # #print(a.policy['010202'])
   print("A win percentage over {0} episodes:  {1}".format(num_episodes,np.round(100*sum(a_wins)/len(a_wins),2)))
   # Save to File
   file_name = "data/foe_q_learner.txt"
   if os.path.exists(file_name):
      os.remove(file_name)
   a_file = open(file_name, "w")
   output = np.reshape(a.error_vector,(-1,len(a.error_vector)))
   for value in output:
      np.savetxt(a_file,value)
   a_file.close()
   # Plot Data:
   print("Friend Q-Learning Training Complete")