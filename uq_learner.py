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
      self.alpha*=0.9999
      if self.alpha<0.001:
         self.alpha = 0.001
      self.epsilon*=0.9999
      if self.epsilon<0.001:
         self.epsilon=0.001

   def select_action(self,state,suggested_action=0):
      if (np.random.random(1) > self.epsilon) and (state in self.Q.keys()):
         action = suggested_action
      else:
         action = np.random.randint(0,high=self.num_actions,size=None,dtype=int)
      return action

   def update_values(self,state,action_i, action_mi,reward,state_prime,pi):
      self.simulation_iteration+=1
      if state not in self.Q.keys():
         self.Q[state] = initializer([self.num_actions,self.num_actions])
      if state_prime not in self.Q.keys():
         self.Q[state_prime] = initializer([self.num_actions,self.num_actions])
      self.Q_old[state] = self.Q[state].copy()
      Vj = np.sum(np.multiply(pi,self.Q[state_prime]))
      if abs(reward) == 100:
         Vj = reward
      self.Q[state][action_mi][action_i] = (1-self.alpha)*self.Q[state][action_mi][action_i]+self.alpha*((1-self.gamma)*reward+self.gamma*Vj)
      error = np.linalg.norm(self.Q[state][STICK][SOUTH] - self.Q_old[state][STICK][SOUTH])
      #error = np.linalg.norm(self.Q[state][action_mi][action_i] - self.Q_old[state][action_mi][action_i])
      if self.simulation_iteration%self.update_rate == 0:
         self.error_vector.append(error)

if __name__ == '__main__':
   # Instantiate the game:
   game = SoccerGameEnv(verbose=False,hard_code_init=False)
   debug = False
   num_steps_allowed = 100
   num_episodes = 100000000
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
         # Set correlated equilibria variables
         num_eq = 42
         num_unknowns = 25
         # Determine equillibrium:
         c = np.zeros([num_unknowns],dtype=float)
         # There are 25 variables:
         x = cp.Variable(num_unknowns,nonneg=True) # Probabilities can't be less than 0
         # Initialize the A matrix:
         A = np.zeros([num_eq,num_unknowns]) # 44 linear inequalities, 26 variables
         num_equations_rational = 40 # Number of inequalities from rationality
         count_index = np.array([[0,5,10,15,20],[1,6,11,16,21],[2,7,12,17,22],[3,8,13,18,23],[4,9,14,19,24]])
         # For each equation
         eq = 0
         while eq < num_equations_rational:
            for i in range(number_actions):
               # Figure out which actions not in current i:
               temp = np.array(range(0,number_actions))
               other_action = temp[temp!=i]
               for other in other_action:
                  for j in range(number_actions):
                     if eq < num_equations_rational/2:
                        A[eq][count_index[j][i]] = a.Q[state_prime][j][other] - a.Q[state_prime][j][i]
                     else:
                        A[eq][count_index[i][j]] = b.Q[state_prime][j][other] - b.Q[state_prime][j][i]
                  eq+=1
         # Create the b vector:
         b_vec = np.zeros([num_eq])
         # Sum of the joint probabilities must be 1 (and negations sum to -1):
         A[eq][:] = 1
         b_vec[eq] = 1
         A[eq+1][:] = -1
         b_vec[eq+1] = -1
         # Objective Function, maximize the sum of the agents' rewards at state s':
         for i in range(number_actions):
            for j in range(number_actions):
               c[count_index[j][i]] += a.Q[state_prime][j][i]
               c[count_index[i][j]] += b.Q[state_prime][j][i]
         # Solve the problem:
         prob = cp.Problem(cp.Maximize(c.T @ x),
                  [A @ x <= b_vec, x>=0.0])
         prob.solve()
         pi = np.zeros([number_actions,number_actions])
         for i in range(number_actions):
            for j in range(number_actions):
               pi[j][i] = x.value[count_index[j][i]]
         # Take a sample to get the suggested value from the original version:
         probs = x.value
         probs /= sum(probs) # Normalize the trailing decimals (~10E-8)
         action_index = np.random.choice(len(probs),1, p=probs)
         b_suggested_action,a_suggested_action = np.where(count_index==action_index[0])
         # Update state-action values:
         a.update_values(state,a_action,b_action,rewardA,state_prime,pi)
         b.update_values(state,b_action,a_action,rewardB,state_prime,pi.T)
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
      if a.simulation_iteration > num_iterations:
         break
      else:
         print(a.simulation_iteration)
      # Increment the seed:
      start_seed+=1
      # Update for next step
      a.prepare_episode()
      b.prepare_episode()
   print("A win percentage over {0} iterations:  {1}".format(num_iterations,np.round(100*sum(a_wins)/len(a_wins),2)))
   file_name = "data/uq_learner.txt"
   if os.path.exists(file_name):
      os.remove(file_name)
   a_file = open(file_name, "w")
   output = np.reshape(a.error_vector,(-1,len(a.error_vector)))
   for value in output:
      np.savetxt(a_file,value)
   a_file.close()
   # Plot Data:
   print("Friend Q-Learning Training Complete")