import numpy as np
import matplotlib.pyplot as plt

def filter_data(some_data,tol):
   i = 1
   while i < len(some_data):
      if abs(some_data[i]) < tol:
         some_data[i] = some_data[i-1]
      i+=1
   return some_data

if __name__ == '__main__':
   # Flags for what to plot:
   q_learner = False
   friend_q = False
   foe_q = True
   uq_q = False
   save = False # Not used
   save_path = "deliverable/figures" # Not used
   # Common Variables:
   color = 'k'
   y_limit = 0.5
   x_limit = 1000000
   tolerance_q = 0.01
   tolerance_frq = 0.0000001
   # Q Learning Plotting:
   if q_learner:
      # Load the data:
      q_learning_data = np.loadtxt("data/q_learner_data.txt")
      # Filter Data:
      q_learning_data = filter_data(q_learning_data,tolerance_q)
      # Plot Data:
      plt.plot(q_learning_data,color=color)
      plt.xlabel("Simulation Iteration",fontsize=14)
      plt.ylabel("Q-Value Difference",fontsize=14)
      plt.ticklabel_format(axis="x", style="sci",scilimits=(1,5))
      plt.ylim([0, y_limit])
      plt.xlim([0,x_limit])
      plt.show()
   if friend_q:
      # Load the data:
      friend_q_learning_data = np.loadtxt("data/friend_q_learner.txt")
      # Filter Data:
      #friend_q_learning_data = filter_data(friend_q_learning_data,tolerance_frq)
      # Plot Data:
      print(len(friend_q_learning_data))
      plt.plot(friend_q_learning_data,color=color)
      plt.xlabel("Simulation Iteration",fontsize=14)
      plt.ylabel("Q-Value Difference",fontsize=14)
      plt.ticklabel_format(axis="x", style="sci",scilimits=(1,5))
      plt.ylim([0, y_limit])
      plt.xlim([0,x_limit])
      plt.show()
   if foe_q:
      # Load the data:
      foe_q_learning_data = np.loadtxt("data/foe_q_learner.txt")
      # Filter Data:
      #friend_q_learning_data = filter_data(friend_q_learning_data,tolerance_q)
      # Plot Data:
      print(len(foe_q_learning_data))
      plt.plot(foe_q_learning_data,color=color)
      plt.xlabel("Simulation Iteration",fontsize=14)
      plt.ylabel("Q-Value Difference",fontsize=14)
      plt.ticklabel_format(axis="x", style="sci",scilimits=(1,5))
      plt.ylim([0, y_limit])
      plt.xlim([0, x_limit])
      plt.show()
   if uq_q:
      # Load the data:
      u_q_learning_data = np.loadtxt("data/uq_learner.txt")
      # Filter Data:
      #friend_q_learning_data = filter_data(friend_q_learning_data,tolerance_q)
      # Plot Data:
      print(len(u_q_learning_data))
      plt.plot(u_q_learning_data,color=color)
      plt.xlabel("Simulation Iteration",fontsize=14)
      plt.ylabel("Q-Value Difference",fontsize=14)
      plt.ticklabel_format(axis="x", style="sci",scilimits=(1,5))
      plt.ylim([0, y_limit])
      plt.xlim([0, x_limit])
      plt.show()