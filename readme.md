## Correlated Q-Learning Project
- Title: Correlated Equilibria Q-Learning
- Author: Peter Bradley "Brad" Smith
- Date: 7/25/2021
- Implementation Timeframe: 2 weeks

## Repository Reqiurements:
- Python version 3.6
- numpy==1.18.0
- cvxopt==1.2.0
- cvxpy==1.1.12
- matplotlib==3.3.4

## Repository Contents:
- SoccerGameEnv.py:
   - OpenAI Gym like object to represent the soccer game environment according to Greenwald 2003
- cvxopt_example.py:
   - Example for running cvxopt for debugging purposes
- Files for running training (one for each algorithm of interest:
   - Each a main for running independent of the other algorithms
   - Note: All will generate a file which will be placed in the "data" folder to be read in by process_data.py:
   1. q_learner.py: Q learning algorithm (generates data/q_learner_data.txt)
   2. uq_learner.py: utilitarian CE algorithm (generates data/uq_learner.txt)
   3. foe_q_learner.py: Foe-Q learning algorithm (generates data/foe_q_learner.txt)
   4. friend_q_learner.py: Friend-Q learning algorithm (generates data/friend_q_learner.txt)
- Data Processor:
   - process_data.py
   - Reads each file for the different algorithms and generates plots with consistent axes.
   - Note: Only one major difference is that since q-learning diverges so intensely, 
- docs (directory):
   - Report word doc for the report
   - Inside here, there is a directory called figures which holds the final 4 figures:
      1. Q-Learning: q_learning.png
      2. Correlated Equilibria Q-Learning: uq_learning.png
      3. Friend-Q Learning: friend_q.png
      4. Foe-Q Learning: foe_q_learning.png
- sources (directory):
   - Contains sources used for the report
   - Contains a folder for somes sources used for the paper and others for background on the subject
- data (directory):
   - Location where the training algorithms will be 
   - If desired to run the process_data.py script with data, will need to unzip zipped_up.zip and copy contents out
   - This was done since the data was so big but wanted to be added to the commit for history
   - Then, process_data.py can be ran