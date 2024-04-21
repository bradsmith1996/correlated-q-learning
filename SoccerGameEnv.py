import numpy as np

# Set up gym like agent for the 2 player game
class SoccerGameEnv:
   def __init__(self,verbose=False,hard_code_init=False):
      self.random_seed = 0
      self.verbose_flag = verbose
      self.field_width = 4
      self.field_depth = 2
      if self.verbose_flag:
         print("Verbose flag set to true")
      self.a_coordinate = None
      self.b_coordinate = None
      self.ball_coordinate = None
      self.poss = None
      self.hard_code_init = True
   def render(self):
      # Renders the field to the user, decodes the state vector:
      a_location = (int(self.state[0]),int(self.state[1]))
      b_location = (int(self.state[2]),int(self.state[3]))
      ball_location = (int(self.state[4]),int(self.state[5]))
      field = np.empty((self.field_depth,self.field_width),dtype='object')
      # Will be made impossible in step, but catch and through warning if
      # both players are in the same location:
      string_a_location = "A"
      string_b_location = "B"
      ball_colocated = False
      # Figure out where to "plot" everyone:
      if a_location == b_location:
         print("ERROR! Players A and B are in the same location, not possible!")
         # Fatal error:
         exit(0)
      elif a_location==ball_location:
         # Add on the BALL string to a string:
         string_a_location = "A,BALL"
         ball_colocated = True
      elif b_location==ball_location:
         # Add on the BALL string to a string:
         string_b_location = "B,BALL"
         ball_colocated = True
      # Add displays:
      if ball_colocated:
         field[a_location] = string_a_location
         field[b_location] = string_b_location
      else:
         field[a_location] = string_a_location
         field[b_location] = string_b_location
         field[ball_location] = "BALL"
      print(field)
   def reset(self,seed=0):
      # Set the numpy random seed:
      self.random_seed = seed
      np.random.seed(self.random_seed)
      # Assumption: A and B can start the game in either goal, but the ball cannot!
      # Random reset of the game:
      if not self.hard_code_init:
         random_col = np.random.randint(1,high=self.field_width-1,size=None,dtype=int)
         random_row = np.random.randint(0,high=self.field_depth,size=None,dtype=int)
         self.a_coordinate = (random_row,random_col)
         # The following ensures a and b never start off at the same location:
         self.b_coordinate = self.a_coordinate
         while self.b_coordinate == self.a_coordinate:
            random_col = np.random.randint(1,high=self.field_width-1,size=None,dtype=int)
            random_row = np.random.randint(0,high=self.field_depth,size=None,dtype=int)
            self.b_coordinate = (random_row,random_col)
         # At random, give the ball to A or B at the start of the game:
         if np.random.random(1) > 0.5:
            self.ball_coordinate = self.a_coordinate
            self.poss = "A"
         else:
            self.ball_coordinate = self.b_coordinate
            self.poss = "B"
      else:
         self.a_coordinate = (0,2)
         self.b_coordinate = (0,1)
         self.ball_coordinate = self.a_coordinate
      # Encode as follows: A, B, ball locations, (row, then col)
      self.state = str(self.a_coordinate[0])+str(self.a_coordinate[1])+ \
                   str(self.b_coordinate[0])+str(self.b_coordinate[1])+ \
                   str(self.ball_coordinate[0])+str(self.ball_coordinate[1])
      if self.verbose_flag:
         print(self.state)
      return self.state
   def reward_function(self,state):
      # Check the ball location:
      ball_col = int(self.state[-1])
      goal = False
      rewardA = 0.0 # No reward if didn't score
      rewardB = 0.0 # No reward if didn't score
      if ball_col == 0:
         # A scored!
         rewardB = -100.0
         rewardA = 100.0
         goal = True
         if self.verbose_flag:
            print("GOAL A!")
      elif ball_col == self.field_width-1:
         # B scored!
         rewardB = 100.0
         rewardA = -100.0
         goal = True
         if self.verbose_flag:
            print("GOAL B!")
      return rewardA,rewardB,goal
   def simulate_move(self,location,action):
      output_location = location
      # actions:
      # 0 -> N | 1 -> S | 2 -> E | 3 -> W | 4 -> STICK (don't move)
      if action == 0:
         # Check if can go up (north), otherwise don't modify:
         if location[0]-1 >= 0:
            output_location = (location[0]-1,location[1])
      elif action == 1:
         # Check if can go down (south), otherwise don't modify:
         if location[0]+1 <= self.field_depth-1:
            output_location = (location[0]+1,location[1])
      elif action == 2:
         # Check if can go right (east), otherwise don't modify:
         if location[1]+1 <= self.field_width-1:
            output_location = (location[0],location[1]+1)
      elif action == 3:
         # Check if can go left (west), otherwise don't modify:
         if location[1]-1 >= 0:
            output_location = (location[0],location[1]-1)
      elif action == 4:
         # Location if sticking, not moving, don't modify
         output_location = location
      else:
         print("ERROR! Invalid action requested!")
         exit(0)
      return output_location

   def step(self, a_action, b_action):
      # Randomly determine if a or b gets to move first (0 -> A, 1 -> B)
      whos_move = np.random.randint(0,high=2,size=None,dtype=int)

      # Decode positions from state:
      a_position = (int(self.state[0]),int(self.state[1]))
      b_position = (int(self.state[2]),int(self.state[3]))
      ball_position = (int(self.state[4]),int(self.state[5]))

      # Determine who has posession:
      if a_position == ball_position:
         possession = "A"
         if self.poss != "A" and self.verbose_flag:
            print("B lost possession to A!")
         self.poss = "A"
      elif b_position == ball_position:
         possession = "B"
         if self.poss != "B" and self.verbose_flag:
            print("B lost possession to A!")
         self.poss = "B"
      else:
         print("Error, one of the players must have possession")
         exit(0)
      # Simulate selected moves:
      a_simulated_position = self.simulate_move(a_position,a_action)
      b_simulated_position = self.simulate_move(b_position,b_action)

      # Corrections if bumped into an edge by the other player:
      if (a_simulated_position == a_position) and (a_simulated_position == b_simulated_position):
         # If a made an invalid move and bumps back into same position, b can't knock it out of place
         a_position = a_simulated_position
         b_position = b_position
      elif (b_simulated_position == b_position) and (a_simulated_position == b_simulated_position):
         b_position = b_simulated_position
         a_position = a_position
      # Make corrections for moving into each other (non-edges):
      elif a_simulated_position == b_simulated_position:
         # B moves into A and A has possession:
         if possession == "A" and whos_move == 0:
            # A keeps possession
            a_position = a_simulated_position
            b_position = b_position
            ball_position = a_position
         # A moves into A and B has possession:
         elif possession == "B" and whos_move == 1:
            # B keeps possession
            b_position = b_simulated_position
            a_position = a_position
            ball_position = b_position
         # B moves into A and B has possession:
         elif possession == "B" and whos_move == 0:
            # B loses possession
            a_position = a_simulated_position
            b_position = b_position
            ball_position= a_position
         # A moves into B and A has possession:
         elif possession == "A" and whos_move == 1:
            # A loses possession
            b_position = b_simulated_position
            a_position = a_position
            ball_position = b_position
         else:
            print("none happened>>>>>>>>>>>")
      else:
         a_position = a_simulated_position
         b_position = b_simulated_position
         if self.poss == "A":
            ball_position = a_position
         else:
            ball_position = b_position
      # Re-encode the state:
      self.state = str(a_position[0])+str(a_position[1])+ \
                   str(b_position[0])+str(b_position[1])+ \
                   str(ball_position[0])+str(ball_position[1])
      # Get the rewards of being in the new state:
      rewardA, rewardB, terminate = self.reward_function(self.state)
      return self.state,rewardA,rewardB,terminate