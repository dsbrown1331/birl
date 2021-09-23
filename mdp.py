import IPython
import math
import copy
import numpy as np
import mdp_utils


class MDP:
    def __init__(self, num_rows, num_cols, terminals, rewards, constraints, gamma, noise=0.1):

        """
        Markov Decision Processes (MDP):
        num_rows: number of row in a environment
        num_cols: number of columns in environment
        terminals: terminal states (sink states)
        noise: with probability 2*noise the agent will move perpendicular to desired action split evenly, 
                e.g. if taking up action, then the agent has probability noise of going right and probability noise of going left.
        """
        self.gamma = gamma
        self.num_states = num_rows * num_cols 
        self.num_actions = 4  #up:0, down:1, left:2, right:3
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.terminals = terminals
        self.rewards = rewards  # think of this
        self.constraints = constraints
        self.state_grid_map = {}
        
        temp =  [(a, b) for b in np.arange(num_rows-1,-1,-1) for a in np.arange(num_cols)]
        for i in range(self.num_states):
            self.state_grid_map[i] = temp[i]
       
        
        #initialize transitions given desired noise level
        self.transitions = np.zeros((self.num_states, self.num_actions, self.num_states))
        self.init_transition_probabilities(noise)
        # IPython.embed()


    def init_transition_probabilities(self, noise):
        # 0: up, 1 : down, 2:left, 3:right

        UP = 0
        DOWN = 1
        LEFT = 2
        RIGHT = 3
        # going UP
        for s in range(self.num_states):

            # possibility of going foward

            if s >= self.num_cols:
                self.transitions[s][UP][s - self.num_cols] = 1.0 - (2 * noise)
            else:
                self.transitions[s][UP][s] = 1.0 - (2 * noise)

            # possibility of going left
            # IPython.embed()
            if s % self.num_cols == 0:
                self.transitions[s][UP][s] = noise
            else:
                self.transitions[s][UP][s - 1] = noise

            # possibility of going right

            if s % self.num_cols < self.num_cols - 1:
                self.transitions[s][UP][s + 1] = noise
            else:
                self.transitions[s][UP][s] = noise

            # special case top left corner

            if s < self.num_cols and s % self.num_cols == 0.0:
                self.transitions[s][UP][s] = 1.0 - noise
            elif s < self.num_cols and s % self.num_cols == self.num_cols - 1:
                self.transitions[s][UP][s] = 1.0 - noise

        # going down
        for s in range(self.num_states):

            # self.num_rows = gridHeight
            # self.num_cols = gridwidth

            # possibility of going down
            if s < (self.num_rows - 1) * self.num_cols:
                self.transitions[s][DOWN][s + self.num_cols] = 1.0 - (2 * noise)
            else:
                self.transitions[s][DOWN][s] = 1.0 - (2 * noise)

            # possibility of going left
            if s % self.num_cols == 0:
                self.transitions[s][DOWN][s] = noise
            else:
                self.transitions[s][DOWN][s - 1] = noise

            # possibility of going right
            if s % self.num_cols < self.num_cols - 1:
                self.transitions[s][DOWN][s + 1] = noise
            else:
                self.transitions[s][DOWN][s] = noise

            # checking bottom right corner
            if s >= (self.num_rows - 1) * self.num_cols and s % self.num_cols == 0:
                self.transitions[s][DOWN][s] = 1.0 - noise
            elif (
                s >= (self.num_rows - 1) * self.num_cols
                and s % self.num_cols == self.num_cols - 1
            ):
                self.transitions[s][DOWN][s] = 1.0 - noise

        # going left
        # self.num_rows = gridHeight
        # self.num_cols = gridwidth
        for s in range(self.num_states):
            # possibility of going left

            if s % self.num_cols > 0:
                self.transitions[s][LEFT][s - 1] = 1.0 - (2 * noise)
            else:
                self.transitions[s][LEFT][s] = 1.0 - (2 * noise)

            # possibility of going up

            if s >= self.num_cols:
                self.transitions[s][LEFT][s - self.num_cols] = noise
            else:
                self.transitions[s][LEFT][s] = noise

            # possiblity of going down
            if s < (self.num_rows - 1) * self.num_cols:
                self.transitions[s][LEFT][s + self.num_cols] = noise
            else:
                self.transitions[s][LEFT][s] = noise

            # check  top left corner
            if s < self.num_cols and s % self.num_cols == 0:
                self.transitions[s][LEFT][s] = 1.0 - noise
            elif s >= (self.num_rows - 1) * self.num_cols and s % self.num_cols == 0:
                self.transitions[s][LEFT][s] = 1 - noise

        # going right
        for s in range(self.num_states):

            if s % self.num_cols < self.num_cols - 1:
                self.transitions[s][RIGHT][s + 1] = 1.0 - (2 * noise)
            else:
                self.transitions[s][RIGHT][s] = 1.0 - (2 * noise)

            # possibility of going up

            if s >= self.num_cols:
                self.transitions[s][RIGHT][s - self.num_cols] = noise
            else:
                self.transitions[s][RIGHT][s] = noise

            # possibility of going down

            if s < (self.num_rows - 1) * self.num_cols:
                self.transitions[s][RIGHT][s + self.num_cols] = noise
            else:
                self.transitions[s][RIGHT][s] = noise

            # check top right corner
            if (s < self.num_cols) and (s % self.num_cols == self.num_cols - 1):
                self.transitions[s][RIGHT][s] = 1 - noise
            # check bottom rihgt corner case
            elif (
                s >= (self.num_rows - 1) * self.num_cols
                and s % self.num_cols == self.num_cols - 1
            ):
                self.transitions[s][RIGHT][s] = 1.0 - noise

        for s in range(self.num_states):
            if s in self.terminals:
                for a in range(self.num_actions):
                    for s2 in range(self.num_states):
                        self.transitions[s][a][s2] = 0.0

        # transition function not changed at all, just not allow action to take agent into a constraint
        # for s in range(self.num_states):        
        #     for a in range(self.num_actions):
        #         for s2 in range(self.num_states):
                    
        #             if s2 in np.nonzero(self.constraints)[0]:
        #                 self.transitions[s][a][s2] = 0.0

        # for s in range(self.num_states):
        #     if s in np.nonzero(self.constraints)[0]:
        #         for a in range(self.num_actions):
        #             for s2 in range(self.num_states):
        #                 self.transitions[s][a][s2] = 0.0

    
    def set_rewards(self, _rewards):
        self.rewards = _rewards

    def set_constraints(self, _constraints):
        self.constraints = _constraints

    def set_gamma(self, gamma):
        assert(gamma < 1.0 and gamma > 0.0)
        self.gamma = gamma



class FeatureMDP(MDP):
    """
    Featurized Linear Reward Function Markov Decision Processes (MDP):
    same as MDP but the reward function is now a linear combination of features
    feature_weights: the weights on the features
    state_features: a matrix or list of lists where each row is the state features at a particular state

    e.g. if each cell in a 3x3 non-terminal grid world has a color: red or white, and red states have as reward of -1
    and white states have a reward of +1, then this could be created by having
    feature_weights = np.array([-1.0, +1.0])
    r = np.array([1,0]) #features for red
    w = np.array([0,1]) #features for white
    state_features = [w, w, w,
                        r, r, w
                        r, w, w]
    mdp = FeatureMDP(3,3,[], reward_weights, reward_features, 0.95, noise = 0)

    """
        
    def __init__(self, num_rows, num_cols, terminals, feature_weights, state_features, gamma, noise = 0.0):

        assert(num_rows * num_cols == len(state_features))
        assert(len(state_features[0]) == len(feature_weights))

        #rewards are linear combination of features
        rewards = np.dot(state_features, feature_weights)
        super().__init__(num_rows, num_cols, terminals, rewards, gamma, noise)

        self.feature_weights = feature_weights
        self.state_features = state_features


    
    def set_rewards(self, _feature_weights):
        ''' set reward weights and update state rewards everywhere'''
        #check and make sure reward weights are right size

        # print("len(_reward_weights", len(_reward_weights), _reward_weights)
        # print("len(self.reward_weights)", len(self.reward_weights), self.reward_weights)
        assert(len(_feature_weights) == len(self.feature_weights))

        self.rewards = np.dot(self.state_features, _feature_weights)
        self.feature_weights = _feature_weights



class GridWorld:
    def __init__(self, grid_size_x=3, grid_size_y=3, terminal_states=[], rewards=None, trans_probs=None, gamma = 0.95):
        


        # self.gamma = gamma
        # self.num_states = num_rows * num_cols 
        # self.num_actions = 4  #up:0, down:1, left:2, right:3
        # self.num_rows = num_rows
        # self.num_cols = num_cols
        # self.terminals = terminals
        # self.rewards = rewards  # think of this
        
        # #initialize transitions given desired noise level
        # self.transitions = np.zeros((self.num_states, self.num_actions, self.num_states))
        # self.init_transition_probabilities(noise)


        # Grid size Nx times Ny
        self.Nx = grid_size_x  
        self.Ny = grid_size_y  
        self.state_dim = (self.Nx, self.Ny)
        self.n_states = self.Nx * self.Ny
        self.states = np.arange(self.n_states)
        self.grid = np.zeros((self.Nx, self.Ny))
        self.state_grid_map = {}
        temp =  [(a, b) for a in np.arange(self.Nx) for b in np.arange(self.Ny)]
        for i in range(self.n_states):
            self.state_grid_map[i] = temp[i]


        # temp =  [(a, b) for b in np.arange(num_rows-1,-1,-1) for a in np.arange(num_cols)]
        # for i in range(self.num_states):
        #     self.state_grid_map[i] = temp[i]



        

        # Actions
        self.n_actions = 4
        self.action_dim = (self.n_actions,)  
        self.action_dict = {"up": 0, "right": 1, "down": 2, "left": 3}
        self.action_coords = [(0, 1), (1, 0), (0, -1), (-1, 0)]  

        # discount factor
        self.terminal_states = terminal_states
        self.trans_probs = self.transition_probs()
        self.rewards = np.array([-1, -1, -1, -1, -1, -1, -1, -1, 1])
        self.gamma = gamma
        # IPython.embed()


    def transition_probs(self):
       
        trans_probs = np.zeros(shape=(self.n_states, self.n_actions, self.n_states), dtype=np.float32)
        
        for state in range(self.n_states): # -1 b3cause last one is the money state
            if state not in self.terminal_states:
                for action in range(self.n_actions):

                    state_next, _, _ = self.step(action, self.state_grid_map[state])
                    temp_state = list(self.state_grid_map.keys())[list(self.state_grid_map.values()).index(state_next)]
                    trans_probs[state, action, temp_state] = 1

        return trans_probs


    def step(self, action, state):
        # Evolve agent state
        state_next = (state[0] + self.action_coords[action][0],
                      state[1] + self.action_coords[action][1])
        #print(state,self.state)
        if state[0] == self.Nx - 1 and action == 1:
            state_next = state
        elif state[0] == 0 and action == 3:
            state_next = state
        elif state[1] == self.Ny - 1 and action == 0:
            state_next = state
        elif state[1] == 0 and action == 2:
            state_next = state    

        
        reward = 0#self.get_rewards(state)#################                               check whther state or next state
        done = (state[0] == self.Ny - 1) and (state[1] == self.Nx - 1)
        return state_next, reward, done
    
    def allowed_actions(self):
        # Generate list of actions allowed depending on agent grid location
        actions_allowed = []
        y, x = self.state[0], self.state[1]
        if (y > 0):  # no passing top-boundary
            actions_allowed.append(self.action_dict["up"])
        if (y < self.Ny - 1):  # no passing bottom-boundary
            actions_allowed.append(self.action_dict["down"])
        if (x > 0):  # no passing left-boundary
            actions_allowed.append(self.action_dict["left"])
        if (x < self.Nx - 1):  # no passing right-boundary
            actions_allowed.append(self.action_dict["right"])
        ### ADDED ###
        if (x == self.Nx - 1 and y == self.Ny - 1 ):  # no passing right-boundary
            actions_allowed.append(self.action_dict["up"])
            actions_allowed.append(self.action_dict["down"])
            actions_allowed.append(self.action_dict["left"])
            actions_allowed.append(self.action_dict["right"])
        

        actions_allowed.append(self.action_dict["up"])
        actions_allowed.append(self.action_dict["down"])
        actions_allowed.append(self.action_dict["left"])
        actions_allowed.append(self.action_dict["right"])
        actions_allowed = np.array(actions_allowed, dtype=int)
        actions_allowed = np.unique(actions_allowed)
        return actions_allowed
   
    
    def get_rewards(self, state):
        
        rewards = np.array(self.rewards).reshape((self.Ny, self.Nx))    
        return rewards[state[0], state[1]]

    def reset(self):
        # Reset agent state to top-left grid corner
        self.state = (0, 0)  
        return self.state





if __name__ =="__main__":

    '''Here's a simple example of how to use the FeatureMDP class'''

    
    #three features, red (-1), blue (+1), white (0)
    r = [1,0]
    b = [0,1]
    w = [0,0]
    #create state features for a 2x2 grid (really just an array, but I'm formating it to look like the grid world)
    state_features = [b, r, 
                      w, w]
    feature_weights = [-1.0, 1.0] #red feature has weight -1 and blue feature has weight +1
    gamma = 0.5
    noise = 0.0
    eps = 0.0001
    env = FeatureMDP(2,2,[0],feature_weights, state_features, gamma, noise)
    
    mdp_utils.value_iteration(env)