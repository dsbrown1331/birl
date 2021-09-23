import numpy as np
import IPython
from itertools import product


class GridWorld:
    def __init__(self, grid_size_x=3, grid_size_y=3, rewards=None, trans_probs=None, gamma = 0.95):
        
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



        

        # Actions
        self.n_actions = 4
        self.action_dim = (self.n_actions,)  
        self.action_dict = {"up": 0, "right": 1, "down": 2, "left": 3}
        self.action_coords = [(0, 1), (1, 0), (0, -1), (-1, 0)]  

        # discount factor
        self.terminal_states = [8]
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




if __name__ == '__main__':
    # The expert in this environments loops s1, s2, s3
    trans_probs = np.empty(shape=(4, 2, 4), dtype=np.float32)
    loop_states = [1, 3, 2]
    a1_next_state = [s for s in range(trans_probs.shape[0]) if s not in loop_states][0]
    trans_probs[:, 1] = np.eye(4)[a1_next_state]
    for state in range(trans_probs.shape[0]):
        trans_probs[a1_next_state, 0, state] = 0 if state == a1_next_state else 1/3
    for state, a0_next_state in zip(loop_states, loop_states[1:] + [loop_states[0]]):
        trans_probs[state, 0] = np.eye(4)[a0_next_state]

    env = LoopEnv(rewards=[0, 0, 0, 1], loop_states=loop_states)
    obs = env.reset()
    for _ in range(100):
        a = np.random.randint(env.n_actions)
        obs, reward = env.step(a)
        print('obs: {}, action: {}, reward: {}'.format(obs, a, reward))