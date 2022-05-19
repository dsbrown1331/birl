
import math
import copy
import numpy as np
import mdp_utils


class MDP:
    def __init__(self, num_rows, num_cols, num_actions, terminals, rewards, gamma, noise = 0.1, driving = False):

        """
        Markov Decision Processes (MDP):
        num_rows: number of row in a environment
        num_cols: number of columns in environment
        terminals: terminal states (sink states)
        noise: with probability 2*noise the agent will move perpendicular to desired action split evenly, 
                e.g. if taking up action, then the agent has probability noise of going right and probability noise of going left.
        """
        self.gamma = gamma
        if not driving:
            self.num_states = num_rows * num_cols 
        else: # assuming three speeds for driving
            self.num_states = num_rows * num_cols * 3
        self.num_actions = num_actions #up:0, down:1, left:2, right:3
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.terminals = terminals
        self.rewards = rewards  # think of this
        
        #initialize transitions given desired noise level
        self.transitions = np.zeros((self.num_states, self.num_actions, self.num_states))
        self.init_transition_probabilities(noise, driving)


    def init_transition_probabilities(self, noise, driving = False):
        if not driving:
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

            # handle terminals
            for s in range(self.num_states):
                if s in self.terminals:
                    for a in range(self.num_actions):
                        for s2 in range(self.num_states):
                            self.transitions[s][a][s2] = 0.0
        
        else:
            # 0: left & decelerate, 1: left & cruise, 2: left & accelerate
            # 3: straight & decelerate, 4: straight & cruise, 5: straight & accelerate
            # 6: right & decelerate, 7: right & cruise, 8: right & accelerate
            LEFT_DECEL = 0
            LEFT_CRUISE = 1
            LEFT_ACCEL = 2
            STRAIGHT_DECEL = 3
            STRAIGHT_CRUISE = 4
            STRAIGHT_ACCEL = 5
            RIGHT_DECEL = 6
            RIGHT_CRUISE = 7
            RIGHT_ACCEL = 8

            shift = self.num_rows * self.num_cols
            road_ends = np.array(range(self.num_cols * (self.num_rows - 1), self.num_cols * self.num_rows))
            all_road_ends = np.concatenate((road_ends, road_ends + shift, road_ends + shift * 2))
            speed_0_states = np.array(range(shift))
            speed_1_states = np.array(range(shift, shift * 2))
            speed_2_states = np.array(range(shift * 2, shift * 3))
            def lb(s): # is state left border?
                return s % self.num_cols == 0
            def rb(s): # is state right border?
                return s % self.num_cols == self.num_cols - 1

            # LEFT, DECEL
            for s in range(self.num_states):
                # move to correct location and speed and account for wraparound
                if 0 <= s < shift:
                    if not lb(s):
                        left_state = (s - 1) % self.num_cols if (s - 1) in all_road_ends else s - 1
                    else:
                        left_state = s % self.num_cols if s in all_road_ends else s
                    straight_state = s % self.num_cols if s in all_road_ends else s
                    if not rb(s):
                        right_state = (s + 1) % self.num_cols if (s + 1) in all_road_ends else s + 1
                    else:
                        right_state = s % self.num_cols if s in all_road_ends else s
                elif shift <= s < shift * 2:
                    if not lb(s):
                        left_state = ((s - 1) % self.num_cols if (s - 1) in all_road_ends else s - 1) - shift
                    else:
                        left_state = (s % self.num_cols if s in all_road_ends else s) - shift
                    straight_state = (s % self.num_cols if s in all_road_ends else s) - shift
                    if not rb(s):
                        right_state = ((s + 1) % self.num_cols if (s + 1) in all_road_ends else s + 1) - shift
                    else:
                        right_state = (s % self.num_cols if s in all_road_ends else s) - shift
                else:
                    if not lb(s):
                        left_state = ((s + 4) % self.num_cols if (s + 4) in all_road_ends else s + 4) - shift
                    else:
                        left_state = ((s + 5) % self.num_cols if (s + 5) in all_road_ends else s + 5) - shift
                    straight_state = ((s + 5) % self.num_cols if (s + 5) in all_road_ends else s + 5) - shift
                    if not rb(s):
                        right_state = ((s + 6) % self.num_cols if (s + 6) in all_road_ends else s + 6) - shift
                    else:
                        right_state = ((s + 5) % self.num_cols if (s + 5) in all_road_ends else s + 5) - shift
                # transitions for each lane
                if s % self.num_cols == 0: # left border
                    self.transitions[s][LEFT_DECEL][left_state] = 0.0
                    self.transitions[s][LEFT_DECEL][straight_state] = 0.5
                    self.transitions[s][LEFT_DECEL][right_state] = 0.5
                elif s % self.num_cols == self.num_cols - 1: # right border
                    self.transitions[s][LEFT_DECEL][left_state] = 1.0 - noise
                    self.transitions[s][LEFT_DECEL][straight_state] = noise
                    self.transitions[s][LEFT_DECEL][right_state] = 0.0
                else: # main lanes
                    self.transitions[s][LEFT_DECEL][left_state] = 1.0 - (2 * noise)
                    self.transitions[s][LEFT_DECEL][straight_state] = noise
                    self.transitions[s][LEFT_DECEL][right_state] = noise
            
            # LEFT, CRUISE
            for s in range(self.num_states):
                # move to correct location and speed and account for wraparound
                if 0 <= s < shift:
                    if not lb(s):
                        left_state = (s - 1) % self.num_cols if (s - 1) in all_road_ends else s - 1
                    else:
                        left_state = s % self.num_cols if s in all_road_ends else s
                    straight_state = s % self.num_cols if s in all_road_ends else s
                    if not rb(s):
                        right_state = (s + 1) % self.num_cols if (s + 1) in all_road_ends else s + 1
                    else:
                        right_state = s % self.num_cols if s in all_road_ends else s
                elif shift <= s < shift * 2:
                    if not lb(s):
                        left_state = ((s + 4) % self.num_cols if (s + 4) in all_road_ends else s + 4)
                    else:
                        left_state = ((s + 5) % self.num_cols if (s + 5) in all_road_ends else s + 5)
                    straight_state = ((s + 5) % self.num_cols if (s + 5) in all_road_ends else s + 5)
                    if not rb(s):
                        right_state = ((s + 6) % self.num_cols if (s + 6) in all_road_ends else s + 6)
                    else:
                        right_state = ((s + 5) % self.num_cols if (s + 5) in all_road_ends else s + 5)
                else:
                    if not lb(s):
                        left_state = ((s + 9) % self.num_cols if (s + 9) in all_road_ends or (s + 9) >= shift * 3 else s + 9)
                    else:
                        left_state = ((s + 10) % self.num_cols if (s + 10) in all_road_ends or (s + 10) >= shift * 3 else s + 10)
                    straight_state = ((s + 10) % self.num_cols if (s + 10) in all_road_ends or (s + 10) >= shift * 3 else s + 10)
                    if not rb(s):
                        right_state = ((s + 11) % self.num_cols if (s + 11) in all_road_ends or (s + 11) >= shift * 3 else s + 11)
                    else:
                        right_state = ((s + 10) % self.num_cols if (s + 10) in all_road_ends or (s + 10) >= shift * 3 else s + 10)
                # transitions for each lane
                if s % self.num_cols == 0: # left border
                    self.transitions[s][LEFT_CRUISE][left_state] = 0.0
                    self.transitions[s][LEFT_CRUISE][straight_state] = 0.5
                    self.transitions[s][LEFT_CRUISE][right_state] = 0.5
                elif s % self.num_cols == self.num_cols - 1: # right border
                    self.transitions[s][LEFT_CRUISE][left_state] = 1.0 - noise
                    self.transitions[s][LEFT_CRUISE][straight_state] = noise
                    self.transitions[s][LEFT_CRUISE][right_state] = 0.0
                else: # main lanes
                    self.transitions[s][LEFT_CRUISE][left_state] = 1.0 - (2 * noise)
                    self.transitions[s][LEFT_CRUISE][straight_state] = noise
                    self.transitions[s][LEFT_CRUISE][right_state] = noise
            
            # LEFT, ACCEL
            for s in range(self.num_states):
                # move to correct location and speed and account for wraparound
                if 0 <= s < shift:
                    if not lb(s):
                        left_state = ((s + 4) % self.num_cols if (s + 4) in all_road_ends else s + 4) + shift
                    else:
                        left_state = ((s + 5) % self.num_cols if (s + 5) in all_road_ends else s + 5) + shift
                    straight_state = ((s + 5) % self.num_cols if (s + 5) in all_road_ends else s + 5) + shift
                    if not rb(s):
                        right_state = ((s + 6) % self.num_cols if (s + 6) in all_road_ends else s + 6) + shift
                    else:
                        right_state = ((s + 5) % self.num_cols if (s + 5) in all_road_ends else s + 5) + shift
                elif shift <= s < shift * 2:
                    if not lb(s):
                        left_state = ((s + 9) % self.num_cols if (s + 9) in all_road_ends else s + 9) + shift
                    else:
                        left_state = ((s + 10) % self.num_cols if (s + 10) in all_road_ends else s + 10) + shift
                    if left_state >= shift * 3:
                            left_state %= self.num_cols
                    straight_state = ((s + 10) % self.num_cols if (s + 10) in all_road_ends else s + 10) + shift
                    if straight_state >= shift * 3:
                            straight_state %= self.num_cols
                    if not rb(s):
                        right_state = ((s + 11) % self.num_cols if (s + 11) in all_road_ends else s + 11) + shift
                    else:
                        right_state = ((s + 10) % self.num_cols if (s + 10) in all_road_ends else s + 10) + shift
                    if right_state >= shift * 3:
                            right_state %= self.num_cols
                else:
                    if not lb(s):
                        left_state = ((s + 9) % self.num_cols if (s + 9) in all_road_ends or (s + 9) >= shift * 3 else s + 9)
                    else:
                        left_state = ((s + 10) % self.num_cols if (s + 10) in all_road_ends or (s + 10) >= shift * 3 else s + 10)
                    straight_state = ((s + 10) % self.num_cols if (s + 10) in all_road_ends or (s + 10) >= shift * 3 else s + 10)
                    if not rb(s):
                        right_state = ((s + 11) % self.num_cols if (s + 11) in all_road_ends or (s + 11) >= shift * 3 else s + 11)
                    else:
                        right_state = ((s + 10) % self.num_cols if (s + 10) in all_road_ends or (s + 10) >= shift * 3 else s + 10)
                # transitions for each lane
                if s % self.num_cols == 0: # left border
                    self.transitions[s][LEFT_ACCEL][left_state] = 0.0
                    self.transitions[s][LEFT_ACCEL][straight_state] = 0.5
                    self.transitions[s][LEFT_ACCEL][right_state] = 0.5
                elif s % self.num_cols == self.num_cols - 1: # right border
                    self.transitions[s][LEFT_ACCEL][left_state] = 1.0 - noise
                    self.transitions[s][LEFT_ACCEL][straight_state] = noise
                    self.transitions[s][LEFT_ACCEL][right_state] = 0.0
                else: # main lanes
                    self.transitions[s][LEFT_ACCEL][left_state] = 1.0 - (2 * noise)
                    self.transitions[s][LEFT_ACCEL][straight_state] = noise
                    self.transitions[s][LEFT_ACCEL][right_state] = noise

            # STRAIGHT, DECEL
            for s in range(self.num_states):
                # move to correct location and speed and account for wraparound
                if 0 <= s < shift:
                    if not lb(s):
                        left_state = (s - 1) % self.num_cols if (s - 1) in all_road_ends else s - 1
                    else:
                        left_state = s % self.num_cols if s in all_road_ends else s
                    straight_state = s % self.num_cols if s in all_road_ends else s
                    if not rb(s):
                        right_state = (s + 1) % self.num_cols if (s + 1) in all_road_ends else s + 1
                    else:
                        right_state = s % self.num_cols if s in all_road_ends else s
                elif shift <= s < shift * 2:
                    if not lb(s):
                        left_state = ((s - 1) % self.num_cols if (s - 1) in all_road_ends else s - 1) - shift
                    else:
                        left_state = (s % self.num_cols if s in all_road_ends else s) - shift
                    straight_state = (s % self.num_cols if s in all_road_ends else s) - shift
                    if not rb(s):
                        right_state = ((s + 1) % self.num_cols if (s + 1) in all_road_ends else s + 1) - shift
                    else:
                        right_state = (s % self.num_cols if s in all_road_ends else s) - shift
                else:
                    if not lb(s):
                        left_state = ((s + 4) % self.num_cols if (s + 4) in all_road_ends else s + 4) - shift
                    else:
                        left_state = ((s + 5) % self.num_cols if (s + 5) in all_road_ends else s + 5) - shift
                    straight_state = ((s + 5) % self.num_cols if (s + 5) in all_road_ends else s + 5) - shift
                    if not rb(s):
                        right_state = ((s + 6) % self.num_cols if (s + 6) in all_road_ends else s + 6) - shift
                    else:
                        right_state = ((s + 5) % self.num_cols if (s + 5) in all_road_ends else s + 5) - shift
                # transitions for each lane
                if s % self.num_cols == 0: # left border
                    self.transitions[s][STRAIGHT_DECEL][left_state] = 0.0
                    self.transitions[s][STRAIGHT_DECEL][straight_state] = 1.0 - noise
                    self.transitions[s][STRAIGHT_DECEL][right_state] = noise
                elif s % self.num_cols == self.num_cols - 1: # right border
                    self.transitions[s][STRAIGHT_DECEL][left_state] = noise
                    self.transitions[s][STRAIGHT_DECEL][straight_state] = 1.0 - noise
                    self.transitions[s][STRAIGHT_DECEL][right_state] = 0.0
                else: # main lanes
                    self.transitions[s][STRAIGHT_DECEL][left_state] = noise
                    self.transitions[s][STRAIGHT_DECEL][straight_state] = 1.0 - (2 * noise)
                    self.transitions[s][STRAIGHT_DECEL][right_state] = noise

            # STRAIGHT, CRUISE
            for s in range(self.num_states):
                # move to correct location and speed and account for wraparound
                if 0 <= s < shift:
                    if not lb(s):
                        left_state = (s - 1) % self.num_cols if (s - 1) in all_road_ends else s - 1
                    else:
                        left_state = s % self.num_cols if s in all_road_ends else s
                    straight_state = s % self.num_cols if s in all_road_ends else s
                    if not rb(s):
                        right_state = (s + 1) % self.num_cols if (s + 1) in all_road_ends else s + 1
                    else:
                        right_state = s % self.num_cols if s in all_road_ends else s
                elif shift <= s < shift * 2:
                    if not lb(s):
                        left_state = ((s + 4) % self.num_cols if (s + 4) in all_road_ends else s + 4)
                    else:
                        left_state = ((s + 5) % self.num_cols if (s + 5) in all_road_ends else s + 5)
                    straight_state = ((s + 5) % self.num_cols if (s + 5) in all_road_ends else s + 5)
                    if not rb(s):
                        right_state = ((s + 6) % self.num_cols if (s + 6) in all_road_ends else s + 6)
                    else:
                        right_state = ((s + 5) % self.num_cols if (s + 5) in all_road_ends else s + 5)
                else:
                    if not lb(s):
                        left_state = ((s + 9) % self.num_cols if (s + 9) in all_road_ends or (s + 9) >= shift * 3 else s + 9)
                    else:
                        left_state = ((s + 10) % self.num_cols if (s + 10) in all_road_ends or (s + 10) >= shift * 3 else s + 10)
                    straight_state = ((s + 10) % self.num_cols if (s + 10) in all_road_ends or (s + 10) >= shift * 3 else s + 10)
                    if not rb(s):
                        right_state = ((s + 11) % self.num_cols if (s + 11) in all_road_ends or (s + 11) >= shift * 3 else s + 11)
                    else:
                        right_state = ((s + 10) % self.num_cols if (s + 10) in all_road_ends or (s + 10) >= shift * 3 else s + 10)
                # transitions for each lane
                if s % self.num_cols == 0: # left border
                    self.transitions[s][STRAIGHT_CRUISE][left_state] = 0.0
                    self.transitions[s][STRAIGHT_CRUISE][straight_state] = 1.0 - noise
                    self.transitions[s][STRAIGHT_CRUISE][right_state] = noise
                elif s % self.num_cols == self.num_cols - 1: # right border
                    self.transitions[s][STRAIGHT_CRUISE][left_state] = noise
                    self.transitions[s][STRAIGHT_CRUISE][straight_state] = 1.0 - noise
                    self.transitions[s][STRAIGHT_CRUISE][right_state] = 0.0
                else: # main lanes
                    self.transitions[s][STRAIGHT_CRUISE][left_state] = noise
                    self.transitions[s][STRAIGHT_CRUISE][straight_state] = 1.0 - (2 * noise)
                    self.transitions[s][STRAIGHT_CRUISE][right_state] = noise

            # STRAIGHT, ACCEL
            for s in range(self.num_states):
                # move to correct location and speed and account for wraparound
                if 0 <= s < shift:
                    if not lb(s):
                        left_state = ((s + 4) % self.num_cols if (s + 4) in all_road_ends else s + 4) + shift
                    else:
                        left_state = ((s + 5) % self.num_cols if (s + 5) in all_road_ends else s + 5) + shift
                    straight_state = ((s + 5) % self.num_cols if (s + 5) in all_road_ends else s + 5) + shift
                    if not rb(s):
                        right_state = ((s + 6) % self.num_cols if (s + 6) in all_road_ends else s + 6) + shift
                    else:
                        right_state = ((s + 5) % self.num_cols if (s + 5) in all_road_ends else s + 5) + shift
                elif shift <= s < shift * 2:
                    if not lb(s):
                        left_state = ((s + 9) % self.num_cols if (s + 9) in all_road_ends or (s + 9) >= shift * 3 else s + 9) + shift
                    else:
                        left_state = ((s + 10) % self.num_cols if (s + 10) in all_road_ends or (s + 10) >= shift * 3 else s + 10) + shift
                    if left_state >= shift * 3:
                        left_state %= self.num_cols
                    straight_state = ((s + 10) % self.num_cols if (s + 10) in all_road_ends or (s + 10) >= shift * 3 else s + 10) + shift
                    if straight_state >= shift * 3:
                        straight_state %= self.num_cols
                    if not rb(s):
                        right_state = ((s + 11) % self.num_cols if (s + 11) in all_road_ends or (s + 11) >= shift * 3 else s + 11) + shift
                    else:
                        right_state = ((s + 10) % self.num_cols if (s + 10) in all_road_ends or (s + 10) >= shift * 3 else s + 10) + shift
                    if right_state >= shift * 3:
                        right_state %= self.num_cols
                else:
                    if not lb(s):
                        left_state = ((s + 9) % self.num_cols if (s + 9) in all_road_ends or (s + 9) >= shift * 3 else s + 9)
                    else:
                        left_state = ((s + 10) % self.num_cols if (s + 10) in all_road_ends or (s + 10) >= shift * 3 else s + 10)
                    straight_state = ((s + 10) % self.num_cols if (s + 10) in all_road_ends or (s + 10) >= shift * 3 else s + 10)
                    if not rb(s):
                        right_state = ((s + 11) % self.num_cols if (s + 11) in all_road_ends or (s + 11) >= shift * 3 else s + 11)
                    else:
                        right_state = ((s + 10) % self.num_cols if (s + 10) in all_road_ends or (s + 10) >= shift * 3 else s + 10)
                # transitions for each lane
                if s % self.num_cols == 0: # left border
                    self.transitions[s][STRAIGHT_ACCEL][left_state] = 0.0
                    self.transitions[s][STRAIGHT_ACCEL][straight_state] = 1.0 - noise
                    self.transitions[s][STRAIGHT_ACCEL][right_state] = noise
                elif s % self.num_cols == self.num_cols - 1: # right border
                    self.transitions[s][STRAIGHT_ACCEL][left_state] = noise
                    self.transitions[s][STRAIGHT_ACCEL][straight_state] = 1.0 - noise
                    self.transitions[s][STRAIGHT_ACCEL][right_state] = 0.0
                else: # main lanes
                    self.transitions[s][STRAIGHT_ACCEL][left_state] = noise
                    self.transitions[s][STRAIGHT_ACCEL][straight_state] = 1.0 - (2 * noise)
                    self.transitions[s][STRAIGHT_ACCEL][right_state] = noise
            
            # RIGHT, DECEL
            for s in range(self.num_states):
                # move to correct location and speed and account for wraparound
                if 0 <= s < shift:
                    if not lb(s):
                        left_state = (s - 1) % self.num_cols if (s - 1) in all_road_ends else s - 1
                    else:
                        left_state = s % self.num_cols if s in all_road_ends else s
                    straight_state = s % self.num_cols if s in all_road_ends else s
                    if not rb(s):
                        right_state = (s + 1) % self.num_cols if (s + 1) in all_road_ends else s + 1
                    else:
                        right_state = s % self.num_cols if s in all_road_ends else s
                elif shift <= s < shift * 2:
                    if not lb(s):
                        left_state = ((s - 1) % self.num_cols if (s - 1) in all_road_ends else s - 1) - shift
                    else:
                        left_state = (s % self.num_cols if s in all_road_ends else s) - shift
                    straight_state = (s % self.num_cols if s in all_road_ends else s) - shift
                    if not rb(s):
                        right_state = ((s + 1) % self.num_cols if (s + 1) in all_road_ends else s + 1) - shift
                    else:
                        right_state = (s % self.num_cols if s in all_road_ends else s) - shift
                else:
                    if not lb(s):
                        left_state = ((s + 4) % self.num_cols if (s + 4) in all_road_ends else s + 4) - shift
                    else:
                        left_state = ((s + 5) % self.num_cols if (s + 5) in all_road_ends else s + 5) - shift
                    straight_state = ((s + 5) % self.num_cols if (s + 5) in all_road_ends else s + 5) - shift
                    if not rb(s):
                        right_state = ((s + 6) % self.num_cols if (s + 6) in all_road_ends else s + 6) - shift
                    else:
                        right_state = ((s + 5) % self.num_cols if (s + 5) in all_road_ends else s + 5) - shift
                # transitions for each lane
                if s % self.num_cols == 0: # left border
                    self.transitions[s][RIGHT_DECEL][left_state] = 0.0
                    self.transitions[s][RIGHT_DECEL][straight_state] = noise
                    self.transitions[s][RIGHT_DECEL][right_state] = 1.0 - noise
                elif s % self.num_cols == self.num_cols - 1: # right border
                    self.transitions[s][RIGHT_DECEL][left_state] = 0.5
                    self.transitions[s][RIGHT_DECEL][straight_state] = 0.5
                    self.transitions[s][RIGHT_DECEL][right_state] = 0.0
                else: # main lanes
                    self.transitions[s][RIGHT_DECEL][left_state] = noise
                    self.transitions[s][RIGHT_DECEL][straight_state] = noise
                    self.transitions[s][RIGHT_DECEL][right_state] = 1.0 - (2 * noise)

            # RIGHT, CRUISE
            for s in range(self.num_states):
                # move to correct location and speed and account for wraparound
                if 0 <= s < shift:
                    if not lb(s):
                        left_state = (s - 1) % self.num_cols if (s - 1) in all_road_ends else s - 1
                    else:
                        left_state = s % self.num_cols if s in all_road_ends else s
                    straight_state = s % self.num_cols if s in all_road_ends else s
                    if not rb(s):
                        right_state = (s + 1) % self.num_cols if (s + 1) in all_road_ends else s + 1
                    else:
                        right_state = s % self.num_cols if s in all_road_ends else s
                elif shift <= s < shift * 2:
                    if not lb(s):
                        left_state = ((s + 4) % self.num_cols if (s + 4) in all_road_ends else s + 4)
                    else:
                        left_state = ((s + 5) % self.num_cols if (s + 5) in all_road_ends else s + 5)
                    straight_state = ((s + 5) % self.num_cols if (s + 5) in all_road_ends else s + 5)
                    if not rb(s):
                        right_state = ((s + 6) % self.num_cols if (s + 6) in all_road_ends else s + 6)
                    else:
                        right_state = ((s + 5) % self.num_cols if (s + 5) in all_road_ends else s + 5)
                else:
                    if not lb(s):
                        left_state = ((s + 9) % self.num_cols if (s + 9) in all_road_ends or (s + 9) >= shift * 3 else s + 9)
                    else:
                        left_state = ((s + 10) % self.num_cols if (s + 10) in all_road_ends or (s + 10) >= shift * 3 else s + 10)
                    straight_state = ((s + 10) % self.num_cols if (s + 10) in all_road_ends or (s + 10) >= shift * 3 else s + 10)
                    if not rb(s):
                        right_state = ((s + 11) % self.num_cols if (s + 11) in all_road_ends or (s + 11) >= shift * 3 else s + 11)
                    else:
                        right_state = ((s + 10) % self.num_cols if (s + 10) in all_road_ends or (s + 10) >= shift * 3 else s + 10)
                # transitions for each lane
                if s % self.num_cols == 0: # left border
                    self.transitions[s][RIGHT_CRUISE][left_state] = 0.0
                    self.transitions[s][RIGHT_CRUISE][straight_state] = noise
                    self.transitions[s][RIGHT_CRUISE][right_state] = 1.0 - noise
                elif s % self.num_cols == self.num_cols - 1: # right border
                    self.transitions[s][RIGHT_CRUISE][left_state] = 0.5
                    self.transitions[s][RIGHT_CRUISE][straight_state] = 0.5
                    self.transitions[s][RIGHT_CRUISE][right_state] = 0.0
                else: # main lanes
                    self.transitions[s][RIGHT_CRUISE][left_state] = noise
                    self.transitions[s][RIGHT_CRUISE][straight_state] = noise
                    self.transitions[s][RIGHT_CRUISE][right_state] = 1.0 - (2 * noise)

            # RIGHT, ACCEL
            for s in range(self.num_states):
                # move to correct location and speed and account for wraparound
                if 0 <= s < shift:
                    if not lb(s):
                        left_state = ((s + 4) % self.num_cols if (s + 4) in all_road_ends else s + 4) + shift
                    else:
                        left_state = ((s + 5) % self.num_cols if (s + 5) in all_road_ends else s + 5) + shift
                    straight_state = ((s + 5) % self.num_cols if (s + 5) in all_road_ends else s + 5) + shift
                    if not rb(s):
                        right_state = ((s + 6) % self.num_cols if (s + 6) in all_road_ends else s + 6) + shift
                    else:
                        right_state = ((s + 5) % self.num_cols if (s + 5) in all_road_ends else s + 5) + shift
                elif shift <= s < shift * 2:
                    if not lb(s):
                        left_state = ((s + 9) % self.num_cols if (s + 9) in all_road_ends or (s + 9) >= shift * 3 else s + 9) + shift
                    else:
                        left_state = ((s + 10) % self.num_cols if (s + 10) in all_road_ends or (s + 10) >= shift * 3 else s + 10) + shift
                    if left_state >= shift * 3:
                        left_state %= self.num_cols
                    straight_state = ((s + 10) % self.num_cols if (s + 10) in all_road_ends or (s + 10) >= shift * 3 else s + 10) + shift
                    if straight_state >= shift * 3:
                        straight_state %= self.num_cols
                    if not rb(s):
                        right_state = ((s + 11) % self.num_cols if (s + 11) in all_road_ends or (s + 11) >= shift * 3 else s + 11) + shift
                    else:
                        right_state = ((s + 10) % self.num_cols if (s + 10) in all_road_ends or (s + 10) >= shift * 3 else s + 10) + shift
                    if right_state >= shift * 3:
                        right_state %= self.num_cols
                else:
                    if not lb(s):
                        left_state = ((s + 9) % self.num_cols if (s + 9) in all_road_ends or (s + 9) >= shift * 3 else s + 9)
                    else:
                        left_state = ((s + 10) % self.num_cols if (s + 10) in all_road_ends or (s + 10) >= shift * 3 else s + 10)
                    straight_state = ((s + 10) % self.num_cols if (s + 10) in all_road_ends or (s + 10) >= shift * 3 else s + 10)
                    if not rb(s):
                        right_state = ((s + 11) % self.num_cols if (s + 11) in all_road_ends or (s + 11) >= shift * 3 else s + 11)
                    else:
                        right_state = ((s + 10) % self.num_cols if (s + 10) in all_road_ends or (s + 11) >= shift * 3 else s + 10)
                # transitions for each lane
                if s % self.num_cols == 0: # left border
                    self.transitions[s][RIGHT_ACCEL][left_state] = 0.0
                    self.transitions[s][RIGHT_ACCEL][straight_state] = noise
                    self.transitions[s][RIGHT_ACCEL][right_state] = 1.0 - noise
                elif s % self.num_cols == self.num_cols - 1: # right border
                    self.transitions[s][RIGHT_ACCEL][left_state] = 0.5
                    self.transitions[s][RIGHT_ACCEL][straight_state] = 0.5
                    self.transitions[s][RIGHT_ACCEL][right_state] = 0.0
                else: # main lanes
                    self.transitions[s][RIGHT_ACCEL][left_state] = noise
                    self.transitions[s][RIGHT_ACCEL][straight_state] = noise
                    self.transitions[s][RIGHT_ACCEL][right_state] = 1.0 - (2 * noise)

            # deal with terminal states
            for s in range(self.num_states):
                if s in self.terminals:
                    for a in range(self.num_actions):
                        for s2 in range(self.num_states):
                            self.transitions[s][a][s2] = 0.0

    
    def set_rewards(self, _rewards):
        self.rewards = _rewards

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
        
    def __init__(self, num_rows, num_cols, num_actions, terminals, feature_weights, state_features, gamma, noise = 0.0, driving = False):

        assert(num_rows * num_cols == len(state_features) if not driving else num_rows * num_cols * 3 == len(state_features))
        assert(len(state_features[0]) == len(feature_weights))

        #rewards are linear combination of features
        rewards = np.dot(state_features, feature_weights)
        super().__init__(num_rows, num_cols, num_actions, terminals, rewards, gamma, noise, driving)

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



class ObjectWorld(MDP):
    """
    Like an MDP, but agent must complete tasks before reaching the terminal state.
    Init with tasks: an array of task positions
    """
    def __init__(self, num_rows, num_cols, num_actions, terminals, rewards, gamma, tasks, noise = 0.1):
        self.gamma = gamma
        self.num_states = num_rows * num_cols 
        self.num_actions = num_actions
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.terminals = terminals
        self.rewards = rewards
        self.tasks = tasks
        self.transitions = np.zeros((self.num_states, self.num_actions, self.num_states))
        self.init_transition_probabilities(noise)



class DrivingSimulator(FeatureMDP):
    """
    An "continuous"-state MDP that simulates driving on a three-lane highway.
    Visually, states are arranged in rows of three: |  i  | i+1 | i+2 |
    The purpose is to avoid obstacles on the road i.e. other cars and motorists,
    as well as avoid getting ticketed, all represented by features.
    Actions include STAYing in the current lane, moving LEFT, and moving RIGHT.
    """
    def __init__(self, num_rows, terminals, feature_weights, motorists, police, gamma, noise = 0.0):
        # feature_weights are weights for each of:
        # normal, good speed, best speed, collision, tailgating
        # motorists and police are arrays of locations
        # TODO: add police cars (currently always passing in None)
        normal = np.array([1, 0, 0, 0, 0])
        good_speed = np.array([0, 1, 0, 0, 0])
        best_speed = np.array([0, 0, 1, 0, 0])
        collision = np.array([0, 0, 0, 1, 0])
        tailgate = np.array([0, 0, 0, 0, 1])
        state_features = []
        num_lanes = 3
        num_cols = num_lanes + 2
        num_speeds = 3
        num_actions = num_speeds * num_lanes
        for s in range(num_rows * num_cols * 3):
            if s in motorists or s % num_cols == 0 or s % num_cols == num_cols - 1: # crash into another car or the sides
                state_features.append(collision)
            elif s >= num_rows * num_cols * (num_speeds - 1) and (s + 5) in motorists: # tailgating
                state_features.append(tailgate)
            elif s >= num_rows * num_cols * (num_speeds - 1): # best (highest) speed
                state_features.append(best_speed)
            elif s >= num_rows * num_cols * (num_speeds - 2): # good (medium) speed
                state_features.append(good_speed)
            else: # nothing special
                state_features.append(normal)
        super().__init__(num_rows, num_cols, num_actions, terminals, feature_weights, np.array(state_features), gamma, noise, driving = True)




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