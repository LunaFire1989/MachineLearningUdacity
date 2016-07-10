import random
import numpy as np
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env, gamma, alpha):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.q_values = {}
        self.total_reward = 0
        self.total_succ = 0
        self.gamma = gamma
        self.alpha = alpha

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.total_reward = 0
    
    def inputs_to_state(self, inputs, next_waypoint):
        return (inputs['light'], inputs['oncoming'], inputs['left'], inputs['right'], next_waypoint) 

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        curr_state = self.inputs_to_state(inputs, self.next_waypoint)
        #print "State: ", curr_state
        
        # TODO: Select action according to your policy
        valid_actions = [None, 'forward', 'left', 'right']
        max_q_action = valid_actions[random.randint(0, 3)]
        max_q_value = 0
        for a in valid_actions:
            q = self.q_values.get((curr_state, a), 0)
            if q > max_q_value:
                max_q_value = q
                max_q_action = a
        #print "Max Q Value: ", max_q_value
        #action = valid_actions[random.randint(0, 3)]
        action = max_q_action
        #print "Action: ", action

        # Execute action and get reward
        reward = self.env.act(self, action)
        self.total_reward += reward

        # TODO: Learn policy based on state, action, reward
        next_inputs = self.env.sense(self) 
        next_state = self.inputs_to_state(next_inputs, self.planner.next_waypoint())
        max_q_value = 0
        for a in valid_actions:
            max_q_value = max(max_q_value, self.q_values.get((next_state, a), 0))
            
        gamma = self.gamma
        alpha = self.alpha
        #print "Alpha: ", alpha
          
        curr_q_value = self.q_values.get((curr_state, action), 0)
        self.q_values[(curr_state, action)] = (1.0 - alpha) * curr_q_value + alpha * (reward + gamma * max_q_value)
        #print "Q_Values: ", self.q_values
        
        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""
    succ_recoder = {}
    #for alpha in np.linspace(0.1, 0.9, 9):
    #    for gamma in np.linspace(0.1, 0.9, 9):
    for alpha in [0.9]:
        for gamma in [0.3]:
            # Set up environment and agent
            e = Environment()  # create environment (also adds some dummy traffic)
            a = e.create_agent(LearningAgent, alpha, gamma)  # create agent
            e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
            # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

            # Now simulate it
            sim = Simulator(e, update_delay=0.01, display=False)  # create simulator (uses pygame when display=True, if available)
            # NOTE: To speed up simulation, reduce update_delay and/or set display=False

            sim.run(n_trials=100)  # run for a specified number of trials
            # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
            
            print "Final Q Table: ", a.q_values
            print "Alpha, Gamma: ", (alpha, gamma)
            print "Total Succ Num: ", a.total_succ
            succ_recoder[(alpha, gamma)] = a.total_succ

    print "Param Result: ", succ_recoder

if __name__ == '__main__':
    run()
