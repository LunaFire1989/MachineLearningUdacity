import random
import numpy as np
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env, gamma, alpha, eps):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.q_values = {}
        self.total_reward = 0
        self.total_succ = 0
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps

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
        self.state = curr_state
        #print "State: ", curr_state
        
        # TODO: Select action according to your policy
        init_q_value = 3
        valid_actions = [None, 'forward', 'left', 'right']
        max_q_action = valid_actions[random.randint(0, 3)]
        max_q_value = self.q_values.get((curr_state, max_q_action), init_q_value)
        for a in valid_actions:
            q = self.q_values.get((curr_state, a), init_q_value)
            if q > max_q_value:
                max_q_value = q
                max_q_action = a
        #print "Max Q Value: ", max_q_value
        #action = valid_actions[random.randint(0, 3)]
        eps = self.eps
        if random.random() < eps:
            action = valid_actions[random.randint(0, 3)]
        else:
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
        
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run(alpha, gamma, eps, succ_recoder):
    """Run the agent for a finite number of trials."""
    total_succ_rate = 0
    run_times = 50
    for t in range(run_times):
        # Set up environment and agent
        e = Environment()  # create environment (also adds some dummy traffic)
        a = e.create_agent(LearningAgent, alpha, gamma, eps)  # create agent
        e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
        # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

        # Now simulate it
        sim = Simulator(e, update_delay=0.5, display=True)  # create simulator (uses pygame when display=True, if available)
        # NOTE: To speed up simulation, reduce update_delay and/or set display=False

        sim.run(n_trials=100)  # run for a specified number of trials
        # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
                    
        #print "Final Q Table: ", a.q_values
        total_succ_rate += a.total_succ / 100.0
    
    succ_rate = total_succ_rate / run_times         
    succ_recoder[(alpha, gamma, eps)] = succ_rate       
    print "Alpha, Gamma, Eps: ", (alpha, gamma, eps)
    print "Success Rate: ", succ_rate

if __name__ == '__main__':
    succ_recoder = {}
    '''
    for alpha in np.linspace(0.1, 0.9, 9):
        for gamma in np.linspace(0.1, 0.9, 9):
           for eps in [0.01, 0.05, 0.1, 0.2]:
               run(alpha, gamma, eps, succ_recoder)
    '''
    run(0.2, 0.6, 0.01, succ_recoder)
    print "Param Result: ", succ_recoder
