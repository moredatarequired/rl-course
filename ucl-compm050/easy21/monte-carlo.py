"""
Monte-Carlo control for Easy21.

Instructions: Initialise the value function to zero. Use
a time-varying scalar step-size of α_t = 1/N(s_t, a_t) and an ε-greedy exploration
strategy with ε_t = N_0/(N_0 + N(s_t)), where N_0 = 100 is a constant, N(s) is
the number of times that state s has been visited, and N(s, a) is the number
of times that action a has been selected from state s. Feel free to choose an
alternative value for N_0, if it helps producing better results. Plot the optimal
value function V∗(s) = max_a Q∗(s, a) using similar axes to the figure taken from
Sutton and Barto's Blackjack example.
"""

from collections import defaultdict
from random import choice, random

from easy21 import actions, State, step


N0 = 100
V = defaultdict(float)  # Estimated value of a state.
N = defaultdict(int)  # Number of times a state (or state-action pair) has been visited.


def alpha(state, action):
    return 1 / (N[(state, action)])


def epsilon(state):
    return N0 / (N0 + N[state])


def next_action(state):
    """Given a state, return the next action sampled from the ε-greedy policy."""
    e = epsilon(state)
    if random() < e:  # Choose an action at random.
        return choice(actions)
    # Choose the best-estimated action (argmax of V(s,a)).
    return max(actions, key=lambda a: V[(state, a)])


def trial():
    state = State.new()
    states = [state]
    while not state.terminated:
        action = next_action(state)
        N[state] += 1
        N[(state, action)] += 1
        state, reward = step(state, action)
    update(states, reward)


def update(states, reward):
    pass
