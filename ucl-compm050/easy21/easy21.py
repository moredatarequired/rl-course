"""
An environment that implements Easy21.
"""

from dataclasses import dataclass
from random import randint

"""
Specifically, write a function, named step, which takes as input a state s (dealer’s first
card 1–10 and the player’s sum 1–21), and an action a (hit or stick), and returns a sample
of the next state s' (which may be terminal if the game is finished) and reward r.

We will be using this environment for model-free reinforcement learning, and you should not
explicitly represent the transition matrix for the MDP. There is no discounting (γ = 1).

You should treat the dealer’s moves as part of the environment, i.e. calling step with a 
stick action will play out the dealer’s cards and return the final reward and terminal state.
"""


def draw():
    """Draw a card from the Easy21 deck.

    * cards are sampled with replacement
    * each draw is a numeric value 1-10 (uniformly distributed)
    * each draw is black (positive) with 2/3 probability and red (negative) with 1/3 probability.
    """
    value = randint(1, 10)
    color = -1 if randint(1, 3) == 1 else +1
    return value * color


@dataclass(frozen=True)
class State:
    player: int
    dealer: int
    terminated: bool = False

    @staticmethod
    def new():
        """The player and the dealer each start with a black (positive) card."""
        return State(player=abs(draw()), dealer=abs(draw()))

    def hit(self):
        """Return a new State that is the result of the player 'hitting'."""
        if self.terminated:
            return self
        player = self.player + draw()
        bust = player < 1 or player > 21  # The player goes bust.
        return State(player=player, dealer=self.dealer, terminated=bust)

    def stick(self):
        """Return a new State that is the result of the player 'sticking'.
        
        After a player sticks, the dealer plays, sticking on 17 or higher and hitting
        otherwise. Then the game is terminated.
        """
        if self.terminated:
            return self
        dealer = self.dealer
        while dealer >= 1 and dealer < 17:
            dealer += draw()
        return State(player=self.player, dealer=dealer, terminated=True)

    def value(self):
        """The value of a state."""
        if not self.terminated:
            return 0  # Only terminal states have a value.
        if self.player < 1 or self.player > 21:
            return -1  # The player went bust, and lost.
        if self.dealer < 1 or self.dealer > 21:
            return +1  # The dealer went bust, so the player wins.
        if self.player > self.dealer:
            return +1
        if self.player == self.dealer:
            return 0
        if self.player < self.dealer:
            return -1


actions = ("hit", "stick")


def step(state, action):
    if state.terminated:
        return state, 0
    state = getattr(state, action)
    return state, state.value
