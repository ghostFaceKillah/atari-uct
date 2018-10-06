from dataclasses import dataclass, field
from typing import Dict, List

import gym


N_ACTIONS = 18


@dataclass(eq=False)
class Node:
    parent: 'Node'
    children: Dict[int, 'Node'] = field(default_factory=dict)

    q: float = 0   # current value estimate
    w: float = 0   # total value of my visits
    n: List[int] = 0     # total number of visits in my state
    p: List[float] = 0   # my prior probability of visit


def node_is_terminal(n: Node, env: gym.Env, depth: int):
    return False


def traverse_back_up(n: Node):
    pass


def visit(n: Node, env: gym.Env, depth: int):
    if node_is_terminal(n, env, depth):
        # do some things that you do in the leaf node
        # for example estimate value
        # or get terminal value from the environment

        pass
    else:
        #

        pass



def search():
    """
    1.

    1. choose node to visit


    :return:
    """

    pass


def search(s, game, nnet):
    if game.gameEnded(s): return -game.gameReward(s)

    if s not in visited:
        visited.add(s)
        P[s], v = nnet.predict(s)
        return -v

    max_u, best_a = -float("inf"), -1
    for a in range(game.getValidActions(s)):
        u = Q[s][a] + c_puct * P[s][a] * sqrt(sum(N[s])) / (1 + N[s][a])
        if u > max_u:
            max_u = u
            best_a = a
    a = best_a

    sp = game.nextState(s, a)
    v = search(sp, game, nnet)

    Q[s][a] = (N[s][a] * Q[s][a] + v) / (N[s][a] + 1)
    N[s][a] += 1


    return -v


if __name__ == '__main__':
    pass
    """
    def v
    1. choose 
    """
