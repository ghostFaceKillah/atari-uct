from dataclasses import dataclass, field
from typing import Dict, List

import gym
import tqdm
import numpy as np


N_RUNS = 100000
MAX_DEPTH = 750
C_PUCT = 1.0
GAMMA = 0.99


@dataclass(eq=False)
class Node:
    parent: 'Node'
    n_actions: int

    n: List[int] = field(default_factory=list)

    p_hat: List[float] = field(default_factory=list)
    v_hat: float = 0

    children: Dict[int, 'Node'] = field(default_factory=dict)

    q: List[float] = field(default_factory=list)

    def __post_init__(self):
        self.n = [0] * self.n_actions
        self.p_hat = [1./self.n_actions] * self.n_actions
        self.q = [-0.1] + [0.0] * (self.n_actions - 1)


def visit(node: Node, env: gym.Env, depth: int):
    if depth == MAX_DEPTH:
        return node.v_hat
    else:
        # choose the best action
        max_u, best_a = -float("inf"), -1
        for a in range(env.action_space.n):
            q = node.q[a]
            u = q + C_PUCT * node.p_hat[a] * np.sqrt(sum(node.n)) / (1 + node.n[a])

            if u > max_u:
                max_u = u
                best_a = a
        a = best_a

        # execute the action
        obs, rew, done, _ = env.step(a)
        # env.render()

        if rew != 0:
            print("rew!")

        if done:
            return rew
        else:
            # maybe make the node we are visiting
            if a not in node.children:
                # after we will hook up policy here
                # we will initialize p_hat, v_hat = nnet.predict(obs)
                node.children[a] = Node(node, env.action_space.n)

            # visit the node below us corresponding to the best action
            v = GAMMA * visit(node.children[a], env, depth + 1) + rew
            node.q[a] = (node.n[a] * node.q[a] + v) / (node.n[a] + 1)
            node.n[a] += 1

            return v


def choose_move(env: gym.Env):
    start_state = env.env.clone_full_state()

    root = Node(None, env.action_space.n)

    for _ in tqdm.tqdm(range(N_RUNS)):
        # reset gym to the original state
        env.reset()
        env.env.restore_full_state(start_state)
        visit(root, env, 0)

    env.env.restore_full_state(start_state)

    print("="*80)
    print(root.q)
    print(root.n)
    action = np.argmax(np.array(root.q))
    print(f"Choosing {action}")
    print("="*80)

    return action


"""
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
"""


if __name__ == '__main__':

    env = gym.make('MontezumaRevenge-v4')
    # env = gym.make('Pong-v4')

    env.reset()
    env.step(1)
    env.step(2)
    done = False

    for i in range(30):
        env.step(0)
        # env.render()

    while not done:
        move = choose_move(env)
        env.step(move)
        # env.render()
