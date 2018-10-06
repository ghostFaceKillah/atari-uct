import gym
import time

env = gym.make('MontezumaRevengeNoFrameskip-v4')

env.reset()

i = 0
last = int(time.time())

while True:

    _, _, done, _ = env.step(env.action_space.sample())
    i += 1
    if done:
        env.reset()

    new = int(time.time())

    if new > last:
        print(f"Processed {i} frames")
        i = 0
        last = new

