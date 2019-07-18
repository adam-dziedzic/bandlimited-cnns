import gym
from gym.spaces.discrete import Discrete

env = gym.make('CartPole-v0')
for i_episode in range(20):
    observation = env.reset()
    for t_timestep in range(20):
        env.render()
        print(observation)
        action = env.action_space.sample()
        # action = 0 # Discrete(1)
        observation, reward, done, info = env.step(action)
        if done:
            print('Episode finished after {} timesteps'.format(t_timestep + 1))
            break
env.close()