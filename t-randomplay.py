# credits: https://github.com/tankala

import gym

# Let's create the Cart Pole OpenAI Gym game environment and define some constants
env = gym.make('CartPole-v1')
env.reset()
goal_steps = 500

# Define a random play function first
# Please debug / modify the program to learn what the folowing variables mean
# action, observation, reward, done, info
def play_a_random_game_first():
    for step_index in range(goal_steps):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Steps survived {}:".format(step_index))
            break
    env.reset()

# Play 100 times
for i in range(100):
    play_a_random_game_first()