import gym
import random
import numpy as np
import keras
from keras.models     import Sequential
from keras.layers     import Dense
from keras.optimizers import Adam

# Let's create the Cart Pole OpenAI Gym game environment and define some constants
env = gym.make('CartPole-v1')
env.reset()
goal_steps = 500

trained_model =  keras.models.load_model('t-agent.h5')



scores = []
choices = []
for each_game in range(100):
    score = 0
    prev_obs = []
    for step_index in range(goal_steps):
        env.render()
        if len(prev_obs)==0:
            action = random.randrange(0,2)
        else:
            prev_obs.shape = (1,4)
            policy = trained_model.predict(prev_obs)
            policy.shape = (2,1)
            action = np.argmax(policy, axis=1)[0]
        
        choices.append(action)
        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation
        score+=reward
        if done:
            break

    env.reset()
    scores.append(score)

print(scores)
print('Average Score:', sum(scores)/len(scores))
print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))
