import gym
from actor_model import Actor
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import sys

model_weight_file = sys.argv[1]

sess = tf.Session()
K.set_session(sess)

# Use env = gym.make('CartPole-v1').unwrapped 
# to remove the max_episode_length limit of 500. Episode will run till Pole falls.
env = gym.make('CartPole-v1')
action_dim = env.action_space.n
observation_dim = env.observation_space.shape

# Creating actor model and setting it's weights
actor = Actor(sess, action_dim, observation_dim)
actor.model.load_weights(model_weight_file)

episode_reward = 0
TOTAL_EPISODES = 10

for _ in range(TOTAL_EPISODES):
    state = env.reset()
    done = False
    while not done:
        env.render()
        next_state, reward, done, _ = env.step(np.argmax(actor.model.predict(np.expand_dims(state, axis=0))))
        state = next_state
        episode_reward += reward
print('Average Reward:', episode_reward / 10)
