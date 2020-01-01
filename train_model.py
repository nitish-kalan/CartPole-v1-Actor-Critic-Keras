import gym
from actor_model import Actor
from critic_model import Critic
import tensorflow as tf
from collections import deque
import numpy as np
import random
import tensorflow.keras.backend as K

# setting seed for reproducibility of results. This is not super important.
random.seed(2212)
np.random.seed(2212)
tf.set_random_seed(2212)

# Hyperparameters
REPLAY_MEMORY_SIZE = 1_00_000
MINIMUM_REPLAY_MEMORY = 1_000
MINIBATCH_SIZE = 32
EPSILON = 1
EPSILON_DECAY = 0.999
MIN_EPSILON = 0.001
DISCOUNT = 0.99
EPISODES = 1_00_000
ENV_NAME = 'CartPole-v1'
VISUALIZATION = False

# creating own session to use across all the Keras/Tensorflow models we are using
sess = tf.Session()
K.set_session(sess)

# Environment details
env = gym.make(ENV_NAME).unwrapped
action_dim = env.action_space.n
observation_dim = env.observation_space.shape

# Actor model to take actions 
# state -> action
actor = Actor(sess, action_dim, observation_dim)
# Critic model to evaluate the action taken by the actor
# state + action -> Expected reward to be achieved by taking action in the state.
critic = Critic(sess, action_dim, observation_dim)

# Replay memory to store experiences of the model with the environment
replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

sess.run(tf.initialize_all_variables())

def train_actor_critic(replay_memory, actor, critic):
    minibatch = random.sample(replay_memory, MINIBATCH_SIZE)

    X_states = []
    X_actions = []
    y = []
    for sample in minibatch:
        cur_state, cur_action, reward, next_state, done = sample
        next_actions = actor.model.predict(np.expand_dims(next_state, axis=0))
        if done:
            # If episode ends means we have lost the game so we give -ve reward
            # Q(st, at) = -reward
            reward = -reward
        else:
            # Q(st, at) = reward + DISCOUNT * Q(s(t+1), a(t+1))
            next_reward = critic.model.predict([np.expand_dims(next_state, axis=0), next_actions])[0][0]
            reward = reward + DISCOUNT * next_reward

        X_states.append(cur_state)
        X_actions.append(cur_action)
        y.append(reward)

    X_states = np.array(X_states)
    X_actions = np.array(X_actions)
    X = [X_states, X_actions]
    y = np.array(y)
    y = np.expand_dims(y, axis=1)
    # Train critic model
    critic.model.fit(X, y, batch_size=MINIBATCH_SIZE, verbose = 0)

    # Get the actions for the cur_states from the minibatch.
    # We are doing this because now actor may have learnt more optimal actions for given states
    # as Actor is constantly learning and we are picking the states from the previous experiences.
    X_actions_new = []
    for sample in minibatch:
        X_actions_new.append(actor.model.predict(np.expand_dims(sample[0], axis=0))[0])
    X_actions_new = np.array(X_actions_new)

    # grad(J(actor_weights)) = sum[ grad(log(pi(at | st, actor_weights)) * grad(critic_output, action_input), actor_weights) ]
    critic_gradients_val = critic.get_critic_gradients(X_states, X_actions)
    actor.train(critic_gradients_val, X_states)

max_reward = 0
for episode in range(EPISODES):
    done = False
    cur_state = env.reset()
    episode_reward = 0
    while not done and episode_reward < 1000:
        if VISUALIZATION:
            env.render()
        if np.random.uniform(0, 1) < EPSILON:
            # Taking random action (Exploration)
            action = [0] * action_dim
            action[np.random.randint(0, action_dim)] = 1
            action = np.array(action, dtype=np.float32)
        else:
            # Taking optimal action (Exploitation)
            action = actor.model.predict(np.expand_dims(cur_state, axis=0))[0]

        next_state, reward, done, _ = env.step(np.argmax(action))

        episode_reward += reward

        # Add experience to replay memory
        replay_memory.append((cur_state, action, reward, next_state, done))

        cur_state = next_state

        if len(replay_memory) < MINIMUM_REPLAY_MEMORY:
            continue

        train_actor_critic(replay_memory, actor, critic)

        if EPSILON > MIN_EPSILON and len(replay_memory) >= MINIMUM_REPLAY_MEMORY:
            EPSILON *= EPSILON_DECAY
            EPSILON = max(EPSILON, MIN_EPSILON)

    #some bookkeeping
    if(episode_reward > 400 and episode_reward > max_reward):
        actor.model.save_weights(str(episode_reward)+".h5")
    max_reward = max(max_reward, episode_reward)
    print('Episode:', episode, 'Episodic Reward:', episode_reward, 'Max Reward Achieved:', max_reward, 'EPSILON:', EPSILON)
