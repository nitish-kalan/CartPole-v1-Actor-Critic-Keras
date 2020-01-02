from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Add, Input
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import tensorflow as tf

# setting seed for reproducibility of results. This is not super important.
tf.set_random_seed(2212)

class Critic:
    def __init__(self, sess, action_dim, observation_dim):
        # setting our created session as default session
        K.set_session(sess)
        self.sess = sess
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.state_input, self.action_input, self.output, self.model = self.create_model()
        self.critic_gradients = tf.gradients(self.output, self.action_input)

    def create_model(self):
        state_input = Input(shape=self.observation_dim)
        state_h1 = Dense(24, activation='relu')(state_input)
        state_h2 = Dense(24, activation='relu')(state_h1)

        action_input = Input(shape=(self.action_dim,))
        action_h1 = Dense(24, activation='relu')(action_input)
        action_h2 = Dense(24, activation='relu')(action_h1)

        state_action = Add()([state_h2, action_h2])
        state_action_h1 = Dense(24, activation='relu')(state_action)
        output = Dense(1, activation='linear')(state_action_h1)

        model = Model(inputs=[state_input, action_input], outputs=output)
        model.compile(loss="mse", optimizer=Adam(lr=0.005))
        return state_input, action_input, output, model

    def get_critic_gradients(self, X_states, X_actions):
        # critic gradients with respect to action_input to feed in the weight updation of actor
        critic_gradients_val = self.sess.run(self.critic_gradients, feed_dict={self.state_input:X_states, self.action_input:X_actions})
        return critic_gradients_val[0]
