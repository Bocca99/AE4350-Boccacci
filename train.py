import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import keras

import RLagents
import snakeModel
import swimmer_v4
import RLnoise


#### ENVIRONMENT

# Setting the environment
env = snakeModel.SnakeEnvironment1(render_mode="rgb_array", model_path ="snake_final.xml")

# Getting the observation and action space, and the bounds
n_obs = env.observation_space.shape[0]
n_act = env.action_space.shape[0]
bounds = [[-1,-0.8,-0.6,-0.3,-0.2,-0.1,-0.1,-0.2,-0.3,-0.6,-0.8,-1],[1,0.8,0.6,0.3,0.2,0.1,0.1,0.2,0.3,0.6,0.8,1]]
#bounds = [-1,1]

class OUnoise:
    def __init__(self, mi, sigma, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mi = mi
        self.sigma = sigma
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        dxt = (self.x + self.theta * (self.mi - self.x) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mi.shape))

        self.x = dxt
        return dxt

    def reset(self):
        if self.x_initial is not None:
            self.x = self.x_initial
        else:
            self.x = np.zeros_like(self.mi)


std_dev = 0.5
noise = OUnoise(mi=np.zeros(1), sigma=float(std_dev) * np.ones(1))
#noise = np.random.normal(loc = 0, scale= float(std_dev), size= 12)

# Network definition


def DDPGActor():

    inputs = layers.Input(shape=(n_obs,))
    hidden = layers.Dense(480, activation="relu")(inputs)
    hidden = layers.Dense(480, activation="relu")(hidden)
    hidden = layers.Dense(480, activation="relu")(hidden)
    hidden = layers.Dense(480, activation="relu")(hidden)
    output = layers.Dense(12, activation="tanh")(hidden)

    # Our upper bound is 2.0 for Pendulum.
    output = output * bounds[1]
    actor = tf.keras.Model(inputs, output)
    return actor


def DDPGCritic():
    # State as input
    obsInput = layers.Input(shape=(n_obs))
    obsOut = layers.Dense(256, activation="relu")(obsInput)
    obsOut = layers.Dense(256, activation="relu")(obsOut)
    obsOut = layers.Dense(256, activation="relu")(obsOut)
    obsOut = layers.Dense(256, activation="relu")(obsOut)

    # Action as input
    actInput = layers.Input(shape=(n_act))
    actOut = layers.Dense(256, activation="relu")(actInput)


    # Both are passed through seperate layer before concatenating
    comb = layers.Concatenate()([obsOut, actOut])

    out = layers.Dense(480, activation="relu")(comb)
    out = layers.Dense(480, activation="relu")(out)
    out = layers.Dense(480, activation="relu")(out)
    out = layers.Dense(480, activation="relu")(out)
    outputs = layers.Dense(1)(out)

    # Outputs single value for give state-action
    model = tf.keras.Model([obsInput, actInput], outputs)

    return model

actor = DDPGActor()
critic = DDPGCritic()

targetActor = actor
targetCritic = critic

targetActor.set_weights(actor.get_weights())
targetCritic.set_weights(critic.get_weights())

actorOptimizer = tf.keras.optimizers.legacy.Adam(learning_rate= 0.001)
criticOptimizer = tf.keras.optimizers.legacy.Adam(learning_rate= 0.002)

class replayBuffer: # From pendulum example
    def __init__(self, memorySize=100000, batchSize=50):
        self.memorySize = memorySize # Size of the replay buffer. Has to be large
        self.batch_size = batchSize # Size of the batch from which experiences are taken

        self.counter = 0

        # Tuples are converted in easier to handle numpy arrays
        self.previousObs = np.zeros((self.memorySize, n_obs))
        self.previousAct = np.zeros((self.memorySize, n_act))
        self.previousReward = np.zeros((self.memorySize, 1))
        self.previousObsNext = np.zeros((self.memorySize, n_obs))

    def memorize(self, pastExperience):

        index = self.counter % self.memorySize # Index reset if memory size is exceeded

        self.previousObs[index] = pastExperience[0]
        self.previousAct[index] = pastExperience[1]
        self.previousReward[index] = pastExperience[2]
        self.previousObsNext[index] = pastExperience[3]

        self.counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def gradientDescent(
        self,
        state_batch,
        action_batch,
        reward_batch,
        next_state_batch,
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = targetActor(next_state_batch, training=True)
            y = reward_batch + gamma * targetCritic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = critic([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic.trainable_variables)
        criticOptimizer.apply_gradients(
            zip(critic_grad, critic.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor(state_batch, training=True)
            critic_value = critic([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor.trainable_variables)
        actorOptimizer.apply_gradients(
            zip(actor_grad, actor.trainable_variables)
        )

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.counter, self.memorySize)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.previousObs[batch_indices])
        action_batch = tf.convert_to_tensor(self.previousAct[batch_indices])
        reward_batch = tf.convert_to_tensor(self.previousReward[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.previousObsNext[batch_indices])

        self.gradientDescent(state_batch, action_batch, reward_batch, next_state_batch)

@tf.function
def update_target(target_weights, weights, tau):
    for a, b in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))


def policy(state, noise_object):
    sampled_actions = tf.squeeze(actor(state))
    noise = noise_object
    # Adding noise to action
    sampled_actions = sampled_actions.numpy() + noise

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, bounds[0], bounds[1])

    return [np.squeeze(legal_action)]


n_episodes = 30 # Number of episodes
d_episode = 300 # Duration of a single episode
gamma = 0.99 # Discount rate: favours final episodes
tau = 0.005 # Target learning parameter


EP_REWARD = []
AVG_REWARD = []

buffer = replayBuffer(10000, 50)

for episode in range(n_episodes): # Cycling thorough each episode
    obs0 = env.reset()[0] # Initial observation
    t = 0 # Starting time
    ep_reward = 0

    # Each time instant, the agent acts on the environment
    while t<d_episode:
        obs0 = tf.expand_dims(tf.convert_to_tensor(obs0), 0) # Converts to a tensor object the observation at time 0
        action = np.squeeze(np.array(policy(obs0,noise))) # Evaluates the action, and eliminates redundant dimensions, checks boundaries
        obs, reward, done, info = env.step(action) # Given an action, provided by the policy, the snake moves in a time step

        buffer.memorize((obs0, action, reward, obs)) # Stores values in replay buffer
        ep_reward += reward # Update total episode reward

        buffer.learn() # Apply gradient ascent and find new values for the weights
        update_target(targetActor.variables, actor.variables, tau)
        update_target(targetCritic.variables, critic.variables, tau)

        if done: # If success is reached, end the episode
            break

        obs0 = obs # New initial observation is the past one
        t += 1 # Update step

    #print(info) # More detailed information on the reward
    EP_REWARD.append(ep_reward)
    avg_reward = np.mean(EP_REWARD)
    AVG_REWARD.append(avg_reward)
    print("Episode: {} | Average Reward {} | Episode Reward {}".format(episode,avg_reward,ep_reward))
    print(info)
    #plt.style.use('ggplot')
    #plt.plot(AVG_REWARD, color='r',label="Average reward")
    #plt.plot(EP_REWARD, linestyle=":",color='b',label="Episode reward")
    #plt.xlabel("Episode")
    #plt.ylabel("Average Reward")
    #plt.pause(0.01)
#plt.show()



# Save the weights after training
actor.save_weights("actor_weights.h5")
critic.save_weights("critic_weights.h5")

targetActor.save_weights("actor_target_weights.h5")
targetCritic.save_weights("critic_target_weights.h5")

np.save("episodic_vector",EP_REWARD)
np.save("average_vector",AVG_REWARD)