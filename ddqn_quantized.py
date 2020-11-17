import random
from collections import deque

import gym
import numpy as np
import pylab
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

import tensorflow_model_optimization as tfmot
import tensorflow as tf
import os

import time

EPISODES = 300 #300

MODEL_NAME = "ddqn_quant_24_24"
MODEL_DIR = os.path.dirname(os.path.realpath(__file__)) + "/saved_model/"

MODEL_PATH = MODEL_DIR + MODEL_NAME + ".tflite"

####################################################################################################
# Double DQN Agent for the Cartpole
#
# it uses Neural Network to approximate q function
# and replay memory & target q network
#
# quantization is done through quantized aware training, then finally the quantized model 
# is created afterwards
# see https://www.tensorflow.org/model_optimization/guide/quantization/training_example
####################################################################################################

class DoubleDQNAgent_quant:
    def __init__(self, state_size, action_size, render=False, load_model=False, mode="null"):
        # if you want to see Cartpole learning, then change to True
        self.render = render                    # render Cartpole simulation
        self.load_model = load_model            # load from tflite file
        self.mode = mode                        # train or eval

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # hyper parameters for the Double DQN
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.train_start = 1000

        # create replay memory using deque
        self.memory = deque(maxlen=2000)

        # create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.set_quant_model()

        # initialize target model
        self.update_target_model()

        # load up TFLite model
        if self.load_model and mode == "eval":
            
            self.interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
            self.interpreter.allocate_tensors()

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        model = Sequential()
        model.add(
            Dense(
                24,
                input_dim=self.state_size,
                activation="relu",
                kernel_initializer="he_uniform",
            )
        )
        model.add(Dense(24, activation="relu", kernel_initializer="he_uniform"))
        model.add(
            Dense(
                self.action_size, activation="linear", kernel_initializer="he_uniform"
            )
        )
        model.summary()
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
        return model

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        if self.mode == "train":
            if self.load_model:
                q_value = self.model.predict(state)
                return np.argmax(q_value[0])

            if np.random.rand() <= self.epsilon:
                return random.randrange(self.action_size)
            else:
                q_value = self.model.predict(state)
                return np.argmax(q_value[0])
        elif self.mode == "eval":
            # get details of tensors needed for tflite
            input_details = self.interpreter.get_input_details()
            output_details = self.interpreter.get_output_details()

            # process the input state for tflite interpreter
            input_shape = input_details[0]['shape']
            input_data = np.array(state, dtype=np.float32)

            self.interpreter.set_tensor(input_details[0]['index'], input_data)
            self.interpreter.invoke()

            output_data = self.interpreter.get_tensor(output_details[0]['index'])
            #print(output_data)
            return np.argmax(output_data[0])

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # pick samples randomly from replay memory (with batch_size)
    def train_model(self):
        if len(self.memory) < self.train_start:
            return
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        update_input = np.zeros((batch_size, self.state_size))
        update_target = np.zeros((batch_size, self.state_size))
        action, reward, done = [], [], []

        for i in range(batch_size):
            update_input[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])

        target = self.model.predict(update_input)
        target_next = self.model.predict(update_target)
        target_val = self.target_model.predict(update_target)

        for i in range(self.batch_size):
            # like Q Learning, get maximum Q value at s' But from target model
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # the key point of Double DQN
                # selection of action is from model, update is from target model
                a = np.argmax(target_next[i])
                target[i][action[i]] = reward[i] + self.discount_factor * (
                    target_val[i][a]
                )

        # make minibatch which includes target q value and predicted q value
        # and do the model fit!
        self.model.fit(
            update_input, target, batch_size=self.batch_size, epochs=1, verbose=0
        )

    # set both model and target_model for quantized aware training
    def set_quant_model(self):
        quantize_model = tfmot.quantization.keras.quantize_model

        # q_aware stands for for quantization aware.
        q_aware_model = quantize_model(self.model)

        # `quantize_model` requires a recompile.
        q_aware_model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))

        q_aware_model.summary()

        self.model = q_aware_model

        # q_aware stands for for quantization aware.
        q_aware_target_model = quantize_model(self.target_model)

        # `quantize_model` requires a recompile.
        q_aware_target_model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))

        q_aware_target_model.summary()

        self.target_model = q_aware_target_model

    # save quant aware training model
    def save_quant_model(self, verbose = False):
        # create quantized model for TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        self.quantized_tflite_model = converter.convert()

        with open(MODEL_PATH, 'wb') as f:
            f.write(self.quantized_tflite_model)

        # create regular TFLite model for reference
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        tflite_model = converter.convert()

        regular_model_path = MODEL_DIR + MODEL_NAME + "_regular.tflite"
        with open(regular_model_path, 'wb') as f:
            f.write(tflite_model)

        if verbose:
            model_size_mb = os.path.getsize("ddqn_regular.tflite") / float(2**20)
            quant_model_size_mb = os.path.getsize("ddqn_quant.tflite") / float(2**20)

            print("For reference, quantized model takes {:3f}% ".format((quant_model_size_mb / model_size_mb)))
            print("Float model in Mb:", os.path.getsize("ddqn_regular.tflite") / float(2**20))
            print("Quantized model in Mb:", os.path.getsize("ddqn_quant.tflite") / float(2**20))

# quantized aware training DDQN
# note that this does not produce a quantized model, it must be done separately after this
def train():
    print("doing quantized aware training")
    # In case of CartPole-v1, maximum length of episode is 500
    env = gym.make("CartPole-v1")
    # get size of state and action from environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    print("setting up agent")
    agent = DoubleDQNAgent_quant(state_size, action_size, mode="train")

    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            if agent.render:
                env.render()

            # get action for the current state and go one step in environment
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            # if an action make the episode end, then gives penalty of -100
            reward = reward if not done or score == 499 else -100

            # save the sample <s, a, r, s'> to the replay memory
            agent.append_sample(state, action, reward, next_state, done)
            # every time step do the training
            agent.train_model()
            score += reward
            state = next_state

            if done:
                # every episode update the target model to be same with model
                agent.update_target_model()

                # every episode, plot the play time
                score = score if score == 500 else score + 100
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, "b")
                pylab.savefig("./saved_graph/cartpole_" + MODEL_NAME +".png")
                print(
                    "episode:",
                    e,
                    " score:",
                    score,
                    " memory length:",
                    len(agent.memory),
                    " epsilon:",
                    agent.epsilon,
                )

                # if the mean of scores of last 10 episode is bigger than 490
                # stop training
                if np.mean(scores[-min(10, len(scores)) :]) > 490:
                    break

        # save the model
        if e % 50 == 0:
            agent.save_quant_model()

def predict():
    env = gym.make("CartPole-v1")

    # get size of state and action from environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    load_model = True
    render = True

    agent = DoubleDQNAgent_quant(
        state_size, action_size, render=render, load_model=load_model, mode="eval"
    )

    done = False
    score = 0
    state = env.reset()
    state = np.reshape(state, [1, state_size])

    iterations = 0
    while not done:
        timeVal = time.time()
        if agent.render:
            env.render()

        # get action for the current state and go one step in environment
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])

        score += reward
        state = next_state
        iterations += 1

        timeVal = time.time() - timeVal
        print("time taken:", timeVal)

        if done:
            print(score)

    print("iterations:", iterations)

if __name__ == "__main__":
    train()
    predict()
