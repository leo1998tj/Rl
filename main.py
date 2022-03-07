import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os
import gym

env = gym.make('CartPole-v0')
env.seed(1)
env = env.unwrapped

model = tf.keras.Sequential([
    layers.Dense(16, input_dim=env.observation_space.shape[0], activation="relu"),
    layers.Dense(env.action_space.n, activation='softmax')
])
model.compile(loss="sparse_categorical_crossentropy", optimizer=tf.optimizers.Adam(0.01))

checkpoint_save_path = "./checkpoint/fashion.ckpt"

model.load_weights(checkpoint_save_path)


def choose_action(observation):
    state = np.array([observation])
    action_prob = model.predict(state)  # 获取概率

    # 获取选择左还是右的总概率
    action_prob = np.sum(action_prob, axis=0)
    action_prob /= np.sum(action_prob)
    return np.random.choice([0, 1], p=action_prob)



for i_episode in range(1000):


    step = 0

    while True:
        env.render()

        action = choose_action(observation)

        observation_, reward, done, info = env.step(action)
        step +=1


        # print(reward)

        if done:
            print(step)
            break
    observation = observation_
env.close()

