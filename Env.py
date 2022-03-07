import gym
from PG import PG
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

env = gym.make('CartPole-v0')

checkpoint_save_path = "./checkpoint/fashion.ckpt"
RL = PG(env.observation_space.shape[0], env.action_space.n)

if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    RL.model.load_weights(checkpoint_save_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=False)


episodes = 2000  # 至多2000次
score_list = []  # 记录所有分数
for i in range(episodes):
    state = env.reset()
    score = 0
    while True:
        action = RL.choose_action(state)
        new_state, reward, done, info = env.step(action)

        RL.get_S_R_A(state, reward, action, )
        score += reward
        state = new_state
        # print('episode:', i_episode, 'score:', reward, 'observation:', observation, 'new_observation:', observation_)

        if done:
            vl = RL.learn(cp_callback)
            score_list.append(score)
            print('episode:', i, 'score:', score, 'max:', max(score_list))
            break
    if np.mean(score_list[-10:]) > 195:
        RL.model.save('CartPole-v0-pg_new.h5')
        break
env.close()

