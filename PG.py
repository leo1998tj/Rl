import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


class PG:
    def __init__(self, inputs, outs,
                 learning_rate=0.01,
                 gamma=0.95
                 ):
        self.n_features = inputs
        self.actions = outs

        self.evn_observation, self.evn_reward, self.evn_action = [], [], []

        self.lr = learning_rate
        self.gamma = gamma  # reward衰减量

        self.model = self._built_net()

    def _built_net(self):
        model = tf.keras.Sequential([
            layers.Dense(16, input_dim=self.n_features, activation="relu"),
            layers.Dense(self.actions, activation='softmax')
        ])

        model.compile(loss="sparse_categorical_crossentropy", optimizer=tf.optimizers.Adam(lr=self.lr))
        return model

    # store data
    def get_S_R_A(self, Observation, reward, action):
        self.evn_observation.append(Observation)
        self.evn_reward.append(reward)
        self.evn_action.append(action)

    def choose_action(self, observation):
        state = np.array([observation])
        action_prob = self.model.predict(state)  # 获取概率

        # 获取选择左还是右的总概率
        action_prob = np.sum(action_prob, axis=0)
        action_prob /= np.sum(action_prob)

        return np.random.choice([0, 1], p=action_prob)

    def discount_reward_and_norm(self):
        discount_reward = np.zeros_like(self.evn_reward)
        prior = 0
        for i in reversed(range(len(self.evn_reward))):
            prior = prior * self.gamma + self.evn_reward[i]
            discount_reward[i] = prior
        discount_reward -= np.mean(discount_reward)
        discount_reward /= np.std(discount_reward)

        return discount_reward

    def learn(self,cp_callback):
        discount_reward = self.discount_reward_and_norm()

        history = self.model.fit(np.vstack(self.evn_observation), np.vstack(self.evn_action),
                                 sample_weight=discount_reward,callbacks=[cp_callback])
        self.evn_action, self.evn_observation, self.evn_reward = [], [], []
        return history

