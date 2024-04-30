import tensorflow as tf
import numpy as np

    
class Network:
    def __init__(self, state_size, action_size, learning_rate, gamma, epsilon):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.q = self.q_model()

    def r(self, collision, finish_line, distance_traveled):
        return np.dot(
            [-0.1, 1, 0.01],
            [collision, finish_line, distance_traveled]
        )

    def target_value(self, state, action, next_state, reward):
        y = self.q.predict(np.array([state]))
        y[0][action] = self.r(*reward) + self.gamma * max(self.q.predict(np.array([next_state]))[0])
        return y

    def update_epsilon(self):
        self.epsilon = max(0.01, self.epsilon * 0.998)

    def choose_action(self, state):
        # explore
        if np.random.rand() < self.epsilon:
            action = np.random.randint(low=0, high=self.action_size)
        # exploit
        else:
            action = np.argmax(self.q.predict(np.array([state]), verbose=0)[0])
        return action

    def train(self, state, action, next_state, reward):
        x = np.array([state])
        y = np.array([self.target_value(state, action, next_state, reward)]) # compute labels using bellman equation

        self.q.fit(
            x = x,
            y = y,
            epochs=5,
            batch_size=64,
            verbose=0
            )

    def q_model(self):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(0)

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.state_size,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                    loss='mse')
        return model

    def save(self):
        self.q.save_weights('./checkpoint.weights.h5')

    def load(self):
        self.q.load_weights('./checkpoint.weights.h5')
