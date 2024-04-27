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

    def r(self, state):
        beam_lengths, velocity, max_velocity, time_step = state
        return 10 * ((sum(beam_lengths) / len(beam_lengths))) + 100 * (velocity / max_velocity) - 0.1 * time_step

    def bellman(self, state):
        return self.r(state) + self.gamma * max(self.q.predict(np.array([self.reshape_state(state)])))

    def reshape_state(self, state):
        return state[0] + state[1:]

    def update_epsilon(self):
        self.epsilon = max(0.01, self.epsilon * 0.995)

    def choose_action(self, state):
        # explore
        if np.random.rand() < self.epsilon:
            action = np.random.randint(low=0, high=self.action_size)
        # exploit
        else:
            action = np.argmax(self.q.predict(np.array([self.reshape_state(state)])))
        return action

    def train_batch(self, states):
        features = np.array([self.reshape_state(state) for state in states]) # flatten
        labels = np.array([self.bellman(state) for state in states]) # compute labels using bellman equation

        self.q.fit(
            x = features,
            y = labels,
            epochs=5,
            batch_size=10,
            verbose=0
            )

    def q_model(self):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(0)

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.state_size,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                    loss='mse')
        return model
