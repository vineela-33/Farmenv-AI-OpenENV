import numpy as np
import random
from collections import deque

class FarmAgent:
    def __init__(self):
        self.q_table = {}
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.1
        self.gamma = 0.95
        self.action_space = 5
        self.action_names = [
            "Water Crops",
            "Apply Fertilizer",
            "Apply Pesticide",
            "Do Nothing",
            "Harvest"
        ]

    def state_to_key(self, state):
        water = int(state["water_level"] / 20)
        soil = int(state["soil_health"] / 20)
        pest = int(state["pest_level"] / 20)
        growth = int(state["growth_stage"])
        day = int(state["day"] / 5)
        return f"{water}_{soil}_{pest}_{growth}_{day}"

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 4)
        key = self.state_to_key(state)
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.action_space)
        return int(np.argmax(self.q_table[key]))

    def learn(self, state, action, reward,
              next_state, done):
        key = self.state_to_key(state)
        next_key = self.state_to_key(next_state)

        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.action_space)
        if next_key not in self.q_table:
            self.q_table[next_key] = np.zeros(self.action_space)

        target = reward
        if not done:
            target = reward + self.gamma * \
                np.max(self.q_table[next_key])

        self.q_table[key][action] += self.learning_rate * \
            (target - self.q_table[key][action])

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def run_episode(self, env):
        state = env.reset()
        total_reward = 0
        steps = []
        done = False

        while not done:
            action = self.choose_action(state)
            next_state, reward, done, info = env.step(action)
            self.learn(state, action, reward,
                      next_state, done)

            steps.append({
                "day": state["day"],
                "action": self.action_names[action],
                "reward": round(reward, 2),
                "water": round(state["water_level"], 1),
                "soil": round(state["soil_health"], 1),
                "pest": round(state["pest_level"], 1),
                "growth": round(state["growth_stage"], 2),
                "weather": state["weather"],
                "info": info
            })

            total_reward += reward
            state = next_state

        return round(total_reward, 2), steps