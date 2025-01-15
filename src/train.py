from gymnasium.wrappers import TimeLimit
from matplotlib import pyplot as plt

from env_hiv import HIVPatient
#from fast_env_py import FastHIVPatient
import numpy as np
import xgboost as xgb
import random
import os
from tqdm import tqdm
import joblib
from sklearn.preprocessing import StandardScaler

# The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.

class ProjectAgent:
    def __init__(self):
        self.num_features = 6
        self.num_actions = 4

        #We Set up XGBoost parameters
        self.boosting_params = {
            'objective': 'reg:squarederror',
            'eval_metric': ['rmse', 'mae'],
            'max_depth': 7,
            'eta': 0.05,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'min_child_weight': 4,
            'gamma': 0.1,
            'lambda': 1.5,
            'alpha': 0.4,
            'tree_method': 'hist',
            'max_leaves': 64,
            'seed': 42
        }
        self.q_models = [None for _ in range(self.num_actions)]
        self.state_normalizers = [StandardScaler() for _ in range(self.num_actions)]


        self.start_epsilon = 1.0
        self.min_epsilon = 0.01
        self.epsilon_decay_rate = 0.995
        self.epsilon = self.start_epsilon

        self.random_exploration_steps = 30000
        self.max_rounds_boost = 200
        self.dcnt_factor = 0.995
        self.reward_multiplier = 1e-6
        self.early_stop_rounds = 8

        #Set up the environments so that we can train on both cases with a specified proportion (important to adjust it)
        self.multi_training_ratio = 0.24
        self.current_env = None
        self.env_single = TimeLimit(HIVPatient(domain_randomization=False), max_episode_steps=200)
        self.env_multi = TimeLimit(HIVPatient(domain_randomization=True), max_episode_steps=200)

    def greedy_action(self, state):
        #Compute Q-values for each action and select the best action greedily by using Fitted Q iteration with xgboost
        q_values = []
        reshaped_state = state.reshape(1, -1)
        for action_index in range(self.num_actions):
            #Compute the model prediction for each test
            if self.q_models[action_index] is not None:
                scaled_state = self.state_normalizers[action_index].transform(reshaped_state)
                q_values.append(self.q_models[action_index].predict(xgb.DMatrix(scaled_state))[0])
            else:
                q_values.append(-1e12)
        chosen_action = np.argmax(q_values)
        return chosen_action

    def act(self, state, random_action=False):
        #Explore or take greedy action according to epsilon value
        if random_action or random.random() < self.epsilon:
            chosen_action = random.randint(0, self.num_actions - 1)
        else:
            chosen_action = self.greedy_action(state)
        #Update epsilon value
        self.epsilon = np.max([self.min_epsilon, self.epsilon * self.epsilon_decay_rate])
        return chosen_action

    def gather_experience(self, steps, environment, random_policy=True):
        #Collect the transitions of the environment to gather experience to train on
        experiences = []
        current_state, dt = environment.reset()

        for i in range(steps):
            selected_action = self.act(current_state, random_policy)
            next_state, reward, terminated, truncated, _ = environment.step(selected_action)
            experiences.append((current_state, selected_action, reward, next_state, truncated or terminated))

            if truncated or terminated:
                current_state, _ = environment.reset()
            else:
                current_state = next_state

        return experiences

    def prepare_training_data(self, experiences):
        #Prepare datasets for training XGBoost models for each action
        rewards = np.array([exp[2] for exp in experiences])
        actions = np.array([exp[1] for exp in experiences])
        states = np.vstack([exp[0] for exp in experiences])
        next_states = np.vstack([exp[3] for exp in experiences])
        terminal_flags = np.array([exp[4] for exp in experiences])

        future_q_values = np.zeros((states.shape[0], self.num_actions))
        if self.q_models[0] is not None:
            for action_index in range(self.num_actions):
                scaled_next_states = self.state_normalizers[action_index].transform(next_states)
                future_q_values[:, action_index] = self.q_models[action_index].predict(xgb.DMatrix(scaled_next_states))

        #Scale the observations as xgboost is sensible to scaling
        max_future_q = np.max(future_q_values, axis=1)
        scaled_rewards = rewards * self.reward_multiplier
        datasets_per_action = [[] for i in range(self.num_actions)]
        targets_per_action = [[] for i in range(self.num_actions)]

        for i in range(states.shape[0]):
            action = int(actions[i])
            target = scaled_rewards[i]
            if not terminal_flags[i]:
                target += self.dcnt_factor * max_future_q[i]

            datasets_per_action[action].append(states[i])
            targets_per_action[action].append(target)

        return datasets_per_action, targets_per_action

    def evaluate_performance(self, environment, trials=5):
        rewards = []
        for i in range(trials):
            current_state, _ = environment.reset()
            finished = False
            truncated = False
            episode_reward = 0
            while not finished and not truncated:
                action = self.act(current_state, random_action=False)
                next_state, reward, terminated, truncated, _ = environment.step(action)
                finished = terminated or truncated
                episode_reward += reward
                current_state = next_state
            rewards.append(episode_reward)
        return np.mean(rewards)

    def train_agent(self, total_epochs=6, trials_per_epoch=200):
        best_models = [None for i in range(self.num_actions)]
        highest_reward_single = -1e12
        highest_reward_multi = -1e12

        print("Starting random exploration...")
        gathered_experiences = []

        for env in [self.env_single, self.env_multi]:
            env_transitions = self.gather_experience(
                steps=self.random_exploration_steps,
                random_policy=True, environment=env
            )
            gathered_experiences.extend(env_transitions)

        epoch_rewards = []
        evaluation_scores_single = []
        evaluation_scores_multi = []
        moving_average_rewards = []

        print("\nBeginning FQI training...")
        for epoch in range(total_epochs):
            current_epoch_rewards = []
            epoch_experiences = []

            for trial in tqdm(range(trials_per_epoch), desc=f"Epoch n°{epoch + 1}/{total_epochs}"):
                self.current_env = self.env_multi if np.random.rand() < self.multi_training_ratio else self.env_single
                episode_experiences = self.gather_experience(200, self.current_env, random_policy=False)
                total_episode_reward = sum(exp[2] for exp in episode_experiences)

                epoch_experiences.extend(episode_experiences)
                current_epoch_rewards.append(total_episode_reward)
                epoch_rewards.append(total_episode_reward)

            gathered_experiences.extend(epoch_experiences)

            bootstrapped_indices = np.random.choice(len(gathered_experiences), len(gathered_experiences), replace=True)
            bootstrapped_experiences = [gathered_experiences[i] for i in bootstrapped_indices]

            action_data, action_targets = self.prepare_training_data(bootstrapped_experiences)

            for action_index in range(self.num_actions):
                if len(action_data[action_index]) > 0:
                    feature_data = np.array(action_data[action_index])
                    target_data = np.array(action_targets[action_index])

                    scaled_features = self.state_normalizers[action_index].fit_transform(feature_data)
                    train_split = int(0.8 * len(scaled_features))

                    train_data = xgb.DMatrix(scaled_features[:train_split], label=target_data[:train_split])
                    validation_data = xgb.DMatrix(scaled_features[train_split:], label=target_data[train_split:])

                    self.q_models[action_index] = xgb.train(
                        self.boosting_params,
                        train_data,
                        num_boost_round=self.max_rounds_boost,
                        evals=[(train_data, 'train'), (validation_data, 'val')],
                        early_stopping_rounds=self.early_stop_rounds,
                        verbose_eval=False
                    )

            eval_score_single = self.evaluate_performance(self.env_single, trials=10)
            eval_score_multi = self.evaluate_performance(self.env_multi, trials=10)
            evaluation_scores_single.append(eval_score_single)
            evaluation_scores_multi.append(eval_score_multi)
            avg_epoch_reward = np.mean(epoch_rewards)
            moving_average_rewards.append(avg_epoch_reward)

            print(f"Epoch n°{epoch + 1} Results:")
            print(f"Average Reward: {avg_epoch_reward:.2e}")
            print(f"Evaluation Reward on the single environment: {eval_score_single:.2e}")
            print(f"Evaluation Reward on the multi environment: {eval_score_multi:.2e}\n")

            if eval_score_single > highest_reward_single:
                print(f"Saving new best model on single environment, increasing score from {highest_reward_single}"
                      f"to {eval_score_single}")
                highest_reward_single = eval_score_single
                best_models = [model.copy() if model else None for model in self.q_models]
                self.save(path="trained_models/best_model_single.pt")

            if eval_score_multi > highest_reward_multi:
                print(f"Saving new best model on multi environment, increasing score from {highest_reward_multi}"
                      f"to {eval_score_multi}")
                highest_reward_multi = eval_score_multi
                best_models = [model.copy() if model else None for model in self.q_models]
                self.save(path="trained_models/best_model_multi.pt")
            os.makedirs("trained_models/ckpts", exist_ok=True)
            epoch_models = [model.copy() if model else None for model in self.q_models]
            joblib.dump({
                'models': epoch_models,
                'normalizers': self.state_normalizers
            }, f"trained_models/ckpts/model_weights_epoch{epoch}.pt")

        self.q_models = best_models
        return epoch_rewards, eval_score_single, eval_score_multi, moving_average_rewards

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'models': self.q_models,
            'normalizers': self.state_normalizers
        }, path)

    def load(self, path=os.path.join(os.getcwd(), 'best_model.pt')):
        loaded_data = joblib.load(path)
        self.q_models = loaded_data['models']
        self.state_normalizers = loaded_data['normalizers']


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    agent = ProjectAgent()

    epoch_rewards, evaluation_scores_single, evaluation_scores_multi, moving_average_rewards = agent.train_agent(
        total_epochs=200, trials_per_epoch=30)