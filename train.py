import gymnasium as gym
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from gymnasium import spaces

# Load dataset
df = pd.read_csv('AI_Human.csv')  # Ensure dataset.csv exists
df = df.dropna()

human_df = df[df['generated'] == 0]
ai_df = df[df['generated'] == 1]

# Sampling of 20000 rows of each class (or any sample size that is not too large)
human_df_sampled = human_df.sample(50_000)
ai_df_sampled = ai_df.sample(50_000)

df = pd.concat([human_df_sampled, ai_df_sampled])
print(f'size of data:{df.shape}')
# Convert text to numerical representation
vectorizer = TfidfVectorizer(max_features=5000)  # Limit vocab size
X = vectorizer.fit_transform(df['text']).toarray()
y = df['generated'].values  # 1: AI-generated, 0: Human-written

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class TextClassificationEnv(gym.Env):
    """Custom Environment for Text Classification."""
    metadata = {"render_modes": ["human"], "render_fps": 30}
    
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y
        self.index = 0
        
        self.observation_space = spaces.Box(low=0, high=1, shape=(X.shape[1],), dtype=np.float32)
        self.action_space = spaces.Discrete(2)  # 0 or 1
    
    def step(self, action):
        correct_label = self.y[self.index]
        reward = 1 if action == correct_label else -1
        
        self.index += 1
        done = self.index >= len(self.X)  # End of dataset
        next_obs = self.X[self.index] if not done else np.zeros(self.X.shape[1])
        
        return next_obs, reward, done, False, {}
    
    def reset(self, seed=None, options=None):
        self.index = 0
        return self.X[self.index], {}
    
    def render(self):
        pass
    
    def close(self):
        pass

# Initialize environment for training and testing
env_train = make_vec_env(lambda: TextClassificationEnv(X_train, y_train), n_envs=1)
env_test = make_vec_env(lambda: TextClassificationEnv(X_test, y_test), n_envs=1)



model = PPO("MlpPolicy", env_train, verbose=1, ent_coef=0.02)
model.learn(total_timesteps=10_00_000)
model.save(f"model/text_classifier_rl")
# Evaluate on test set
rewards = []
obs = env_test.reset()
for _ in range(len(X_test)):
    action, _ = model.predict(obs)
    obs, reward, done, _ = env_test.step(action)
    rewards.append(reward)
    if done:
        obs = env_test.reset()
    
print(f"PPO Average Reward on Test Set: {np.mean(rewards)}")
# Test on a new sample
obs = vectorizer.transform(["This is a sample AI-generated text."]).toarray()
predicted_action, _ = model.predict(obs)
print(f"PPO  Prediction: {'AI-generated' if predicted_action[0] == 1 else 'Human-written'}")
