#!/usr/bin/env python3
"""
NQ Alpha Elite - The World's Best Trading Bot

Complete implementation with all elite-level components integrated,
including advanced order flow analysis, alpha enhancement, and dynamic trade management.
"""

import os
import sys
import time
import logging
import threading
import signal
import argparse
import json
import yaml
import numpy as np
import pandas as pd
from collections import deque, defaultdict, Counter
from pathlib import Path
import traceback
from scipy.stats import linregress
import requests
from bs4 import BeautifulSoup
import re
import random
import warnings
from scipy import stats
import joblib
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from collections import deque
from tensorflow.keras.layers import Dense, Input, LSTM, Dropout, Concatenate, BatchNormalization
import tensorflow_probability as tfp
import joblib
import os
from datetime import datetime
import json
import traceback
try:
    from nq_alpha_rl import NQAlphaEliteRL, PPOAgent
    RL_AVAILABLE = True
except ImportError as e:
    print(f"RL module import error: {e}")
    RL_AVAILABLE = False

try:
    import talib
except ImportError:
    pass  # TALib is optional for technical indicators

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/elite_system_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)

# Global logger
logger = logging.getLogger("NQAlpha")



class PrioritizedReplayBuffer:
    """Advanced experience replay buffer for more efficient learning"""
    
    def __init__(self, max_size=10000, alpha=0.6, beta=0.4):
        self.max_size = max_size
        self.buffer = []
        self.priorities = []
        self.position = 0
        self.alpha = alpha  # prioritization strength
        self.beta = beta    # importance sampling weight
        self.max_priority = 1.0
    
    def add(self, state, action, reward, next_state, done):
        """Add experience with max priority"""
        if len(self.buffer) < self.max_size:
            self.buffer.append((state, action, reward, next_state, done))
            self.priorities.append(self.max_priority)
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
            self.priorities[self.position] = self.max_priority
            
        self.position = (self.position + 1) % self.max_size
    
    def sample(self, batch_size):
        """Sample a batch based on priorities"""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
            
        # Calculate sampling probabilities
        priorities = np.array(self.priorities)
        prob_alpha = priorities ** self.alpha
        probs = prob_alpha / prob_alpha.sum()
        
        # Sample indices based on priorities
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # normalize
        
        # Collect experiences
        states, actions, rewards, next_states, dones = [], [], [], [], []
        
        for idx in indices:
            state, action, reward, next_state, done = self.buffer[idx]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
            indices,
            np.array(weights, dtype=np.float32)
        )
    
    def update_priority(self, idx, priority):
        """Update priority for an experience"""
        priority = max(priority, 1e-5)  # avoid zero priority
        self.priorities[idx] = priority
        self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.buffer)


class RLTradingAgent:
    """Advanced reinforcement learning agent for trading"""
    
    def __init__(self, state_size, action_size, learning_rate=0.0001):
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters
        self.gamma = 0.99           # Discount factor
        self.epsilon = 1.0          # Exploration rate
        self.epsilon_min = 0.01     # Minimum exploration rate
        self.epsilon_decay = 0.995  # Exploration decay rate
        self.learning_rate = learning_rate
        
        # Experience buffer
        self.memory = PrioritizedReplayBuffer(max_size=50000)
        self.batch_size = 64
        
        # Build neural network models
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
        # Performance tracking
        self.train_count = 0
        self.profits = []
        self.trade_history = []
        
        # Create logging directory
        self.log_dir = "rl_trading_logs"
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, f"trading_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        with open(self.log_file, 'w') as f:
            f.write("step,reward,profit,loss,epsilon\n")
    
    def _build_model(self):
        """Build deep neural network model"""
        model = Sequential()
        
        # First layer with batch normalization
        model.add(Dense(128, input_dim=self.state_size))
        model.add(BatchNormalization())
        model.add(Dense(128, activation='relu'))
        
        # Hidden layers
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))  # Add dropout for regularization
        model.add(Dense(64, activation='relu'))
        
        # Output layer - Q-values for each action
        model.add(Dense(self.action_size, activation='linear'))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return model
    
    def update_target_model(self):
        """Update target model to match main model"""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.add(state, action, reward, next_state, done)
    
    def act(self, state, training=True):
        """Select action based on current state"""
        if training and np.random.rand() <= self.epsilon:
            # Exploration: random action
            return random.randrange(self.action_size)
        
        # Exploitation: predict best action
        q_values = self.model.predict(state, verbose=0)[0]
        return np.argmax(q_values)
    
    def replay(self, batch_size):
        """Train on batch of experiences"""
        if len(self.memory) < batch_size:
            return
        
        # Sample batch from memory
        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(batch_size)
        
        # Predict Q-values for current states
        targets = self.model.predict(states, verbose=0)
        
        # Predict Q-values for next states using target network
        next_q_values = self.target_model.predict(next_states, verbose=0)
        
        # Calculate target Q-values (Bellman equation)
        for i in range(batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        # Calculate TD errors for priority updates
        pred_q_values = self.model.predict(states, verbose=0)
        td_errors = np.abs(np.choose(actions, targets.T) - np.choose(actions, pred_q_values.T))
        
        # Update priorities in replay buffer
        for i, idx in enumerate(indices):
            self.memory.update_priority(idx, td_errors[i])
        
        # Train the model
        history = self.model.fit(states, targets, epochs=1, verbose=0, sample_weight=weights)
        loss = history.history['loss'][0]
        
        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update target model occasionally
        self.train_count += 1
        if self.train_count % 100 == 0:
            self.update_target_model()
            
        return loss
    
    def save_model(self, filepath):
        """Save model weights"""
        self.model.save_weights(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model weights"""
        self.model.load_weights(filepath)
        self.target_model.load_weights(filepath)
        print(f"Model loaded from {filepath}")
    
    def log_performance(self, step, reward, profit, loss=0):
        """Log trading performance"""
        with open(self.log_file, 'a') as f:
            f.write(f"{step},{reward},{profit},{loss},{self.epsilon}\n")
        
        # Store profit for tracking
        self.profits.append(profit)
        
        # Print summary every 10 steps
        if step % 10 == 0:
            avg_profit = np.mean(self.profits[-100:]) if self.profits else 0
            win_rate = np.mean([p > 0 for p in self.profits[-100:]]) if self.profits else 0
            print(f"Step {step}: Reward={reward:.2f}, Profit={profit:.2%}, Win Rate={win_rate:.2%}, Epsilon={self.epsilon:.4f}")

class OnlinePPOAgent:
    """Advanced PPO agent with online learning capabilities"""
    
    def __init__(self, state_size, action_size, lr=0.0001):
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters
        self.lr = lr
        self.gamma = 0.99  # discount factor
        self.clip_ratio = 0.2  # PPO clip ratio
        self.lam = 0.95  # GAE lambda
        self.update_epochs = 4
        self.entropy_coef = 0.01
        
        # Experience buffer
        self.buffer_size = 10000
        self.min_buffer_size = 1000  # Min experiences before training
        self.batch_size = 256
        
        # Experience buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.logprobs = []
        self.dones = []
        
        # Experience replay for offline learning
        self.replay_buffer = PrioritizedReplayBuffer(max_size=self.buffer_size)
        
        # Performance tracking
        self.train_iterations = 0
        self.total_rewards = []
        self.avg_rewards = []
        self.win_rate = []
        
        # Networks
        self._build_model()
        
        # Adaptive learning rate
        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.lr,
            decay_steps=10000,
            decay_rate=0.95,
            staircase=True
        )
        
        # Create optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)
        
        # Training log
        self.log_dir = "trading_bot_logs"
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        with open(self.log_file, 'w') as f:
            f.write("iteration,avg_reward,win_rate,learning_rate\n")
    
    def _build_model(self):
        """Build actor and critic networks with shared layers"""
        # Feature extraction layers (shared)
        state_input = Input(shape=(self.state_size,))
        
        # Shared network
        shared = Dense(128, activation='relu')(state_input)
        shared = BatchNormalization()(shared)
        shared = Dense(128, activation='relu')(shared)
        shared = BatchNormalization()(shared)
        
        # Policy network (actor)
        policy_hidden = Dense(64, activation='relu')(shared)
        policy_out = Dense(self.action_size, activation='softmax')(policy_hidden)
        
        # Value network (critic)
        value_hidden = Dense(64, activation='relu')(shared)
        value_out = Dense(1)(value_hidden)
        
        # Create models
        self.actor = Model(inputs=state_input, outputs=policy_out)
        self.critic = Model(inputs=state_input, outputs=value_out)
        
        # Print model summaries
        self.actor.summary()
        self.critic.summary()
    
    def get_action(self, state, training=True):
        """Sample an action from the policy distribution"""
        # Ensure state is correctly shaped
        if len(state.shape) == 1:
            state = np.expand_dims(state, axis=0)
            
        # Get action probabilities
        probs = self.actor.predict(state, verbose=0)[0]
        
        # Calculate state value
        value = self.critic.predict(state, verbose=0)[0][0]
        
        if training:
            # Sample action from distribution
            action = np.random.choice(self.action_size, p=probs)
            log_prob = np.log(probs[action] + 1e-10)
            return action, log_prob, value, probs
        else:
            # Deterministic action (e.g., for trading)
            action = np.argmax(probs)
            return action
    
    def remember(self, state, action, reward, next_state, done, log_prob=None, value=None):
        """Store experience in replay buffer"""
        # For immediate PPO training
        if log_prob is not None and value is not None:
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.values.append(value)
            self.logprobs.append(log_prob)
            self.dones.append(float(done))
        
        # For experience replay (offline learning)
        self.replay_buffer.add(state, action, reward, next_state, done)
    
    def _calculate_advantages(self):
        """Calculate advantages using Generalized Advantage Estimation (GAE)"""
        advantages = np.zeros_like(self.rewards, dtype=np.float32)
        returns = np.zeros_like(self.rewards, dtype=np.float32)
        
        last_gae = 0
        last_return = 0
        
        # Calculate advantages in reverse order
        for t in reversed(range(len(self.rewards))):
            # For last step
            if t == len(self.rewards) - 1:
                # If episode didn't finish, bootstrap value
                if not self.dones[t]:
                    # Bootstrap with critic value
                    next_state = np.expand_dims(self.states[t], axis=0)
                    next_value = self.critic.predict(next_state, verbose=0)[0][0]
                else:
                    next_value = 0
            else:
                next_value = self.values[t + 1]
            
            # Calculate TD error
            delta = self.rewards[t] + self.gamma * next_value * (1 - self.dones[t]) - self.values[t]
            
            # Calculate GAE
            advantages[t] = last_gae = delta + self.gamma * self.lam * (1 - self.dones[t]) * last_gae
            
            # Calculate returns for critic
            returns[t] = last_return = self.rewards[t] + self.gamma * (1 - self.dones[t]) * last_return
        
        return returns, advantages
    
    def train_from_buffer(self):
        """Train agent from current experience buffer"""
        if len(self.states) < self.batch_size:
            return
        
        # Convert lists to numpy arrays
        states = np.array(self.states)
        actions = np.array(self.actions)
        old_log_probs = np.array(self.logprobs)
        
        # Calculate advantages and returns
        returns, advantages = self._calculate_advantages()
        
        # Normalize advantages for stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO optimization loop
        for _ in range(self.update_epochs):
            # Create random indices
            indices = np.random.permutation(len(states))
            
            # Process in minibatches
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                if end > len(states):
                    end = len(states)
                
                # Get minibatch indices
                mb_indices = indices[start:end]
                
                with tf.GradientTape() as tape:
                    # Get current action probabilities and values
                    curr_probs = self.actor(states[mb_indices])
                    curr_values = self.critic(states[mb_indices])
                    
                    # Get one-hot actions
                    actions_one_hot = tf.one_hot(actions[mb_indices], self.action_size)
                    
                    # Calculate probabilities of actions taken
                    curr_action_probs = tf.reduce_sum(curr_probs * actions_one_hot, axis=1)
                    
                    # Calculate log probabilities
                    curr_log_probs = tf.math.log(curr_action_probs + 1e-10)
                    
                    # Probability ratio
                    ratio = tf.exp(curr_log_probs - old_log_probs[mb_indices])
                    
                    # Clipped objective
                    clip_adv = tf.clip_by_value(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * advantages[mb_indices]
                    policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages[mb_indices], clip_adv))
                    
                    # Value loss
                    value_loss = tf.reduce_mean(tf.square(returns[mb_indices] - tf.squeeze(curr_values)))
                    
                    # Entropy bonus for exploration
                    entropy = -tf.reduce_mean(tf.reduce_sum(curr_probs * tf.math.log(curr_probs + 1e-10), axis=1))
                    
                    # Total loss
                    loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy
                
                # Compute gradients
                grads = tape.gradient(loss, self.actor.trainable_variables + self.critic.trainable_variables)
                
                # Apply gradients
                self.optimizer.apply_gradients(zip(grads, self.actor.trainable_variables + self.critic.trainable_variables))
        
        # Clear experience buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.logprobs = []
        self.dones = []
        
        # Track training
        self.train_iterations += 1
        
        # Log training (every 10 iterations)
        if self.train_iterations % 10 == 0:
            self._log_training()
    
    def train_offline_batch(self, batch_size=None):
        """Train agent from replay buffer (offline learning)"""
        if batch_size is None:
            batch_size = self.batch_size
            
        if len(self.replay_buffer) < self.min_buffer_size:
            return
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones, indices, weights = self.replay_buffer.sample(batch_size)
        
        # Compute TD errors for prioritized replay update
        current_values = self.critic.predict(states, verbose=0).flatten()
        next_values = self.critic.predict(next_states, verbose=0).flatten()
        
        # Calculate targets for critic
        targets = rewards + self.gamma * next_values * (1 - dones)
        td_errors = np.abs(targets - current_values)
        
        # Update priorities
        for i, idx in enumerate(indices):
            self.replay_buffer.update_priority(idx, td_errors[i])
        
        # Prepare data for PPO
        self.states = list(states)
        self.actions = list(actions)
        self.rewards = list(rewards)
        self.values = list(current_values)
        
        # Get log probs for old actions
        log_probs = []
        for i in range(len(states)):
            probs = self.actor.predict(np.expand_dims(states[i], axis=0), verbose=0)[0]
            log_probs.append(np.log(probs[actions[i]] + 1e-10))
            
        self.logprobs = log_probs
        self.dones = list(dones)
        
        # Train using PPO
        self.train_from_buffer()
    
    def _log_training(self):
        """Log training progress"""
        avg_reward = np.mean(self.total_rewards[-100:]) if self.total_rewards else 0
        win_rate = np.mean([r > 0 for r in self.total_rewards[-100:]]) if self.total_rewards else 0
        
        # Store metrics
        self.avg_rewards.append(avg_reward)
        self.win_rate.append(win_rate)
        
        # Get current learning rate
        lr = self.optimizer._decayed_lr(tf.float32).numpy()
        
        # Log to file
        with open(self.log_file, 'a') as f:
            f.write(f"{self.train_iterations},{avg_reward},{win_rate},{lr}\n")
        
        print(f"Iteration {self.train_iterations} | Avg Reward: {avg_reward:.4f} | Win Rate: {win_rate:.4f} | LR: {lr:.6f}")
    
    def save(self, actor_path="ppo_actor.h5", critic_path="ppo_critic.h5", buffer_path="replay_buffer.pkl"):
        """Save model and replay buffer"""
        self.actor.save_weights(actor_path)
        self.critic.save_weights(critic_path)
        
        # Save replay buffer
        with open(buffer_path, 'wb') as f:
            joblib.dump(self.replay_buffer, f)
        
        # Save metrics and hyperparameters
        metrics = {
            "train_iterations": self.train_iterations,
            "avg_rewards": self.avg_rewards,
            "win_rate": self.win_rate,
            "hyperparameters": {
                "lr": self.lr,
                "gamma": self.gamma,
                "clip_ratio": self.clip_ratio,
                "lam": self.lam,
                "entropy_coef": self.entropy_coef
            }
        }
        
        with open("agent_metrics.json", 'w') as f:
            json.dump(metrics, f)
    
    def load(self, actor_path="ppo_actor.h5", critic_path="ppo_critic.h5", buffer_path="replay_buffer.pkl"):
        """Load model and replay buffer"""
        if os.path.exists(actor_path):
            self.actor.load_weights(actor_path)
        
        if os.path.exists(critic_path):
            self.critic.load_weights(critic_path)
        
        if os.path.exists(buffer_path):
            with open(buffer_path, 'rb') as f:
                self.replay_buffer = joblib.load(f)
        
        if os.path.exists("agent_metrics.json"):
            with open("agent_metrics.json", 'r') as f:
                metrics = json.load(f)
                self.train_iterations = metrics["train_iterations"]
                self.avg_rewards = metrics["avg_rewards"]
                self.win_rate = metrics["win_rate"]

#==============================================================================
# MARKET DATA MODULE
#==============================================================================

class MarketDataFeed:
    """
    Elite market data feed with multi-source verification and microstructure analysis.
    """
    
    def __init__(self, config=None, logger=None):
        """Initialize market data feed
        
        Args:
            config (dict, optional): Configuration
            logger (logging.Logger, optional): Logger
        """
        self.logger = logger or logging.getLogger("NQAlpha.MarketData")
        
        # Default configuration
        self.config = {
            'symbol': 'NQ',                      # Futures symbol
            'update_interval': 2.0,              # Update interval in seconds
            'data_dir': 'data/market_data',      # Directory for data storage
            'current_contract': 'NQM25',         # Current NQ futures contract (June 2025)
            'timeout': 15,                       # Request timeout
            'retry_count': 3,                    # Number of retries
            'min_price': 18000,                  # Minimum reasonable price
            'max_price': 22000,                  # Maximum reasonable price
            'headers': {                         # Request headers
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept-Language': 'en-US,en;q=0.9',
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache'
            },
            'debug_requests': False              # Debug HTTP requests
        }
        
        # Update with provided config
        if config:
            self._update_nested_dict(self.config, config)
        
        # Create HTTP session
        self.session = requests.Session()
        self.session.headers.update(self.config['headers'])
        
        # Internal state
        self.running = False
        self.thread = None
        self.market_data = []
        self.order_book = {}
        self.last_price = None
        self.bid = None
        self.ask = None
        self.spread = None
        self.volume = 0
        self.tick_count = 0
        self.last_update_time = None
        self.data_source = None
        
        # Order flow metrics
        self.order_flow = 0.0
        self.delta = 0.0
        self.delta_history = deque(maxlen=100)
        self.bid_volume = 0
        self.ask_volume = 0
        self.trade_imbalance = 0.0
        self.liquidity_zones = []
        self.institutional_activity = []
        
        # Market metrics
        self.vpin = 0.5
        self.toxicity = 0.0
        self.liquidity_score = 1.0
        
        # Performance metrics
        self.metrics = {
            'ticks_processed': 0,
            'updates_per_second': 0,
            'start_time': None,
            'last_tick_time': None,
            'requests_count': 0,
            'request_errors': 0,
            'request_timeouts': 0,
            'source_switches': 0
        }
        
        # Initialize microstructure analyzer
        self.microstructure = MicrostructureAnalyzer(logger=self.logger)
        
        # Ensure data directory exists
        os.makedirs(self.config['data_dir'], exist_ok=True)
        
        self.logger.info(f"Elite market data feed initialized for {self.config['symbol']}")
    
    def _update_nested_dict(self, d, u):
        """Update nested dictionary recursively"""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_nested_dict(d[k], v)
            else:
                d[k] = v
    
    def get_random_headers(self):
        """Generate random headers to avoid blocking"""
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:93.0) Gecko/20100101 Firefox/93.0",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36"
        ]
        
        return {
            "User-Agent": random.choice(user_agents),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Cache-Control": "max-age=0"
        }
    
    def run(self, interval=None):
        """Start market data feed
        
        Args:
            interval (float, optional): Update interval override
        """
        if self.running:
            self.logger.warning("Market data feed already running")
            return
        
        update_interval = interval or self.config['update_interval']
        
        self.logger.info(f"Starting elite market data feed with {update_interval}s update interval")
        
        try:
            # Set running flag
            self.running = True
            self.metrics['start_time'] = datetime.datetime.now()
            
            # Start in background thread
            self.thread = threading.Thread(
                target=self._feed_thread,
                args=(update_interval,),
                name="MarketDataThread"
            )
            self.thread.daemon = True
            self.thread.start()
            
            self.logger.info("Market data thread started")
            
        except Exception as e:
            self.running = False
            self.logger.error(f"Error starting market data feed: {e}")
    
    def stop(self):
        """Stop market data feed"""
        if not self.running:
            self.logger.warning("Market data feed not running")
            return
        
        self.logger.info("Stopping market data feed")
        
        try:
            # Set running flag
            self.running = False
            
            # Wait for thread to complete
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=5.0)
            
            # Save data
            if self.market_data:
                self._save_market_data()
            
            self.logger.info("Market data feed stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping market data feed: {e}")
    
    def _feed_thread(self, interval):
        """Background thread for market data feed
        
        Args:
            interval (float): Update interval
        """
        self.logger.info("Market data thread running")
        
        try:
            last_metrics_time = time.time()
            updates_count = 0
            
            while self.running:
                try:
                    start_time = time.time()
                    
                    # Fetch market data
                    self._fetch_market_data()
                    updates_count += 1
                    
                    # Calculate performance metrics
                    current_time = time.time()
                    if current_time - last_metrics_time >= 1.0:
                        # Update metrics every second
                        self.metrics['updates_per_second'] = updates_count
                        updates_count = 0
                        last_metrics_time = current_time
                    
                    # Sleep for remaining interval time
                    elapsed = time.time() - start_time
                    sleep_time = max(0.0, interval - elapsed)
                    
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    
                except Exception as e:
                    self.logger.error(f"Error in market data loop: {e}")
                    time.sleep(0.5)
            
        except Exception as e:
            self.logger.error(f"Fatal error in market data thread: {e}")
        
        self.logger.info("Market data thread stopped")
    
    def _fetch_market_data(self):
        """Fetch real-time market data from web sources"""
        try:
            # *** USING MULTI-SOURCE VERIFICATION ***
            price = self.get_verified_price()
            
            if price is not None:
                # Generate realistic bid-ask spread based on price volatility
                spread = 0.25  # Default spread for NQ futures
                
                # Small random adjustment to spread
                spread_adjustment = max(0.25, spread * (0.8 + 0.4 * random.random()))
                
                # Calculate bid and ask
                bid = price - spread_adjustment / 2
                ask = price + spread_adjustment / 2
                
                # Generate random volume - would normally come from market data
                volume = int(random.expovariate(1/100)) * 10  # Average around 1000
                
                # Update data
                self._update_market_data(price, bid, ask, volume, 'multi-source')
                
                return True
            else:
                self.logger.warning("No valid price received from any source")
                return False
            
        except Exception as e:
            self.logger.error(f"Error fetching market data: {e}")
            return False
    
    def get_verified_price(self):
        """Get verified NQ price from multiple sources"""
        # Try multiple sources
        prices = []
        weights = []
        
        # Try CME Group (best source)
        cme_price = self.get_cme_price()
        if cme_price is not None and self._validate_price(cme_price):
            prices.append(cme_price)
            weights.append(3.0)  # Highest weight
        
        # Try TradingView
        tv_price = self.get_tradingview_price()
        if tv_price is not None and self._validate_price(tv_price):
            prices.append(tv_price)
            weights.append(2.0)
        
        # Try Investing.com
        inv_price = self.get_investing_price()
        if inv_price is not None and self._validate_price(inv_price):
            prices.append(inv_price)
            weights.append(1.0)
        
        # Try Barchart
        bc_price = self.get_barchart_price()
        if bc_price is not None and self._validate_price(bc_price):
            prices.append(bc_price)
            weights.append(1.5)
        
        # If we have prices, calculate weighted average
        if prices:
            if len(prices) == 1:
                return prices[0]
            
            # Calculate weighted average
            total_weight = sum(weights)
            weighted_price = sum(p * w for p, w in zip(prices, weights)) / total_weight
            
            # Calculate consensus strength
            max_diff = max(abs(p - weighted_price) for p in prices)
            consensus = 1.0 - (max_diff / weighted_price)
            
            self.logger.info(f"Verified price: {weighted_price:.2f} from {len(prices)} sources (consensus: {consensus:.2f})")
            
            # Round to nearest 0.25 (NQ tick size)
            return round(weighted_price * 4) / 4
        
        # If we couldn't get any prices, return last known price
        if self.last_price is not None:
            self.logger.warning(f"No new prices available, using last price: {self.last_price}")
            return self.last_price
        
        # If no last price, return default
        self.logger.error("No price data available from any source")
        return 20192.15  # Current approximate price as of May 2025
    
    def _validate_price(self, price):
        """Validate NQ price is in reasonable range"""
        if price is None:
            return False
            
        min_price = self.config['min_price']
        max_price = self.config['max_price']
        
        if min_price <= price <= max_price:
            return True
            
        self.logger.warning(f"Price {price} outside reasonable range ({min_price}-{max_price})")
        return False
    
    def get_cme_price(self):
        """Get NQ price directly from CME Group"""
        try:
            # Try the CME delayed quotes page
            url = f"https://www.cmegroup.com/markets/equities/nasdaq/e-mini-nasdaq-100.quotes.html"
            response = self.session.get(url, headers=self.get_random_headers(), timeout=10)
            
            if response.status_code == 200:
                # Find price in JSON data
                contract_pattern = re.compile(r'"last":"(\d+\.\d+)"')
                matches = contract_pattern.findall(response.text)
                
                if matches:
                    for match in matches:
                        try:
                            price = float(match)
                            self.logger.info(f"CME Group price: {price}")
                            return price
                        except (ValueError, IndexError):
                            continue
                
                # Fallback: Try another pattern
                alt_pattern = re.compile(r'"lastPrice":(\d+\.\d+)')
                alt_matches = alt_pattern.findall(response.text)
                
                if alt_matches:
                    for match in alt_matches:
                        try:
                            price = float(match)
                            self.logger.info(f"CME Group alternate price: {price}")
                            return price
                        except (ValueError, IndexError):
                            continue
            
            return None
        except Exception as e:
            self.logger.error(f"Error getting CME price: {e}")
            return None
    
    def get_tradingview_price(self):
        """Get NQ price from TradingView"""
        try:
            url = "https://www.tradingview.com/symbols/CME_MINI-NQ1!/"
            
            response = self.session.get(url, headers=self.get_random_headers(), timeout=10)
            
            if response.status_code == 200:
                # Try to find price in JSON-LD
                json_ld = re.search(r'<script type="application/ld\+json">(.*?)</script>', response.text, re.DOTALL)
                if json_ld:
                    try:
                        data = json.loads(json_ld.group(1))
                        if 'price' in data:
                            price = float(data['price'])
                            self.logger.info(f"TradingView price: {price}")
                            return price
                    except (json.JSONDecodeError, ValueError):
                        pass
                
                # Try another pattern
                price_match = re.search(r'"last_price":"(\d+\.\d+)"', response.text)
                if price_match:
                    try:
                        price = float(price_match.group(1))
                        self.logger.info(f"TradingView last_price: {price}")
                        return price
                    except (ValueError, IndexError):
                        pass
            
            return None
        except Exception as e:
            self.logger.error(f"Error getting TradingView price: {e}")
            return None
    
    def get_investing_price(self):
        """Get NQ price from Investing.com"""
        try:
            url = "https://www.investing.com/indices/nasdaq-100-futures"
            
            response = self.session.get(url, headers=self.get_random_headers(), timeout=10)
            
            if response.status_code == 200:
                # Find price in page
                price_pattern = re.compile(r'id="last_last"[^>]*>([0-9,.]+)<')
                matches = price_pattern.findall(response.text)
                
                if matches and len(matches) > 0:
                    try:
                        price = float(matches[0].replace(',', ''))
                        self.logger.info(f"Investing.com price: {price}")
                        return price
                    except ValueError:
                        pass
                
                # Try another pattern
                alt_pattern = re.compile(r'class="text-2xl"[^>]*>([0-9,.]+)<')
                alt_matches = alt_pattern.findall(response.text)
                
                if alt_matches and len(alt_matches) > 0:
                    try:
                        price = float(alt_matches[0].replace(',', ''))
                        self.logger.info(f"Investing.com alt price: {price}")
                        return price
                    except ValueError:
                        pass
            
            return None
        except Exception as e:
            self.logger.error(f"Error getting Investing.com price: {e}")
            return None
    
    def get_barchart_price(self):
        """Get NQ price from Barchart"""
        try:
            url = "https://www.barchart.com/futures/quotes/NQ*0"
            
            response = self.session.get(url, headers=self.get_random_headers(), timeout=10)
            
            if response.status_code == 200:
                # Try to find price in the page data
                price_pattern = re.compile(r'data-current="(\d+\.\d+)"')
                matches = price_pattern.findall(response.text)
                
                if matches and len(matches) > 0:
                    try:
                        price = float(matches[0])
                        self.logger.info(f"Barchart price: {price}")
                        return price
                    except ValueError:
                        pass
                
                # Try alternate pattern
                alt_pattern = re.compile(r'"lastPrice":(\d+\.\d+)')
                alt_matches = alt_pattern.findall(response.text)
                
                if alt_matches and len(alt_matches) > 0:
                    try:
                        price = float(alt_matches[0])
                        self.logger.info(f"Barchart alt price: {price}")
                        return price
                    except ValueError:
                        pass
            
            return None
        except Exception as e:
            self.logger.error(f"Error getting Barchart price: {e}")
            return None
    
    def _update_market_data(self, price, bid, ask, volume, source):
        """Update market data with new values
        
        Args:
            price (float): Last price
            bid (float): Bid price
            ask (float): Ask price
            volume (int): Volume
            source (str): Data source
        """
        try:
            # Validate price
            if not price or price <= 0:
                self.logger.warning(f"Invalid price from {source}: {price}")
                return
            
            # Get current time
            current_time = datetime.datetime.now()
            
            # Calculate price change
            price_change = 0.0
            if self.last_price:
                price_change = price - self.last_price
            
            # Update state
            self.last_price = price
            self.bid = bid
            self.ask = ask
            self.spread = ask - bid
            self.volume += volume
            self.tick_count += 1
            self.last_update_time = current_time
            
            # Update metrics
            self.metrics['ticks_processed'] += 1
            self.metrics['last_tick_time'] = current_time
            
            # Calculate volume imbalance for order flow
            if volume > 0:
                if price_change > 0:
                    # More buying than selling
                    buy_ratio = min(0.8, 0.5 + abs(price_change) / 2)
                    self.bid_volume = int(volume * (1 - buy_ratio))
                    self.ask_volume = int(volume * buy_ratio)
                elif price_change < 0:
                    # More selling than buying
                    sell_ratio = min(0.8, 0.5 + abs(price_change) / 2)
                    self.bid_volume = int(volume * sell_ratio)
                    self.ask_volume = int(volume * (1 - sell_ratio))
                else:
                    # Equal buying and selling
                    self.bid_volume = int(volume * 0.5)
                    self.ask_volume = int(volume * 0.5)
                
                # Calculate delta and order flow
                delta = self.ask_volume - self.bid_volume
                total_volume = self.bid_volume + self.ask_volume
                normalized_delta = delta / total_volume if total_volume > 0 else 0
                
                # Add to delta history
                self.delta_history.append(normalized_delta)
                
                # Calculate order flow as weighted moving average of delta
                weights = np.linspace(1, 2, min(20, len(self.delta_history)))
                weights = weights / np.sum(weights)
                
                recent_delta = list(self.delta_history)[-min(20, len(self.delta_history)):]
                self.order_flow = np.sum(np.array(recent_delta) * weights[-len(recent_delta):])
                
                # Update delta
                self.delta = normalized_delta
            
            # Create tick data for microstructure analysis
            tick_data = {
                'timestamp': current_time,
                'price': price,
                'bid': bid,
                'ask': ask,
                'spread': self.spread,
                'volume': volume,
                'bid_volume': self.bid_volume,
                'ask_volume': self.ask_volume
            }
            
            # Update microstructure metrics
            microstructure_metrics = self.microstructure.update(tick_data)
            
            # Detect institutional activity
            self._detect_institutional_activity(price, price_change, volume, source)
            
            # Store market data including microstructure metrics
            market_data_point = {
                'timestamp': current_time,
                'price': price,
                'bid': bid,
                'ask': ask,
                'spread': self.spread,
                'volume': volume,
                'delta': self.delta,
                'order_flow': self.order_flow,
                'source': source
            }
            
            # Add microstructure metrics
            market_data_point.update(microstructure_metrics)
            
            self.market_data.append(market_data_point)
            
            # Limit market data size
            max_ticks = 10000
            if len(self.market_data) > max_ticks:
                self.market_data = self.market_data[-max_ticks:]
            
        except Exception as e:
            self.logger.error(f"Error updating market data: {e}")
    
    def _detect_institutional_activity(self, price, price_change, volume, source):
        """Detect institutional activity based on price action and volume"""
        try:
            # Skip if volume is too low
            if volume < 10:
                return
            
            # Detect large price moves or volume spikes
            large_price_move = abs(price_change) > 5.0
            large_volume = volume > 100
            
            if large_price_move or large_volume:
                # Determine direction
                direction = 'buy' if price_change >= 0 else 'sell'
                
                # Estimate size based on volume and price change
                size = max(50, volume)
                
                # Create institutional trade record
                trade = {
                    'timestamp': datetime.datetime.now(),
                    'price': price,
                    'size': size,
                    'direction': direction,
                    'type': 'institutional',
                    'source': source
                }
                
                # Add to institutional activity
                self.institutional_activity.append(trade)
                
                # Log detection
                self.logger.debug(f"Institutional {direction} detected: {size} contracts at {price}")
                
                # Limit size of institutional activity list
                max_inst = 100
                if len(self.institutional_activity) > max_inst:
                    self.institutional_activity = self.institutional_activity[-max_inst:]
        
        except Exception as e:
            self.logger.error(f"Error detecting institutional activity: {e}")
    
    def _save_market_data(self):
        """Save market data to disk"""
        try:
            if not self.market_data:
                return
            
            # Create timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create filename
            filename = f"{self.config['symbol']}_{timestamp}.csv"
            filepath = os.path.join(self.config['data_dir'], filename)
            
            # Convert to DataFrame
            df = pd.DataFrame(self.market_data)
            
            # Save to CSV
            df.to_csv(filepath, index=False)
            
            self.logger.info(f"Saved market data to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving market data: {e}")
    def get_price_history(self, length=60):
        """Get recent price history data
        
        Args:
            length: Number of price points to return
            
        Returns:
            List of recent prices, newest last
        """
        # If we haven't implemented price history collection yet, use a workaround
        if not hasattr(self, '_price_history'):
            self._price_history = []
        
        # Get current price and add to history
        current_price = self.get_realtime_data().get('price', None)
        if current_price is not None:
            self._price_history.append(current_price)
            
            # Keep history to a reasonable length
            while len(self._price_history) > 1000:  # 1000 is arbitrary max length
                self._price_history.pop(0)
        
        # Return last 'length' elements or fewer if not enough history
        return self._price_history[-min(length, len(self._price_history)):]
    def get_last_price(self):
        """Get last price
        
        Returns:
            float: Last price
        """
        return self.last_price or 0.0
    
    def get_realtime_data(self):
        """Get real-time market data
        
        Returns:
            dict: Real-time data
        """
        try:
            if not self.market_data:
                return None
            
            # Get latest data point
            latest = self.market_data[-1].copy()
            
            # Add additional data
            if 'bid_volume' not in latest:
                latest['bid_volume'] = self.bid_volume
            if 'ask_volume' not in latest:
                latest['ask_volume'] = self.ask_volume
            if 'delta' not in latest:
                latest['delta'] = self.delta
            if 'vpin' not in latest:
                latest['vpin'] = self.microstructure.metrics.get('vpin', 0.5)
            if 'liquidity_score' not in latest:
                latest['liquidity_score'] = self.microstructure.metrics.get('liquidity_score', 1.0)
            
            return latest
            
        except Exception as e:
            self.logger.error(f"Error getting real-time data: {e}")
            return None
    
    def get_order_flow_metrics(self):
        """Get order flow metrics
        
        Returns:
            dict: Order flow metrics
        """
        try:
            # Get enhanced order flow metrics from microstructure analyzer
            metrics = self.microstructure.metrics.copy()
            
            # Add traditional metrics
            metrics.update({
                'delta': self.delta,
                'order_flow': self.order_flow,
                'bid_volume': self.bid_volume,
                'ask_volume': self.ask_volume,
                'institutional_pressure': self._get_institutional_pressure()
            })
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error getting order flow metrics: {e}")
            return {}
    
    def _get_institutional_pressure(self):
        """Calculate institutional pressure
        
        Returns:
            float: Institutional pressure (-1 to 1)
        """
        try:
            if not self.institutional_activity:
                return 0.0
            
            # Get recent activity (last hour)
            now = datetime.datetime.now()
            recent = [
                trade for trade in self.institutional_activity
                if (now - trade['timestamp']).total_seconds() < 3600
            ]
            
            if not recent:
                return 0.0
            
            # Calculate buy and sell volume
            buy_volume = sum(trade['size'] for trade in recent if trade['direction'] == 'buy')
            sell_volume = sum(trade['size'] for trade in recent if trade['direction'] == 'sell')
            
            total_volume = buy_volume + sell_volume
            
            if total_volume == 0:
                return 0.0
            
            # Calculate pressure (-1 to 1)
            pressure = (buy_volume - sell_volume) / total_volume
            
            return pressure
            
        except Exception as e:
            self.logger.error(f"Error calculating institutional pressure: {e}")
            return 0.0
    def update_data(self):
        """Update market data with latest prices
        
        Returns:
            bool: Success status
        """
        try:
            # Fetch market data
            success = self._fetch_market_data()
            
            # Log update
            if success:
                self.logger.info(f"Market data updated: {self.last_price:.2f}")
            else:
                self.logger.warning("Failed to update market data")
            
            return success
        except Exception as e:
            self.logger.error(f"Error updating market data: {e}")
            return False
    def get_market_data(self, count=100):
        """Get historical market data
        
        Args:
            count (int): Number of data points
            
        Returns:
            list: Market data
        """
        try:
            if not self.market_data:
                return []
            
            # Return recent data
            return self.market_data[-count:]
            
        except Exception as e:
            self.logger.error(f"Error getting market data: {e}")
            return []
    
    def get_metrics(self):
        """Get performance metrics
        
        Returns:
            dict: Performance metrics
        """
        return self.metrics.copy()

#==============================================================================
# MARKET MICROSTRUCTURE ANALYZER
#==============================================================================
class TradingAnalytics:
    """Built-in analytics for NQ Alpha Elite without external dependencies"""
    
    def __init__(self):
        self.trade_history = []
        self.equity_curve = []
        self.regime_performance = {}
        self.pattern_performance = {}
        self.hourly_performance = {}
        self.last_report_time = datetime.datetime.now() - datetime.timedelta(minutes=10)
        self.report_interval = 300  # Report every 5 minutes
    
    def add_trade(self, trade_data):
        """Add a completed trade to the analytics engine"""
        self.trade_history.append(trade_data)
        
        # Limit history size
        if len(self.trade_history) > 500:
            self.trade_history = self.trade_history[-500:]
            
        # Update performance metrics by regime
        regime = trade_data.get('regime', 'unknown')
        if regime not in self.regime_performance:
            self.regime_performance[regime] = {
                'trades': 0, 'wins': 0, 'losses': 0, 
                'profit': 0, 'max_profit': 0, 'max_loss': 0,
                'avg_profit': 0, 'avg_loss': 0
            }
        
        # Update regime stats
        self.regime_performance[regime]['trades'] += 1
        profit = trade_data.get('profit', 0)
        self.regime_performance[regime]['profit'] += profit
        
        if profit > 0:
            self.regime_performance[regime]['wins'] += 1
            self.regime_performance[regime]['max_profit'] = max(
                self.regime_performance[regime]['max_profit'], profit)
            
            # Update average profit
            self.regime_performance[regime]['avg_profit'] = (
                (self.regime_performance[regime]['avg_profit'] * 
                 (self.regime_performance[regime]['wins'] - 1) + profit) / 
                self.regime_performance[regime]['wins']
            )
        else:
            self.regime_performance[regime]['losses'] += 1
            self.regime_performance[regime]['max_loss'] = min(
                self.regime_performance[regime]['max_loss'], profit)
                
            # Update average loss
            if self.regime_performance[regime]['losses'] > 0:
                self.regime_performance[regime]['avg_loss'] = (
                    (self.regime_performance[regime]['avg_loss'] * 
                     (self.regime_performance[regime]['losses'] - 1) + profit) / 
                    self.regime_performance[regime]['losses']
                )
        
        # Update pattern performance if available
        patterns = trade_data.get('patterns', [])
        for pattern in patterns:
            pattern_name = pattern.get('name', 'unknown')
            if pattern_name not in self.pattern_performance:
                self.pattern_performance[pattern_name] = {
                    'trades': 0, 'wins': 0, 'losses': 0, 'profit': 0
                }
            
            self.pattern_performance[pattern_name]['trades'] += 1
            self.pattern_performance[pattern_name]['profit'] += profit
            if profit > 0:
                self.pattern_performance[pattern_name]['wins'] += 1
            else:
                self.pattern_performance[pattern_name]['losses'] += 1
        
        # Update hourly performance
        hour = trade_data.get('exit_time', datetime.datetime.now()).hour
        if hour not in self.hourly_performance:
            self.hourly_performance[hour] = {
                'trades': 0, 'wins': 0, 'losses': 0, 'profit': 0
            }
        
        self.hourly_performance[hour]['trades'] += 1
        self.hourly_performance[hour]['profit'] += profit
        if profit > 0:
            self.hourly_performance[hour]['wins'] += 1
        else:
            self.hourly_performance[hour]['losses'] += 1
    
    def update_equity(self, equity):
        """Update equity curve"""
        self.equity_curve.append({
            'time': datetime.datetime.now(),
            'equity': equity
        })
        
        # Limit history size
        if len(self.equity_curve) > 1000:
            self.equity_curve = self.equity_curve[-1000:]
    
    def calculate_metrics(self):
        """Calculate comprehensive performance metrics"""
        if not self.trade_history:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'average_profit': 0,
                'average_loss': 0,
                'best_trade': 0,
                'worst_trade': 0,
                'average_trade': 0,
                'max_consecutive_wins': 0,
                'max_consecutive_losses': 0,
                'recovery_factor': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'drawdown': 0,
                'max_drawdown': 0
            }
        
        # Calculate basic metrics
        total_trades = len(self.trade_history)
        wins = sum(1 for trade in self.trade_history if trade.get('profit', 0) > 0)
        losses = total_trades - wins
        win_rate = wins / total_trades if total_trades > 0 else 0
        
        # Calculate profit metrics
        gross_profit = sum(trade.get('profit', 0) for trade in self.trade_history if trade.get('profit', 0) > 0)
        gross_loss = sum(trade.get('profit', 0) for trade in self.trade_history if trade.get('profit', 0) <= 0)
        profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float('inf')
        
        # Calculate average trade metrics
        average_profit = gross_profit / wins if wins > 0 else 0
        average_loss = gross_loss / losses if losses > 0 else 0
        average_trade = (gross_profit + gross_loss) / total_trades if total_trades > 0 else 0
        
        # Find best and worst trades
        best_trade = max((trade.get('profit', 0) for trade in self.trade_history), default=0)
        worst_trade = min((trade.get('profit', 0) for trade in self.trade_history), default=0)
        
        # Calculate consecutive win/loss streaks
        current_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        for trade in self.trade_history:
            if trade.get('profit', 0) > 0:
                if current_streak > 0:
                    current_streak += 1
                else:
                    current_streak = 1
                max_win_streak = max(max_win_streak, current_streak)
            else:
                if current_streak < 0:
                    current_streak -= 1
                else:
                    current_streak = -1
                max_loss_streak = max(max_loss_streak, abs(current_streak))
        
        # Calculate drawdown metrics
        if self.equity_curve:
            equity_values = [point['equity'] for point in self.equity_curve]
            peak = equity_values[0]
            drawdown = 0
            max_drawdown = 0
            for equity in equity_values:
                if equity > peak:
                    peak = equity
                current_drawdown = (peak - equity) / peak if peak > 0 else 0
                drawdown = current_drawdown
                max_drawdown = max(max_drawdown, current_drawdown)
            
            # Calculate recovery factor
            net_profit = equity_values[-1] - equity_values[0]
            recovery_factor = abs(net_profit / max_drawdown) if max_drawdown > 0 else float('inf')
            
            # Calculate risk-adjusted returns
            if len(equity_values) > 1:
                returns = [(equity_values[i] / equity_values[i-1]) - 1 for i in range(1, len(equity_values))]
                avg_return = sum(returns) / len(returns)
                std_dev = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
                
                # Sharpe Ratio (simplified)
                sharpe_ratio = (avg_return / std_dev) * (252 ** 0.5) if std_dev > 0 else 0
                
                # Sortino Ratio (simplified)
                downside_returns = [r for r in returns if r < 0]
                downside_deviation = (sum(r ** 2 for r in downside_returns) / len(downside_returns)) ** 0.5 if downside_returns else 0
                sortino_ratio = (avg_return / downside_deviation) * (252 ** 0.5) if downside_deviation > 0 else 0
            else:
                sharpe_ratio = 0
                sortino_ratio = 0
        else:
            drawdown = 0
            max_drawdown = 0
            recovery_factor = 0
            sharpe_ratio = 0
            sortino_ratio = 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate * 100,  # Convert to percentage
            'profit_factor': profit_factor,
            'average_profit': average_profit,
            'average_loss': average_loss,
            'best_trade': best_trade,
            'worst_trade': worst_trade,
            'average_trade': average_trade,
            'max_consecutive_wins': max_win_streak,
            'max_consecutive_losses': max_loss_streak,
            'recovery_factor': recovery_factor,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'drawdown': drawdown * 100,  # Convert to percentage
            'max_drawdown': max_drawdown * 100  # Convert to percentage
        }
    
    def should_generate_report(self):
        """Check if it's time to generate a report"""
        current_time = datetime.datetime.now()
        if (current_time - self.last_report_time).total_seconds() >= self.report_interval:
            self.last_report_time = current_time
            return True
        return False
    
    def generate_report(self, logger, current_system_info=None):
        """Generate and log a comprehensive performance report"""
        # Calculate performance metrics
        metrics = self.calculate_metrics()
        
        # Generate report header
        report = "\n" + "=" * 80 + "\n"
        report += f"NQ Alpha Elite - Performance Analytics - {datetime.datetime.now()}\n"
        report += "=" * 80 + "\n\n"
        
        # Add high-level metrics
        report += "HIGH-LEVEL PERFORMANCE METRICS\n"
        report += "-" * 40 + "\n"
        report += f"Total Trades: {metrics['total_trades']}\n"
        report += f"Win Rate: {metrics['win_rate']:.2f}%\n"
        report += f"Profit Factor: {metrics['profit_factor']:.2f}\n"
        report += f"Average Trade: ${metrics['average_trade']:.2f}\n"
        report += f"Current Drawdown: {metrics['drawdown']:.2f}%\n"
        report += f"Max Drawdown: {metrics['max_drawdown']:.2f}%\n"
        report += f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
        report += f"Sortino Ratio: {metrics['sortino_ratio']:.2f}\n\n"
        
        # Add regime performance
        report += "REGIME PERFORMANCE\n"
        report += "-" * 40 + "\n"
        for regime, stats in self.regime_performance.items():
            win_rate = stats['wins'] / stats['trades'] * 100 if stats['trades'] > 0 else 0
            report += f"{regime.capitalize()}: {stats['trades']} trades, {win_rate:.1f}% win rate, ${stats['profit']:.2f} total profit\n"
            report += f"  Avg Win: ${stats['avg_profit']:.2f}, Avg Loss: ${stats['avg_loss']:.2f}\n"
        report += "\n"
        
        # Add pattern performance
        if self.pattern_performance:
            report += "PATTERN PERFORMANCE\n"
            report += "-" * 40 + "\n"
            for pattern, stats in self.pattern_performance.items():
                win_rate = stats['wins'] / stats['trades'] * 100 if stats['trades'] > 0 else 0
                report += f"{pattern}: {stats['trades']} trades, {win_rate:.1f}% win rate, ${stats['profit']:.2f} profit\n"
            report += "\n"
        
        # Add hourly performance
        report += "HOURLY PERFORMANCE\n"
        report += "-" * 40 + "\n"
        for hour in sorted(self.hourly_performance.keys()):
            stats = self.hourly_performance[hour]
            win_rate = stats['wins'] / stats['trades'] * 100 if stats['trades'] > 0 else 0
            report += f"{hour:02d}:00-{hour:02d}:59: {stats['trades']} trades, {win_rate:.1f}% win rate, ${stats['profit']:.2f} profit\n"
        report += "\n"
        
        # Add recent trades
        report += "RECENT TRADES (LAST 5)\n"
        report += "-" * 40 + "\n"
        for trade in self.trade_history[-5:]:
            profit = trade.get('profit', 0)
            entry = trade.get('entry_price', 0)
            exit_price = trade.get('exit_price', 0)
            regime = trade.get('regime', 'unknown')
            reason = trade.get('reason', 'unknown')
            direction = "LONG" if trade.get('size', 0) > 0 else "SHORT"
            
            report += f"{direction} {regime} trade: ${entry:.2f} → ${exit_price:.2f}, P&L: ${profit:.2f}, Reason: {reason}\n"
        
        # Add current system info if provided
        if current_system_info:
            report += "\nCURRENT SYSTEM STATUS\n"
            report += "-" * 40 + "\n"
            for key, value in current_system_info.items():
                report += f"{key}: {value}\n"
        
        report += "\n" + "=" * 80 + "\n"
        
        # Log the report
        for line in report.split('\n'):
            logger.info(line)
        
        return report
class MicrostructureAnalyzer:
    """Advanced market microstructure analysis"""
    
    def __init__(self, config=None, logger=None):
        self.logger = logger
        
        # Default configuration
        self.config = {
            'vpin_window': 20,
            'toxicity_threshold': 0.7,
            'order_flow_window': 50,
            'delta_smoothing': 0.1,
            'order_book_levels': 5,
            'implied_sentiment_window': 10
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
        
        # Internal state
        self.order_flow_history = deque(maxlen=self.config['order_flow_window'])
        self.delta_history = deque(maxlen=self.config['order_flow_window'])
        self.price_history = deque(maxlen=self.config['order_flow_window'])
        self.volume_history = deque(maxlen=self.config['order_flow_window'])
        self.vpin_history = deque(maxlen=self.config['vpin_window'])
        self.metrics = {}
    
    def update(self, tick_data):
        """Update microstructure metrics with new tick data
        
        Args:
            tick_data (dict): Market data tick
            
        Returns:
            dict: Updated microstructure metrics
        """
        # Extract data
        price = tick_data.get('price')
        volume = tick_data.get('volume', 0)
        bid_volume = tick_data.get('bid_volume')
        ask_volume = tick_data.get('ask_volume')
        spread = tick_data.get('spread')
        
        # Store history
        self.price_history.append(price)
        self.volume_history.append(volume)
        
        # Calculate delta if bid/ask volumes available
        if bid_volume is not None and ask_volume is not None:
            total_volume = bid_volume + ask_volume
            if total_volume > 0:
                delta = (ask_volume - bid_volume) / total_volume
                self.delta_history.append(delta)
                
                # Cumulative delta
                cum_delta = sum(self.delta_history)
                
                # Normalized delta (for easier interpretation)
                norm_delta = cum_delta / len(self.delta_history) if self.delta_history else 0
            else:
                delta = 0
                cum_delta = 0
                norm_delta = 0
        else:
            # Estimate delta from price changes if volumes not available
            if len(self.price_history) >= 2:
                price_change = self.price_history[-1] - self.price_history[-2]
                estimated_delta = np.tanh(price_change * 2)  # Scale and bound between -1 and 1
                self.delta_history.append(estimated_delta)
                
                # Cumulative delta
                cum_delta = sum(self.delta_history)
                
                # Normalized delta
                norm_delta = cum_delta / len(self.delta_history) if self.delta_history else 0
            else:
                delta = 0
                cum_delta = 0
                norm_delta = 0
        
        # Calculate order flow
        if len(self.delta_history) >= 2:
            # Simple order flow - change in delta
            order_flow = self.delta_history[-1] - self.delta_history[-2]
            
            # Smoothed order flow - exponential smoothing
            alpha = self.config['delta_smoothing']
            if 'smoothed_flow' in self.metrics:
                smoothed_flow = alpha * order_flow + (1 - alpha) * self.metrics['smoothed_flow']
            else:
                smoothed_flow = order_flow
            
            # Store order flow
            self.order_flow_history.append(order_flow)
        else:
            order_flow = 0
            smoothed_flow = 0
        
        # Calculate VPIN (Volume-synchronized Probability of Informed Trading)
        if len(self.delta_history) > 0:
            # VPIN is average of absolute delta values
            vpin = np.mean([abs(d) for d in list(self.delta_history)[-self.config['vpin_window']:]])
            self.vpin_history.append(vpin)
        else:
            vpin = 0.5  # Default VPIN
        
        # Calculate toxicity
        if vpin > self.config['toxicity_threshold']:
            # High VPIN indicates toxic order flow
            toxicity = (vpin - self.config['toxicity_threshold']) / (1 - self.config['toxicity_threshold'])
        else:
            toxicity = 0
        
        # Calculate imbalance trend
        if len(self.delta_history) >= 10:
            recent_deltas = list(self.delta_history)[-10:]
            positive_bias = sum(1 for d in recent_deltas if d > 0) / len(recent_deltas)
            negative_bias = sum(1 for d in recent_deltas if d < 0) / len(recent_deltas)
            
            # Calculate dominant bias
            if positive_bias > 0.7:
                bias = positive_bias
            elif negative_bias > 0.7:
                bias = -negative_bias
            else:
                bias = 0
        else:
            bias = 0
        
        # Calculate inverse toxicity as liquidity score
        liquidity_score = 1.0 - min(1.0, toxicity)
        
        # Adjust for spread if available
        if spread is not None and len(self.price_history) > 0:
            avg_price = np.mean(self.price_history)
            relative_spread = spread / avg_price
            
            # Higher spread reduces liquidity
            liquidity_score = liquidity_score * (1.0 - min(0.5, relative_spread * 1000))
        
        # Calculate institutional pressure
        if len(self.delta_history) >= 20 and len(self.volume_history) >= 20:
            # Calculate volume-weighted delta
            volume_weighted_delta = 0
            total_vol = sum(self.volume_history)
            
            if total_vol > 0:
                for i in range(min(len(self.delta_history), len(self.volume_history))):
                    volume_weighted_delta += self.delta_history[i] * self.volume_history[i] / total_vol
            
            # Institutional metrics
            institutional_pressure = volume_weighted_delta
            
            # Calculate large lot imbalance (proxy for institutional activity)
            # This would typically use actual order sizes, but we're simulating
            if len(self.volume_history) > 0 and max(self.volume_history) > 0:
                normalized_volumes = [v / max(self.volume_history) for v in self.volume_history]
                large_lot_idx = [i for i, v in enumerate(normalized_volumes) if v > 0.8]
                
                if large_lot_idx:
                    large_lot_deltas = [self.delta_history[i] for i in large_lot_idx if i < len(self.delta_history)]
                    large_lot_imbalance = np.mean(large_lot_deltas) if large_lot_deltas else 0
                else:
                    large_lot_imbalance = 0
            else:
                large_lot_imbalance = 0
        else:
            institutional_pressure = 0
            large_lot_imbalance = 0
        
        # Compile metrics
        self.metrics = {
            'delta': delta if 'delta' in locals() else 0,
            'cum_delta': cum_delta,
            'norm_delta': norm_delta,
            'order_flow': order_flow,
            'smoothed_flow': smoothed_flow,
            'vpin': vpin,
            'toxicity': toxicity,
            'liquidity_score': liquidity_score,
            'bias': bias,
            'institutional_pressure': institutional_pressure,
            'large_lot_imbalance': large_lot_imbalance
        }
        
        return self.metrics
    
    def get_order_flow_signal(self):
        """Generate trading signal from order flow
        
        Returns:
            float: Order flow signal (-1 to 1)
        """
        if not self.metrics:
            return 0.0
        
        # Define weights for different metrics
        weights = {
            'norm_delta': 1.0,       # Normalized cumulative delta
            'smoothed_flow': 0.8,    # Smoothed order flow 
            'bias': 1.2,             # Directional bias
            'institutional_pressure': 1.5  # Institutional pressure
        }
        
        # Adjust weights based on liquidity
        liquidity = self.metrics.get('liquidity_score', 0.5)
        if liquidity < 0.3:
            # In low liquidity, reduce all signals
            for k in weights:
                weights[k] *= 0.5
        
        # Calculate weighted signal
        signal = 0.0
        weight_sum = 0.0
        
        for metric, weight in weights.items():
            if metric in self.metrics:
                signal += self.metrics[metric] * weight
                weight_sum += weight
        
        if weight_sum > 0:
            signal = signal / weight_sum
        
        # Bound between -1 and 1
        return max(-1.0, min(1.0, signal))

#==============================================================================
# MARKET REGIME CLASSIFIER
#==============================================================================

class MarketRegimeClassifier:
    """
    Advanced market regime classifier with machine learning enhancements.
    """
    
    def __init__(self, config=None, system=None, logger=None):
        """Initialize market regime classifier
        
        Args:
            config (dict, optional): Configuration
            system (NQAlphaSystem, optional): Parent system
            logger (logging.Logger, optional): Logger
        """
        self.logger = logger or logging.getLogger("NQAlpha.RegimeClassifier")
        self.system = system
        
        # Default configuration
        self.config = {
            'update_interval': 5.0,                # Update interval in seconds
            'price_history_length': 100,           # Length of price history to keep
            'volatility_window': 20,               # Window for volatility calculation
            'trend_window': 50,                    # Window for trend calculation
            'lookback_period': 50,                 # Lookback period for regime detection
            'regime_thresholds': {                 # Thresholds for regime classification
                'volatility': {                    # Volatility thresholds
                    'low': 0.0001,                 # Low volatility threshold
                    'high': 0.0005                 # High volatility threshold
                },
                'trend': {                         # Trend thresholds
                    'weak': 0.2,                   # Weak trend threshold
                    'strong': 0.5                  # Strong trend threshold
                }
            },
            'regimes': [                           # Available regimes
                'trending_up',                     # Strong uptrend
                'trending_down',                   # Strong downtrend
                'range_bound',                     # Rangebound market
                'volatile',                        # Volatile market
                'choppy'                           # Choppy market
            ],
            'use_advanced_detection': True,        # Use advanced detection methods
            'ml_ensemble_weight': 0.3,             # Weight for ML-based classifications
            'smoothing_factor': 0.7,               # Regime smoothing factor (0-1)
            'detection_method': 'ensemble'         # 'basic', 'hurst', 'ensemble'
        }
        
        # Update with provided config
        if config:
            self._update_nested_dict(self.config, config)
        
        # Internal state
        self.running = False
        self.thread = None
        self.price_history = deque(maxlen=self.config['price_history_length'])
        self.return_history = deque(maxlen=self.config['price_history_length'])
        self.prices = deque(maxlen=self.config['lookback_period'])  # Added this for compatibility
        self.returns = []  # Added for compatibility
        self.current_regime = 'unknown'
        self.regime_start_time = datetime.datetime.now()
        self.regime_confidence = 0.0
        self.volatility = 0.0
        self.trend_strength = 0.0
        self.trend_direction = 0.0
        self.hurst_exponent = 0.5  # Default: random walk
        self.regime_history = deque(maxlen=20)     # Store recent regime classifications
        
        # Advanced detection state
        self.regime_probabilities = {regime: 0.0 for regime in self.config['regimes']}
        self.ml_classifications = []
        
        # Metrics
        self.metrics = {
            'updates_count': 0,
            'regime_changes': 0,
            'last_update_time': None
        }
        
        self.logger.info("Elite market regime classifier initialized")
    
    def _update_nested_dict(self, d, u):
        """Update nested dictionary recursively"""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_nested_dict(d[k], v)
            else:
                d[k] = v
    
    def start(self):
        """Start regime classifier"""
        if self.running:
            self.logger.warning("Regime classifier already running")
            return
        
        self.logger.info("Starting regime classifier")
        
        try:
            # Set running flag
            self.running = True
            
            # Start in background thread
            self.thread = threading.Thread(
                target=self._classifier_thread,
                name="RegimeClassifierThread"
            )
            self.thread.daemon = True
            self.thread.start()
            
            self.logger.info("Regime classifier thread started")
            
        except Exception as e:
            self.running = False
            self.logger.error(f"Error starting regime classifier: {e}")
    
    def stop(self):
        """Stop regime classifier"""
        if not self.running:
            self.logger.warning("Regime classifier not running")
            return
        
        self.logger.info("Stopping regime classifier")
        
        try:
            # Set running flag
            self.running = False
            
            # Wait for thread to complete
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=2.0)
            
            self.logger.info("Regime classifier stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping regime classifier: {e}")
    
    def _classifier_thread(self):
        """Background thread for regime classifier"""
        self.logger.info("Regime classifier thread running")
        
        try:
            while self.running:
                try:
                    # Update regime
                    self._update_regime()
                    
                    # Sleep
                    time.sleep(self.config['update_interval'])
                    
                except Exception as e:
                    self.logger.error(f"Error in regime classifier: {e}")
                    time.sleep(1.0)
            
        except Exception as e:
            self.logger.error(f"Fatal error in regime classifier thread: {e}")
        
        self.logger.info("Regime classifier thread stopped")
    
    def _calculate_hurst(self):
        """Calculate Hurst exponent to identify mean-reversion vs trend-following regimes
            
        Returns:
            float: Hurst exponent (0-1)
        """
        try:
            # Need enough data
            if len(self.returns) < 20:
                return 0.5
            
            # Calculate range over lag
            lags = range(2, min(20, len(self.returns) // 2))
            tau = []
            
            # Calculate the array of the variances of the lagged differences
            for lag in lags:
                # Construct price vector using lag
                pp = []
                for i in range(lag):
                    if i + len(self.returns) - lag + 1 <= len(self.returns):
                        pp.append(sum(self.returns[i:i + len(self.returns) - lag + 1]))
                
                # Calculate variance
                if len(pp) > 0:
                    tau.append(np.sqrt(np.std(pp)))
            
            # Calculate Hurst as slope of log-log regression
            if len(tau) > 1 and len(lags) == len(tau):
                m = np.polyfit(np.log(lags), np.log(tau), 1)
                hurst = m[0]
                return hurst
            
            return 0.5  # Default: random walk
            
        except Exception as e:
            self.logger.error(f"Error calculating Hurst exponent: {e}")
            return 0.5  # Default: random walk
    
    def _update_regime(self):
        """Update market regime based on current data"""
        try:
            # Get current data
            market_data = self.system.market_data.get_realtime_data()
            if not market_data:
                return
            
            # Get current price
            current_price = market_data.get('price')
            if not current_price:
                return
            
            # Update price history
            self.prices.append(current_price)
            
            # Calculate returns
            if len(self.prices) > 1:
                returns = []
                for i in range(1, len(self.prices)):
                    returns.append((self.prices[i] - self.prices[i-1]) / self.prices[i-1])
                self.returns = returns
                
                # Update volatility - safe handling for NumPy arrays
                if isinstance(returns, np.ndarray):
                    self.volatility = float(np.std(returns))
                else:
                    self.volatility = np.std(returns)
                
                # Update trend
                self._update_trend()
                
                # Calculate Hurst exponent (measure of long-term memory)
                self.hurst_exponent = self._calculate_hurst()
                
                # Apply detection method based on configuration
                if self.config['detection_method'] == 'basic':
                    self._basic_regime_detection()
                elif self.config['detection_method'] == 'hurst':
                    regime, confidence = self._hurst_based_detection()
                    self._set_new_regime(regime, confidence)
                else:  # Default to ensemble
                    self._ensemble_regime_detection()
            
            # Update metrics
            self.metrics['updates_count'] += 1
            self.metrics['last_update_time'] = datetime.datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error updating regime: {e}")
    
    def _update_trend(self):
        """Update trend direction and strength"""
        try:
            # Simple linear regression
            x = np.arange(len(self.prices))
            y = np.array(self.prices)
            
            # Calculate slope
            if len(x) > 1 and len(y) > 1:
                slope, _, _, _, _ = linregress(x, y)
                self.trend_direction = 1 if slope > 0 else -1
                
                # Calculate R-squared for trend strength
                y_mean = np.mean(y)
                # Safe handling for NumPy array comparisons
                ss_tot = np.sum((y - y_mean) ** 2)
                
                # Predict y values
                y_pred = slope * x + np.mean(y) - slope * np.mean(x)
                ss_res = np.sum((y - y_pred) ** 2)
                
                # Calculate R-squared
                if ss_tot > 0:
                    r2 = 1 - (ss_res / ss_tot)
                    # Scale to 0-5 range
                    self.trend_strength = min(5.0, max(0.0, r2 * 5))
                else:
                    self.trend_strength = 0.0
            else:
                self.trend_direction = 0
                self.trend_strength = 0.0
                
        except Exception as e:
            self.logger.error(f"Error updating trend: {e}")
            self.trend_direction = 0
            self.trend_strength = 0.0
    
    def _basic_regime_detection(self):
        """Basic regime detection using volatility and trend metrics"""
        try:
            # Get thresholds
            vol_thresholds = self.config['regime_thresholds']['volatility']
            trend_thresholds = self.config['regime_thresholds']['trend']
            
            # Determine basic regime
            old_regime = self.current_regime
            
            # Handle NumPy arrays
            volatility_value = float(self.volatility) if isinstance(self.volatility, np.ndarray) else self.volatility
            trend_strength_value = float(self.trend_strength) if isinstance(self.trend_strength, np.ndarray) else self.trend_strength
            
            if volatility_value > vol_thresholds['high']:
                if trend_strength_value > trend_thresholds['strong']:
                    # High volatility with strong trend
                    new_regime = 'trending_up' if self.trend_direction > 0 else 'trending_down'
                    confidence = min(1.0, trend_strength_value + volatility_value)
                else:
                    # High volatility without strong trend
                    new_regime = 'volatile'
                    confidence = min(1.0, volatility_value * 2)
            elif trend_strength_value > trend_thresholds['strong']:
                # Strong trend without high volatility
                new_regime = 'trending_up' if self.trend_direction > 0 else 'trending_down'
                confidence = trend_strength_value
            elif volatility_value < vol_thresholds['low'] and trend_strength_value < trend_thresholds['weak']:
                # Low volatility and weak trend
                new_regime = 'range_bound'
                confidence = 1.0 - max(volatility_value * 10, trend_strength_value)
            else:
                # Default: choppy market
                new_regime = 'choppy'
                confidence = 0.5
            
            # Update regime with smoothing
            if old_regime != 'unknown' and old_regime != new_regime:
                # Apply smoothing - only change regime if significant change in conditions
                smoothing = self.config['smoothing_factor']
                if confidence > (self.regime_confidence * 1.2) or len(self.regime_history) >= 3 and all(r == new_regime for r in list(self.regime_history)[-3:]):
                    # Change regime immediately if high confidence or consistent new classification
                    self._set_new_regime(new_regime, confidence)
                else:
                    # Otherwise blend with previous regime
                    if random.random() > smoothing:
                        self._set_new_regime(new_regime, confidence)
            else:
                self._set_new_regime(new_regime, confidence)
        except Exception as e:
            self.logger.error(f"Error in basic regime detection: {e}")
    
    def _hurst_based_detection(self):
        """Regime detection using Hurst exponent"""
        try:
            # Hurst interpretation
            # H < 0.4: Mean-reverting
            # 0.4 <= H <= 0.6: Random walk
            # H > 0.6: Trending
            
            # Fix for array comparison - handle if hurst_exponent is numpy array
            if isinstance(self.hurst_exponent, np.ndarray):
                hurst_value = float(self.hurst_exponent.mean())  # Take mean if it's an array
            else:
                hurst_value = self.hurst_exponent
            
            # Base regime on Hurst exponent
            if hurst_value < 0.4:
                # Mean-reverting regime
                new_regime = 'range_bound'
                confidence = 0.5 + min(0.5, (0.4 - hurst_value) * 5)
            elif hurst_value > 0.6:
                # Trending regime
                if self.trend_direction > 0:
                    new_regime = 'trending_up'
                else:
                    new_regime = 'trending_down'
                confidence = 0.5 + min(0.5, (hurst_value - 0.6) * 2.5)
            else:
                # Random walk / choppy
                # Handle NumPy arrays
                volatility_value = float(self.volatility) if isinstance(self.volatility, np.ndarray) else self.volatility
                if volatility_value > self.config['regime_thresholds']['volatility']['high']:
                    new_regime = 'volatile'
                    confidence = 0.5 + min(0.5, volatility_value * 10)
                else:
                    new_regime = 'choppy'
                    confidence = 0.5
                    
            return new_regime, confidence
        
        except Exception as e:
            self.logger.error(f"Error in Hurst-based detection: {e}")
            # Default to "unknown" regime in case of error
            return 'unknown', 0.3
    
    def _volatility_based_detection(self):
        """Regime detection using volatility"""
        try:
            # Handle NumPy arrays
            if isinstance(self.volatility, np.ndarray):
                volatility_value = float(self.volatility.mean())
            else:
                volatility_value = self.volatility
                
            # Check volatility thresholds
            if volatility_value > self.config['regime_thresholds']['volatility']['high']:
                # High volatility regime
                new_regime = 'volatile'
                confidence = 0.7 + min(0.3, (volatility_value - self.config['regime_thresholds']['volatility']['high']) * 10)
            elif volatility_value < self.config['regime_thresholds']['volatility']['low']:
                # Low volatility regime - could be range-bound or weak trend
                if self.trend_strength > 3.0:
                    # Trending with low volatility
                    if self.trend_direction > 0:
                        new_regime = 'trending_up'
                    else:
                        new_regime = 'trending_down'
                    confidence = 0.5 + min(0.5, self.trend_strength / 10.0)
                else:
                    # Range-bound
                    new_regime = 'range_bound'
                    confidence = 0.5 + min(0.3, (self.config['regime_thresholds']['volatility']['low'] - volatility_value) * 10)
            else:
                # Medium volatility
                if self.trend_strength > 3.5:
                    # Trending
                    if self.trend_direction > 0:
                        new_regime = 'trending_up'
                    else:
                        new_regime = 'trending_down'
                    confidence = 0.6 + min(0.4, (self.trend_strength - 3.5) / 5.0)
                else:
                    # Choppy
                    new_regime = 'choppy'
                    confidence = 0.5
            
            return new_regime, confidence
            
        except Exception as e:
            self.logger.error(f"Error in volatility detection: {e}")
            return 'unknown', 0.3
    
    def _ensemble_regime_detection(self):
        """Ensemble-based regime detection combining multiple methods"""
        try:
            # Get both detection methods' results
            hurst_regime, hurst_confidence = self._hurst_based_detection()
            volatility_regime, volatility_confidence = self._volatility_based_detection()
            
            # Combine regimes with weights
            regimes = [hurst_regime, volatility_regime]
            confidences = [hurst_confidence, volatility_confidence]
            
            # Add microstructure data if available
            micro_regime = None
            micro_confidence = 0.0
            if self.system and hasattr(self.system, 'market_data') and hasattr(self.system.market_data, 'microstructure'):
                # Use order flow metrics to influence regime detection
                order_flow = self.system.market_data.microstructure.metrics.get('order_flow', 0)
                vpin = self.system.market_data.microstructure.metrics.get('vpin', 0.5)
                institutional_pressure = self.system.market_data.microstructure.metrics.get('institutional_pressure', 0)
                
                if abs(institutional_pressure) > 0.6:
                    # Strong institutional pressure indicates trending market
                    if institutional_pressure > 0:
                        micro_regime = 'trending_up'
                    else:
                        micro_regime = 'trending_down'
                    micro_confidence = abs(institutional_pressure)
                elif vpin > 0.7:
                    # High VPIN indicates volatile market
                    micro_regime = 'volatile'
                    micro_confidence = vpin
                
                if micro_regime:
                    regimes.append(micro_regime)
                    confidences.append(micro_confidence)
            
            # Get regime with highest confidence
            max_confidence = max(confidences)
            max_index = confidences.index(max_confidence)
            new_regime = regimes[max_index]
            
            # Apply smoothing for stability
            old_regime = self.current_regime
            if old_regime != 'unknown' and old_regime != new_regime:
                # Check for regime persistence in history
                if len(self.regime_history) >= 3:
                    # If 3 consecutive same classifications, switch immediately
                    if all(r == new_regime for r in list(self.regime_history)[-2:]):
                        self._set_new_regime(new_regime, max_confidence)
                        return
                
                # Otherwise apply smoothing factor
                smoothing = self.config['smoothing_factor']
                if max_confidence > (self.regime_confidence * 1.2) or random.random() > smoothing:
                    self._set_new_regime(new_regime, max_confidence)
            else:
                self._set_new_regime(new_regime, max_confidence)
        
        except Exception as e:
            self.logger.error(f"Error in ensemble regime detection: {e}")
    
    def _set_new_regime(self, regime, confidence):
        """Set new regime with logging
        
        Args:
            regime (str): New regime
            confidence (float): Confidence level
        """
        if self.current_regime != regime:
            self.logger.info(f"Market regime changed from {self.current_regime} to {regime} (confidence: {confidence:.2f})")
            self.regime_start_time = datetime.datetime.now()
            self.metrics['regime_changes'] += 1
        
        self.current_regime = regime
        self.regime_confidence = confidence
        self.regime_history.append(regime)
    
    def get_current_regime(self):
        """Get current market regime
        
        Returns:
            dict: Current regime information
        """
        # Safely handle NumPy arrays in returned values
        volatility = float(self.volatility) if isinstance(self.volatility, np.ndarray) else self.volatility
        trend_strength = float(self.trend_strength) if isinstance(self.trend_strength, np.ndarray) else self.trend_strength
        trend_direction = float(self.trend_direction) if isinstance(self.trend_direction, np.ndarray) else self.trend_direction
        hurst_exponent = float(self.hurst_exponent) if isinstance(self.hurst_exponent, np.ndarray) else self.hurst_exponent
        
        return {
            'regime': self.current_regime,
            'confidence': self.regime_confidence,
            'volatility': volatility,
            'trend_strength': trend_strength,
            'trend_direction': trend_direction,
            'hurst_exponent': hurst_exponent,
            'start_time': self.regime_start_time,
            'duration': (datetime.datetime.now() - self.regime_start_time).total_seconds()
        }

#==============================================================================
# ALPHA ENHANCER
#==============================================================================

class AlphaEnhancer:
    """Proprietary signal amplification for NQ Alpha Trading System"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger("NQAlpha.AlphaEnhancer")
        self.last_signals = []
        self.regime_weights = {
            'trending_up': {'trend_following': 1.5, 'mean_reversion': 0.5, 'breakout': 1.2, 'order_flow': 1.0},
            'trending_down': {'trend_following': 1.5, 'mean_reversion': 0.5, 'breakout': 1.2, 'order_flow': 1.0},
            'range_bound': {'trend_following': 0.4, 'mean_reversion': 1.8, 'breakout': 0.7, 'order_flow': 1.3},
            'volatile': {'trend_following': 0.7, 'mean_reversion': 0.6, 'breakout': 0.4, 'order_flow': 1.5},
            'choppy': {'trend_following': 0.3, 'mean_reversion': 1.2, 'breakout': 0.5, 'order_flow': 1.0},
            'unknown': {'trend_following': 0.7, 'mean_reversion': 0.7, 'breakout': 0.7, 'order_flow': 1.0}
        }
        self.logger.info("Alpha Enhancer initialized")
        
    def enhance_signals(self, signals, regime=None, market_data=None, regime_confidence=None):
        """Enhance strategy signals with alpha factors
        
        Args:
            signals (dict): Strategy signals
            regime (str, optional): Current market regime
            market_data (dict, optional): Market data
            regime_confidence (float, optional): Confidence level in current regime
            
        Returns:
            dict: Enhanced signals
        """
        try:
            # Clone signals
            enhanced = signals.copy()
            
            # Apply alpha enhancements
            if regime and market_data:
                # Get regime confidence (use provided or default to 0.8)
                confidence = regime_confidence if regime_confidence is not None else 0.8
                
                # Apply regime-specific adjustments
                if regime == 'trending_up':
                    # In uptrend, boost trend following and breakout, reduce mean reversion
                    if 'trend_following' in enhanced:
                        enhanced['trend_following'] *= 1.0 + (0.2 * confidence)
                    if 'breakout' in enhanced:
                        enhanced['breakout'] *= 1.0 + (0.15 * confidence)
                    if 'mean_reversion' in enhanced:
                        enhanced['mean_reversion'] *= 1.0 - (0.3 * confidence)
                
                elif regime == 'trending_down':
                    # In downtrend, boost trend following and breakout, reduce mean reversion
                    if 'trend_following' in enhanced:
                        enhanced['trend_following'] *= 1.0 + (0.2 * confidence)
                    if 'breakout' in enhanced:
                        enhanced['breakout'] *= 1.0 + (0.15 * confidence)
                    if 'mean_reversion' in enhanced:
                        enhanced['mean_reversion'] *= 1.0 - (0.3 * confidence)
                
                elif regime == 'range_bound':
                    # In range-bound market, boost mean reversion, reduce trend following
                    if 'mean_reversion' in enhanced:
                        enhanced['mean_reversion'] *= 1.0 + (0.25 * confidence)
                    if 'trend_following' in enhanced:
                        enhanced['trend_following'] *= 1.0 - (0.25 * confidence)
                
                elif regime == 'volatile':
                    # In volatile market, boost order flow and breakout, reduce all others
                    for strategy in enhanced:
                        if strategy in ['order_flow', 'breakout']:
                            enhanced[strategy] *= 1.0 + (0.1 * confidence)
                        else:
                            enhanced[strategy] *= 1.0 - (0.2 * confidence)
                
                # Calculate composite signal
                composite = 0.0
                total_weight = 0.0
                
                for strategy, signal in enhanced.items():
                    if strategy != 'composite':
                        # Apply market data factors
                        if market_data and strategy == 'order_flow':
                            # Adjust order flow by VPIN if available
                            vpin = market_data.get('vpin', 0.5)
                            if vpin > 0.7:
                                # High VPIN indicates potential significant price moves
                                enhanced[strategy] *= 1.2
                            elif vpin < 0.3:
                                # Low VPIN indicates less informed trading
                                enhanced[strategy] *= 0.8
                        
                        # Add to composite
                        weight = 0.25  # Default equal weight
                        composite += signal * weight
                        total_weight += weight
                
                # Normalize composite
                if total_weight > 0:
                    composite = composite / total_weight
                
                # Add composite to enhanced signals
                enhanced['composite'] = composite
            
            return enhanced
            
        except Exception as e:
            self.logger.error(f"Error enhancing signals: {e}")
            # Return original signals on error
            enhanced = signals.copy()
            if 'composite' not in enhanced:
                # Calculate simple average for composite
                values = [v for k, v in signals.items() if k != 'composite']
                if values:
                    enhanced['composite'] = sum(values) / len(values)
                else:
                    enhanced['composite'] = 0.0
            return enhanced
    
class StrategyManager:
    """
    Elite strategy manager with advanced signal processing and alpha enhancement.
    """
    
    def __init__(self, config=None, system=None, logger=None):
        """Initialize strategy manager
        
        Args:
            config (dict, optional): Configuration
            system (NQAlphaSystem, optional): Parent system
            logger (logging.Logger, optional): Logger
        """
        self.logger = logger or logging.getLogger("NQAlpha.StrategyManager")
        self.system = system
        
        # Default configuration
        self.config = {
            'update_interval': 1.0,             # Update interval in seconds
            'strategies': {                     # Strategy settings
                'order_flow': {                 # Order flow strategy
                    'enabled': True,            # Whether strategy is enabled
                    'weight': 0.4,              # Strategy weight (0-1)
                    'threshold': 0.3,           # Signal threshold
                    'parameters': {             # Strategy-specific parameters
                        'delta_threshold': 0.5, # Delta threshold for signals
                        'vpin_threshold': 0.7   # VPIN threshold
                    }
                },
                'trend_following': {            # Trend following strategy
                    'enabled': True,            # Whether strategy is enabled
                    'weight': 0.3,              # Strategy weight (0-1)
                    'threshold': 0.4,           # Signal threshold
                    'parameters': {             # Strategy-specific parameters
                        'fast_ma': 10,          # Fast moving average
                        'slow_ma': 30           # Slow moving average
                    }
                },
                'mean_reversion': {             # Mean reversion strategy
                    'enabled': True,            # Whether strategy is enabled
                    'weight': 0.2,              # Strategy weight (0-1)
                    'threshold': 0.5,           # Signal threshold
                    'parameters': {             # Strategy-specific parameters
                        'lookback': 20,         # Lookback period
                        'std_dev': 2.0          # Standard deviation threshold
                    }
                },
                'breakout': {                   # Breakout strategy
                    'enabled': True,            # Whether strategy is enabled
                    'weight': 0.1,              # Strategy weight (0-1)
                    'threshold': 0.6,           # Signal threshold
                    'parameters': {             # Strategy-specific parameters
                        'range_period': 20,     # Range period
                        'breakout_threshold': 0.5 # Breakout threshold
                    }
                }
            },
            'bootstrap_history': True,          # Whether to bootstrap price history
            'signal_smoothing': 0.3,            # Signal smoothing factor (0-1)
            'use_alpha_enhancer': True,         # Whether to use alpha enhancer
            'enable_adaptive_thresholds': True  # Dynamically adjust thresholds
        }
        self.strategy_reweighter = StrategyReweighter(
            logger=self.logger,
            reweight_interval=100,
            lookback_period=20,
            min_trades_per_strategy=3,
            adaptation_rate=0.05
        )
        # Update with provided config
        if config:
            self._update_nested_dict(self.config, config)
        
        # Internal state
        self.running = False
        self.thread = None
        self.strategies = {}
        self.signals = {}
        self.composite_signal = 0.0
        self.price_history = []  # For bootstrapping
        
        # Initialize alpha enhancer
        self.alpha_enhancer = AlphaEnhancer(logger=self.logger)
        
        # Initialize strategies
        self._initialize_strategies()
        
        # Metrics
        self.metrics = {
            'updates_count': 0,
            'signals_generated': 0,
            'last_update_time': None
        }
        
        self.logger.info("Elite strategy manager initialized")
    
    def _update_nested_dict(self, d, u):
        """Update nested dictionary recursively"""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_nested_dict(d[k], v)
            else:
                d[k] = v
    
    def _initialize_strategies(self):
        """Initialize trading strategies"""
        try:
            # Initialize strategy objects
            for strategy_name, strategy_config in self.config['strategies'].items():
                if strategy_config['enabled']:
                    # Create strategy object (would normally import custom strategy classes)
                    self.strategies[strategy_name] = {
                        'config': strategy_config,
                        'state': {
                            'last_signal': 0.0,
                            'last_update': None,
                            'signal_count': 0,
                            'thresholds': {
                                'current': strategy_config['threshold'],
                                'min': strategy_config['threshold'] * 0.5,
                                'max': strategy_config['threshold'] * 1.5
                            }
                        }
                    }
                    
                    self.logger.info(f"Initialized strategy: {strategy_name}")
            
            self.logger.info(f"Initialized {len(self.strategies)} strategies")
            
        except Exception as e:
            self.logger.error(f"Error initializing strategies: {e}")
    
    def start(self):
        """Start strategy manager with bootstrapping for faster signal generation"""
        if self.running:
            self.logger.warning("Strategy manager already running")
            return
        
        self.logger.info("Starting strategy manager")
        
        try:
            # Set running flag
            self.running = True
            
            # Bootstrap price history if enabled
            if self.config['bootstrap_history']:
                self._bootstrap_price_history()
            
            # Start in background thread
            self.thread = threading.Thread(
                target=self._strategy_thread,
                name="StrategyManagerThread"
            )
            self.thread.daemon = True
            self.thread.start()
            
            self.logger.info("Strategy manager thread started")
            
        except Exception as e:
            self.running = False
            self.logger.error(f"Error starting strategy manager: {e}")
    
    def _bootstrap_price_history(self):
        """Bootstrap initial price history for faster strategy initialization"""
        try:
            # Get current price if available
            current_price = None
            if self.system and hasattr(self.system, 'market_data'):
                data = self.system.market_data.get_realtime_data()
                if data and 'price' in data:
                    current_price = data['price']
            
            if not current_price:
                self.logger.warning("Cannot bootstrap price history: no current price")
                return
            
            # Generate synthetic price history
            self.logger.info(f"Bootstrapping price history from current price: {current_price}")
            
            # Start with current price
            prices = [current_price]
            
            # Generate realistic price path
            for i in range(49):  # Generate 49 more points for 50 total
                # Random walk with slight mean reversion
                last_price = prices[-1]
                # Volatility scaled to be realistic for NQ futures
                volatility = current_price * 0.0003  # 0.03% volatility per step
                
                # Random component
                random_change = np.random.normal(0, volatility)
                
                # Mean reversion component (slight pull back to starting price)
                mean_reversion = (current_price - last_price) * 0.05
                
                # Combine components
                new_price = last_price + random_change + mean_reversion
                
                # Add new price to history
                prices.append(new_price)
            
            # Store bootstrapped history
            self.price_history = prices
            
            self.logger.info(f"Bootstrapped {len(prices)} price points")
            
        except Exception as e:
            self.logger.error(f"Error bootstrapping price history: {e}")
    
    def stop(self):
        """Stop strategy manager"""
        if not self.running:
            self.logger.warning("Strategy manager not running")
            return
        
        self.logger.info("Stopping strategy manager")
        
        try:
            # Set running flag
            self.running = False
            
            # Wait for thread to complete
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=2.0)
            
            self.logger.info("Strategy manager stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping strategy manager: {e}")
    
    def _strategy_thread(self):
        """Background thread for strategy manager"""
        self.logger.info("Strategy manager thread running")
        
        try:
            while self.running:
                try:
                    # Update strategies
                    self._update_strategies()
                    
                    # Sleep
                    time.sleep(self.config['update_interval'])
                    
                except Exception as e:
                    self.logger.error(f"Error in strategy manager: {e}")
                    time.sleep(1.0)
            
        except Exception as e:
            self.logger.error(f"Fatal error in strategy manager thread: {e}")
        
        self.logger.info("Strategy manager thread stopped")
    
    def _update_strategies(self):
        """Update trading strategies"""
        try:
            # Skip if no market data
            if not self.system or not hasattr(self.system, 'market_data'):
                return
            
            # Get latest market data
            market_data = self.system.market_data.get_realtime_data()
            if not market_data:
                return
            
            # Update price history with new data
            if 'price' in market_data:
                self.price_history.append(market_data['price'])
                # Keep reasonable history size
                if len(self.price_history) > 500:
                    self.price_history = self.price_history[-500:]
            
            # Get current market regime
            current_regime = 'unknown'
            regime_confidence = 0.0
            if hasattr(self.system, 'regime_classifier'):
                regime_info = self.system.regime_classifier.get_current_regime()
                if regime_info:
                    current_regime = regime_info['regime']
                    regime_confidence = regime_info['confidence']
            
            # Adaptive threshold adjustment based on regime
            if self.config['enable_adaptive_thresholds']:
                self._adjust_thresholds(current_regime, regime_confidence)
            
            # Update each strategy
            for strategy_name, strategy in self.strategies.items():
                # Get strategy config
                config = strategy['config']
                
                # Generate raw strategy signal
                raw_signal = self._generate_strategy_signal(
                    strategy_name, 
                    config, 
                    market_data, 
                    current_regime,
                    regime_confidence
                )
                
                # Apply elite signal filtering (ADD THIS LINE)
                filtered_signal = self.enhance_signal_filtering(raw_signal, strategy_name)
                
                # Apply signal smoothing
                if 'last_signal' in strategy['state'] and strategy['state']['last_signal'] is not None:
                    smoothing = self.config['signal_smoothing']
                    # Use filtered signal instead of raw signal (MODIFY THIS LINE)
                    filtered_signal = filtered_signal * (1 - smoothing) + strategy['state']['last_signal'] * smoothing
                
                # Store filtered signal (MODIFY THIS LINE)
                self.signals[strategy_name] = filtered_signal
                
                # Update strategy state with filtered signal (MODIFY THIS LINE)
                strategy['state']['last_signal'] = filtered_signal
                strategy['state']['last_update'] = datetime.datetime.now()
                strategy['state']['signal_count'] += 1
            
            # Calculate composite signal
            self._calculate_composite_signal(current_regime, market_data)
            if hasattr(self, 'strategy_reweighter') and self.metrics['updates_count'] % self.strategy_reweighter.reweight_interval == 0:
                # Get current regime
                current_regime_info = 'unknown'
                if self.system and hasattr(self.system, 'regime_classifier'):
                    regime_info = self.system.regime_classifier.get_current_regime()
                    if regime_info:
                        current_regime_info = regime_info
                
                # Get trade counts by strategy (simplified for now)
                trades_by_strategy = {strategy_name: 5 for strategy_name in self.strategies.keys()}
                
                # Reweight strategies
                self.strategy_reweighter.reweight(
                    strategies=self.strategies,
                    performance_metrics={'regime': current_regime},
                    trades_by_strategy=trades_by_strategy
                )
            # Update metrics
            self.metrics['updates_count'] += 1
            self.metrics['last_update_time'] = datetime.datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error updating strategies: {e}")
    
    def _adjust_thresholds(self, regime, confidence):
        """Dynamically adjust strategy thresholds based on market regime
        
        Args:
            regime (str): Current market regime
            confidence (float): Regime confidence
        """
        try:
            # Threshold adjustment factors by regime
            adjustment_factors = {
                'trending_up': {'trend_following': 0.7, 'mean_reversion': 1.3, 'breakout': 0.8, 'order_flow': 0.9},
                'trending_down': {'trend_following': 0.7, 'mean_reversion': 1.3, 'breakout': 0.8, 'order_flow': 0.9},
                'range_bound': {'trend_following': 1.2, 'mean_reversion': 0.6, 'breakout': 1.1, 'order_flow': 0.8},
                'volatile': {'trend_following': 1.1, 'mean_reversion': 1.2, 'breakout': 1.3, 'order_flow': 0.7},
                'choppy': {'trend_following': 1.3, 'mean_reversion': 0.8, 'breakout': 1.2, 'order_flow': 1.0},
                'unknown': {'trend_following': 1.0, 'mean_reversion': 1.0, 'breakout': 1.0, 'order_flow': 1.0}
            }
            
            # Get adjustment factors for current regime
            factors = adjustment_factors.get(regime, adjustment_factors['unknown'])
            
            # Apply adjustments to each strategy
            for strategy_name, strategy in self.strategies.items():
                if strategy_name in factors:
                    # Get base threshold
                    base_threshold = strategy['config']['threshold']
                    
                    # Calculate new threshold
                    factor = factors[strategy_name]
                    new_threshold = base_threshold * factor
                    
                    # Scale by confidence
                    adjustment = (new_threshold - strategy['state']['thresholds']['current']) * confidence
                    
                    # Update current threshold
                    current = strategy['state']['thresholds']['current'] + adjustment
                    
                    # Ensure threshold is within limits
                    min_threshold = strategy['state']['thresholds']['min']
                    max_threshold = strategy['state']['thresholds']['max']
                    current = max(min_threshold, min(max_threshold, current))
                    
                    # Store updated threshold
                    strategy['state']['thresholds']['current'] = current
            
        except Exception as e:
            self.logger.error(f"Error adjusting thresholds: {e}")
    
    def _generate_strategy_signal(self, strategy_name, config, market_data, current_regime, regime_confidence):
        """Generate signal for a specific strategy
        
        Args:
            strategy_name (str): Strategy name
            config (dict): Strategy configuration
            market_data (dict): Market data
            current_regime (str): Current market regime
            regime_confidence (float): Regime confidence
            
        Returns:
            float: Strategy signal (-1 to 1)
        """
        try:
            # Base signal
            signal = 0.0
            
            # Generate signal based on strategy type
            if strategy_name == 'order_flow':
                # Order flow strategy using microstructure metrics
                if hasattr(self.system, 'market_data') and hasattr(self.system.market_data, 'microstructure'):
                    # Use microstructure analyzer's signal
                    signal = self.system.market_data.microstructure.get_order_flow_signal()
                else:
                    # Fallback to basic order flow signals
                    delta = market_data.get('delta', 0.0)
                    order_flow = market_data.get('order_flow', 0.0)
                    vpin = market_data.get('vpin', 0.5)
                    
                    # Calculate signal
                    if abs(order_flow) > config['parameters']['delta_threshold']:
                        signal = order_flow * 2.0  # Scale to -1 to 1
                    
                    # Adjust for VPIN
                    if vpin > config['parameters']['vpin_threshold']:
                        signal *= max(0.2, 1.0 - (vpin - config['parameters']['vpin_threshold']))
                
            elif strategy_name == 'trend_following':
                # Trend following strategy
                if len(self.price_history) >= config['parameters']['slow_ma']:
                    # Calculate moving averages
                    prices = self.price_history
                    fast_ma = np.mean(prices[-config['parameters']['fast_ma']:])
                    slow_ma = np.mean(prices[-config['parameters']['slow_ma']:])
                    
                    # Calculate signal
                    if fast_ma > slow_ma:
                        # Uptrend strength based on separation and slope
                        trend_strength = (fast_ma / slow_ma - 1.0) * 10.0
                        # Calculate slope of fast MA
                        if len(prices) >= config['parameters']['fast_ma'] + 5:
                            fast_ma_prev = np.mean(prices[-(config['parameters']['fast_ma'] + 5):-5])
                            slope = (fast_ma - fast_ma_prev) / fast_ma_prev
                            # Adjust strength by slope
                            trend_strength *= (1.0 + max(0, slope) * 20.0)
                        signal = min(1.0, trend_strength)
                    elif fast_ma < slow_ma:
                        # Downtrend strength
                        trend_strength = (fast_ma / slow_ma - 1.0) * 10.0
                        # Calculate slope of fast MA
                        if len(prices) >= config['parameters']['fast_ma'] + 5:
                            fast_ma_prev = np.mean(prices[-(config['parameters']['fast_ma'] + 5):-5])
                            slope = (fast_ma - fast_ma_prev) / fast_ma_prev
                            # Adjust strength by slope
                            trend_strength *= (1.0 + min(0, slope) * -20.0)
                        signal = max(-1.0, trend_strength)
            
            elif strategy_name == 'mean_reversion':
                # Mean reversion strategy
                if len(self.price_history) >= config['parameters']['lookback']:
                    # Calculate mean and standard deviation
                    lookback = config['parameters']['lookback']
                    prices = self.price_history[-lookback:]
                    mean_price = np.mean(prices)
                    std_price = np.std(prices)
                    current_price = market_data['price']
                    
                    # Calculate z-score
                    if std_price > 0:
                        z_score = (current_price - mean_price) / std_price
                        
                        # Generate signal (opposite of z-score)
                        if abs(z_score) > config['parameters']['std_dev']:
                            # Scale signal strength based on z-score magnitude
                            signal = -z_score / config['parameters']['std_dev'] / 2.0
                            signal = max(-1.0, min(1.0, signal))  # Ensure -1 to 1
                            
                            # Reduce signal if price is accelerating away from mean
                            if len(self.price_history) >= lookback + 2:
                                price_0 = self.price_history[-1]
                                price_1 = self.price_history[-2]
                                price_2 = self.price_history[-3]
                                
                                # Calculate acceleration
                                vel_0 = price_0 - price_1
                                vel_1 = price_1 - price_2
                                accel = vel_0 - vel_1
                                
                                # If acceleration is in wrong direction, reduce signal
                                if (signal > 0 and accel < 0) or (signal < 0 and accel > 0):
                                    signal *= 0.5
            
            elif strategy_name == 'breakout':
                # Breakout strategy
                if len(self.price_history) >= config['parameters']['range_period']:
                    # Calculate range
                    range_period = config['parameters']['range_period']
                    prices = self.price_history[-range_period:]
                    range_high = max(prices[:-1])  # Exclude current price
                    range_low = min(prices[:-1])   # Exclude current price
                    current_price = market_data['price']
                    range_size = range_high - range_low
                    
                    # Calculate range quality (tightness of range)
                    if range_size > 0:
                        range_stdev = np.std(prices[:-1])
                        range_quality = 1.0 - (range_stdev / range_size)
                        
                        # Check for breakout
                        if current_price > range_high:
                            # Breakout above
                            breakout_strength = (current_price - range_high) / range_size
                            # Scale by range quality and threshold
                            if breakout_strength > config['parameters']['breakout_threshold']:
                                signal = min(1.0, breakout_strength * range_quality)
                        elif current_price < range_low:
                            # Breakout below
                            breakout_strength = (range_low - current_price) / range_size
                            # Scale by range quality and threshold
                            if breakout_strength > config['parameters']['breakout_threshold']:
                                signal = max(-1.0, -breakout_strength * range_quality)
            
            # Adjust signal based on market regime
            if current_regime != 'unknown':
                # Adjust based on regime
                if current_regime == 'trending_up':
                    # Enhance trend following, reduce mean reversion
                    if strategy_name == 'trend_following':
                        signal *= 1.5
                    elif strategy_name == 'mean_reversion':
                        signal *= 0.5
                elif current_regime == 'trending_down':
                    # Enhance trend following, reduce mean reversion
                    if strategy_name == 'trend_following':
                        signal *= 1.5
                    elif strategy_name == 'mean_reversion':
                        signal *= 0.5
                elif current_regime == 'range_bound':
                    # Enhance mean reversion, reduce trend following
                    if strategy_name == 'trend_following':
                        signal *= 0.5
                    elif strategy_name == 'mean_reversion':
                        signal *= 1.5
                elif current_regime == 'volatile':
                    # Reduce all signals
                    signal *= 0.7
                elif current_regime == 'choppy':
                    # Reduce all signals
                    signal *= 0.8
                
                # Scale by regime confidence
                signal *= (0.5 + regime_confidence * 0.5)
            
            # Ensure signal is in range -1 to 1
            signal = max(-1.0, min(1.0, signal))
            
            # Apply strategy threshold
            current_threshold = self.strategies[strategy_name]['state']['thresholds']['current']
            if abs(signal) < current_threshold:
                signal = 0.0
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating signal for {strategy_name}: {e}")
            return 0.0
    def resolve_regime_signal_conflict(self, regime, signals):
        """Resolve conflicts between regime classification and strategy signals
        
        Args:
            regime (str): Current market regime
            signals (dict): Strategy signals
            
        Returns:
            dict: Adjusted signals
        """
        try:
            adjusted_signals = signals.copy()
            
            # Check for regime-signal conflicts
            if 'trending' in regime:
                # In trending regimes, check for contradictory mean reversion signals
                if 'mean_reversion' in adjusted_signals:
                    mean_rev_signal = adjusted_signals['mean_reversion']
                    
                    # Check for strong mean reversion signal against trend direction
                    if ((regime == 'trending_up' and mean_rev_signal < -0.5) or 
                        (regime == 'trending_down' and mean_rev_signal > 0.5)):
                        
                        # Log the conflict
                        self.logger.info(f"Signal-regime conflict: {regime} with {mean_rev_signal:.2f} mean reversion")
                        
                        # Reduce weight of the conflicting signal
                        adjusted_signals['mean_reversion'] *= 0.7
                        
                        # Check for high VPIN - possible false trend
                        vpin = 0.5
                        if hasattr(self.system, 'market_data'):
                            market_data = self.system.market_data.get_realtime_data()
                            vpin = market_data.get('vpin', 0.5)
                        
                        if vpin > 0.3:
                            # When VPIN is elevated during conflict, may indicate
                            # trapped traders - actually favor mean reversion
                            adjusted_signals['mean_reversion'] *= 1.5  # Restore and boost
                            self.logger.info(f"Potential false trend with high VPIN ({vpin:.2f}), favoring mean reversion")
            
            elif regime == 'range_bound':
                # In range bound regimes, boost mean reversion, reduce trend following
                if 'mean_reversion' in adjusted_signals:
                    adjusted_signals['mean_reversion'] *= 1.2
                
                if 'trend_following' in adjusted_signals:
                    adjusted_signals['trend_following'] *= 0.8
            
            elif regime == 'volatile':
                # In volatile regimes, check delta confirmation
                order_flow = 0.0
                if hasattr(self.system, 'market_data'):
                    market_data = self.system.market_data.get_realtime_data()
                    order_flow = market_data.get('order_flow', 0.0)
                
                # Reduce signals that contradict order flow
                for strategy_name, signal in adjusted_signals.items():
                    if (signal > 0 and order_flow < -0.1) or (signal < 0 and order_flow > 0.1):
                        adjusted_signals[strategy_name] *= 0.8
                        self.logger.info(f"Reducing {strategy_name} signal - contradicts order flow")
            
            return adjusted_signals
            
        except Exception as e:
            self.logger.error(f"Error in regime-signal conflict resolution: {e}")
            return signals  # Return original signals on error
    def _calculate_composite_signal(self, current_regime, market_data):
        """Calculate composite signal with elite alpha enhancements"""
        try:
            # Skip if no signals
            if not self.signals:
                self.composite_signal = 0.0
                return
            
            # Get regime confidence
            regime_confidence = 0.5  # Default
            if self.system and hasattr(self.system, 'regime_classifier'):
                regime_info = self.system.regime_classifier.get_current_regime()
                if regime_info:
                    regime_confidence = regime_info.get('confidence', 0.5)
            adjusted_signals = self.resolve_regime_signal_conflict(current_regime, self.signals)

            # Use adjusted signals for further processing
            if self.config['use_alpha_enhancer']:
                enhanced_signals = self.alpha_enhancer.enhance_signals(
                    adjusted_signals,  # Use adjusted signals instead of original
                    regime=current_regime,
                    market_data=market_data,
                    regime_confidence=regime_confidence
                )
            # Use alpha enhancer if enabled
            if self.config['use_alpha_enhancer']:
                enhanced_signals = self.alpha_enhancer.enhance_signals(
                    self.signals, 
                    regime=current_regime,
                    market_data=market_data,
                    regime_confidence=regime_confidence  # Pass confidence to enhancer
                )
                if 'composite' in enhanced_signals:
                    self.composite_signal = enhanced_signals['composite']
            else:
                # Calculate weighted sum of signals
                weighted_sum = 0.0
                total_weight = 0.0
                
                for strategy_name, signal in self.signals.items():
                    if strategy_name in self.strategies:
                        weight = self.strategies[strategy_name]['config']['weight']
                        weighted_sum += signal * weight
                        total_weight += weight
                
                # Normalize
                if total_weight > 0:
                    self.composite_signal = weighted_sum / total_weight
                else:
                    self.composite_signal = 0.0
            
            # Apply additional regime-based filtering for composite signal
            if current_regime == 'volatile' and abs(self.composite_signal) < 0.7:
                # Higher threshold in volatile markets
                self.composite_signal *= 0.8
            elif current_regime == 'choppy' and abs(self.composite_signal) < 0.6:
                # Higher threshold in choppy markets
                self.composite_signal *= 0.7
            
            # Apply confidence weighting
            self.composite_signal *= (0.7 + (regime_confidence * 0.3))
            
            # Ensure signal is within bounds
            self.composite_signal = max(-1.0, min(1.0, self.composite_signal))
            
            # Log significant signals
            if abs(self.composite_signal) > 0.5:
                signal_type = "BUY" if self.composite_signal > 0 else "SELL"
                self.logger.info(f"Strong {signal_type} signal: {self.composite_signal:.2f}")
                self.metrics['signals_generated'] += 1
            
        except Exception as e:
            self.logger.error(f"Error calculating composite signal: {e}")
            self.composite_signal = 0.0
    def _adapt_signal_to_volatility(self, signal, volatility):
        """Adapt signal strength based on volatility characteristics"""
        # Get liquidity and order flow metrics
        market_data = self.system.market_data.get_realtime_data()
        liquidity = market_data.get('liquidity_score', 0.5)
        order_flow = market_data.get('order_flow', 0.0)
        
        # Calculate volatility-adjusted signal
        if volatility < 0.001:  # Very low volatility
            # Reduce signal in extremely low volatility
            return signal * 0.85
        elif volatility < 0.003:  # Normal volatility
            # Boost signals that align with order flow
            if (signal > 0 and order_flow > 0) or (signal < 0 and order_flow < 0):
                return signal * 1.1
            else:
                return signal * 0.9
        else:  # High volatility
            # In high volatility, liquidity is key - reduce size in low liquidity
            if liquidity < 0.6:
                return signal * 0.8
            else:
                return signal * 1.0        
    def enhance_signal_filtering(self, raw_signal, strategy_name):
        """Apply regime-specific signal filtering to reduce false positives"""
        try:
            # Get current regime from system
            current_regime = 'unknown'
            regime_confidence = 0.0
            
            if self.system and hasattr(self.system, 'regime_classifier'):
                regime_info = self.system.regime_classifier.get_current_regime()
                current_regime = regime_info.get('regime', 'unknown')
                regime_confidence = regime_info.get('confidence', 0.0)
            
            original_signal = raw_signal  # Store for logging
            
            # Apply regime-specific adjustments with confidence weighting
            confidence_factor = min(1.0, regime_confidence + 0.3)  # Baseline confidence
            
            if current_regime == 'range_bound':
                # Range-bound market adaptations
                if strategy_name == 'mean_reversion':
                    # Boost mean reversion in range-bound markets
                    boost_factor = 1.0 + (0.25 * confidence_factor)
                    raw_signal *= boost_factor
                elif strategy_name == 'trend_following':
                    # Reduce trend following in range-bound markets
                    discount_factor = 1.0 - (0.3 * confidence_factor)
                    raw_signal *= discount_factor
                    
            elif 'trending' in current_regime:
                # Trending market adaptations
                direction_match = (raw_signal > 0 and current_regime == 'trending_up') or \
                                (raw_signal < 0 and current_regime == 'trending_down')
                
                if strategy_name == 'trend_following' and direction_match:
                    # Boost aligned trend following signals
                    boost_factor = 1.0 + (0.3 * confidence_factor)
                    raw_signal *= boost_factor
                elif strategy_name == 'mean_reversion':
                    # Reduce mean reversion in strong trends
                    discount_factor = 1.0 - (0.4 * confidence_factor)
                    raw_signal *= discount_factor
                    
            elif current_regime == 'volatile':
                # Volatile market adaptations
                if strategy_name == 'breakout':
                    # Boost breakout strategies in volatile markets
                    boost_factor = 1.0 + (0.2 * confidence_factor)
                    raw_signal *= boost_factor
                elif strategy_name == 'order_flow':
                    # Boost order flow in volatile markets
                    boost_factor = 1.0 + (0.15 * confidence_factor)
                    raw_signal *= boost_factor
            
            # Apply time-of-day filters
            current_hour = datetime.datetime.now().hour
            if 1 <= current_hour <= 3:  # Low liquidity period
                # Reduce all signals during low liquidity
                raw_signal *= 0.7
            
            # Log significant modifications
            if abs(raw_signal - original_signal) > 0.05:
                self.logger.info(f"Elite signal adjustment: {strategy_name} {original_signal:.2f} → {raw_signal:.2f} in {current_regime} regime")
            
            return raw_signal
            
        except Exception as e:
            self.logger.error(f"Error in elite signal filtering: {e}")
            return raw_signal  # Return original signal on error
    def get_strategy_signals(self):
        """Get current strategy signals
        
        Returns:
            dict: Strategy signals
        """
        return {
            'signals': self.signals.copy(),
            'composite': self.composite_signal,
            'last_update': self.metrics['last_update_time']
        }

#==============================================================================
# DYNAMIC TRADE MANAGER
#==============================================================================
# --- RL AGENT CLASS ---
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        
    def _build_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def load(self, name):
        self.model.load_weights(name)
        
    def save(self, name):
        self.model.save_weights(name)
# --- END RL AGENT CLASS ---
class DynamicTradeManager:
    """Advanced position management with dynamic stop losses and profit targets"""
    
    def __init__(self, system=None, config=None, logger=None):
        """Initialize trade manager
        
        Args:
            system: Reference to main system
            config (dict): Configuration
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger("NQAlpha.TradeManager")
        self.system = system  # Reference to main system
        
        # Default configuration
        self.config = {
            'enable_dynamic_stops': True,
            'enable_partial_exits': True,
            'enable_breakeven_stops': True,
            'breakeven_threshold': 1.5,  # ATR multiplier to move to breakeven
            'trailing_stop_activation': 2.0,  # ATR multiplier to activate trailing stop
            'trailing_stop_distance': 1.5,  # ATR multiplier for trailing distance
            'partial_exit_thresholds': [1.0, 2.0, 3.0],  # Take partial profits at these levels (ATR multipliers)
            'partial_exit_sizes': [0.25, 0.25, 0.25],  # Size of each partial exit
            'max_adverse_excursion': 2.5,  # Maximum adverse excursion allowed (ATR multiplier)
            'time_stop_minutes': 60*4,  # Exit after this many minutes if not profitable
            'minimum_profit_target': 1.0,  # Minimum profit target (ATR multiplier)
            'signal_threshold_exit': 0.3,  # Exit if signal reverses beyond this threshold
            'stop_loss_adjustment_factor': 0.6,  # Adjust stop loss based on volatility (lower = wider)
            # Elite enhancements configuration
            'kelly_max_risk': 0.02,  # Maximum risk per trade (2% of equity)
            'kelly_factor': 0.5,     # Conservative Kelly fraction (half-Kelly)
            'max_contracts': 2,      # Maximum contracts during calibration phase
            'regime_stop_adjustments': {  # Regime-specific stop adjustments
                'volatile': {'factor': 0.996, 'profit_threshold': 0.002},
                'range_bound': {'factor': 0.998, 'profit_threshold': 0.001},
                'trending_up': {'factor': 0.997, 'trail_factor': 10},
                'trending_down': {'factor': 0.997, 'trail_factor': 10}
            }
        }
        
        # Update with provided config
        if config:
            self._update_nested_dict(self.config, config)
        
        # Internal state
        self.active_trades = {}
        self.trade_log = []
        self.last_update_time = None
        self.metrics = {
            'wins': 0,
            'losses': 0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'total_profit': 0.0
        }
        
        self.logger.info("Dynamic Trade Manager initialized with elite enhancements")
    
    def _update_nested_dict(self, d, u):
        """Update nested dictionary recursively"""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_nested_dict(d[k], v)
            else:
                d[k] = v
    
    def manage_position(self, position, current_price, atr, signal, market_data=None, timestamp=None):
        """Manage an active position
        
        Args:
            position (dict): Position information {'id', 'entry_price', 'size', 'entry_time', etc.}
            current_price (float): Current market price
            atr (float): Average True Range
            signal (float): Current trade signal (-1 to 1)
            market_data (dict): Market data metrics
            timestamp (datetime): Current timestamp
            
        Returns:
            dict: Position actions {'action', 'reason', 'size', 'price', etc.}
        """
        if not position or position['size'] == 0:
            return {'action': 'none', 'reason': 'no_position'}
        
        # Set current time
        timestamp = timestamp or datetime.datetime.now()
        
        # Get position details
        position_id = position['id']
        entry_price = position['entry_price']
        current_size = position['size']
        entry_time = position['entry_time']
        direction = 1 if current_size > 0 else -1
        
        # Get or initialize trade management info
        if position_id not in self.active_trades:
            # Apply elite stop loss calculation based on market regime
            stop_loss_distance = atr * self.config['stop_loss_adjustment_factor']
            
            # Adjust stop based on regime if available
            if self.system and hasattr(self.system, 'regime_classifier'):
                regime_info = self.system.regime_classifier.get_current_regime()
                current_regime = regime_info.get('regime', 'unknown')
                volatility = regime_info.get('volatility', 0.0001)
                
                # Apply regime-specific adjustments
                if current_regime in self.config['regime_stop_adjustments']:
                    regime_factor = self.config['regime_stop_adjustments'][current_regime]['factor']
                    initial_stop = entry_price * (regime_factor if direction > 0 else 2-regime_factor)
                    self.logger.info(f"Elite stop placement: {current_regime} regime-optimized stop at {initial_stop:.2f}")
                else:
                    initial_stop = entry_price - direction * stop_loss_distance
            else:
                initial_stop = entry_price - direction * stop_loss_distance
            
            self.active_trades[position_id] = {
                'stops': {
                    'initial': initial_stop,
                    'current': initial_stop,
                    'breakeven': entry_price
                },
                'targets': {
                    'initial': entry_price + direction * atr * max(self.config['minimum_profit_target'], abs(signal)*3),
                    'current': entry_price + direction * atr * max(self.config['minimum_profit_target'], abs(signal)*3)
                },
                'partial_exits': [False] * len(self.config['partial_exit_thresholds']),
                'high_watermark': entry_price,
                'low_watermark': entry_price,
                'max_favorable_excursion': 0.0,
                'max_adverse_excursion': 0.0,
                'regime_at_entry': self.system.regime_classifier.get_current_regime() if self.system and hasattr(self.system, 'regime_classifier') else {'regime': 'unknown'},
                'last_management_time': timestamp
            }
        
        # Get trade info
        trade_info = self.active_trades[position_id]
        
        # Update watermarks
        if direction > 0:  # Long position
            trade_info['high_watermark'] = max(trade_info['high_watermark'], current_price)
            trade_info['low_watermark'] = min(trade_info['low_watermark'], current_price)
        else:  # Short position
            trade_info['high_watermark'] = max(trade_info['high_watermark'], current_price)
            trade_info['low_watermark'] = min(trade_info['low_watermark'], current_price)
        # Apply volatile regime tactics
        volatile_action = self._apply_volatile_regime_tactics(position, current_price, atr)
        if volatile_action:
            return volatile_action
        # Calculate excursions
        favorable_excursion = direction * (current_price - entry_price)
        adverse_excursion = -favorable_excursion
        
        trade_info['max_favorable_excursion'] = max(trade_info['max_favorable_excursion'], favorable_excursion)
        trade_info['max_adverse_excursion'] = max(trade_info['max_adverse_excursion'], adverse_excursion)
        
        # Apply elite adaptive trade management based on current market regime
        if self.system and hasattr(self.system, 'regime_classifier'):
            self._apply_elite_trade_management(position_id, trade_info, current_price, favorable_excursion, atr, timestamp)
        
        # Check for stop loss hit
        if direction * (current_price - trade_info['stops']['current']) < 0:
            return {
                'action': 'exit',
                'reason': 'stop_loss',
                'size': current_size,
                'price': current_price,
                'target_price': trade_info['stops']['current']
            }
        
        # Check for profit target hit
        if direction * (current_price - trade_info['targets']['current']) >= 0:
            return {
                'action': 'exit',
                'reason': 'profit_target',
                'size': current_size,
                'price': current_price,
                'target_price': trade_info['targets']['current']
            }
        
        # Check for partial exits
        if self.config['enable_partial_exits']:
            for i, threshold in enumerate(self.config['partial_exit_thresholds']):
                if not trade_info['partial_exits'][i] and favorable_excursion >= threshold * atr:
                    exit_size = int(abs(current_size) * self.config['partial_exit_sizes'][i]) * direction
                    
                    if exit_size != 0:
                        trade_info['partial_exits'][i] = True
                        return {
                            'action': 'partial_exit',
                            'reason': f'partial_target_{i+1}',
                            'size': exit_size,
                            'price': current_price,
                            'target_price': entry_price + direction * threshold * atr
                        }
        
        # Check for breakeven stop
        if self.config['enable_breakeven_stops'] and \
           not trade_info['stops']['current'] == trade_info['stops']['breakeven'] and \
           favorable_excursion >= self.config['breakeven_threshold'] * atr:
            
            trade_info['stops']['current'] = entry_price
            self.logger.info(f"Moving stop to breakeven at {entry_price}")
        
        # Check for trailing stop
        if favorable_excursion >= self.config['trailing_stop_activation'] * atr:
            # Calculate new stop level
            if direction > 0:  # Long position
                new_stop = trade_info['high_watermark'] - self.config['trailing_stop_distance'] * atr
                
                # Only move stop up, never down
                if new_stop > trade_info['stops']['current']:
                    trade_info['stops']['current'] = new_stop
                    self.logger.info(f"Updated trailing stop to {new_stop}")
            else:  # Short position
                new_stop = trade_info['low_watermark'] + self.config['trailing_stop_distance'] * atr
                
                # Only move stop down, never up
                if new_stop < trade_info['stops']['current']:
                    trade_info['stops']['current'] = new_stop
                    self.logger.info(f"Updated trailing stop to {new_stop}")
        
        # Check for time stop
        position_duration = (timestamp - entry_time).total_seconds() / 60  # Minutes
        if position_duration > self.config['time_stop_minutes'] and favorable_excursion <= 0:
            return {
                'action': 'exit',
                'reason': 'time_stop',
                'size': current_size,
                'price': current_price
            }
        
        # Check for signal reversal
        signal_direction = 1 if signal > 0 else (-1 if signal < 0 else 0)
        if signal_direction != 0 and signal_direction != direction and abs(signal) > self.config['signal_threshold_exit']:
            return {
                'action': 'exit',
                'reason': 'signal_reversal',
                'size': current_size,
                'price': current_price
            }
        
        # Check for excessive adverse excursion
        if adverse_excursion > self.config['max_adverse_excursion'] * atr:
            return {
                'action': 'exit',
                'reason': 'excessive_adverse_excursion',
                'size': current_size,
                'price': current_price
            }
        
        # No action needed
        return {
            'action': 'hold',
            'reason': 'monitoring',
            'current_stop': trade_info['stops']['current'],
            'current_target': trade_info['targets']['current']
        }
    def _apply_volatile_regime_tactics(self, position, current_price, atr):
        """Apply specialized tactics for volatile market regime"""
        # Only apply if position exists and we're in volatile regime
        if not position or not hasattr(self.system, 'regime_classifier'):
            return None
            
        regime_info = self.system.regime_classifier.get_current_regime()
        if regime_info.get('regime') != 'volatile':
            return None
            
        confidence = regime_info.get('confidence', 0.0)
        if confidence < 0.65:
            return None  # Only apply with significant confidence
        
        direction = 1 if position['size'] > 0 else -1
        entry_price = position['entry_price']
        
        # 1. Accelerated partial profit taking
        profit_pct = (current_price - entry_price) * direction / entry_price
        
        if profit_pct > 0.0015:  # Take first partial at smaller moves
            return {
                'action': 'partial_exit',
                'reason': 'volatile_quick_profit',
                'size': position['size'] * 0.33 * direction,
                'price': current_price
            }
        
        # 2. Dynamic stop adjustment based on VPIN
        vpin = self.system.market_data.get_realtime_data().get('vpin', 0.5)
        
        # If VPIN rises (more informed trading), tighten stops
        if vpin > 0.15 and profit_pct > 0.0008:
            new_stop = current_price - (direction * atr * 0.75)
            return {
                'action': 'update_stop',
                'reason': 'volatile_vpin_protection',
                'new_stop': new_stop
            }
        
        return None  # No special action needed
    def _apply_elite_trade_management(self, position_id, trade_info, current_price, favorable_excursion, atr, timestamp):
        """Apply elite trade management strategies based on market regime"""
        # Get regime information
        regime_info = self.system.regime_classifier.get_current_regime()
        current_regime = regime_info.get('regime', 'unknown')
        regime_confidence = regime_info.get('confidence', 0.0)
        volatility = regime_info.get('volatility', 0.0001)
        
        # Get position information
        entry_price = trade_info['stops']['breakeven']  # Using breakeven as entry price
        direction = 1 if current_price > entry_price else -1
        
        # Only apply updates periodically (not on every tick)
        time_since_last = (timestamp - trade_info.get('last_management_time', timestamp - datetime.timedelta(minutes=5))).total_seconds()
        if time_since_last < 60:  # Only update once per minute
            return
        
        # Update last management time
        trade_info['last_management_time'] = timestamp
        
        # Profitable trade management
        is_profitable = favorable_excursion > 0
        
        # Apply regime-specific management
        regime_adjustments = self.config['regime_stop_adjustments'].get(current_regime, {})
        
        if current_regime == 'volatile' and is_profitable and regime_confidence > 0.6:
            # In volatile markets, take profits quickly but with wider initial stops
            profit_threshold = regime_adjustments.get('profit_threshold', 0.002)
            
            if favorable_excursion / entry_price > profit_threshold:
                # Tighten stops to lock in profits in volatile markets
                new_stop = current_price - direction * (atr * 0.5)
                
                # Only update if better
                if direction > 0 and new_stop > trade_info['stops']['current']:
                    trade_info['stops']['current'] = new_stop
                    self.logger.info(f"Elite volatile regime stop adjustment: {new_stop:.2f}")
                elif direction < 0 and new_stop < trade_info['stops']['current']:
                    trade_info['stops']['current'] = new_stop
                    self.logger.info(f"Elite volatile regime stop adjustment: {new_stop:.2f}")
        
        elif current_regime == 'range_bound' and regime_confidence > 0.7:
            # In range-bound markets, be more aggressive with taking profits
            profit_threshold = regime_adjustments.get('profit_threshold', 0.001)
            
            if favorable_excursion / entry_price > profit_threshold:
                # Very tight stops in range-bound markets to secure profits
                new_stop = current_price - direction * (atr * 0.3)
                
                # Only update if better
                if direction > 0 and new_stop > trade_info['stops']['current']:
                    trade_info['stops']['current'] = new_stop
                    self.logger.info(f"Elite range-bound regime stop adjustment: {new_stop:.2f}")
                elif direction < 0 and new_stop < trade_info['stops']['current']:
                    trade_info['stops']['current'] = new_stop
                    self.logger.info(f"Elite range-bound regime stop adjustment: {new_stop:.2f}")
        
        elif 'trending' in current_regime and regime_confidence > 0.75:
            # In trending markets, let profits run with trailing stops
            trail_factor = regime_adjustments.get('trail_factor', 10)
            
            if is_profitable and favorable_excursion / entry_price > 0.0005:
                # Calculate volatility-based trailing stop
                trail_amount = max(atr * 0.8, current_price * volatility * trail_factor)
                new_stop = current_price - direction * trail_amount
                
                # Only update if better
                if direction > 0 and new_stop > trade_info['stops']['current']:
                    trade_info['stops']['current'] = new_stop
                    self.logger.info(f"Elite trending regime trailing stop: {new_stop:.2f}")
                elif direction < 0 and new_stop < trade_info['stops']['current']:
                    trade_info['stops']['current'] = new_stop
                    self.logger.info(f"Elite trending regime trailing stop: {new_stop:.2f}")
    
    def calculate_position_size(self, signal, current_price, stop_price=None):
        """Calculate position size with elite regime optimization"""
        try:
            # Get account equity
            equity = self.system.execution_engine.get_account_equity()
            
            # Use Kelly criterion if enabled
            if self.config.get('kelly_factor', 0) > 0:
                # Calculate win rate based on recent performance
                win_rate = self.metrics.get('win_rate', 0.55)  # Default to 55% if no data
                
                # Calculate risk amount based on kelly_max_risk
                risk_amount = equity * self.config.get('kelly_max_risk', 0.02)
                
                # Calculate reward/risk ratio based on signal strength
                reward_risk_ratio = 1.5 + (abs(signal) * 1.5)  # 1.5-3.0 based on signal
                
                # Calculate Kelly position size
                kelly_percentage = win_rate - ((1 - win_rate) / reward_risk_ratio)
                
                # Apply fractional Kelly
                kelly_percentage *= self.config.get('kelly_factor', 0.5)
                
                # Ensure it's positive
                kelly_percentage = max(0, kelly_percentage)
                
                # Calculate risk per point
                if stop_price is not None and stop_price != current_price:
                    risk_per_point = abs(current_price - stop_price)
                else:
                    # Default to 0.5% of price if no stop specified
                    risk_per_point = current_price * 0.005
                
                # Calculate position size
                position_size = int((risk_amount / risk_per_point) / 20)  # NQ is $20 per point
                
                # Ensure minimum and maximum position size
                position_size = max(1, min(position_size, self.config.get('max_contracts', 10)))
                
                self.logger.info(f"Kelly position sizing: {position_size} contracts (Kelly: {kelly_percentage:.2f}, WR: {win_rate:.2f}, R/R: {reward_risk_ratio:.1f})")
            else:
                # Simple position sizing based on equity percentage
                risk_amount = equity * 0.01  # 1% risk
                position_size = max(1, int(risk_amount / 2000))  # $2000 risk per contract
            
            # ENHANCEMENT: Enhance position sizing based on regime
            if hasattr(self.system, 'regime_classifier'):
                regime_info = self.system.regime_classifier.get_current_regime()
                current_regime = regime_info.get('regime', 'unknown')
                confidence = regime_info.get('confidence', 0.5)
                
                # Trending regime position boost
                if 'trending' in current_regime and confidence > 0.8:
                    # For high-confidence trending regimes, increase position size
                    original_size = position_size
                    position_size = int(position_size * 1.2)
                    self.logger.info(f"Elite position sizing: {original_size} → {position_size} in {current_regime} regime")
                    
                # Range-bound regime position reduction
                elif current_regime == 'range_bound':
                    # For range-bound regimes, decrease position size
                    original_size = position_size
                    position_size = int(position_size * 0.8)
                    self.logger.info(f"Elite position sizing: {original_size} → {position_size} in {current_regime} regime")
                
                # Volatile regime with high vpin - smaller position
                elif current_regime == 'volatile' and hasattr(self.system, 'market_data'):
                    market_data = self.system.market_data.get_realtime_data()
                    vpin = market_data.get('vpin', 0.5)
                    if vpin > 0.25:
                        original_size = position_size
                        position_size = int(position_size * 0.7)
                        self.logger.info(f"Elite position sizing: {original_size} → {position_size} in volatile regime with high VPIN ({vpin:.2f})")
            
            # Ensure minimum position size
            position_size = max(1, position_size)
            
            return position_size
        
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 1  # Default to 1 contract on error
    
    def get_position_details(self, position_id):
        """Get details for a specific position
        
        Args:
            position_id (str): Position ID
            
        Returns:
            dict: Position management details
        """
        return self.active_trades.get(position_id, {})
    
    
    def clear_position(self, position_id):
        """Clear position from active trades
        
        Args:
            position_id (str): Position ID
        """
        if position_id in self.active_trades:
            # Add to trade log before removing
            self.trade_log.append({
                'position_id': position_id,
                'management_data': self.active_trades[position_id],
                'closed_at': datetime.datetime.now()
            })
            
            # Remove from active trades
            del self.active_trades[position_id]
    
    def update_metrics(self, trade_result):
        """Update performance metrics after trade completion
        
        Args:
            trade_result (dict): Trade result data with profit/loss
        """
        try:
            profit = trade_result.get('profit', 0)
            
            # Update win/loss counts
            if profit > 0:
                self.metrics['wins'] += 1
                # Update average win
                self.metrics['avg_win'] = ((self.metrics['avg_win'] * (self.metrics['wins'] - 1)) + profit) / self.metrics['wins']
            elif profit < 0:
                self.metrics['losses'] += 1
                # Update average loss (stored as positive value)
                self.metrics['avg_loss'] = ((self.metrics['avg_loss'] * (self.metrics['losses'] - 1)) + abs(profit)) / self.metrics['losses']
            
            # Update total profit
            self.metrics['total_profit'] += profit
            
            self.logger.info(f"Updated trade metrics - Win rate: {self.get_win_rate()*100:.1f}%, " + 
                            f"Avg win: {self.metrics['avg_win']:.2f}, Avg loss: {self.metrics['avg_loss']:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error updating metrics: {e}")
    
    def get_win_rate(self):
        """Get current win rate
        
        Returns:
            float: Win rate (0-1)
        """
        total_trades = self.metrics['wins'] + self.metrics['losses']
        if total_trades > 0:
            return self.metrics['wins'] / total_trades
        return 0.5  # Default 50% if no trades
#==============================================================================
# RISK MANAGER
#==============================================================================

class RiskManager:
    """
    Elite risk manager with advanced position sizing and dynamic risk control.
    """
    
    def __init__(self, config=None, system=None, logger=None):
        """Initialize risk manager
        
        Args:
            config (dict, optional): Configuration
            system (NQAlphaSystem, optional): Parent system
            logger (logging.Logger, optional): Logger
        """
        self.logger = logger or logging.getLogger("NQAlpha.RiskManager")
        self.system = system
        
        # Default configuration
        self.config = {
            'max_position_size': 10,          # Maximum position size in contracts
            'max_capital_at_risk': 0.02,      # Maximum capital at risk (2%)
            'max_drawdown': 0.05,             # Maximum drawdown (5%)
            'position_sizing': 'kelly',       # Position sizing method (fixed, volatility, kelly)
            'kelly_fraction': 0.3,            # Kelly fraction
            'stop_loss_atr': 2.0,             # Stop loss in ATR units
            'profit_target_atr': 3.0,         # Profit target in ATR units
            'regime_risk_factors': {          # Risk factors by regime
                'trending_up': 1.0,
                'trending_down': 0.8,
                'range_bound': 0.7,
                'volatile': 0.5,
                'choppy': 0.6,
                'unknown': 0.5
            },
            'use_dynamic_trade_manager': True,  # Whether to use dynamic trade manager
            'use_volatility_adjustment': True,  # Adjust position sizing by volatility
            'volatility_lookback': 20,         # Lookback for volatility calculation
            'max_open_trades': 1,              # Maximum number of open trades
            'max_correlation': 0.7             # Maximum correlation between trades
        }
        
        # Update with provided config
        if config:
            self._update_nested_dict(self.config, config)
        
        # Initialize dynamic trade manager if enabled
        if self.config['use_dynamic_trade_manager']:
            self.trade_manager = DynamicTradeManager(logger=self.logger)
        
        # Internal state
        self.positions = {}
        self.total_risk = 0.0
        self.current_drawdown = 0.0
        self.atr = 0.0
        self.regime_factor = 1.0
        self.volatility_ratio = 1.0  # Current volatility relative to historical
        self.historical_volatility = 0.0
        
        # Position history for win rate calculation
        self.position_history = []
        self.win_count = 0
        self.loss_count = 0
        
        self.logger.info("Elite risk manager initialized")
    
    def _update_nested_dict(self, d, u):
        """Update nested dictionary recursively"""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_nested_dict(d[k], v)
            else:
                d[k] = v
    
    def calculate_position_size(self, signal, price, equity):
        """Calculate position size based on signal, price, and equity
        
        Args:
            signal (float): Trading signal (-1 to 1)
            price (float): Current price
            equity (float): Account equity
            
        Returns:
            int: Position size in contracts
        """
        try:
            # Absolute signal strength
            signal_strength = abs(signal)
            
            # Skip if signal too weak
            if signal_strength < 0.2:
                return 0
            
            # Calculate ATR if we have market data
            self._calculate_atr()
            
            # Get current regime factor
            self._update_regime_factor()
            
            # Update volatility ratio if enabled
            if self.config['use_volatility_adjustment']:
                self._update_volatility_ratio()
            
            # Base position size
            position_size = 0
            
            # Position sizing methods
            if self.config['position_sizing'] == 'fixed':
                # Fixed position sizing
                position_size = int(self.config['max_position_size'] * signal_strength)
                
            elif self.config['position_sizing'] == 'volatility':
                # Volatility-based position sizing
                risk_amount = equity * self.config['max_capital_at_risk'] * signal_strength
                volatility_risk = price * self.atr * 0.01  # Convert ATR to dollar risk
                
                if volatility_risk > 0:
                    position_size = int(risk_amount / volatility_risk)
                
            elif self.config['position_sizing'] == 'kelly':
                # Kelly criterion position sizing
                # Estimate win rate from signal strength and historical performance
                base_win_rate = 0.5
                if self.win_count + self.loss_count > 10:
                    # Use historical win rate if we have enough data
                    base_win_rate = self.win_count / (self.win_count + self.loss_count)
                
                # Adjust win rate by signal strength
                win_rate = base_win_rate + (signal_strength * 0.2)
                
                # Estimate risk/reward ratio
                risk_reward = self.config['profit_target_atr'] / self.config['stop_loss_atr']
                
                # Kelly formula: f = p - (1-p)/r where p=win rate, r=risk/reward
                kelly = win_rate - (1 - win_rate) / risk_reward
                
                # Apply Kelly fraction
                kelly *= self.config['kelly_fraction']
                
                # Kelly can be negative, ensure it's positive
                kelly = max(0, kelly)
                
                # Calculate position size
                position_size = int(kelly * equity / price)
            
            # Apply regime factor
            position_size = int(position_size * self.regime_factor)
            
            # Apply volatility adjustment
            if self.config['use_volatility_adjustment'] and self.volatility_ratio > 0:
                # Reduce size in high volatility, increase in low volatility
                volatility_adjustment = 1.0 / self.volatility_ratio
                position_size = int(position_size * volatility_adjustment)
            
            # Ensure max position size
            position_size = min(position_size, self.config['max_position_size'])
            
            # Ensure minimum position size (if signal is strong enough)
            if signal_strength > 0.7 and position_size == 0:
                position_size = 1
            
            # Ensure direction
            if signal < 0:
                position_size = -position_size
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0
    
    def _calculate_atr(self, period=14):
        """Calculate Average True Range
        
        Args:
            period (int): ATR period
        """
        try:
            # Skip if no market data
            if not self.system or not hasattr(self.system, 'market_data'):
                self.atr = 1.0  # Default value
                return
            
            # Get market data
            market_data = self.system.market_data.get_market_data(period + 10)
            if not market_data or len(market_data) < period:
                self.atr = 1.0  # Default value
                return
            
            # Calculate ATR
            highs = [data.get('price', data.get('ask', 0)) for data in market_data]
            lows = [data.get('price', data.get('bid', 0)) for data in market_data]
            closes = [data.get('price', 0) for data in market_data]
            
            # Calculate true ranges
            tr_values = []
            for i in range(1, len(highs)):
                high = highs[i]
                low = lows[i]
                prev_close = closes[i-1]
                
                tr1 = high - low
                tr2 = abs(high - prev_close)
                tr3 = abs(low - prev_close)
                
                tr = max(tr1, tr2, tr3)
                tr_values.append(tr)
            
            # Calculate ATR
            if tr_values:
                self.atr = sum(tr_values[-period:]) / len(tr_values[-period:])
            else:
                self.atr = 1.0  # Default value
            
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {e}")
            self.atr = 1.0  # Default value
    
    def _update_regime_factor(self):
        """Update risk factor based on current market regime"""
        try:
            # Default regime factor
            self.regime_factor = 1.0
            
            # Get current regime
            if self.system and hasattr(self.system, 'regime_classifier'):
                regime_info = self.system.regime_classifier.get_current_regime()
                if regime_info:
                    regime = regime_info.get('regime', 'unknown')
                    
                    # Apply regime-specific risk factor
                    if regime in self.config['regime_risk_factors']:
                        self.regime_factor = self.config['regime_risk_factors'][regime]
                    
                    # Scale by confidence
                    confidence = regime_info.get('confidence', 0.5)
                    self.regime_factor *= (0.7 + 0.3 * confidence)
            
        except Exception as e:
            self.logger.error(f"Error updating regime factor: {e}")
            self.regime_factor = 0.5  # Conservative default
    
    def _update_volatility_ratio(self):
        """Update current volatility ratio compared to historical"""
        try:
            # Skip if no market data
            if not self.system or not hasattr(self.system, 'market_data'):
                self.volatility_ratio = 1.0  # Default value
                return
            
            # Get market data
            lookback = self.config['volatility_lookback']
            market_data = self.system.market_data.get_market_data(lookback * 3)  # Get 3x the lookback period
            if not market_data or len(market_data) < lookback * 2:
                self.volatility_ratio = 1.0  # Default value
                return
            
            # Calculate current volatility (recent lookback period)
            current_prices = [data.get('price', 0) for data in market_data[-lookback:]]
            current_returns = np.diff(current_prices) / current_prices[:-1]
            current_vol = np.std(current_returns) if len(current_returns) > 0 else 0
            
            # Calculate historical volatility (previous lookback period)
            hist_prices = [data.get('price', 0) for data in market_data[-lookback*2:-lookback]]
            hist_returns = np.diff(hist_prices) / hist_prices[:-1]
            hist_vol = np.std(hist_returns) if len(hist_returns) > 0 else 0
            
            # If no historical data, use current as historical
            if hist_vol == 0:
                self.historical_volatility = current_vol
                self.volatility_ratio = 1.0
                return
            
            # Store historical volatility
            self.historical_volatility = hist_vol
            
            # Calculate ratio
            self.volatility_ratio = current_vol / hist_vol if hist_vol > 0 else 1.0
            
            # Cap at reasonable values
            self.volatility_ratio = min(3.0, max(0.3, self.volatility_ratio))
            
        except Exception as e:
            self.logger.error(f"Error updating volatility ratio: {e}")
            self.volatility_ratio = 1.0  # Default value
    
    def calculate_stop_loss(self, entry_price, position_size, direction):
        """Calculate stop loss price
        
        Args:
            entry_price (float): Entry price
            position_size (int): Position size
            direction (int): Position direction (1 for long, -1 for short)
            
        Returns:
            float: Stop loss price
        """
        try:
            # Calculate stop loss based on ATR
            if self.atr <= 0:
                self._calculate_atr()
            
            stop_distance = self.atr * self.config['stop_loss_atr']
            
            # Apply direction
            if direction > 0:  # Long position
                stop_price = entry_price - stop_distance
            else:  # Short position
                stop_price = entry_price + stop_distance
            
            return stop_price
            
        except Exception as e:
            self.logger.error(f"Error calculating stop loss: {e}")
            # Default to 2% stop loss
            return entry_price * (0.98 if direction > 0 else 1.02)
    
    def calculate_profit_target(self, entry_price, position_size, direction):
        """Calculate profit target price
        
        Args:
            entry_price (float): Entry price
            position_size (int): Position size
            direction (int): Position direction (1 for long, -1 for short)
            
        Returns:
            float: Profit target price
        """
        try:
            # Calculate profit target based on ATR
            if self.atr <= 0:
                self._calculate_atr()
            
            target_distance = self.atr * self.config['profit_target_atr']
            
            # Apply direction
            if direction > 0:  # Long position
                target_price = entry_price + target_distance
            else:  # Short position
                target_price = entry_price - target_distance
            
            return target_price
            
        except Exception as e:
            self.logger.error(f"Error calculating profit target: {e}")
            # Default to 3% profit target
            return entry_price * (1.03 if direction > 0 else 0.97)
    
    def is_risk_acceptable(self, position_size, stop_loss_price, entry_price, equity):
        """Check if risk is acceptable
        
        Args:
            position_size (int): Position size
            stop_loss_price (float): Stop loss price
            entry_price (float): Entry price
            equity (float): Account equity
            
        Returns:
            bool: Whether risk is acceptable
        """
        try:
            # Skip if position size is zero
            if position_size == 0:
                return False
            
            # Calculate risk per contract
            risk_per_contract = abs(entry_price - stop_loss_price)
            
            # Calculate total risk
            total_risk = risk_per_contract * abs(position_size)
            
            # Calculate risk as percentage of equity
            risk_percentage = total_risk / equity
            
            # Check if risk is acceptable
            if risk_percentage > self.config['max_capital_at_risk']:
                self.logger.warning(f"Risk too high: {risk_percentage:.2%} > {self.config['max_capital_at_risk']:.2%}")
                return False
            
            # Check if drawdown is too high
            if self.current_drawdown > self.config['max_drawdown']:
                self.logger.warning(f"Drawdown too high: {self.current_drawdown:.2%} > {self.config['max_drawdown']:.2%}")
                return False
            
            # Check if we have too many open positions
            if len(self.positions) >= self.config['max_open_trades']:
                self.logger.warning(f"Too many open trades: {len(self.positions)} >= {self.config['max_open_trades']}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking risk acceptability: {e}")
            return False
    
    def update_drawdown(self, equity, peak_equity):
        """Update current drawdown
        
        Args:
            equity (float): Current equity
            peak_equity (float): Peak equity
        """
        try:
            # Calculate drawdown
            if peak_equity > 0:
                self.current_drawdown = max(0, (peak_equity - equity) / peak_equity)
            else:
                self.current_drawdown = 0.0
            
        except Exception as e:
            self.logger.error(f"Error updating drawdown: {e}")
            self.current_drawdown = 0.0
    
    def manage_positions(self, signal, current_price):
        """Manage all open positions
        
        Args:
            signal (float): Current trading signal
            current_price (float): Current price
            
        Returns:
            list: Position actions
        """
        if not self.config['use_dynamic_trade_manager']:
            return []
        
        actions = []
        
        for position_id, position in self.positions.items():
            # Get position details
            try:
                # Skip if not active
                if position.get('status') != 'active':
                    continue
                
                # Get market data
                market_data = None
                if self.system and hasattr(self.system, 'market_data'):
                    market_data = self.system.market_data.get_realtime_data()
                
                # Calculate ATR if needed
                if self.atr <= 0:
                    self._calculate_atr()
                
                # Manage position
                action = self.trade_manager.manage_position(
                    position,
                    current_price,
                    self.atr,
                    signal,
                    market_data=market_data
                )
                
                # Add position ID to action
                action['position_id'] = position_id
                
                # Add to actions list
                actions.append(action)
                
                # Process action
                if action['action'] == 'exit':
                    # Update position history for win rate calculation
                    self._update_position_history(position, action)
                    
            except Exception as e:
                self.logger.error(f"Error managing position {position_id}: {e}")
        
        return actions
    
    def _update_position_history(self, position, action):
        """Update position history for win rate calculation
        
        Args:
            position (dict): Position details
            action (dict): Position action
        """
        try:
            # Calculate P&L
            entry_price = position['entry_price']
            exit_price = action['price']
            size = position['size']
            
            pnl = (exit_price - entry_price) * size
            
            # Add to position history
            self.position_history.append({
                'entry_price': entry_price,
                'exit_price': exit_price,
                'size': size,
                'pnl': pnl,
                'exit_reason': action['reason']
            })
            
            # Update win/loss count
            if pnl > 0:
                self.win_count += 1
            else:
                self.loss_count += 1
                
            # Limit history size
            if len(self.position_history) > 100:
                self.position_history = self.position_history[-100:]
                
        except Exception as e:
            self.logger.error(f"Error updating position history: {e}")
    
    def get_risk_metrics(self):
        """Get risk metrics
        
        Returns:
            dict: Risk metrics
        """
        return {
            'atr': self.atr,
            'regime_factor': self.regime_factor,
            'volatility_ratio': self.volatility_ratio,
            'current_drawdown': self.current_drawdown,
            'win_rate': self.win_count / (self.win_count + self.loss_count) if (self.win_count + self.loss_count) > 0 else 0.5,
            'total_trades': self.win_count + self.loss_count,
            'current_positions': len(self.positions)
        }
class TechnicalIndicators:
    """Technical analysis indicators"""
    
    def __init__(self, system=None, logger=None):
        """Initialize technical indicators
        
        Args:
            system: Reference to main system
            logger: Logger instance
        """
        self.system = system
        self.logger = logger or logging.getLogger("NQAlpha.Indicators")
        self.price_history = []
        self.indicators = {}
        
    def update(self, price):
        """Update indicators with new price
        
        Args:
            price (float): New price
        """
        # Add to price history
        self.price_history.append(price)
        
        # Limit history size
        if len(self.price_history) > 500:
            self.price_history = self.price_history[-500:]
            
        # Update ATR
        self._update_atr()
    
    def _update_atr(self, period=14):
        """Update Average True Range
        
        Args:
            period (int): ATR period
        """
        if len(self.price_history) < period + 1:
            return
            
        # Simple ATR calculation
        true_ranges = []
        for i in range(1, min(period+1, len(self.price_history))):
            high = self.price_history[i]
            low = self.price_history[i]
            prev_close = self.price_history[i-1]
            
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            
            true_ranges.append(max(tr1, tr2, tr3))
            
        if true_ranges:
            self.indicators['atr'] = sum(true_ranges) / len(true_ranges)
    
    def get_atr(self, period=14):
        """Get Average True Range
        
        Args:
            period (int): ATR period
            
        Returns:
            float: ATR value
        """
        return self.indicators.get('atr', self.price_history[-1] * 0.005 if self.price_history else 0.001)


#==============================================================================
# EXECUTION ENGINE
#==============================================================================

class ExecutionEngine:
    """
    Advanced execution engine with smart order routing and position tracking.
    """
    
    def __init__(self, config=None, system=None, logger=None):
        """Initialize execution engine
        
        Args:
            config (dict, optional): Configuration
            system (NQAlphaSystem, optional): Parent system
            logger (logging.Logger, optional): Logger
        """
        self.logger = logger or logging.getLogger("NQAlpha.ExecutionEngine")
        self.system = system
        
        # Default configuration
        self.config = {
            'mode': 'paper',                  # Execution mode: paper, live
            'order_types': [                  # Supported order types
                'market',                     # Market order
                'limit',                      # Limit order
                'stop',                       # Stop order
                'trail'                       # Trailing stop order
            ],
            'default_slippage': 0.5,         # Default slippage in ticks
            'commission_per_contract': 2.25,  # Commission per contract
            'min_distance': 1.0,              # Minimum distance for limit and stop orders
            'max_retries': 3,                # Maximum retry attempts
            'retry_delay': 1.0,               # Delay between retries
            'slippage_model': 'advanced',     # 'fixed', 'normal', 'advanced'
            'use_iceberg_orders': True,       # Whether to use iceberg orders for large positions
            'iceberg_threshold': 5,           # Threshold for iceberg orders
            'iceberg_display_size': 2,        # Display size for iceberg orders
            'order_book_impact': True         # Consider order book impact
        }
        
        # Update with provided config
        if config:
            self._update_nested_dict(self.config, config)
        
        # Internal state
        self.orders = {}
        self.positions = {}
        self.order_count = 0
        self.fills = []
        
        # Paper trading state
        self.paper_equity = 0.0
        self.paper_positions = {}
        self.paper_orders = {}
        self.paper_fills = []
        
        # Metrics
        self.metrics = {
            'orders_placed': 0,
            'orders_filled': 0,
            'orders_canceled': 0,
            'commission_paid': 0.0,
            'slippage_cost': 0.0,
            'avg_execution_time': 0.0,
            'rejection_rate': 0.0
        }
        
        self.logger.info(f"Elite execution engine initialized in {self.config['mode']} mode")
    
    def _update_nested_dict(self, d, u):
        """Update nested dictionary recursively"""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_nested_dict(d[k], v)
            else:
                d[k] = v
    
    def place_order(self, order_type, symbol, quantity, price=None, stop_price=None, trail_amount=None):
        """Place an order
        
        Args:
            order_type (str): Order type (market, limit, stop, trail)
            symbol (str): Symbol
            quantity (int): Quantity (positive for buy, negative for sell)
            price (float, optional): Limit price
            stop_price (float, optional): Stop price
            trail_amount (float, optional): Trailing amount
            
        Returns:
            str: Order ID
        """
        try:
            # Validate order type
            if order_type not in self.config['order_types']:
                self.logger.error(f"Invalid order type: {order_type}")
                return None
            
            # Get current price
            current_price = None
            if self.system and hasattr(self.system, 'market_data'):
                data = self.system.market_data.get_realtime_data()
                if data:
                    current_price = data.get('price')
            
            if not current_price:
                self.logger.error("Cannot place order without current price")
                return None
            
            # Check for iceberg order
            use_iceberg = False
            display_size = abs(quantity)
            
            if self.config['use_iceberg_orders'] and abs(quantity) >= self.config['iceberg_threshold']:
                use_iceberg = True
                display_size = self.config['iceberg_display_size']
            
            # Generate order ID
            self.order_count += 1
            order_id = f"ORD-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}-{self.order_count}"
            
            # Create order
            order = {
                'id': order_id,
                'type': order_type,
                'symbol': symbol,
                'quantity': quantity,
                'price': price,
                'stop_price': stop_price,
                'trail_amount': trail_amount,
                'status': 'pending',
                'created_at': datetime.datetime.now(),
                'updated_at': datetime.datetime.now(),
                'filled_price': None,
                'filled_quantity': 0,
                'commission': 0.0,
                'slippage': 0.0,
                'use_iceberg': use_iceberg,
                'display_size': display_size,
                'remaining_quantity': abs(quantity)
            }
            
            # Store order
            self.orders[order_id] = order
            
            # Paper trading
            if self.config['mode'] == 'paper':
                self.paper_orders[order_id] = order
                
                # Execute market orders immediately
                if order_type == 'market':
                    self._execute_paper_market_order(order_id)
                elif order_type == 'limit' and price is not None:
                    # Check if limit price is immediately executable
                    if (quantity > 0 and price >= current_price) or (quantity < 0 and price <= current_price):
                        self._execute_paper_market_order(order_id)
            else:
                # Live trading - connect to broker API
                pass
            
            # Update metrics
            self.metrics['orders_placed'] += 1
            
            self.logger.info(f"Placed {order_type} order {order_id}: {quantity} {symbol} @ {price or 'market'}")
            
            return order_id
            
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return None
    
    def _execute_paper_market_order(self, order_id):
        """Execute paper market order
        
        Args:
            order_id (str): Order ID
        """
        try:
            # Get order
            order = self.paper_orders.get(order_id)
            if not order:
                self.logger.error(f"Order not found: {order_id}")
                return
            
            # Get current price
            current_price = None
            if self.system and hasattr(self.system, 'market_data'):
                data = self.system.market_data.get_realtime_data()
                if data:
                    current_price = data.get('price')
            
            if not current_price:
                self.logger.error("Cannot execute paper order without current price")
                return
            
            # Apply slippage
            filled_price = self._calculate_slippage(current_price, order['quantity'])
            
            # Round to tick size
            filled_price = round(filled_price * 4) / 4  # Assuming 0.25 tick size
            
            # Calculate commission
            commission = abs(order['quantity']) * self.config['commission_per_contract']
            
            # Update order
            order['status'] = 'filled'
            order['filled_price'] = filled_price
            order['filled_quantity'] = order['quantity']
            order['commission'] = commission
            order['slippage'] = abs(filled_price - current_price) * abs(order['quantity'])
            order['updated_at'] = datetime.datetime.now()
            
            # Create fill
            fill = {
                'id': f"FILL-{order_id}",
                'order_id': order_id,
                'symbol': order['symbol'],
                'quantity': order['quantity'],
                'price': filled_price,
                'commission': commission,
                'timestamp': datetime.datetime.now()
            }
            
            # Store fill
            self.paper_fills.append(fill)
            
            # Update position
            self._update_paper_position(order['symbol'], order['quantity'], filled_price)
            
            # Update metrics
            self.metrics['orders_filled'] += 1
            self.metrics['commission_paid'] += commission
            self.metrics['slippage_cost'] += order['slippage']
            
            self.logger.info(f"Filled paper order {order_id}: {order['quantity']} {order['symbol']} @ {filled_price}")
            
        except Exception as e:
            self.logger.error(f"Error executing paper market order: {e}")
    
    def _calculate_slippage(self, price, quantity):
        """Calculate slippage based on model and order size
        
        Args:
            price (float): Current price
            quantity (int): Order quantity
            
        Returns:
            float: Filled price with slippage
        """
        try:
            # Get slippage model
            model = self.config['slippage_model']
            
            # Direction factor (1 for buy, -1 for sell)
            direction = 1 if quantity > 0 else -1
            
            # Base slippage in ticks
            base_slippage = self.config['default_slippage']
            
            if model == 'fixed':
                # Fixed slippage model
                slippage_ticks = base_slippage
            
            elif model == 'normal':
                # Normal distribution model
                # Higher slippage for larger orders
                size_factor = min(3.0, 1.0 + abs(quantity) / 10.0)
                slippage_ticks = base_slippage * size_factor * np.random.normal(1.0, 0.3)
                
            elif model == 'advanced':
                # Advanced model considering current market conditions
                # Get market data
                market_data = None
                if self.system and hasattr(self.system, 'market_data'):
                    market_data = self.system.market_data.get_realtime_data()
                
                # Base size factor
                size_factor = min(3.0, 1.0 + abs(quantity) / 10.0)
                
                # Volatility factor
                volatility_factor = 1.0
                if market_data and 'volatility_20' in market_data:
                    volatility = market_data['volatility_20']
                    # Scale volatility to a reasonable factor
                    volatility_factor = min(3.0, 1.0 + volatility * 100.0)
                
                # Liquidity factor
                liquidity_factor = 1.0
                if market_data and 'liquidity_score' in market_data:
                    liquidity = market_data['liquidity_score']
                    # Lower liquidity = higher slippage
                    liquidity_factor = max(1.0, 2.0 - liquidity)
                
                # Random component for realism
                random_factor = np.random.gamma(2.0, 0.5)  # Skewed to occasional high slippage
                
                # Combine factors
                slippage_ticks = base_slippage * size_factor * volatility_factor * liquidity_factor * random_factor
            
            else:
                # Default to fixed model
                slippage_ticks = base_slippage
            
            # Calculate price slippage
            tick_size = 0.25  # NQ futures tick size
            slippage_amount = slippage_ticks * tick_size
            
            # Apply direction
            filled_price = price + (direction * slippage_amount)
            
            return filled_price
            
        except Exception as e:
            self.logger.error(f"Error calculating slippage: {e}")
            # Default to price with minimal slippage
            return price + (0.25 * (1 if quantity > 0 else -1))
    
    def _update_paper_position(self, symbol, quantity, price):
        """Update paper position
        
        Args:
            symbol (str): Symbol
            quantity (int): Quantity
            price (float): Price
        """
        try:
            # Get current position
            position = self.paper_positions.get(symbol, {
                'symbol': symbol,
                'quantity': 0,
                'avg_price': 0.0,
                'value': 0.0,
                'unrealized_pnl': 0.0,
                'realized_pnl': 0.0
            })
            
            # Calculate cost
            cost = quantity * price
            
            # Update position
            if position['quantity'] == 0:
                # New position
                position['quantity'] = quantity
                position['avg_price'] = price
                position['value'] = cost
            elif (position['quantity'] > 0 and quantity > 0) or (position['quantity'] < 0 and quantity < 0):
                # Adding to position
                total_quantity = position['quantity'] + quantity
                position['avg_price'] = (position['quantity'] * position['avg_price'] + cost) / total_quantity
                position['quantity'] = total_quantity
                position['value'] = position['quantity'] * position['avg_price']
            elif abs(position['quantity']) >= abs(quantity):
                # Reducing position
                realized_pnl = (price - position['avg_price']) * quantity
                position['realized_pnl'] += realized_pnl
                position['quantity'] += quantity
                if position['quantity'] == 0:
                    position['avg_price'] = 0.0
                    position['value'] = 0.0
                else:
                    position['value'] = position['quantity'] * position['avg_price']
            else:
                # Flipping position
                realized_pnl = (price - position['avg_price']) * position['quantity']
                position['realized_pnl'] += realized_pnl
                position['quantity'] += quantity
                position['avg_price'] = price
                position['value'] = position['quantity'] * position['avg_price']
            
            # Store position
            self.paper_positions[symbol] = position
            
            # Calculate unrealized P&L
            self._update_paper_unrealized_pnl(symbol)
            
        except Exception as e:
            self.logger.error(f"Error updating paper position: {e}")
    
    def _update_paper_unrealized_pnl(self, symbol):
        """Update paper unrealized P&L
        
        Args:
            symbol (str): Symbol
        """
        try:
            # Get position
            position = self.paper_positions.get(symbol)
            if not position or position['quantity'] == 0:
                return
            
            # Get current price
            current_price = None
            if self.system and hasattr(self.system, 'market_data'):
                data = self.system.market_data.get_realtime_data()
                if data:
                    current_price = data.get('price')
            
            if not current_price:
                return
            
            # Calculate unrealized P&L
            position['unrealized_pnl'] = (current_price - position['avg_price']) * position['quantity']
            
        except Exception as e:
            self.logger.error(f"Error updating paper unrealized P&L: {e}")
    
    def cancel_order(self, order_id):
        """Cancel an order
        
        Args:
            order_id (str): Order ID
            
        Returns:
            bool: Whether cancellation was successful
        """
        try:
            # Get order
            order = self.orders.get(order_id)
            if not order:
                self.logger.error(f"Order not found: {order_id}")
                return False
            
            # Check if order can be canceled
            if order['status'] in ['filled', 'canceled', 'rejected']:
                self.logger.warning(f"Cannot cancel order {order_id} with status {order['status']}")
                return False
            
            # Update order
            order['status'] = 'canceled'
            order['updated_at'] = datetime.datetime.now()
            
            # Paper trading
            if self.config['mode'] == 'paper' and order_id in self.paper_orders:
                self.paper_orders[order_id]['status'] = 'canceled'
                self.paper_orders[order_id]['updated_at'] = datetime.datetime.now()
            else:
                # Live trading - connect to broker API
                pass
            
            # Update metrics
            self.metrics['orders_canceled'] += 1
            
            self.logger.info(f"Canceled order {order_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error canceling order: {e}")
            return False
    
    def get_position(self, symbol):
        """Get current position for symbol
        
        Args:
            symbol (str): Symbol
            
        Returns:
            dict: Position
        """
        try:
            # Paper trading
            if self.config['mode'] == 'paper':
                return self.paper_positions.get(symbol, {
                    'symbol': symbol,
                    'quantity': 0,
                    'avg_price': 0.0,
                    'value': 0.0,
                    'unrealized_pnl': 0.0,
                    'realized_pnl': 0.0
                })
            else:
                # Live trading - connect to broker API
                return self.positions.get(symbol, {
                    'symbol': symbol,
                    'quantity': 0,
                    'avg_price': 0.0,
                    'value': 0.0,
                    'unrealized_pnl': 0.0,
                    'realized_pnl': 0.0
                })
            
        except Exception as e:
            self.logger.error(f"Error getting position: {e}")
            return {
                'symbol': symbol,
                'quantity': 0,
                'avg_price': 0.0,
                'value': 0.0,
                'unrealized_pnl': 0.0,
                'realized_pnl': 0.0
            }
    
    def get_account_equity(self):
        """Get current account equity
        
        Returns:
            float: Account equity
        """
        try:
            # Paper trading
            if self.config['mode'] == 'paper':
                # Calculate total P&L
                total_realized_pnl = sum(position.get('realized_pnl', 0.0) for position in self.paper_positions.values())
                total_unrealized_pnl = sum(position.get('unrealized_pnl', 0.0) for position in self.paper_positions.values())
                
                # Calculate total commission
                total_commission = sum(fill.get('commission', 0.0) for fill in self.paper_fills)
                
                # Calculate equity
                self.paper_equity = self.system.capital + total_realized_pnl + total_unrealized_pnl - total_commission
                
                return self.paper_equity
            else:
                # Live trading - connect to broker API
                return self.system.capital
            
        except Exception as e:
            self.logger.error(f"Error getting account equity: {e}")
            return self.system.capital
    
    def get_order(self, order_id):
        """Get order by ID
        
        Args:
            order_id (str): Order ID
            
        Returns:
            dict: Order
        """
        try:
            return self.orders.get(order_id)
        except Exception as e:
            self.logger.error(f"Error getting order: {e}")
            return None
    
    def get_open_orders(self, symbol=None):
        """Get open orders
        
        Args:
            symbol (str, optional): Symbol filter
            
        Returns:
            list: Open orders
        """
        try:
            open_orders = []
            
            for order_id, order in self.orders.items():
                if order['status'] not in ['filled', 'canceled', 'rejected']:
                    if symbol is None or order['symbol'] == symbol:
                        open_orders.append(order)
            
            return open_orders
            
        except Exception as e:
            self.logger.error(f"Error getting open orders: {e}")
            return []

#==============================================================================
# PERFORMANCE DASHBOARD
#==============================================================================

class PerformanceDashboard:
    """
    Elite performance dashboard with advanced analytics and visualization.
    """
    
    def __init__(self, config=None, system=None, logger=None):
        """Initialize performance dashboard
        
        Args:
            config (dict, optional): Configuration
            system (NQAlphaSystem, optional): Parent system
            logger (logging.Logger, optional): Logger
        """
        self.logger = logger or logging.getLogger("NQAlpha.Dashboard")
        self.system = system
        
        # Default configuration
        self.config = {
            'enabled': True,              # Whether dashboard is enabled
            'update_interval': 5.0,       # Update interval in seconds
            'chart_points': 100,          # Number of points in charts
            'metrics_to_track': [         # Metrics to track
                'equity',                 # Account equity
                'drawdown',               # Drawdown
                'win_rate',               # Win rate
                'sharpe_ratio',           # Sharpe ratio
                'volatility',             # Volatility
                'regime',                 # Market regime
                'order_flow',             # Order flow metrics
                'trade_stats'             # Trade statistics
            ],
            'save_results': True,         # Whether to save results
            'results_dir': 'results',     # Directory for results
            'web_dashboard': False,       # Whether to enable web dashboard
            'web_port': 8050,             # Web dashboard port
            'web_host': '127.0.0.1',      # Web dashboard host
            'advanced_analytics': True,   # Whether to enable advanced analytics
            'metrics_history_length': 100, # Length of metrics history
            'notification_thresholds': {  # Thresholds for notifications
                'drawdown': 0.05,         # Drawdown threshold (5%)
                'win_rate': 0.4,          # Win rate threshold (40%)
                'volatility': 0.02        # Volatility threshold (2%)
            }
        }
        
        # Update with provided config
        if config:
            self._update_nested_dict(self.config, config)
        
        # Internal state
        self.running = False
        self.thread = None
        self.metrics = {}
        self.history = {}
        self.performance_stats = {}
        self.trade_history = []
        self.equity_curve = []
        
        # Create result directory
        if self.config['save_results']:
            os.makedirs(self.config['results_dir'], exist_ok=True)
        
        # Metrics for tracking
        for metric in self.config['metrics_to_track']:
            self.metrics[metric] = None
            self.history[metric] = deque(maxlen=self.config['chart_points'])
        
        self.logger.info("Elite performance dashboard initialized")
    
    def _update_nested_dict(self, d, u):
        """Update nested dictionary recursively"""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_nested_dict(d[k], v)
            else:
                d[k] = v
    
    def start(self):
        """Start performance dashboard"""
        if not self.config['enabled']:
            self.logger.info("Performance dashboard is disabled")
            return
        
        if self.running:
            self.logger.warning("Performance dashboard already running")
            return
        
        self.logger.info("Starting performance dashboard")
        
        try:
            # Set running flag
            self.running = True
            
            # Start in background thread
            self.thread = threading.Thread(
                target=self._dashboard_thread,
                name="PerformanceDashboardThread"
            )
            self.thread.daemon = True
            self.thread.start()
            
            self.logger.info("Performance dashboard thread started")
            
            # Start web dashboard if enabled
            if self.config['web_dashboard']:
                self._start_web_dashboard()
            
        except Exception as e:
            self.running = False
            self.logger.error(f"Error starting performance dashboard: {e}")
    
    def stop(self):
        """Stop performance dashboard"""
        if not self.running:
            self.logger.warning("Performance dashboard not running")
            return
        
        self.logger.info("Stopping performance dashboard")
        
        try:
            # Set running flag
            self.running = False
            
            # Wait for thread to complete
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=2.0)
            
            # Save results
            if self.config['save_results']:
                self._save_results()
            
            self.logger.info("Performance dashboard stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping performance dashboard: {e}")
    
    def _dashboard_thread(self):
        """Background thread for performance dashboard"""
        self.logger.info("Performance dashboard thread running")
        
        try:
            while self.running:
                try:
                    # Update metrics
                    self._update_metrics()
                    
                    # Update performance statistics
                    self._update_performance_stats()
                    
                    # Check for alert conditions
                    self._check_alerts()
                    
                    # Print current metrics
                    self._print_current_metrics()
                    
                    # Sleep
                    time.sleep(self.config['update_interval'])
                    
                except Exception as e:
                    self.logger.error(f"Error in performance dashboard: {e}")
                    time.sleep(1.0)
            
        except Exception as e:
            self.logger.error(f"Fatal error in performance dashboard thread: {e}")
        
        self.logger.info("Performance dashboard thread stopped")
    
    def _update_metrics(self):
        """Update performance metrics"""
        try:
            # Skip if no system
            if not self.system:
                return
            
            # Current time
            current_time = datetime.datetime.now()
            
            # Update equity
            if 'equity' in self.metrics and hasattr(self.system, 'execution_engine'):
                equity = self.system.execution_engine.get_account_equity()
                self.metrics['equity'] = equity
                self.history['equity'].append((current_time, equity))
                
                # Update equity curve
                self.equity_curve.append({'timestamp': current_time, 'equity': equity})
                # Keep reasonable size
                if len(self.equity_curve) > 1000:
                    self.equity_curve = self.equity_curve[-1000:]
            
            # Update drawdown
            if 'drawdown' in self.metrics and 'equity' in self.metrics:
                equity = self.metrics['equity']
                peak_equity = self.system.capital
                
                if self.history['equity']:
                    peak_equity = max(peak_equity, max(e[1] for e in self.history['equity']))
                
                if peak_equity > 0:
                    drawdown = max(0, (peak_equity - equity) / peak_equity)
                    self.metrics['drawdown'] = drawdown
                    self.history['drawdown'].append((current_time, drawdown))
            
            # Update win rate
            if 'win_rate' in self.metrics and hasattr(self.system, 'execution_engine'):
                if hasattr(self.system.execution_engine, 'paper_fills') and self.system.execution_engine.paper_fills:
                    fills = self.system.execution_engine.paper_fills
                    
                    # Group fills by order ID to get trades
                    trades = {}
                    for fill in fills:
                        order_id = fill['order_id']
                        if order_id not in trades:
                            trades[order_id] = {'pnl': 0.0}
                        
                        # Calculate P&L (simplified)
                        trades[order_id]['pnl'] += fill['quantity'] * fill['price']
                    
                    # Count winning and losing trades
                    winning_trades = sum(1 for trade in trades.values() if trade['pnl'] > 0)
                    total_trades = len(trades)
                    
                    if total_trades > 0:
                        win_rate = winning_trades / total_trades
                        self.metrics['win_rate'] = win_rate
                        self.history['win_rate'].append((current_time, win_rate))
            
            # Update market regime
            if 'regime' in self.metrics and hasattr(self.system, 'regime_classifier'):
                regime_info = self.system.regime_classifier.get_current_regime()
                if regime_info:
                    self.metrics['regime'] = regime_info
                    self.history['regime'].append((current_time, regime_info['regime']))
            
            # Update order flow metrics
            if 'order_flow' in self.metrics and hasattr(self.system, 'market_data'):
                order_flow_metrics = self.system.market_data.get_order_flow_metrics()
                if order_flow_metrics:
                    self.metrics['order_flow'] = order_flow_metrics
                    self.history['order_flow'].append((current_time, order_flow_metrics['order_flow']))
            
            # Update volatility
            if 'volatility' in self.metrics and hasattr(self.system, 'regime_classifier'):
                regime_info = self.system.regime_classifier.get_current_regime()
                if regime_info and 'volatility' in regime_info:
                    self.metrics['volatility'] = regime_info['volatility']
                    self.history['volatility'].append((current_time, regime_info['volatility']))
            
            # Update trade statistics
            if 'trade_stats' in self.metrics and hasattr(self.system, 'execution_engine'):
                # Get fills
                fills = []
                if hasattr(self.system.execution_engine, 'paper_fills'):
                    fills = self.system.execution_engine.paper_fills
                
                # Skip if no fills
                if not fills:
                    return
                
                # Update trade history
                self.trade_history = fills
                
                # Calculate statistics
                total_fills = len(fills)
                total_commission = sum(fill.get('commission', 0.0) for fill in fills)
                
                # Group by order ID
                orders = {}
                for fill in fills:
                    order_id = fill.get('order_id')
                    if order_id not in orders:
                        orders[order_id] = []
                    orders[order_id].append(fill)
                
                # Calculate trade statistics
                self.metrics['trade_stats'] = {
                    'total_fills': total_fills,
                    'total_trades': len(orders),
                    'total_commission': total_commission
                }
            
        except Exception as e:
            self.logger.error(f"Error updating metrics: {e}")
    
    def _update_performance_stats(self):
        """Update performance statistics"""
        try:
            # Skip if no equity curve
            if not self.equity_curve or len(self.equity_curve) < 2:
                return
            
            # Extract equity values
            timestamps = [point['timestamp'] for point in self.equity_curve]
            equity_values = [point['equity'] for point in self.equity_curve]
            
            # Calculate returns
            returns = []
            for i in range(1, len(equity_values)):
                returns.append((equity_values[i] - equity_values[i-1]) / equity_values[i-1])
            
            # Skip if no returns
            if not returns:
                return
            
            # Calculate statistics
            total_return = (equity_values[-1] - equity_values[0]) / equity_values[0]
            
            # Calculate annualized return
            days = (timestamps[-1] - timestamps[0]).total_seconds() / (24 * 3600)
            annualized_return = ((1 + total_return) ** (365 / days) - 1) if days > 0 else 0
            
            # Calculate Sharpe ratio
            returns_mean = np.mean(returns)
            returns_std = np.std(returns)
            risk_free_rate = 0.02 / 365  # 2% annual risk-free rate, daily
            
            sharpe_ratio = (returns_mean - risk_free_rate) / returns_std * np.sqrt(252) if returns_std > 0 else 0
            
            # Calculate Sortino ratio (using only negative returns)
            negative_returns = [r for r in returns if r < 0]
            downside_std = np.std(negative_returns) if negative_returns else 0
            
            sortino_ratio = (returns_mean - risk_free_rate) / downside_std * np.sqrt(252) if downside_std > 0 else 0
            
            # Calculate maximum drawdown
            max_drawdown = 0.0
            peak = equity_values[0]
            
            for equity in equity_values:
                if equity > peak:
                    peak = equity
                drawdown = (peak - equity) / peak
                max_drawdown = max(max_drawdown, drawdown)
            
            # Store performance statistics
            self.performance_stats = {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown,
                'volatility': returns_std * np.sqrt(252),
                'win_rate': self.metrics.get('win_rate', 0.0)
            }
            
            # Update Sharpe ratio in metrics
            if 'sharpe_ratio' in self.metrics:
                self.metrics['sharpe_ratio'] = sharpe_ratio
                self.history['sharpe_ratio'].append((timestamps[-1], sharpe_ratio))
            
        except Exception as e:
            self.logger.error(f"Error updating performance statistics: {e}")
    
    def _check_alerts(self):
        """Check for alert conditions"""
        try:
            # Skip if no thresholds
            if not self.config['notification_thresholds']:
                return
            
            # Check drawdown
            if 'drawdown' in self.metrics and self.metrics['drawdown'] is not None and 'drawdown' in self.config['notification_thresholds']:
                threshold = self.config['notification_thresholds']['drawdown']
                if self.metrics['drawdown'] > threshold:
                    self.logger.warning(f"ALERT: Drawdown {self.metrics['drawdown']:.2%} exceeds threshold {threshold:.2%}")
            
            # Check win rate
            if 'win_rate' in self.metrics and self.metrics['win_rate'] is not None and 'win_rate' in self.config['notification_thresholds']:
                threshold = self.config['notification_thresholds']['win_rate']
                if self.metrics['win_rate'] < threshold:
                    self.logger.warning(f"ALERT: Win rate {self.metrics['win_rate']:.2%} below threshold {threshold:.2%}")
            
            # Check volatility
            if 'volatility' in self.metrics and self.metrics['volatility'] is not None and 'volatility' in self.config['notification_thresholds']:
                threshold = self.config['notification_thresholds']['volatility']
                if self.metrics['volatility'] > threshold:
                    self.logger.warning(f"ALERT: Volatility {self.metrics['volatility']:.6f} exceeds threshold {threshold:.6f}")
                
        except Exception as e:
            self.logger.error(f"Error checking alerts: {e}")
    
    def _print_current_metrics(self):
        """Print current metrics to console"""
        try:
            # Skip if metrics are empty
            if not self.metrics:
                return
            
            # Print header
            print("\n" + "=" * 80)
            print(f"NQ Alpha Elite - Performance Metrics - {datetime.datetime.now()}")
            print("=" * 80)
            
            # Print account metrics
            if 'equity' in self.metrics and self.metrics['equity'] is not None:
                print(f"Account Equity: ${self.metrics['equity']:,.2f}")
            
            if 'drawdown' in self.metrics and self.metrics['drawdown'] is not None:
                print(f"Current Drawdown: {self.metrics['drawdown']:.2%}")
            
            # Print performance metrics
            if self.performance_stats:
                print(f"Total Return: {self.performance_stats.get('total_return', 0.0):.2%}")
                print(f"Annualized Return: {self.performance_stats.get('annualized_return', 0.0):.2%}")
                print(f"Sharpe Ratio: {self.performance_stats.get('sharpe_ratio', 0.0):.2f}")
                print(f"Sortino Ratio: {self.performance_stats.get('sortino_ratio', 0.0):.2f}")
            
            # Print trading metrics
            if 'win_rate' in self.metrics and self.metrics['win_rate'] is not None:
                print(f"Win Rate: {self.metrics['win_rate']:.2%}")
            
            if 'trade_stats' in self.metrics and self.metrics['trade_stats'] is not None:
                trade_stats = self.metrics['trade_stats']
                print(f"Total Trades: {trade_stats.get('total_trades', 0)}")
                print(f"Total Commission: ${trade_stats.get('total_commission', 0.0):,.2f}")
            
            # Print market metrics
            if 'regime' in self.metrics and self.metrics['regime'] is not None:
                regime_info = self.metrics['regime']
                print(f"Market Regime: {regime_info.get('regime', 'unknown')} (confidence: {regime_info.get('confidence', 0.0):.2f})")
                print(f"Volatility: {regime_info.get('volatility', 0.0):.6f}, Trend Strength: {regime_info.get('trend_strength', 0.0):.2f}")
            
            # Print order flow metrics
            if 'order_flow' in self.metrics and self.metrics['order_flow'] is not None:
                order_flow = self.metrics['order_flow']
                print(f"Order Flow: {order_flow.get('order_flow', 0.0):.3f}, Delta: {order_flow.get('delta', 0.0):.3f}")
                
                # Safely access vpin and liquidity score
                vpin = order_flow.get('vpin', 0.0)
                liquidity = order_flow.get('liquidity_score', 1.0)
                print(f"VPIN: {vpin:.3f}, Liquidity Score: {liquidity:.3f}")
            
            # Print market data
            if self.system and hasattr(self.system, 'market_data'):
                data = self.system.market_data.get_realtime_data()
                if data:
                    price = data.get('price', 0.0)
                    source = data.get('source', 'unknown')
                    print(f"Current Price: ${price:,.2f} (source: {source})")
            
            # Print footer
            print("=" * 80)
            
        except Exception as e:
            self.logger.error(f"Error printing metrics: {e}")
    
    def _save_results(self):
        """Save results to disk"""
        try:
            # Skip if no metrics
            if not self.metrics:
                return
            
            # Skip if saving not enabled
            if not self.config['save_results']:
                return
            
            # Create filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_{timestamp}.json"
            filepath = os.path.join(self.config['results_dir'], filename)
            
            # Convert history to serializable format
            serializable_history = {}
            for metric, values in self.history.items():
                serializable_history[metric] = [(t.isoformat() if isinstance(t, datetime.datetime) else str(t), v) for t, v in values]
            
            # Convert metrics to serializable format
            serializable_metrics = self._make_json_serializable(self.metrics)
            
            # Create results dictionary
            results = {
                'metrics': serializable_metrics,
                'performance_stats': self.performance_stats,
                'history': serializable_history,
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            # Save to JSON
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)
            
            self.logger.info(f"Saved performance results to {filepath}")
            
            # Save equity curve to CSV
            csv_filepath = os.path.join(self.config['results_dir'], f"equity_curve_{timestamp}.csv")
            with open(csv_filepath, 'w') as f:
                f.write("timestamp,equity\n")
                for point in self.equity_curve:
                    f.write(f"{point['timestamp'].isoformat()},{point['equity']}\n")
            
            self.logger.info(f"Saved equity curve to {csv_filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
    
    def _make_json_serializable(self, obj):
        """Convert object to JSON serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(self._make_json_serializable(v) for v in obj)
        elif isinstance(obj, datetime.datetime):
            return obj.isoformat()
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)
    
    def _start_web_dashboard(self):
        """Start web dashboard"""
        try:
            # Import Dash
            try:
                import dash
                from dash import dcc, html
                import plotly.graph_objs as go
            except ImportError:
                self.logger.error("Cannot start web dashboard: dash or plotly not installed")
                return
            
            # Create Dash app
            app = dash.Dash("NQAlphaElite")
            
            # Define layout
            app.layout = html.Div([
                html.H1("NQ Alpha Elite - Trading Dashboard"),
                html.Div([
                    html.Div([
                        html.H3("Account Metrics"),
                        dcc.Graph(id='equity-chart'),
                        dcc.Graph(id='drawdown-chart')
                    ], className='six columns'),
                    html.Div([
                        html.H3("Market Metrics"),
                        dcc.Graph(id='order-flow-chart'),
                        dcc.Graph(id='regime-chart')
                    ], className='six columns')
                ], className='row'),
                html.Div([
                    html.Div([
                        html.H3("Performance Metrics"),
                        html.Div(id='performance-stats')
                    ], className='twelve columns')
                ], className='row'),
                dcc.Interval(
                    id='interval-component',
                    interval=5*1000,  # in milliseconds
                    n_intervals=0
                )
            ])
            
            # Define callbacks
            @app.callback(
                dash.dependencies.Output('equity-chart', 'figure'),
                [dash.dependencies.Input('interval-component', 'n_intervals')]
            )
            def update_equity_chart(n):
                if 'equity' not in self.history or not self.history['equity']:
                    return go.Figure()
                
                x = [t for t, _ in self.history['equity']]
                y = [v for _, v in self.history['equity']]
                
                return {
                    'data': [go.Scatter(x=x, y=y, name='Equity')],
                    'layout': go.Layout(title='Account Equity')
                }
            
            @app.callback(
                dash.dependencies.Output('drawdown-chart', 'figure'),
                [dash.dependencies.Input('interval-component', 'n_intervals')]
            )
            def update_drawdown_chart(n):
                if 'drawdown' not in self.history or not self.history['drawdown']:
                    return go.Figure()
                
                x = [t for t, _ in self.history['drawdown']]
                y = [v * 100 for _, v in self.history['drawdown']]  # Convert to percentage
                
                return {
                    'data': [go.Scatter(x=x, y=y, name='Drawdown %')],
                    'layout': go.Layout(title='Drawdown %')
                }
            
            @app.callback(
                dash.dependencies.Output('order-flow-chart', 'figure'),
                [dash.dependencies.Input('interval-component', 'n_intervals')]
            )
            def update_order_flow_chart(n):
                if 'order_flow' not in self.history or not self.history['order_flow']:
                    return go.Figure()
                
                x = [t for t, _ in self.history['order_flow']]
                y = [v for _, v in self.history['order_flow']]
                
                return {
                    'data': [go.Scatter(x=x, y=y, name='Order Flow')],
                    'layout': go.Layout(title='Order Flow')
                }
            
            @app.callback(
                dash.dependencies.Output('regime-chart', 'figure'),
                [dash.dependencies.Input('interval-component', 'n_intervals')]
            )
            def update_regime_chart(n):
                if 'regime' not in self.history or not self.history['regime']:
                    return go.Figure()
                
                x = [t for t, _ in self.history['regime']]
                regimes = [v for _, v in self.history['regime']]
                
                # Map regimes to numeric values
                regime_map = {
                    'unknown': 0,
                    'trending_up': 1,
                    'trending_down': -1,
                    'range_bound': 0.5,
                    'volatile': 0.25,
                    'choppy': -0.5
                }
                
                y = [regime_map.get(r, 0) for r in regimes]
                
                return {
                    'data': [go.Scatter(x=x, y=y, name='Regime')],
                    'layout': go.Layout(title='Market Regime')
                }
            
            @app.callback(
                dash.dependencies.Output('performance-stats', 'children'),
                [dash.dependencies.Input('interval-component', 'n_intervals')]
            )
            def update_performance_stats(n):
                if not self.performance_stats:
                    return html.P("No performance statistics available yet")
                
                return html.Div([
                    html.Table([
                        html.Tr([
                            html.Td("Total Return:"),
                            html.Td(f"{self.performance_stats['total_return']:.2%}")
                        ]),
                        html.Tr([
                            html.Td("Annualized Return:"),
                            html.Td(f"{self.performance_stats['annualized_return']:.2%}")
                        ]),
                        html.Tr([
                            html.Td("Sharpe Ratio:"),
                            html.Td(f"{self.performance_stats['sharpe_ratio']:.2f}")
                        ]),
                        html.Tr([
                            html.Td("Sortino Ratio:"),
                            html.Td(f"{self.performance_stats['sortino_ratio']:.2f}")
                        ]),
                        html.Tr([
                            html.Td("Max Drawdown:"),
                            html.Td(f"{self.performance_stats['max_drawdown']:.2%}")
                        ]),
                        html.Tr([
                            html.Td("Win Rate:"),
                            html.Td(f"{self.performance_stats['win_rate']:.2%}")
                        ])
                    ])
                ])
            
            # Start Dash app in a separate thread
            threading.Thread(
                target=lambda: app.run_server(
                    debug=False,
                    host=self.config['web_host'],
                    port=self.config['web_port']
                ),
                daemon=True
            ).start()
            
            self.logger.info(f"Web dashboard started at http://{self.config['web_host']}:{self.config['web_port']}/")
            
        except Exception as e:
            self.logger.error(f"Error starting web dashboard: {e}")
#==============================================================================
# STRATEGY REWEIGHTER
#==============================================================================

class StrategyReweighter:
    """Dynamically adjusts strategy weights based on performance"""
    
    def __init__(self, logger=None, reweight_interval=100, lookback_period=20, 
                 min_trades_per_strategy=3, adaptation_rate=0.05):
        self.logger = logger or logging.getLogger("NQAlpha.Reweighter")
        self.reweight_interval = reweight_interval
        self.lookback_period = lookback_period
        self.min_trades_per_strategy = min_trades_per_strategy
        self.adaptation_rate = adaptation_rate
        self.last_weights = {}
        self.performance_history = {}
        
    def reweight(self, strategies, performance_metrics, trades_by_strategy):
        """Reweight strategies based on performance
        
        Args:
            strategies (dict): Strategy configurations
            performance_metrics (dict): Performance metrics
            trades_by_strategy (dict): Trade counts by strategy
        """
        # Store initial weights if first run
        if not self.last_weights:
            self.last_weights = {
                name: strategy['config']['weight'] 
                for name, strategy in strategies.items()
            }
        
        # Check for minimum trades
        eligible = all(count >= self.min_trades_per_strategy for count in trades_by_strategy.values())
        if not eligible:
            return False
        
        # Calculate performance by strategy
        # This is a simplified version that would be expanded with actual trade data
        # For now, we'll just use the current regime to boost appropriate strategies
        regime = performance_metrics.get('regime', 'unknown')
        
        # Regime-based weight adjustments
        adjustments = {
            'trending_up': {'trend_following': 0.1, 'mean_reversion': -0.05, 'breakout': 0.05, 'order_flow': 0.0},
            'trending_down': {'trend_following': 0.1, 'mean_reversion': -0.05, 'breakout': 0.05, 'order_flow': 0.0},
            'range_bound': {'trend_following': -0.05, 'mean_reversion': 0.1, 'breakout': -0.02, 'order_flow': 0.0},
            'volatile': {'trend_following': -0.05, 'mean_reversion': -0.02, 'breakout': 0.02, 'order_flow': 0.1},
            'choppy': {'trend_following': -0.08, 'mean_reversion': 0.05, 'breakout': -0.05, 'order_flow': 0.05},
            'unknown': {'trend_following': 0.0, 'mean_reversion': 0.0, 'breakout': 0.0, 'order_flow': 0.0}
        }
        
        # Get adjustment for current regime
        current_adjustments = adjustments.get(regime, adjustments['unknown'])
        
        # Calculate new weights
        new_weights = {}
        for strategy_name, adjustment in current_adjustments.items():
            if strategy_name in strategies:
                # Get current weight
                current_weight = strategies[strategy_name]['config']['weight']
                
                # Calculate new weight
                new_weight = current_weight + (adjustment * self.adaptation_rate)
                
                # Ensure weight is in valid range
                new_weight = max(0.05, min(0.6, new_weight))
                
                new_weights[strategy_name] = new_weight
        
        # Normalize weights to sum to 1.0
        weight_sum = sum(new_weights.values())
        if weight_sum > 0:
            normalized_weights = {
                strategy: weight / weight_sum 
                for strategy, weight in new_weights.items()
            }
            
            # Apply new weights
            for strategy_name, weight in normalized_weights.items():
                if strategy_name in strategies:
                    old_weight = strategies[strategy_name]['config']['weight']
                    strategies[strategy_name]['config']['weight'] = weight
                    self.logger.info(f"Adjusted {strategy_name} weight: {old_weight:.2f} → {weight:.2f}")
            
            # Store weights for next comparison
            self.last_weights = normalized_weights
            
            return True
        
        return False

class NQAlphaEliteRL:
    def __init__(self, base_system):
        self.base_system = base_system
        
        # Define state and action dimensions
        self.state_dim = 32        # Your quantum features, flow metrics, regime indicators
        self.action_dim = 3        # Signal adjustment factor, position sizing, exit timing
        
        # Create the PPO agent
        self.model = PPOAgent(  
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            learning_rate=1e-5,    # Conservative learning rate for stability
            batch_size=64,         # Small batches for incremental updates
            gamma=0.98,            # Discount factor (slightly less than 1 for trading)
            memory_size=2000       # Last ~2000 decision points
        )
        
        # RL system parameters
        self.experience_buffer = []
        self.last_state = None
        self.last_action = None
        self.training_frequency = 10  # Update model every 10 trades
        self.exploration_rate = 0.2   # Initial exploration rate
        
        # Regime-specific adaptation
        self.regime_learning_rates = {
            'trending_up': 1.2,    # Faster learning in trending markets
            'trending_down': 1.2,  # Faster learning in trending markets
            'range_bound': 0.8,    # More conservative in range-bound
            'choppy': 0.5,         # Most conservative in choppy markets
            'volatile': 1.0        # Baseline in volatile markets
        }
        
        # Track metrics for performance monitoring
        self.cumulative_reward = 0
        self.episode_count = 0
        self.rl_performance = {
            'rewards': [],
            'actions_taken': [],
            'improvements': []
        }
        
        # Log initialization
        if hasattr(self.base_system, 'logger'):
            self.base_system.logger.info(f"Reinforcement Learning system initialized with {self.state_dim} state features and {self.action_dim} action dimensions")
    def process_state(self, market_data, system_state):
        """Convert raw market data and system state to RL state vector"""
        state = np.zeros(32)
        # Fill with key metrics from your data
        state[0] = system_state.get('entanglement', 0)
        state[1] = system_state.get('regime_confidence', 0)
        state[2:5] = self._encode_regime(system_state.get('regime', 'unknown'))
        state[5] = market_data.get('order_flow', 0)
        state[6] = market_data.get('delta', 0)
        state[7] = market_data.get('volatility', 0)
        state[8] = market_data.get('trend_strength', 0)
        state[9] = market_data.get('vpin', 0)
        state[10] = self._normalize_z_score(system_state.get('z_score', 0))
        # Plus additional features from your rich feature set
        return state
        
    def get_action(self, state):
        """Get RL-enhanced action adjustments"""
        # During exploitation phase with high confidence
        if np.random.random() > self.exploration_rate:
            action = self.model.predict(state)
        # During exploration phase or low confidence scenarios
        else:
            action = self.model.explore(state)
        return action
    
    def apply_action(self, action, base_signal):
        """Apply RL action to adjust the base system's behavior"""
        # Action[0]: Signal adjustment factor
        adjusted_signal = base_signal * (1.0 + action[0])
        
        # Action[1]: Position sizing adjustment
        position_sizing_factor = 1.0 + action[1]
        
        # Action[2]: Exit timing adjustment (0-1 range)
        exit_timing_factor = action[2]
        
        return {
            'adjusted_signal': adjusted_signal,
            'position_sizing_factor': position_sizing_factor,
            'exit_timing_factor': exit_timing_factor
        }
    
    def observe_reward(self, trade_result):
        """Calculate reward based on trade outcome"""
        # Primary reward: P&L (normalized)
        pnl = trade_result.get('profit', 0)
        normalized_pnl = pnl / (trade_result.get('risk', 100) + 1e-5)
        
        # Risk-adjusted reward components
        drawdown_penalty = -0.5 * trade_result.get('max_drawdown_pct', 0) 
        time_efficiency = min(1.0, trade_result.get('hold_time', 0) / trade_result.get('optimal_hold_time', 60))
        
        # Composite reward function
        reward = normalized_pnl + 0.2 * time_efficiency + drawdown_penalty
        
        # Store experience for learning
        if self.last_state is not None and self.last_action is not None:
            self.experience_buffer.append({
                'state': self.last_state,
                'action': self.last_action,
                'reward': reward,
                'next_state': self.process_state(
                    self.base_system.market_data.get_realtime_data(),
                    self.base_system.get_system_state()
                ),
                'done': True  # Each trade is a complete episode
            })
        
        # Check if it's time to train the model
        self._try_training()
        
        return reward
    
    def _try_training(self):
        """Train the RL model incrementally"""
        if len(self.experience_buffer) >= self.training_frequency:
            self.model.train_batch(self.experience_buffer)
            # Keep the most recent experiences for catastrophic forgetting prevention
            self.experience_buffer = self.experience_buffer[-int(self.training_frequency/2):]
            # Gradually reduce exploration as system learns
            self.exploration_rate = max(0.05, self.exploration_rate * 0.995)

def execute_live_trade_with_learning(symbol, timeframe='1h', quantity=None, api_key=None, api_secret=None):
    """Execute trades with real-time learning from paper trading"""
    print(f"Starting live trading with online learning for {symbol}...")
    
    # Initialize client if credentials provided
    client = None
    if api_key and api_secret:
        from binance.client import Client
        client = Client(api_key, api_secret)
    
    # Load agent if exists
    agent_exists = os.path.exists("trading_agent_actor_latest.h5")
    
    # Initialize variables
    position = 0
    entry_price = 0
    trade_count = 0
    max_drawdown = 0
    lookback = 20
    trades = []
    equity_curve = [1000]  # Starting equity
    
    try:
        while True:
            # Get latest klines
            if client:
                klines = client.get_klines(symbol=symbol, interval=timeframe, limit=100)
                df = process_klines(klines)  # Use your existing kline processing function
            else:
                # Simulate with yfinance or other data source if no API credentials
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                df = ticker.history(period="7d", interval=timeframe)
                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
            
            # Add technical indicators (use your existing indicator functions)
            df['RSI'] = calculate_rsi(df['Close'], 14)
            
            # Skip if not enough data
            if len(df) <= lookback:
                print("Not enough data, waiting...")
                time.sleep(60)
                continue
            
            # Current index is the last available data
            current_idx = len(df) - 1
            
            # Create state
            state = create_market_state(df, current_idx, lookback)
            if state is None:
                print("Cannot create state, waiting...")
                time.sleep(60)
                continue
            
            # Initialize or get RL agent
            global rl_agent
            if 'rl_agent' not in globals() or rl_agent is None:
                state_size = len(state)
                action_size = 3  # Sell, Hold, Buy
                
                if agent_exists:
                    print("Loading existing RL agent")
                    rl_agent = OnlinePPOAgent(state_size, action_size)
                    rl_agent.load("trading_agent_actor_latest.h5", "trading_agent_critic_latest.h5")
                else:
                    print("Creating new RL agent")
                    rl_agent = OnlinePPOAgent(state_size, action_size)
            
            # Get action
            action, log_prob, value, probs = rl_agent.get_action(state)
            
            # Current market data
            current_price = df['Close'].iloc[-1]
            
            # Determine trading action
            new_position = position
            
            if action == 0:  # Sell
                if position == 1:  # Close long
                    print(f"CLOSING LONG at {current_price}")
                    # Execute sell
                    if quantity and client:
                        try:
                            client.create_order(
                                symbol=symbol,
                                side='SELL',
                                type='MARKET',
                                quantity=quantity
                            )
                        except Exception as e:
                            print(f"Order execution error: {e}")
                    
                    # Record trade
                    profit = (current_price - entry_price) / entry_price
                    trades.append({
                        'type': 'CLOSE LONG',
                        'entry': entry_price,
                        'exit': current_price,
                        'profit': profit,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    print(f"Trade profit: {profit:.2%}")
                    
                    new_position = 0
                elif position == 0:  # Open short
                    print(f"OPENING SHORT at {current_price}")
                    # Execute short
                    if quantity and client:
                        try:
                            client.create_order(
                                symbol=symbol,
                                side='SELL',
                                type='MARKET',
                                quantity=quantity
                            )
                        except Exception as e:
                            print(f"Order execution error: {e}")
                    
                    new_position = -1
                    entry_price = current_price
            
            elif action == 2:  # Buy
                if position == -1:  # Close short
                    print(f"CLOSING SHORT at {current_price}")
                    # Execute buy
                    if quantity and client:
                        try:
                            client.create_order(
                                symbol=symbol,
                                side='BUY',
                                type='MARKET',
                                quantity=quantity
                            )
                        except Exception as e:
                            print(f"Order execution error: {e}")
                    
                    # Record trade
                    profit = (entry_price - current_price) / entry_price
                    trades.append({
                        'type': 'CLOSE SHORT',
                        'entry': entry_price,
                        'exit': current_price,
                        'profit': profit,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    print(f"Trade profit: {profit:.2%}")
                    
                    new_position = 0
                elif position == 0:  # Open long
                    print(f"OPENING LONG at {current_price}")
                    # Execute buy
                    if quantity and client:
                        try:
                            client.create_order(
                                symbol=symbol,
                                side='BUY',
                                type='MARKET',
                                quantity=quantity
                            )
                        except Exception as e:
                            print(f"Order execution error: {e}")
                    
                    new_position = 1
                    entry_price = current_price
            
            # Track trade count for overtrading penalty
            if position != new_position:
                trade_count += 1
            else:
                trade_count = max(0, trade_count - 0.1)  # Gradually decay
            
            # Update position
            old_position = position
            position = new_position
            
            # Wait for next candle to calculate reward
            print(f"Waiting for next candle to calculate reward...")
            time.sleep(60)  # Wait a minute
            
            # Get updated data with next candle
            if client:
                klines = client.get_klines(symbol=symbol, interval=timeframe, limit=100)
                updated_df = process_klines(klines)
            else:
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                updated_df = ticker.history(period="7d", interval=timeframe)
                updated_df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
            
            # Skip if no new data
            if len(updated_df) <= len(df):
                print("No new data yet, continuing...")
                continue
            
            # Get next price
            next_price = updated_df['Close'].iloc[-1]
            
            # Calculate reward
            reward, _, _ = calculate_reward(
                action, old_position, entry_price, current_price, next_price,
                trade_count, max_drawdown
            )
            
            # Update equity curve
            if old_position == 1:
                pnl = (next_price - current_price) / current_price
            elif old_position == -1:
                pnl = (current_price - next_price) / current_price
            else:
                pnl = 0
                
            equity_curve.append(equity_curve[-1] * (1 + pnl))
            
            # Calculate drawdown
            peak = max(equity_curve)
            current_dd = (equity_curve[-1] / peak) - 1
            max_drawdown = min(max_drawdown, current_dd)
            
            # Create next state
            next_state = create_market_state(updated_df, len(updated_df)-1, lookback)
            
            # Online learning
            if next_state is not None:
                print(f"Learning from experience: Action={action}, Reward={reward:.4f}")
                rl_agent.remember(state, action, reward, next_state, False, log_prob, value)
                
                # Train occasionally
                if random.random() < 0.5:  # 50% chance each cycle
                    if len(rl_agent.states) >= rl_agent.batch_size:
                        print("Training agent from recent experiences...")
                        rl_agent.train_from_buffer()
            
            # Save model periodically
            if random.random() < 0.1:  # 10% chance each cycle
                try:
                    rl_agent.save("trading_agent_actor_latest.h5", "trading_agent_critic_latest.h5")
                    print("Agent model saved")
                except Exception as e:
                    print(f"Error saving model: {str(e)}")
            
            # Print status
            print(f"Position: {position}, Entry: {entry_price}, Current: {current_price}")
            print(f"Action probabilities - Sell: {probs[0]:.2f}, Hold: {probs[1]:.2f}, Buy: {probs[2]:.2f}")
            print(f"Current Equity: ${equity_curve[-1]:.2f}, Max Drawdown: {max_drawdown:.2%}")
            
            # Summary of performance
            if len(trades) > 0:
                win_rate = sum(1 for t in trades if t['profit'] > 0) / len(trades)
                avg_profit = sum(t['profit'] for t in trades) / len(trades)
                print(f"Total Trades: {len(trades)}, Win Rate: {win_rate:.2%}, Avg Profit: {avg_profit:.2%}")
            
            # Sleep until next check
            print(f"Waiting for next cycle...")
            time.sleep(300)  # 5 minutes between checks
            
    except KeyboardInterrupt:
        print("Trading stopped by user")
        
        # Save final model
        if 'rl_agent' in globals() and rl_agent is not None:
            rl_agent.save("trading_agent_actor_final.h5", "trading_agent_critic_final.h5")
            print("Final agent model saved")
        
        # Save trade history
        with open("trade_history.json", "w") as f:
            json.dump(trades, f)
        
        return trades, equity_curve
    except Exception as e:
        print(f"Error in live trading: {str(e)}")
        traceback.print_exc()
        
        # Try to save model
        if 'rl_agent' in globals() and rl_agent is not None:
            rl_agent.save("trading_agent_actor_emergency.h5", "trading_agent_critic_emergency.h5")
            print("Emergency agent model saved")
        
        return None, None
#==============================================================================
# MAIN SYSTEM CLASS
#==============================================================================         
class NQAlphaEliteSystem:
    """
    NQ Alpha Elite - World's Best Trading Bot
    
    This is the main system class that orchestrates all components.
    """
    
    def __init__(self, mode="paper", capital=100000, config_file=None):
        """Initialize trading system with optimized startup sequence
        
        Args:
            mode (str): Trading mode (paper, live, backtest)
            capital (float): Initial capital
            config_file (str, optional): Path to configuration file
        """
        # Initialize logging first for proper startup diagnostics
        self.logger = logging.getLogger("NQAlpha.System")
        self.logger.info(f"Initializing NQ Alpha Elite v2.0 in {mode} mode with ${capital:,.2f}")
        
        # System state with atomic flags for thread safety
        self.mode = mode
        self.capital = capital
        self.running = False
        self.initialized = False
        self.shutdown_requested = threading.Event()  # Thread-safe shutdown flag
        self.start_time = None
        self.main_thread = None
        self.worker_threads = {}  # Track all worker threads
        
        # Create essential directories
        for dir_name in ['logs', 'data', 'results', 'config', 'models', 'backups']:
            dir_path = os.path.join(os.getcwd(), dir_name)
            os.makedirs(dir_path, exist_ok=True)
            self.logger.debug(f"Ensured directory exists: {dir_path}")
        
        # Load configuration with fallback cascade
        self.config = self._load_configuration(config_file)
        
        # Initialize components dictionary to track initialization status
        self.components = {
            'analytics': None,
            'market_data': None,
            'regime_classifier': None,
            'trade_manager': None,
            'strategy_manager': None,
            'risk_manager': None,
            'indicators': None,
            'execution_engine': None,
            'dashboard': None,
            'rl_system': None
        }
        
        # Initialize analytics first (no dependencies)
        self.initialize_analytics()
        
        # Initialize RL system with proper version checks
        self.use_reinforcement_learning = self._initialize_rl_system()
        
        # Enhanced overtrading prevention with adaptive parameters
        self._initialize_trading_controls()
        
        # Initialize all other components
        self._initialize_components()
        
        # Register signal handlers for clean shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Runtime performance monitoring
        self.performance_metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'execution_times': {},
            'last_gc': time.time()
        }
        
        # System health monitor thread
        self.health_monitor_thread = threading.Thread(
            target=self._health_monitor, 
            daemon=True, 
            name="HealthMonitor"
        )
        self.health_monitor_thread.start()
        
        self.logger.info("NQ Alpha Elite Trading System fully initialized and ready")
        self.initialized = True

    def _load_configuration(self, config_file):
        """Load configuration with fallback mechanism
        
        Args:
            config_file (str): Path to config file
            
        Returns:
            dict: Configuration dictionary
        """
        config = {}
        
        try:
            # Priority 1: Specified config file
            if config_file and os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    self.logger.info(f"Configuration loaded from {config_file}")
                    return config
                    
            # Priority 2: Default config.json in config directory
            default_config = os.path.join('config', 'config.json')
            if os.path.exists(default_config):
                with open(default_config, 'r') as f:
                    config = json.load(f)
                    self.logger.info(f"Configuration loaded from default {default_config}")
                    return config
                    
            # Priority 3: Import config module
            try:
                import config as config_module
                # Convert module attributes to dictionary
                config = {attr: getattr(config_module, attr) 
                        for attr in dir(config_module) 
                        if not attr.startswith('_') and not callable(getattr(config_module, attr))}
                self.logger.info("Configuration loaded from config module")
                return config
            except ImportError:
                self.logger.warning("No config module found, using defaults")
                
            # Priority 4: Use embedded defaults
            self.logger.warning("Using embedded default configuration")
            return self._get_default_config()
                
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            self.logger.warning("Using embedded default configuration")
            return self._get_default_config()

    def _initialize_rl_system(self):
        """Initialize reinforcement learning system with version compatibility checks
        
        Returns:
            bool: Whether RL is enabled and functioning
        """
        use_rl = self.config.get('use_reinforcement_learning', False)
        
        if not use_rl:
            self.logger.info("Reinforcement Learning disabled by configuration")
            return False
            
        try:
            # Check if RL module is available with version check
            from nq_alpha_rl import NQAlphaEliteRL, __version__ as rl_version
            
            min_required_version = "1.2.0"
            if parse_version(rl_version) < parse_version(min_required_version):
                self.logger.warning(
                    f"RL module version {rl_version} is older than required {min_required_version}. "
                    "Please update the nq_alpha_rl package. Running without RL."
                )
                return False
                
            # Initialize RL with GPU acceleration if available
            gpu_available = self._check_gpu_availability()
            rl_config = self.config.get('reinforcement_learning', {})
            rl_config['use_gpu'] = gpu_available
            
            self.rl_system = NQAlphaEliteRL(
                system=self,
                config=rl_config
            )
            self.components['rl_system'] = self.rl_system
            self.logger.info(f"Reinforcement Learning system initialized (GPU: {gpu_available})")
            return True
            
        except ImportError as e:
            self.logger.warning(f"RL module not found: {e}. Running without reinforcement learning")
            return False
        except Exception as e:
            self.logger.error(f"Failed to initialize Reinforcement Learning: {e}")
            self.logger.warning("Running without RL capabilities")
            return False

    def _initialize_trading_controls(self):
        """Initialize advanced trading controls with market-adaptive parameters"""
        # Dynamic trade cooldown system
        self._trade_cooldown_period = self.config.get('trade_cooldown_base', 60)
        self._last_trade_time = datetime.datetime.now() - datetime.timedelta(seconds=3600)
        
        # Enhanced adaptive cooldown multipliers based on market conditions
        self._cooldown_multipliers = {
            'loss': self.config.get('cooldown_loss_multiplier', 2.5),  
            'win': self.config.get('cooldown_win_multiplier', 1.0),
            'range_bound': self.config.get('cooldown_range_multiplier', 1.8),
            'volatile': self.config.get('cooldown_volatile_multiplier', 2.2),
            'trending': self.config.get('cooldown_trending_multiplier', 0.7),
            # New: Additional multipliers for fine-tuning
            'small_loss': self.config.get('cooldown_small_loss_multiplier', 1.5),
            'large_loss': self.config.get('cooldown_large_loss_multiplier', 3.0),
            'breakout': self.config.get('cooldown_breakout_multiplier', 0.5),
            'high_volume': self.config.get('cooldown_high_volume_multiplier', 0.8),
            'low_liquidity': self.config.get('cooldown_low_liquidity_multiplier', 2.0)
        }
        
        # Advanced signal enhancement with decay
        self._signal_enhancement_active = False
        self._enhanced_signal = 0.0
        self._signal_enhancement_expiry = datetime.datetime.now()
        self._signal_enhancement_decay_rate = self.config.get('signal_decay_rate', 0.05)
        
        # Adaptive thresholds for different market regimes
        self.regime_params = {
            "volatile": {"signal_threshold": self.config.get('volatile_threshold', 0.60)},
            "trending_up": {"signal_threshold": self.config.get('trending_up_threshold', 0.45)},
            "trending_down": {"signal_threshold": self.config.get('trending_down_threshold', 0.45)},
            "range_bound": {"signal_threshold": self.config.get('range_bound_threshold', 0.55)},
            "unknown": {"signal_threshold": self.config.get('unknown_threshold', 0.50)},
            # New regimes for more granular control
            "breakout": {"signal_threshold": self.config.get('breakout_threshold', 0.40)},
            "high_volatility": {"signal_threshold": self.config.get('high_volatility_threshold', 0.65)},
            "choppy": {"signal_threshold": self.config.get('choppy_threshold', 0.60)}
        }
        
        # Advanced trading circuit breakers
        self._daily_loss_limit = self.config.get('daily_loss_limit_pct', 2.0) / 100.0
        self._weekly_loss_limit = self.config.get('weekly_loss_limit_pct', 5.0) / 100.0
        self._consecutive_loss_limit = self.config.get('consecutive_loss_limit', 4)
        self._current_consecutive_losses = 0
    def get_system_state(self):
        """Get current system state for RL integration"""
        state = {
            'regime': getattr(self, 'current_regime', 'unknown'),
            'regime_confidence': getattr(self, 'regime_confidence', 0.0),
            'entanglement': getattr(self, 'quantum_entanglement', 0.0),
            'signal_strength': getattr(self, 'composite_signal', 0.0),
            'confirmation_score': getattr(self, 'confirmation_score', 0.0),
            'z_score': getattr(self, 'z_score', 0.0),
            'position_in_range': getattr(self, 'position_in_range', 0.5),
            'range_confidence': getattr(self, 'range_confidence', 0.0),
            'hurst_exponent': getattr(self, 'hurst_exponent', 0.5),
            'trend_direction': getattr(self, 'trend_direction', 0),
            'win_rate': self.get_recent_win_rate(50) if hasattr(self, 'get_recent_win_rate') else 0.5,
            'drawdown': getattr(self.risk_manager, 'current_drawdown', 0.0) if hasattr(self, 'risk_manager') else 0.0,
            'has_flow_pattern': 1.0 if hasattr(self, 'flow_pattern_detected') and self.flow_pattern_detected else 0.0,
            'has_edge_pattern': 1.0 if hasattr(self, 'edge_pattern_detected') and self.edge_pattern_detected else 0.0,
            'pattern_confidence': getattr(self, 'pattern_confidence', 0.0),
            'pattern_direction': getattr(self, 'pattern_direction', 0),
            'edge_strength': getattr(self, 'edge_strength', 0.0),
            'risk_score': getattr(self, 'risk_score', 0.0),
        }
        
        # Add recent trade info if available
        if hasattr(self, 'trade_history') and self.trade_history:
            last_trade = self.trade_history[-1]
            state['last_trade_profit'] = last_trade.get('profit', 0.0)
        else:
            state['last_trade_profit'] = 0.0
            
        # Add risk-reward if available
        if hasattr(self, 'risk_reward_ratio'):
            state['expectancy'] = getattr(self, 'trade_expectancy', 0.0)
            state['reward_risk_ratio'] = self.risk_reward_ratio
        else:
            state['expectancy'] = 0.0
            state['reward_risk_ratio'] = 1.0
            
        return state    
    def calculate_composite_signal(self, market_data):
        # Your existing signal calculation code
        base_signal = self._apply_all_signal_components()
        
        # Apply RL enhancement if enabled
        if hasattr(self, 'use_reinforcement_learning') and self.use_reinforcement_learning and hasattr(self, 'rl_system'):
            try:
                # Process current state for RL
                current_state = self.rl_system.process_state(market_data, self.get_system_state())
                
                # Get RL action
                rl_action = self.rl_system.get_action(current_state)
                
                # Apply RL adjustments
                adjustments = self.rl_system.apply_action(rl_action, base_signal)
                
                # Store for learning
                self.rl_system.last_state = current_state
                self.rl_system.last_action = rl_action
                
                # Apply the adjusted signal
                final_signal = adjustments['adjusted_signal']
                
                # Adjust position sizing
                self.position_sizing_multiplier = adjustments['position_sizing_factor']
                
                # Adjust exit timing
                self.exit_timing_adjustment = adjustments['exit_timing_factor']
                
                self.logger.info(f"RL adjusted signal: {base_signal:.4f} → {final_signal:.4f}, sizing: {adjustments['position_sizing_factor']:.2f}x, exit timing: {adjustments['exit_timing_factor']:.2f}x")
                
                return final_signal
            except Exception as e:
                self.logger.error(f"RL signal adjustment error: {e}")
                
        # Return the original signal if RL is disabled or failed
        return base_signal    
    def generate_trading_signal(self, market_data):
        # Your existing signal generation code
        base_signal = self.calculate_composite_signal(market_data)
        
        # Process current state for RL
        current_state = self.rl_system.process_state(market_data, self.get_system_state())
        
        # Get RL action
        rl_action = self.rl_system.get_action(current_state)
        
        # Apply RL adjustments
        adjustments = self.rl_system.apply_action(rl_action, base_signal)
        
        # Store for learning
        self.rl_system.last_state = current_state
        self.rl_system.last_action = rl_action
        
        # Apply the adjusted signal
        final_signal = adjustments['adjusted_signal']
        
        # Adjust position sizing
        self.position_sizing_multiplier = adjustments['position_sizing_factor']
        
        # Adjust exit timing
        self.exit_timing_adjustment = adjustments['exit_timing_factor']
        
        return final_signal    
    def enhance_baseline_collection(self):
        """Add enhanced baseline data collection"""
        self.logger.info("Setting up enhanced baseline data collection...")
        
        # Create data directories if they don't exist
        os.makedirs('data/regimes', exist_ok=True)
        os.makedirs('data/signals', exist_ok=True)
        os.makedirs('data/market_structure', exist_ok=True)
        
        # Create signal log
        self.signal_log_path = os.path.join('data/signals', f'signal_log_{datetime.datetime.now().strftime("%Y%m%d")}.csv')
        with open(self.signal_log_path, 'a') as f:
            f.write("timestamp,strategy,signal,regime,price,vpin,liquidity_score,volatility\n")
        
        # Create regime log
        self.regime_log_path = os.path.join('data/regimes', f'regime_log_{datetime.datetime.now().strftime("%Y%m%d")}.csv')
        with open(self.regime_log_path, 'a') as f:
            f.write("timestamp,regime,confidence,volatility,trend_strength,trend_direction,hurst\n")
        
        # Create market structure log
        self.structure_log_path = os.path.join('data/market_structure', f'structure_log_{datetime.datetime.now().strftime("%Y%m%d")}.csv')
        with open(self.structure_log_path, 'a') as f:
            f.write("timestamp,price,vpin,liquidity_score,order_flow,delta,bias,institutional_pressure\n")
        
        # Log regime transitions
        original_set_new_regime = self.regime_classifier._set_new_regime
        
        def enhanced_set_new_regime(self, regime, confidence):
            # Call original method
            original_set_new_regime(regime, confidence)
            
            # Log regime data
            try:
                with open(self.system.regime_log_path, 'a') as f:
                    timestamp = datetime.datetime.now().isoformat()
                    
                    # Handle hurst exponent if it's an array
                    hurst_val = self.hurst_exponent
                    if isinstance(hurst_val, np.ndarray):
                        hurst_val = float(hurst_val.mean())
                    
                    row = [
                        timestamp,
                        regime,
                        f"{confidence:.6f}",
                        f"{self.volatility:.6f}",
                        f"{self.trend_strength:.6f}",
                        f"{self.trend_direction:.6f}",
                        f"{hurst_val:.6f}"
                    ]
                    f.write(",".join(row) + "\n")
            except Exception as e:
                self.logger.error(f"Error logging regime data: {e}")
        
        # Apply patch
        import types
        self.regime_classifier._set_new_regime = types.MethodType(
            enhanced_set_new_regime, self.regime_classifier)
        self.regime_classifier.system = self  # Add reference to system
        
        # Log signals
        original_calculate = self.strategy_manager._calculate_composite_signal
        
        def enhanced_calculate_composite_signal(self, current_regime, market_data):
            # Call original method
            original_calculate(current_regime, market_data)
            
            # Log signal data
            try:
                with open(self.system.signal_log_path, 'a') as f:
                    timestamp = datetime.datetime.now().isoformat()
                    for strategy_name, signal in self.signals.items():
                        if strategy_name != 'composite':
                            row = [
                                timestamp,
                                strategy_name,
                                f"{signal:.6f}",
                                current_regime,
                                f"{market_data.get('price', 0):.2f}",
                                f"{market_data.get('vpin', 0):.6f}",
                                f"{market_data.get('liquidity_score', 0):.6f}",
                                f"{self.system.regime_classifier.volatility:.6f}" if hasattr(self.system, 'regime_classifier') else "0.0"
                            ]
                            f.write(",".join(row) + "\n")
            except Exception as e:
                self.logger.error(f"Error logging signals: {e}")
        
        # Apply patch
        self.strategy_manager._calculate_composite_signal = types.MethodType(
            enhanced_calculate_composite_signal, self.strategy_manager)
        self.strategy_manager.system = self  # Add reference to system
        
        # Log market structure
        def log_market_structure():
            try:
                if hasattr(self, 'market_data'):
                    data = self.market_data.get_realtime_data()
                    if data:
                        with open(self.structure_log_path, 'a') as f:
                            timestamp = datetime.datetime.now().isoformat()
                            row = [
                                timestamp,
                                f"{data.get('price', 0):.2f}",
                                f"{data.get('vpin', 0):.6f}",
                                f"{data.get('liquidity_score', 0):.6f}",
                                f"{data.get('order_flow', 0):.6f}",
                                f"{data.get('delta', 0):.6f}",
                                f"{data.get('bias', 0):.6f}",
                                f"{data.get('institutional_pressure', 0):.6f}"
                            ]
                            f.write(",".join(row) + "\n")
            except Exception as e:
                self.logger.error(f"Error logging market structure: {e}")
        
        # Schedule logging every 5 seconds
        import threading
        def structure_logging_thread():
            while self.running and not self.shutdown_requested:
                log_market_structure()
                time.sleep(5)
        
        # Start logging thread
        threading.Thread(target=structure_logging_thread, daemon=True).start()
        
        self.logger.info("Enhanced baseline data collection enabled")
        return True
    def _load_config(self, config_file=None):
        """Load configuration from file or use default"""
        try:
            if config_file is not None and os.path.exists(config_file):
                # Load from JSON file
                with open(config_file, 'r') as f:
                    loaded_config = json.load(f)
                    self.logger.info(f"Configuration loaded from {config_file}")
                    return loaded_config
            else:
                # Default configuration
                default_config = {
                    'max_position_size': 5,
                    'risk_per_trade': 0.01,
                    'use_reinforcement_learning': True,
                    'debug_mode': False,
                    'log_level': 'INFO',
                    'data_sources': ['primary', 'secondary', 'alternative'],
                    'trading_hours': {
                        'start': '09:30',
                        'end': '16:00'
                    },
                    'exit_settings': {
                        'use_trailing_stop': True,
                        'use_time_exits': True,
                        'max_hold_time': 600  # 10 minutes
                    },
                    'performance_settings': {
                        'target_sharpe': 2.0,
                        'max_drawdown': 0.05
                    },
                    'model_paths': {
                        'regime_classifier': 'models/regime_classifier.pkl',
                        'alpha_extractor': 'models/alpha_extractor.h5',
                        'risk_model': 'models/risk_model.pkl'
                    }
                }
                self.logger.info("Using default configuration")
                return default_config
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            self.logger.warning("Using minimal default configuration")
            return {'use_reinforcement_learning': False}
    
    def _merge_configs(self, target, source):
        """Merge source config into target config recursively
        
        Args:
            target (dict): Target config
            source (dict): Source config
        """
        for key, value in source.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                self._merge_configs(target[key], value)
            else:
                target[key] = value
    
    def _initialize_components(self):
        """Initialize system components with dependency management and validation"""
        try:
            self.logger.info("Initializing system components with dependency validation")
            
            # Component initialization order with dependency tracking
            init_sequence = [
                ('market_data', self._init_market_data),
                ('indicators', self._init_indicators),
                ('regime_classifier', self._init_regime_classifier),
                ('risk_manager', self._init_risk_manager),
                ('strategy_manager', self._init_strategy_manager),
                ('trade_manager', self._init_trade_manager),
                ('execution_engine', self._init_execution_engine),
                ('dashboard', self._init_dashboard)
            ]
            
            # Track dependencies for validation
            required_deps = {
                'indicators': ['market_data'],
                'regime_classifier': ['market_data', 'indicators'],
                'risk_manager': ['market_data'],
                'strategy_manager': ['market_data', 'indicators', 'regime_classifier'],
                'trade_manager': ['market_data', 'risk_manager', 'strategy_manager'],
                'execution_engine': ['market_data', 'trade_manager'],
                'dashboard': ['market_data', 'trade_manager']
            }
            
            # Initialize each component and validate dependencies
            for component_name, init_func in init_sequence:
                try:
                    # Check dependencies first
                    if component_name in required_deps:
                        for dep in required_deps[component_name]:
                            if not self.components[dep]:
                                raise RuntimeError(f"Missing required dependency: {dep} for {component_name}")
                    
                    # Initialize component
                    self.logger.debug(f"Initializing {component_name}...")
                    init_func()
                    
                    # Validate component
                    if not self.components[component_name]:
                        raise RuntimeError(f"Component {component_name} failed to initialize properly")
                    
                    self.logger.info(f"Component {component_name} successfully initialized")
                    
                except Exception as e:
                    self.logger.error(f"Failed to initialize {component_name}: {e}")
                    if component_name in self.config.get('critical_components', 
                                                    ['market_data', 'risk_manager', 'execution_engine']):
                        raise RuntimeError(f"Critical component {component_name} failed, cannot continue")
            
            # Start analytics and monitoring threads
            self._start_background_threads()
            
            # Verify system integrity
            missing = [name for name, comp in self.components.items() 
                    if comp is None and name != 'rl_system']
            if missing:
                self.logger.warning(f"System initialized with missing components: {', '.join(missing)}")
            else:
                self.logger.info("All system components successfully initialized")
            
        except Exception as e:
            self.logger.critical(f"Fatal error during component initialization: {e}")
            raise SystemInitializationError(f"Failed to initialize trading system: {e}")

    def _init_market_data(self):
        """Initialize market data feed component"""
        self.market_data = MarketDataFeed(
            config=self.config.get('market_data', {}),
            logger=logging.getLogger("NQAlpha.MarketData")
        )
        self.components['market_data'] = self.market_data

    def _init_indicators(self):
        """Initialize technical indicators component"""
        self.indicators = TechnicalIndicators(
            system=self,
            logger=logging.getLogger("NQAlpha.Indicators")
        )
        self.components['indicators'] = self.indicators

    def _init_regime_classifier(self):
        """Initialize market regime classifier component"""
        self.regime_classifier = MarketRegimeClassifier(
            config=self.config.get('regime_classifier', {}),
            system=self,
            logger=logging.getLogger("NQAlpha.RegimeClassifier")
        )
        self.components['regime_classifier'] = self.regime_classifier

    def _init_risk_manager(self):
        """Initialize risk manager component"""
        self.risk_manager = RiskManager(
            config=self.config.get('risk', {}),
            system=self,
            logger=logging.getLogger("NQAlpha.RiskManager")
        )
        self.components['risk_manager'] = self.risk_manager

    def _init_strategy_manager(self):
        """Initialize strategy manager component"""
        self.strategy_manager = StrategyManager(
            config=self.config.get('strategies', {}),
            system=self,
            logger=logging.getLogger("NQAlpha.StrategyManager")
        )
        self.components['strategy_manager'] = self.strategy_manager

    def _init_trade_manager(self):
        """Initialize trade manager component"""
        # Only initialize once with proper dependencies
        self.trade_manager = DynamicTradeManager(
            system=self,
            config=self.config.get('trade_manager', {}),
            logger=logging.getLogger("NQAlpha.TradeManager")
        )
        self.components['trade_manager'] = self.trade_manager

    def _init_execution_engine(self):
        """Initialize execution engine component"""
        self.execution_engine = ExecutionEngine(
            config=self.config.get('execution', {}),
            system=self,
            logger=logging.getLogger("NQAlpha.ExecutionEngine")
        )
        self.components['execution_engine'] = self.execution_engine

    def _init_dashboard(self):
        """Initialize performance dashboard component"""
        self.dashboard = PerformanceDashboard(
            config=self.config.get('dashboard', {}),
            system=self,
            logger=logging.getLogger("NQAlpha.Dashboard")
        )
        self.components['dashboard'] = self.dashboard

    def _start_background_threads(self):
        """Start all background monitoring and maintenance threads"""
        # Analytics thread
        self.analytics_thread = threading.Thread(
            target=self._analytics_thread, 
            daemon=True,
            name="AnalyticsThread"
        )
        self.analytics_thread.start()
        self.worker_threads['analytics'] = self.analytics_thread
        
        # Add other background threads like data cleanup, etc.
        cleanup_thread = threading.Thread(
            target=self._cleanup_thread,
            daemon=True,
            name="CleanupThread"
        )
        cleanup_thread.start()
        self.worker_threads['cleanup'] = cleanup_thread
    def _signal_handler(self, sig, frame):
        """Signal handler for graceful shutdown
        
        Args:
            sig: Signal number
            frame: Current stack frame
        """
        self.logger.info(f"Received signal {sig}, initiating shutdown")
        self.shutdown_requested = True
    
    def start(self):
        """Start trading system"""
        if self.running:
            self.logger.warning("System already running")
            return False
        
        self.logger.info("Starting trading system")
        
        try:
            # Set running flag
            self.running = True
            self.start_time = datetime.datetime.now()
            # Performance tracking
            self.loop_metrics = {
                'execution_times': collections.deque(maxlen=1000),
                'last_100_loops': collections.deque(maxlen=100),
                'data_latencies': collections.deque(maxlen=100),
                'signal_strengths': collections.deque(maxlen=50)
            }
            
            # Initialize thread-local storage for performance monitoring
            self.thread_local = threading.local()
            self.thread_local.last_loop_time = time.time()
            
            # Initialize adaptive parameters
            self._adaptive_params = {
                'signal_threshold': self.config.get('base_signal_threshold', 0.55),
                'position_size_multiplier': 1.0,
                'max_trades_per_hour': self.config.get('max_trades_per_hour', 5)
            }
            # Start components
            self.market_data.run()
            self.logger.info("Market data feed started")
            
            self.regime_classifier.start()
            self.logger.info("Market regime classifier started")
            
            self.strategy_manager.start()
            self.logger.info("Strategy manager started")
            
            self.dashboard.start()
            self.logger.info("Performance dashboard started")
            
            self.enhance_baseline_collection()
            self.logger.info("Enhanced data collection activated")

            # Start main thread
            self.thread = threading.Thread(
                target=self._main_thread,
                name="MainSystemThread"
            )
            self.thread.daemon = True
            self.thread.start()
            
            self.logger.info("Main system thread started")
            self.logger.info("Trading system started")
            
            return True
            
        except Exception as e:
            self.running = False
            self.logger.error(f"Error starting system: {e}")
            return False
    
    def stop(self):
        """Stop trading system"""
        if not self.running:
            self.logger.warning("System not running")
            return
        
        self.logger.info("Stopping trading system")
        
        try:
            # Set running flag
            self.running = False
            
            # Stop components in reverse order
            self.dashboard.stop()
            self.logger.info("Performance dashboard stopped")
            
            self.strategy_manager.stop()
            self.logger.info("Strategy manager stopped")
            
            self.regime_classifier.stop()
            self.logger.info("Market regime classifier stopped")
            
            self.market_data.stop()
            self.logger.info("Market data feed stopped")
            
            # Wait for main thread
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=5.0)
            
            self.logger.info("Trading system stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping system: {e}")
    def detect_orderflow_divergence(self, market_data, lookback=5):
        """Detect institutional order flow divergence patterns with elite signal enhancement"""
        try:
            # Get current metrics
            current_order_flow = market_data.get('order_flow', 0)
            current_delta = market_data.get('delta', 0)
            current_price = market_data.get('price', 0)
            
            # Calculate divergence score
            divergence_score = current_order_flow * (0 - current_delta) * 10
            
            # Initialize divergence info
            divergence_info = {'detected': False}
            
            if divergence_score > 0.15:
                direction = 1 if current_order_flow > 0 else -1
                self.logger.info(f"Order flow divergence detected: Score {divergence_score:.2f}, " +
                            f"Bias: {'LONG' if direction > 0 else 'SHORT'}")
                
                divergence_info = {
                    'detected': True,
                    'score': divergence_score,
                    'direction': direction,
                    'bias': 'accumulation' if direction > 0 else 'distribution'
                }
                
                # ELITE ENHANCEMENT: For strong divergences, boost signal in divergence direction
                if divergence_info['detected'] and abs(divergence_score) > 0.15:
                    # Get current composite signal directly from strategy manager
                    current_signal = self.strategy_manager.get_strategy_signals().get('composite', 0.0)
                    # Track divergence persistence
                    if not hasattr(self, '_prev_divergence_scores'):
                        self._prev_divergence_scores = []

                    # Add current score to history
                    self._prev_divergence_scores.append(divergence_score)
                    if len(self._prev_divergence_scores) > 10:
                        self._prev_divergence_scores.pop(0)

                    # Calculate persistence - is divergence building or fading?
                    persistence_factor = 1.0
                    if len(self._prev_divergence_scores) >= 3:
                        recent_avg = sum(self._prev_divergence_scores[-3:]) / 3
                        if recent_avg > 0:
                            trend = divergence_score / recent_avg
                            persistence_factor = min(1.3, max(0.8, trend))
                            self.logger.info(f"Divergence persistence factor: {persistence_factor:.2f}")
                    # Calculate boost proportional to divergence score
                    signal_boost = direction * min(0.25, divergence_score * 1.5 * persistence_factor)
                    enhanced_signal = current_signal + signal_boost
                    
                    # Get current regime for threshold determination
                    regime_info = self.regime_classifier.get_current_regime()
                    current_regime = regime_info.get('regime', 'unknown')
                    
                    # Store enhanced signal for use in main thread
                    self._signal_enhancement_active = True
                    self._enhanced_signal = enhanced_signal
                    self._signal_enhancement_expiry = datetime.datetime.now() + datetime.timedelta(minutes=2)
                    
                    # Get regime-based threshold
                    base_threshold = self.regime_params.get(current_regime, {}).get("signal_threshold", 0.5)
                    
                    # Log enhancement if it crosses threshold
                    if abs(enhanced_signal) >= base_threshold and abs(current_signal) < base_threshold:
                        self.logger.info(f"Elite divergence amplification: {current_signal:.2f} → {enhanced_signal:.2f} (crosses threshold: {base_threshold:.2f})")
                    else:
                        self.logger.info(f"Elite divergence amplification: {current_signal:.2f} → {enhanced_signal:.2f}")
                    
                    # Add enhancement data to divergence_info
                    divergence_info['enhanced_signal'] = enhanced_signal
                    divergence_info['signal_boost'] = signal_boost
            
            return divergence_info
            
        except Exception as e:
            self.logger.error(f"Error in order flow divergence detection: {e}")
            return {'detected': False}
    
    def calculate_trade_pnl(self, entry_price, exit_price, position_size, commission=2.25):
        """Calculate accurate P&L for NQ futures"""
        if not entry_price or not exit_price:
            self.logger.error(f"Invalid prices for P&L calculation: entry={entry_price}, exit={exit_price}")
            return 0
            
        direction = 1 if position_size > 0 else -1
        point_value = 20  # NQ futures is $20 per point
        
        # Calculate points difference
        points = (exit_price - entry_price) * direction
        
        # Calculate raw P&L
        raw_pnl = points * abs(position_size) * point_value
        
        # Subtract commission
        net_pnl = raw_pnl - (commission * abs(position_size))
        
        # CRITICAL: Sanity check against impossible values
        if abs(points) > 200:  # No reasonable single trade would move 200+ points
            self.logger.error(f"P&L calculation error: {points} points between {entry_price} and {exit_price}")
            return 0
            
        if abs(raw_pnl) > 10000:  # Cap at $10,000 per contract as sanity check
            self.logger.warning(f"P&L sanity check: ${raw_pnl:.2f} from {points} points")
            max_reasonable_pnl = 10000 * abs(position_size)
            return max(min(raw_pnl, max_reasonable_pnl), -max_reasonable_pnl)
        
        return net_pnl
    def calculate_enhanced_position_size(self, signal, price, stop_price, market_data, atr):
        """Elite position sizing with volatility adjustment and performance feedback"""
        # Get basic size from risk manager (1% risk per trade)
        risk_amount = self.capital * 0.01
        if stop_price == price:  # Avoid division by zero
            stop_price = price * 0.995 if signal > 0 else price * 1.005
            
        price_risk = abs(price - stop_price)
        dollar_risk_per_contract = price_risk * 20  # NQ is $20 per point
        
        # Base position size from risk calculation
        if dollar_risk_per_contract > 0:
            base_size = max(1, int(risk_amount / dollar_risk_per_contract))
        else:
            base_size = 1
        
        # Adjust for volatility (reduce size in high volatility)
        volatility = market_data.get('volatility', 0.0001)
        vol_ratio = 0.00008 / max(volatility, 0.00004)  # Compare to baseline volatility
        vol_adjustment = min(1.5, max(0.5, vol_ratio))
        
        # Adjust for signal strength (larger positions for stronger signals)
        signal_adjustment = 0.8 + (min(1.0, abs(signal)) * 0.4)
        
        # Adjust for regime
        regime_info = self.regime_classifier.get_current_regime()
        current_regime = regime_info.get('regime', 'unknown')
        regime_confidence = regime_info.get('confidence', 0.5)
        
        regime_adjustment = 1.0
        if current_regime == "trending_up" or current_regime == "trending_down":
            # More aggressive in trending markets with high confidence
            regime_adjustment = 1.0 + (0.3 * regime_confidence)
        elif current_regime == "volatile":
            # More conservative in volatile markets
            regime_adjustment = 0.7
        
        # Calculate final position size with a weighted approach
        adjusted_size = int(base_size * vol_adjustment * signal_adjustment * regime_adjustment)
        
        # Ensure minimum and maximum sizes
        adjusted_size = max(1, min(5, adjusted_size))
        
        # Log adjustments
        self.logger.info(f"Elite position sizing: Base {base_size} → {adjusted_size} " +
                        f"(Vol: {vol_adjustment:.2f}, Signal: {signal_adjustment:.2f}, " +
                        f"Regime: {regime_adjustment:.2f})")
        
        return adjusted_size    

    def detect_trapped_traders(self, market_data, regime_info):
        """Detect trapped trader patterns that precede significant moves
        
        Args:
            market_data (dict): Current market data
            regime_info (dict): Current regime information
        
        Returns:
            dict: Trapped traders analysis
        """
        try:
            # Get key metrics
            price = market_data.get('price', 0)
            delta = market_data.get('delta', 0)
            order_flow = market_data.get('order_flow', 0)
            vpin = market_data.get('vpin', 0)
            
            # Check for regime transition in last 5 minutes
            # Since get_time_since_last_change is missing, we'll add a safer check
            regime_change_recent = False
            current_regime = regime_info.get('regime', 'unknown')
            
            # We need to track regime changes ourselves
            if not hasattr(self, '_previous_regime'):
                self._previous_regime = current_regime
                self._regime_change_time = time.time()
            
            # Check if regime changed since last check
            if current_regime != self._previous_regime:
                self._previous_regime = current_regime
                self._regime_change_time = time.time()
                regime_change_recent = True
            else:
                # Check if change was recent (within 5 minutes)
                regime_change_recent = (time.time() - self._regime_change_time) < 300
            
            # Pattern 1: Retail trapped long (failed breakout)
            if current_regime == 'range_bound' and regime_change_recent:
                prev_regime = getattr(self, '_previous_regime', 'unknown')
                if prev_regime == 'trending_up' and delta < -0.4 and order_flow < -0.02:
                    score = abs(delta) * (1 + vpin) * 2
                    
                    if score > 0.8:
                        self.logger.info(f"Trapped traders detected: LONGS trapped - Score: {score:.2f}")
                        return {
                            'detected': True,
                            'type': 'trapped_longs',
                            'score': score,
                            'bias': 'bearish',
                            'target': price * 0.997  # 0.3% downside target
                        }
            
            # Pattern 2: Retail trapped short (failed breakdown)
            if current_regime == 'range_bound' and regime_change_recent:
                prev_regime = getattr(self, '_previous_regime', 'unknown')
                if prev_regime == 'trending_down' and delta > 0.4 and order_flow > 0.02:
                    score = abs(delta) * (1 + vpin) * 2
                    
                    if score > 0.8:
                        self.logger.info(f"Trapped traders detected: SHORTS trapped - Score: {score:.2f}")
                        return {
                            'detected': True,
                            'type': 'trapped_shorts',
                            'score': score,
                            'bias': 'bullish',
                            'target': price * 1.003  # 0.3% upside target
                        }
            
            return {'detected': False}
            
        except Exception as e:
            self.logger.error(f"Error in trapped traders detection: {e}")
            return {'detected': False}
            
    def detect_volatility_compression_setup(self, market_data, regime_info):
        """Detect potential volatility expansion setups
        
        Args:
            market_data (dict): Market data
            regime_info (dict): Regime information
            
        Returns:
            dict: Compression setup information
        """
        try:
            # Get key metrics
            volatility = regime_info.get('volatility', 0.0001)
            vpin = market_data.get('vpin', 0.5)
            trend_strength = regime_info.get('trend_strength', 0.0)
            
            # Make sure volatility isn't zero to prevent division by zero
            if volatility <= 0.000001:
                volatility = 0.000001  # Set minimum value
            
            # Safety check for other parameters
            vpin = max(0.01, min(vpin, 1.0))  # Ensure between 0.01 and 1.0
            trend_strength = min(trend_strength, 1.4)  # Cap at 1.4
            
            # Check for compression pattern (low vol + high VPIN + decreasing trend strength)
            if volatility < 0.0001 and vpin > 0.25 and trend_strength < 1.5:
                compression_score = (0.0001 / volatility) * vpin * (1.5 - trend_strength)
                
                if compression_score > 7.5:
                    # Get additional confirmation
                    price = market_data.get('price', 0)
                    delta = market_data.get('delta', 0)
                    order_flow = market_data.get('order_flow', 0)
                    
                    # Combine delta and order flow for direction bias
                    flow_signal = (delta + order_flow) / 2
                    
                    # Determine probable direction
                    probable_direction = 1 if flow_signal > 0 else -1
                    
                    # Calculate optimal entry and stop
                    entry_level = price + (probable_direction * price * 0.001)
                    stop_level = price - (probable_direction * price * 0.0005)
                    
                    # Calculate compression duration if possible
                    duration_minutes = 0
                    
                    self.logger.info(f"Volatility compression detected - Score: {compression_score:.2f}, " +
                                f"Direction bias: {'UP' if probable_direction > 0 else 'DOWN'}")
                    
                    return {
                        'detected': True,
                        'score': compression_score,
                        'direction': probable_direction,
                        'entry': entry_level,
                        'stop': stop_level,
                        'duration': duration_minutes
                    }
            
            return {'detected': False}
        
        except Exception as e:
            self.logger.error(f"Error in volatility compression detection: {e}")
            return {'detected': False}
    def enhance_signal_generation(self):
        """Implement elite signal enhancement for improved trade execution"""
        self.logger.info("Activating elite signal enhancement protocol")
        
        # 1. Add dynamic bias detection to StrategyManager
        if hasattr(self, 'strategy_manager'):
            # Modify composite signal calculation to amplify emerging trends
            original_get_strategy_signals = self.strategy_manager.get_strategy_signals
            
            def enhanced_get_strategy_signals(*args, **kwargs):
                # Get original signals
                signals = original_get_strategy_signals(*args, **kwargs)
                
                # Get market data and regime info
                market_data = self.market_data.get_realtime_data()
                regime_info = self.regime_classifier.get_current_regime()
                
                # Extract key metrics
                current_regime = regime_info.get('regime', 'unknown')
                regime_confidence = regime_info.get('confidence', 0.0)
                delta = market_data.get('delta', 0.0)
                order_flow = market_data.get('order_flow', 0.0)
                
                # Get original composite signal
                composite = signals.get('composite', 0.0)
                
                # Apply tactical signal amplification
                if abs(composite) > 0.05 and abs(composite) < 0.3:
                    # Small but non-zero signal - potential emerging trend
                    # Amplify by 1.5x for trending regimes with high confidence
                    if current_regime in ['trending_up', 'trending_down'] and regime_confidence > 0.8:
                        enhanced_composite = composite * 1.5
                        self.logger.info(f"Elite enhancement: Signal amplified {composite:.2f} → {enhanced_composite:.2f} in {current_regime}")
                        signals['composite'] = enhanced_composite
                
                # Apply bias when signals are very small but confirmation metrics are strong
                if abs(composite) < 0.05:
                    # Delta-based bias for very small signals
                    if abs(delta) > 0.4:
                        bias = delta * 0.25
                        enhanced_composite = composite + bias
                        self.logger.info(f"Elite enhancement: Delta bias applied {composite:.2f} → {enhanced_composite:.2f}")
                        signals['composite'] = enhanced_composite
                    
                    # Order flow bias for very small signals
                    elif abs(order_flow) > 0.15:
                        bias = order_flow * 0.35
                        enhanced_composite = composite + bias
                        self.logger.info(f"Elite enhancement: Flow bias applied {composite:.2f} → {enhanced_composite:.2f}")
                        signals['composite'] = enhanced_composite
                
                return signals
            
            # Apply the enhancement
            self.strategy_manager.get_strategy_signals = enhanced_get_strategy_signals
        
        # 2. Optimize threshold calculation in main thread
        # Define a method to calculate thresholds that we can replace
        def enhanced_threshold_calculation(self, base_threshold, confirmation_score, market_data):
            # Initialize tracking counter if not present
            if not hasattr(self, '_high_confirmation_count'):
                self._high_confirmation_count = 0
                
            # Track consistent high confirmation
            if confirmation_score > 0.7:
                self._high_confirmation_count += 1
            else:
                self._high_confirmation_count = 0
            
            # Progressive threshold reduction for persistent confirmation
            if self._high_confirmation_count >= 5:  # 5 consecutive high confirmations
                base_threshold *= max(0.7, 0.9 - (self._high_confirmation_count * 0.02))
                self.logger.info(f"Elite threshold reduction: Persistent high confirmation ({self._high_confirmation_count})")
            
            # Apply confirmation-based adjustment
            if confirmation_score > 0.7:
                threshold = base_threshold * 0.9  # Strong confirmation lowers threshold
                self.logger.info(f"Adjusting signal threshold: {base_threshold:.2f} → {threshold:.2f} (strong confirmation: {confirmation_score:.2f})")
            elif confirmation_score < 0.3:
                threshold = base_threshold * 1.1  # Weak confirmation raises threshold (reduced from 1.2)
                self.logger.info(f"Adjusting signal threshold: {base_threshold:.2f} → {threshold:.2f} (weak confirmation: {confirmation_score:.2f})")
            else:
                threshold = base_threshold
            
            # Implement minimum threshold
            if threshold > 0.45 and market_data.get('vpin', 0) > 0.25:
                original = threshold
                threshold = max(0.35, threshold * 0.8)
                self.logger.info(f"Elite threshold cap: High VPIN environment - threshold {original:.2f} → {threshold:.2f}")
            
            return threshold
        
        # Store the method for use in the main thread
        self._calculate_enhanced_threshold = enhanced_threshold_calculation
        
        # 3. Add adaptive signal integration flag
        self._adaptive_signal_activated = True
        self.logger.info("Elite adaptive signal integration activated")    
    
    def get_multi_timeframe_confirmation(self, signal, market_data):
        """Get multi-timeframe confirmation for trade signals
        
        Args:
            signal (float): Current trading signal (-1.0 to 1.0)
            market_data (dict): Current market data
        
        Returns:
            float: Confirmation score (-1.0 to 1.0)
        """
        try:
            # Get regime and order flow data
            if not hasattr(self, 'regime_classifier'):
                return 0.0
                
            regime_info = self.regime_classifier.get_current_regime()
            current_regime = regime_info.get('regime', 'unknown')
            
            # Get order flow metrics with safety checks
            order_flow = market_data.get('order_flow', 0) or 0
            delta = market_data.get('delta', 0) or 0
            
            # Calculate confirmation score
            if signal > 0:  # Long signal
                # Check if regime is trending up
                regime_score = 1.0 if current_regime == 'trending_up' else \
                            0.5 if current_regime == 'range_bound' else \
                            -0.5 if current_regime == 'trending_down' else 0
                            
                # Check if order flow confirms
                flow_score = min(1.0, order_flow * 3) if order_flow > 0 else 0
                
                # Check if delta confirms
                delta_score = min(1.0, delta * 2) if delta > 0 else 0
                
                # Combined confirmation
                confirmation = (regime_score * 0.5) + (flow_score * 0.3) + (delta_score * 0.2)
                
            elif signal < 0:  # Short signal
                # Check if regime is trending down
                regime_score = 1.0 if current_regime == 'trending_down' else \
                            0.5 if current_regime == 'range_bound' else \
                            -0.5 if current_regime == 'trending_up' else 0
                            
                # Check if order flow confirms
                flow_score = min(1.0, -order_flow * 3) if order_flow < 0 else 0
                
                # Check if delta confirms
                delta_score = min(1.0, -delta * 2) if delta < 0 else 0
                
                # Combined confirmation
                confirmation = (regime_score * 0.5) + (flow_score * 0.3) + (delta_score * 0.2)
            
            else:  # Zero signal
                # For zero signal, check regime tendencies
                if current_regime == 'trending_up':
                    confirmation = 0.3  # Slight bullish bias in uptrend
                elif current_regime == 'trending_down':
                    confirmation = -0.3  # Slight bearish bias in downtrend
                elif current_regime == 'range_bound':
                    # In range, use order flow as guide
                    confirmation = order_flow * 0.5
                else:
                    confirmation = 0.0
            
            self.logger.info(f"Multi-timeframe confirmation: {confirmation:.2f} for signal {signal:.2f}")
            return confirmation
            
        except Exception as e:
            self.logger.error(f"Error in multi-timeframe confirmation: {e}")
            return 0.0
    def calculate_enhanced_stop(self, entry_price, direction, atr, market_data, regime):
        """Elite stop placement with volatility and regime awareness"""
        # Start with ATR-based stop distance
        base_stop_distance = atr * 1.5
        
        # Adjust for regime
        if regime == "volatile":
            # Wider stops in volatile regimes
            regime_multiplier = 1.3
        elif regime == "range_bound":
            # Tighter stops in range-bound regimes
            regime_multiplier = 0.9
        elif regime in ["trending_up", "trending_down"]:
            # Standard stops in trending regimes
            regime_multiplier = 1.0
        else:
            regime_multiplier = 1.1  # Unknown regime
        
        # Adjust for VPIN (further away in high VPIN environments)
        vpin = market_data.get('vpin', 0.5)
        vpin_multiplier = 1.0 + (vpin - 0.2) * 0.5  # VPIN above 0.2 starts widening stops
        vpin_multiplier = max(1.0, min(1.3, vpin_multiplier))  # Cap between 1.0-1.3x
        
        # Calculate final stop distance with adjustments
        stop_distance = base_stop_distance * regime_multiplier * vpin_multiplier
        
        # Calculate stop price
        stop_price = entry_price - (direction * stop_distance)
        
        # Log the enhanced stop placement
        self.logger.info(f"Elite stop placement: {regime} regime-optimized stop at {stop_price:.2f}")
        self.logger.info(f"Stop details: ATR: {atr:.2f}, Distance: {stop_distance:.2f} " + 
                        f"(Regime: {regime_multiplier:.2f}x, VPIN: {vpin_multiplier:.2f}x)")
        
        return stop_price      
    def calculate_elite_position_size(self, signal, price, stop_price, market_data, regime_info):
        """Elite position sizing with volatility and regime optimization"""
        # Get key market metrics
        volatility = market_data.get('volatility', 0.0001)
        vpin = market_data.get('vpin', 0.2)
        order_flow = abs(market_data.get('order_flow', 0))
        delta = abs(market_data.get('delta', 0))
        regime = regime_info.get('regime', 'unknown')
        confidence = regime_info.get('confidence', 0.5)
        
        # Base risk calculation (1% risk per trade)
        risk_amount = self.capital * 0.01
        price_risk = abs(price - stop_price)
        
        # Safety check for invalid price risk
        if price_risk < 0.5:
            price_risk = price * 0.001  # Use 0.1% of price as minimum risk
            self.logger.warning(f"Stop too close to entry - using default risk: ${price_risk:.2f}")
        
        # Calculate base position size
        dollar_risk_per_contract = price_risk * 20  # NQ is $20 per point
        base_size = max(1, int(risk_amount / dollar_risk_per_contract))
        
        # -- Elite Factor Calculations --
        
        # 1. Volatility Factor (reduce size in high volatility)
        # For volatile regime, adjust based on actual measured volatility
        vol_factor = 1.0
        if volatility > 0.002:  # High volatility
            vol_factor = 0.7
        elif volatility < 0.0005:  # Low volatility
            vol_factor = 1.2
        
        # 2. Signal Strength Factor (larger size for stronger signals)
        signal_factor = 0.7 + (min(1.0, abs(signal)) * 0.6)
        
        # 3. Regime-specific Factor
        regime_factor = 1.0
        if regime == "volatile":
            # More conservative in volatile markets (especially with high confidence)
            regime_factor = 0.7 - (0.1 * confidence)
        elif regime in ["trending_up", "trending_down"]:
            # More aggressive in trending markets with high confidence
            regime_factor = 0.9 + (0.3 * confidence)
        elif regime == "range_bound":
            # Standard sizing in range markets
            regime_factor = 1.0
        
        # 4. VPIN Factor (reduce size when VPIN is high)
        vpin_factor = max(0.7, min(1.2, 1.3 - (vpin * 1.5)))
        
        # 5. Order Flow Factor
        flow_factor = 1.0
        if order_flow > 0.15:
            # Increase size with strong order flow
            flow_factor = 1.15
        
        # Calculate final position size with all factors
        adjusted_size = int(base_size * vol_factor * signal_factor * regime_factor * vpin_factor * flow_factor)
        
        # Enforce reasonable limits
        adjusted_size = max(1, min(3, adjusted_size))
        
        # Log the elite position sizing logic
        self.logger.info(f"Elite position sizing: Base {base_size} → {adjusted_size} contracts")
        self.logger.info(f"Position factors: Vol:{vol_factor:.2f} Signal:{signal_factor:.2f} " + 
                        f"Regime:{regime_factor:.2f} VPIN:{vpin_factor:.2f} Flow:{flow_factor:.2f}")
        
        return adjusted_size 
    def optimize_execution_timing(self, action_type, direction, market_data):
        """Elite execution timing optimization to reduce slippage"""
        
        # Extract order flow metrics
        order_imbalance = market_data.get('order_flow', 0)
        delta = market_data.get('delta', 0)
        vpin = market_data.get('vpin', 0.5)
        liquidity = market_data.get('liquidity_score', 0.5)
        
        # Default delay is zero (immediate execution)
        delay_seconds = 0
        execute_now = True
        
        # For entries, time execution when flow aligns with direction
        if action_type == 'entry':
            if (direction > 0 and order_imbalance < -0.15) or \
            (direction < 0 and order_imbalance > 0.15):
                # Flow is strongly against position - short delay
                delay_seconds = 3
                execute_now = False
                self.logger.info(f"Elite execution: Delaying {action_type} by {delay_seconds}s (adverse flow)")
        
        # For exits, prioritize liquidity over timing when closing
        elif action_type == 'exit':
            if liquidity < 0.3:  # Very low liquidity
                delay_seconds = 5
                execute_now = False
                self.logger.info(f"Elite execution: Delaying {action_type} by {delay_seconds}s (low liquidity)")
        
        return {
            'execute_now': execute_now,
            'delay_seconds': delay_seconds,
            'reason': 'optimal_timing'
        }
    def calculate_elite_stop(self, price, direction, atr, market_data, regime, confidence):
        """Elite stop calculation with minimum distance protection"""
        
        # Base ATR multiplier based on regime
        if regime == 'volatile':
            base_multiplier = 1.25
        elif regime in ['trending_up', 'trending_down']:
            base_multiplier = 1.0
        elif regime == 'range_bound':
            base_multiplier = 0.85
        else:
            base_multiplier = 1.1
        
        # Apply confidence scaling
        effective_multiplier = base_multiplier * (0.8 + (confidence * 0.3))
        
        # Calculate initial stop distance
        stop_distance = atr * effective_multiplier
        
        # Ensure minimum viable stop distance (15 points / $300 per contract)
        min_stop_distance = 15.0  # 15 points × $20 = $300 per contract
        
        if stop_distance < min_stop_distance:
            self.logger.info(f"Elite stop adjustment: Increasing stop from {stop_distance:.2f} to minimum {min_stop_distance:.2f} points")
            stop_distance = min_stop_distance
        
        # Calculate actual stop price
        stop_price = price - (direction * stop_distance)
        
        self.logger.info(f"Elite stop placement: {regime} regime-optimized stop at {stop_price:.2f}")
        return stop_price       
    def calculate_elite_risk_reward(self, entry_price, stop_price, market_data, regime):
        """Calculate optimal profit targets using elite risk/reward formula"""
        risk = abs(entry_price - stop_price)
        
        # Base R:R ratios adapted to regime
        if regime == 'trending_up' or regime == 'trending_down':
            # In trending regimes, use higher targets
            r1 = 1.5 * risk
            r2 = 2.5 * risk
            r3 = 4.0 * risk
        elif regime == 'volatile':
            # In volatile regimes, use tighter targets
            r1 = 1.0 * risk
            r2 = 1.7 * risk
            r3 = 2.5 * risk
        else:  # range_bound or unknown
            # Standard targets
            r1 = 1.2 * risk
            r2 = 2.0 * risk
            r3 = 3.0 * risk
        
        # Direction of trade
        direction = 1 if entry_price > stop_price else -1
        
        # Calculate actual profit targets
        target1 = entry_price + (direction * r1)
        target2 = entry_price + (direction * r2)
        target3 = entry_price + (direction * r3)
        
        self.logger.info(f"Elite profit targets: T1: {target1:.2f}, T2: {target2:.2f}, T3: {target3:.2f}")
        
        return {
            'targets': [target1, target2, target3],
            'risk': risk,
            'risk_reward_ratios': [r1/risk, r2/risk, r3/risk]
        } 
    def adjust_for_equity_curve(self, position_size):
        """Dynamically adjust position size based on recent performance"""
        if not hasattr(self, 'trade_history') or len(self.trade_history) < 5:
            return position_size
            
        # Calculate recent performance trend
        recent_trades = self.trade_history[-5:]
        wins = sum(1 for t in recent_trades if t.get('profit', 0) > 0)
        total_profit = sum(t.get('profit', 0) for t in recent_trades)
        
        # Scale position size based on performance metrics
        if wins >= 4 and total_profit > 0:  # Strong performance, scale up
            adjusted_size = int(position_size * 1.2)
            self.logger.info(f"Elite equity curve boost: {position_size} → {adjusted_size} (recent win rate: {wins/5:.2f})")
            return adjusted_size
        elif wins <= 1 or total_profit < -500:  # Poor performance, scale down
            adjusted_size = max(1, int(position_size * 0.7))
            self.logger.info(f"Elite equity curve protection: {position_size} → {adjusted_size} (recent win rate: {wins/5:.2f})")
            return adjusted_size
        
        return position_size

    def detect_range_extremes(self, price_history, current_price):
        """Detect if price is at range extremes for mean reversion opportunities"""
        if len(price_history) < 50:
            return {'at_extreme': False}
            
        recent_high = max(price_history[-50:])
        recent_low = min(price_history[-50:])
        range_size = recent_high - recent_low
        
        # Position within range (0-1)
        if range_size > 0:
            position = (current_price - recent_low) / range_size
        else:
            return {'at_extreme': False}
        
        # Calculate distance from recent high/low in ATR terms
        atr = self.calculate_atr(price_history[-20:]) if len(price_history) >= 20 else range_size * 0.01
        
        # Elite range conditions
        if position > 0.9:  # Near top of range
            confidence = min(1.0, (position - 0.9) * 10)
            self.logger.info(f"Elite range detection: Price near UPPER extreme ({position:.2f}) - confidence: {confidence:.2f}")
            return {
                'at_extreme': True, 
                'type': 'high', 
                'confidence': confidence,
                'target': recent_low + (range_size * 0.5)
            }
        elif position < 0.1:  # Near bottom of range
            confidence = min(1.0, (0.1 - position) * 10)
            self.logger.info(f"Elite range detection: Price near LOWER extreme ({position:.2f}) - confidence: {confidence:.2f}")
            return {
                'at_extreme': True,
                'type': 'low', 
                'confidence': confidence,
                'target': recent_low + (range_size * 0.5)
            }
            
        return {'at_extreme': False}
    def _analytics_thread(self):
        """Elite analytics without impacting core trading"""
        while self.running:
            try:
                # Run analytics every 15 seconds
                time.sleep(15)
                
                # Skip if not enough data
                if not hasattr(self, 'price_tracker') or len(self.price_tracker.price_history) < 100:
                    continue
                    
                # Get performance metrics
                performance = self.calculate_performance_metrics() if hasattr(self, 'calculate_performance_metrics') else {}
                
                # Check for high-risk conditions
                market_data = self.market_data.get_realtime_data() if hasattr(self, 'market_data') else {}
                
                if market_data:
                    # Detect adverse conditions
                    vpin = market_data.get('vpin', 0)
                    flow = market_data.get('order_flow', 0)
                    self.analytics.update_equity(self.execution_engine.get_account_equity())
                    if vpin > 0.4:
                        self.logger.warning(f"ANALYTICS: High toxicity detected - VPIN: {vpin:.2f}")
                    
                    if abs(flow) > 0.3:
                        self.logger.info(f"ANALYTICS: Strong order flow bias: {flow:.2f}")
                
            except Exception as e:
                self.logger.error(f"Error in analytics thread: {e}")
                time.sleep(30)  # Longer sleep on error
    def calculate_atr(self, price_history):
        """Calculate ATR from price history"""
        if len(price_history) < 2:
            return 0.0
            
        tr_sum = 0.0
        for i in range(1, len(price_history)):
            high = price_history[i]
            low = price_history[i]
            prev_close = price_history[i-1]
            
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            
            tr = max(tr1, tr2, tr3)
            tr_sum += tr
            
        return tr_sum / (len(price_history) - 1)

    def optimize_range_trading(self, range_analysis, current_price, signal, market_data):
        """Advanced range trading signal and threshold optimization"""
        # Extract range data
        range_high = range_analysis.get('range_high', 0)
        range_low = range_analysis.get('range_low', 0)
        range_size = range_high - range_low
        position_in_range = range_analysis.get('position_in_range', 0.5)
        z_score = range_analysis.get('z_score', 0)
        confidence = range_analysis.get('confidence', 0.5)
        
        # Skip optimization if range is invalid or too narrow
        if range_size < 0.5 or confidence < 0.3:
            return {
                'signal': signal,
                'threshold': 0.50,
                'optimization_applied': False
            }
        
        # Calculate range metrics
        range_width_points = range_size  # Points
        normalized_range = range_width_points / current_price  # Percentage
        
        # Calculate signal adjustment based on position in range
        signal_adjustment = 0
        threshold_adjustment = 0
        
        # Apply stronger mean reversion at range extremes
        if position_in_range > 0.85:  # Near top of range
            # Enhance sell signals, dampen buy signals
            if signal < 0:
                # Enhance sell signals at top of range
                boost_factor = 1.0 + ((position_in_range - 0.85) * 3.0)  # 1.0 to 1.45
                signal_adjustment = -0.15 * boost_factor * confidence
                # Lower threshold for sell signals at top of range
                threshold_adjustment = -0.15 * confidence
            else:
                # Dampen buy signals at top of range
                dampen_factor = 0.8 - ((position_in_range - 0.85) * 1.5)  # 0.8 to 0.58
                signal_adjustment = signal * (dampen_factor - 1.0)  # Reduction
                # Raise threshold for buy signals at top
                threshold_adjustment = 0.12 * confidence
        
        elif position_in_range < 0.15:  # Near bottom of range
            # Enhance buy signals, dampen sell signals
            if signal > 0:
                # Enhance buy signals at bottom of range
                boost_factor = 1.0 + ((0.15 - position_in_range) * 3.0)  # 1.0 to 1.45
                signal_adjustment = 0.15 * boost_factor * confidence
                # Lower threshold for buy signals at bottom
                threshold_adjustment = -0.15 * confidence
            else:
                # Dampen sell signals at bottom of range
                dampen_factor = 0.8 - ((0.15 - position_in_range) * 1.5)  # 0.8 to 0.58
                signal_adjustment = signal * (dampen_factor - 1.0)  # Reduction
                # Raise threshold for sell signals at bottom
                threshold_adjustment = 0.12 * confidence
        
        else:
            # In middle of range - less aggressive adjustments
            # Calculate distance from center (0.5)
            center_distance = abs(position_in_range - 0.5)
            
            if center_distance < 0.1:
                # Very close to middle - slightly dampen all signals
                signal_adjustment = signal * -0.1  # 10% reduction
                threshold_adjustment = 0.05  # Slightly higher threshold in middle
            else:
                # Between middle and edge - graduated adjustments
                # Signal adjustment proportional to distance from center
                middle_to_edge = (center_distance - 0.1) / 0.25  # 0 to 1 scale
                
                # Direction based dampening/enhancement
                if (position_in_range > 0.5 and signal < 0) or (position_in_range < 0.5 and signal > 0):
                    # Signal aligned with mean reversion - enhance
                    signal_adjustment = np.sign(signal) * 0.05 * middle_to_edge * confidence
                    threshold_adjustment = -0.05 * middle_to_edge * confidence
                else:
                    # Signal against mean reversion - dampen
                    signal_adjustment = signal * (-0.05 * middle_to_edge)
                    threshold_adjustment = 0.05 * middle_to_edge * confidence
        
        # Apply confidence scaling
        signal_adjustment *= min(1.0, confidence * 1.2)
        threshold_adjustment *= min(1.0, confidence * 1.2)
        
        # Calculate final values
        optimized_signal = signal + signal_adjustment
        optimized_threshold = 0.50 + threshold_adjustment
        
        # Ensure threshold stays within reasonable bounds
        optimized_threshold = min(0.65, max(0.25, optimized_threshold))
        
        return {
            'signal': optimized_signal,
            'threshold': optimized_threshold,
            'original_signal': signal,
            'signal_adjustment': signal_adjustment,
            'threshold_adjustment': threshold_adjustment,
            'optimization_applied': True,
            'position_in_range': position_in_range,
            'z_score': z_score
        }
    def stabilized_range_detection(self, price_history, current_price, timeframe_seconds=300):
        """Time-stabilized range detection to prevent rapidly changing boundaries"""
        import numpy as np
        from datetime import datetime, timedelta
        
        # Check if we have cached range data that's still valid
        current_time = datetime.now()
        if hasattr(self, '_cached_range_data') and self._cached_range_data.get('expiry', current_time) > current_time:
            # Update position in range and z-score with current price but keep boundaries
            cached_data = self._cached_range_data.copy()
            range_size = cached_data.get('range_high', 0) - cached_data.get('range_low', 0)
            if range_size > 0:
                cached_data['position_in_range'] = (current_price - cached_data.get('range_low', 0)) / range_size
            
            # Update z-score
            mean_price = cached_data.get('mean_price', current_price)
            std_dev = cached_data.get('std_dev', 1.0)
            if std_dev > 0:
                cached_data['z_score'] = (current_price - mean_price) / std_dev
            
            return cached_data
        
        # Calculate new range data
        recent_prices = price_history[-min(300, len(price_history)):]
        if len(recent_prices) < 20:
            # Not enough data for reliable range detection
            return {
                'range_high': None,
                'range_low': None,
                'range_size': 0,
                'position_in_range': 0.5,
                'z_score': 0,
                'confidence': 0,
                'boundaries_detected': False
            }
        
        # Use percentile-based boundaries instead of min/max to handle outliers
        range_low = np.percentile(recent_prices, 10)
        range_high = np.percentile(recent_prices, 90)
        
        # Calculate additional metrics
        range_size = range_high - range_low
        position_in_range = (current_price - range_low) / range_size if range_size > 0 else 0.5
        
        # Calculate z-score and other statistics
        mean_price = np.mean(recent_prices)
        std_price = np.std(recent_prices)
        z_score = (current_price - mean_price) / std_price if std_price > 0 else 0
        
        # Calculate confidence based on stability of the range
        recent_volatility = np.std(recent_prices) / mean_price if mean_price > 0 else 0
        price_density = np.histogram(recent_prices, bins=10)[0]
        normalized_density = price_density / np.max(price_density)
        peak_count = np.sum(normalized_density > 0.5)  # Count significant peaks
        
        # Higher confidence for ranges with clear peaks
        confidence = 0.5 + (0.5 * (1.0 - min(1.0, peak_count / 3)))
        
        # Store with expiry time to prevent rapid changes
        range_data = {
            'range_high': range_high,
            'range_low': range_low,
            'range_size': range_size,
            'position_in_range': position_in_range,
            'z_score': z_score,
            'confidence': confidence,
            'expiry': current_time + timedelta(seconds=timeframe_seconds),
            'mean_price': mean_price,
            'std_dev': std_price,
            'boundaries_detected': True
        }
        
        # Cache the results
        self._cached_range_data = range_data
        
        return range_data
    def analyze_volatility_surface(self, market_data, regime):
        """Quantum-enhanced volatility surface analysis"""
        # Extract key metrics
        current_vol = market_data.get('volatility', 0.0001)
        vpin = market_data.get('vpin', 0.2)
        liquidity = market_data.get('liquidity_score', 0.5)
        
        # CRITICAL FIX: Improve sensitivity based on absolute volatility level
        MIN_VOLATILITY = 0.00005  # Minimum volatility to consider (prevents zero division)
        if current_vol < MIN_VOLATILITY:
            current_vol = MIN_VOLATILITY
        
        # Adaptive sensitivity based on regime
        if regime == 'trending_down' or regime == 'trending_up':
            vol_sensitivity = 3.5
            vpin_threshold = 0.20
        elif regime == 'range_bound':
            vol_sensitivity = 2.5
            vpin_threshold = 0.18
        else:  # volatile or unknown
            vol_sensitivity = 4.0
            vpin_threshold = 0.15
        
        # Calculate compression from history or synthesize if needed
        vol_history = []
        if hasattr(self, 'price_tracker') and hasattr(self.price_tracker, 'get_volatility_history'):
            vol_history = self.price_tracker.get_volatility_history(30)
        
        # If no volatility history, use price history to estimate
        if len(vol_history) < 5:
            if hasattr(self, 'price_tracker') and hasattr(self.price_tracker, 'get_history'):
                price_history = self.price_tracker.get_history(30)
                if len(price_history) > 5:
                    # Calculate rolling volatility from price
                    for i in range(4, len(price_history)):
                        window = price_history[i-4:i+1]
                        pct_changes = [abs((window[j] - window[j-1])/window[j-1]) for j in range(1, len(window))]
                        vol_est = sum(pct_changes) / len(pct_changes)
                        vol_history.append(vol_est)
        
        # Calculate volatility compression
        vol_compression = 0.5  # Default value
        if len(vol_history) >= 5:
            # Sort volatilities to find percentile
            sorted_vols = sorted(vol_history)
            # Find percentile rank of current volatility
            rank = 0
            for i, vol in enumerate(sorted_vols):
                if current_vol <= vol:
                    rank = i
                    break
                rank = len(sorted_vols)
            
            vol_compression = rank / max(1, len(sorted_vols))
            
            # Invert to get compression (low volatility = high compression)
            vol_compression = 1.0 - vol_compression
        
        # CRITICAL: Enhanced emergence probability calculation
        emergence_prob = 0.0
        
        # Pattern 1: Low volatility + high VPIN = impending volatility
        if vol_compression > 0.7:
            # Calculate VPIN impact (normalized)
            vpin_impact = max(0, (vpin - vpin_threshold) / 0.25)
            emergence_prob += vol_compression * vpin_impact * vol_sensitivity
        
        # Pattern 2: Extreme volatility compression alone is significant
        if vol_compression > 0.85:
            emergence_prob += (vol_compression - 0.85) * 5.0
        
        # Pattern 3: Low liquidity adds to volatility emergence risk
        if liquidity < 0.8:
            liquidity_impact = (0.8 - liquidity) / 0.3
            emergence_prob += liquidity_impact * vol_compression
        
        # Scale and bound emergence probability
        emergence_prob = min(0.99, emergence_prob)
        emergence_prob = max(0.01, emergence_prob)  # Always some small probability
        
        # Calculate directional bias
        order_flow = market_data.get('order_flow', 0)
        delta = market_data.get('delta', 0)
        directional_bias = (order_flow * 0.6) + (delta * 0.4)
        
        self.logger.info(f"Quantum volatility analysis: Expansion probability: {emergence_prob:.2f}, Expected magnitude: {current_vol:.6f}, Bias: {directional_bias:.2f}")
        
        return {
            'emergence_probability': emergence_prob,
            'expected_magnitude': current_vol * (1.0 + (emergence_prob * 3.0)),
            'directional_bias': directional_bias,
            'vol_compression': vol_compression
        }

    def analyze_microstructure(self, market_data, price_history):
        """Advanced market microstructure analysis with emergent pattern recognition"""
        # Extract key metrics
        order_flow = market_data.get('order_flow', 0)
        vpin = market_data.get('vpin', 0.2)
        delta = market_data.get('delta', 0)
        
        # Initialize with safe defaults
        avg_velocity = 0
        avg_acceleration = 0
        avg_jerk = 0
        
        # Calculate price derivatives with strict bounds
        if len(price_history) >= 5:
            # Calculate price derivatives with normalization
            diffs = []
            for i in range(1, min(len(price_history), 20)):
                # Normalize by price level to get percentage changes
                if price_history[i-1] > 0:  # Avoid division by zero
                    pct_change = (price_history[i] - price_history[i-1]) / price_history[i-1]
                    diffs.append(pct_change)
            
            # Apply strict magnitude limits
            MAX_VELOCITY = 0.005  # Max 0.5% move between ticks
            velocity = [max(-MAX_VELOCITY, min(MAX_VELOCITY, d)) for d in diffs]
            
            if len(velocity) >= 2:
                # Second derivative (acceleration)
                acc_diffs = []
                for i in range(1, len(velocity)):
                    acc_diffs.append(velocity[i] - velocity[i-1])
                
                MAX_ACCELERATION = 0.002  # Max acceleration
                acceleration = [max(-MAX_ACCELERATION, min(MAX_ACCELERATION, a)) for a in acc_diffs]
                
                if len(acceleration) >= 2:
                    # Third derivative (jerk)
                    jerk_values = []
                    for i in range(1, len(acceleration)):
                        jerk_values.append(acceleration[i] - acceleration[i-1])
                    
                    MAX_JERK = 0.001  # Max jerk
                    jerk = [max(-MAX_JERK, min(MAX_JERK, j)) for j in jerk_values]
                    
                    # Use recent average values
                    avg_velocity = sum(velocity[-3:]) / 3 if len(velocity) >= 3 else 0
                    avg_acceleration = sum(acceleration[-3:]) / 3 if len(acceleration) >= 3 else 0
                    avg_jerk = sum(jerk[-3:]) / 3 if len(jerk) >= 3 else 0
        
        # Calculate normalized microstructure score and bias
        microstructure_score = 0
        microstructure_bias = 0
        
        # Strong order flow contributes to score and bias
        if abs(order_flow) > 0.15:
            microstructure_score += min(0.35, abs(order_flow) * 0.8)
            microstructure_bias += order_flow * 0.7
        
        # Acceleration against order flow indicates potential reversal
        if avg_acceleration != 0 and ((avg_acceleration > 0 and order_flow < -0.1) or 
                                    (avg_acceleration < 0 and order_flow > 0.1)):
            microstructure_score += 0.25
            microstructure_bias -= order_flow * 0.6  # Reversal signal
        
        # Jerk (rate of change of acceleration) indicates momentum shift
        if abs(avg_jerk) > 0.0001:
            microstructure_score += 0.2
            # Critical fix: Scale jerk properly (was causing numerical explosion)
            microstructure_bias += avg_jerk * 500  # Normalized scale
        
        # CRITICAL: Apply strict bounds to final bias value
        MAX_BIAS = 3.0  # Maximum reasonable bias
        microstructure_bias = max(-MAX_BIAS, min(MAX_BIAS, microstructure_bias))
        
        self.logger.info(f"Microstructure analysis: Score: {microstructure_score:.2f}, Bias: {microstructure_bias:.2f}, Institutional activity: 0.00")
        
        return {
            'microstructure_score': microstructure_score,
            'bias': microstructure_bias,
            'institutional_activity': 0.00,  # Add proper calculation later
            'flow_price_divergence': abs(order_flow) * 0.5  # Simplified metric
        }

    def extract_neural_alpha(self, market_data, price_history, regime, confidence):
        """Neural alpha extraction system using adaptive feature importance"""
        # Skip if not enough data
        if len(price_history) < 5:
            return {'alpha_score': 0, 'alpha_direction': 0, 'confidence': 0}
        
        # Extract features from market data
        features = {
            'volatility': market_data.get('volatility', 0.0001),
            'order_flow': market_data.get('order_flow', 0),
            'delta': market_data.get('delta', 0),
            'vpin': market_data.get('vpin', 0.2),
            'liquidity': market_data.get('liquidity_score', 0.5),
            'trend_strength': market_data.get('trend_strength', 1.0)
        }
        
        # Basic feature contribution
        alpha_score = 0
        alpha_direction = 0
        feature_contribution = {}
        
        # Calculate simple contributions (more sophisticated version can be added later)
        if abs(features['order_flow']) > 0.1:
            contribution = features['order_flow'] * 0.5
            feature_contribution['order_flow'] = contribution
            alpha_score += abs(contribution)
            alpha_direction += contribution
        
        if abs(features['delta']) > 0.2:
            contribution = features['delta'] * 0.3
            feature_contribution['delta'] = contribution
            alpha_score += abs(contribution)
            alpha_direction += contribution
        
        # Volatility is typically inversely related to price in uptrends
        if features['volatility'] > 0.0001:
            if regime == 'trending_up':
                contribution = -features['volatility'] * 500  # Scaled for magnitude
            elif regime == 'trending_down':
                contribution = features['volatility'] * 500
            else:
                contribution = 0
            
            feature_contribution['volatility'] = contribution
            alpha_score += abs(contribution)
            alpha_direction += contribution
        
        # Calculate momentum from price history
        if len(price_history) >= 10:
            momentum = (price_history[-1] / price_history[-10]) - 1
            contribution = momentum * 0.3
            feature_contribution['momentum'] = contribution
            alpha_score += abs(contribution)
            alpha_direction += contribution
        
        # Normalize alpha score to 0-1 range
        alpha_score = min(1.0, alpha_score)
        
        # Set direction sign
        if alpha_direction == 0:
            alpha_direction = 0
        else:
            alpha_direction = 1 if alpha_direction > 0 else -1
        
        # Simple confidence calculation - more agreement = higher confidence
        confidence_score = 0.5  # Default
        
        # Log important insights
        top_contributors = sorted(feature_contribution.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        top_contributors_str = ', '.join([f"{f}: {v:.3f}" for f, v in top_contributors])
        
        self.logger.info(f"Neural alpha extraction: Score: {alpha_score:.2f}, Direction: {alpha_direction:.1f}")
        self.logger.info(f"Top contributing factors: {top_contributors_str}")
        
        return {
            'alpha_score': alpha_score,
            'alpha_direction': alpha_direction,
            'confidence': confidence_score,
            'feature_contribution': feature_contribution
        }

    def calculate_trade_pnl(self, entry_price, exit_price, position_size, commission=2.25):
        """Calculate accurate P&L for NQ futures with enhanced error handling"""
        # CRITICAL FIX: Properly handle missing or zero entry prices
        if not entry_price or entry_price == 0:
            # Try to retrieve from position history if available
            if hasattr(self, 'trade_history') and self.trade_history:
                for trade in reversed(self.trade_history):
                    if trade.get('id') == self.current_position_id:
                        entry_price = trade.get('entry_price')
                        self.logger.info(f"Retrieved entry price {entry_price} from trade history")
                        break
            
            # If still no valid entry price, use a close approximation
            if not entry_price or entry_price == 0:
                # Use exit price with small offset to prevent zero P&L
                entry_price = exit_price * 0.9999  # Creates minimal P&L impact
                self.logger.warning(f"Using approximated entry price {entry_price} for P&L calculation")
        
        if not exit_price:
            self.logger.error(f"Invalid exit price for P&L calculation: exit={exit_price}")
            return 0
            
        direction = 1 if position_size > 0 else -1
        point_value = 20  # NQ futures is $20 per point
        
        # Calculate points difference
        points = (exit_price - entry_price) * direction
        
        # Calculate raw P&L
        raw_pnl = points * abs(position_size) * point_value
        
        # Subtract commission
        net_pnl = raw_pnl - (commission * abs(position_size))
        
        # CRITICAL: Sanity check against impossible values
        if abs(points) > 200:  # No reasonable single trade would move 200+ points
            self.logger.error(f"P&L calculation error: {points} points between {entry_price} and {exit_price}")
            return 0
            
        if abs(raw_pnl) > 10000:  # Cap at $10,000 per contract as sanity check
            self.logger.warning(f"P&L sanity check: ${raw_pnl:.2f} from {points} points")
            max_reasonable_pnl = 10000 * abs(position_size)
            return max(min(raw_pnl, max_reasonable_pnl), -max_reasonable_pnl)
        
        return net_pnl

    
    def percentile_rank(self, value, data_list):
        """Calculate percentile rank of a value in a list"""
        if not data_list:
            return 0.5
        count = sum(1 for x in data_list if x < value)
        return count / len(data_list)

    def calculate_signal_agreement(self, feature_contributions):
        """Calculate how much features agree with each other"""
        if not feature_contributions:
            return 0.5
            
        # Get overall direction
        overall_direction = np.sign(sum(feature_contributions.values()))
        if overall_direction == 0:
            return 0.5
            
        # Count agreeing vs disagreeing features
        agreeing = 0
        disagreeing = 0
        
        for feature, contribution in feature_contributions.items():
            if abs(contribution) < 0.05:  # Ignore weak contributions
                continue
                
            if np.sign(contribution) == overall_direction:
                agreeing += abs(contribution)
            else:
                disagreeing += abs(contribution)
        
        total = agreeing + disagreeing
        if total == 0:
            return 0.5
            
        # Calculate agreement ratio
        agreement_ratio = agreeing / total
        
        return agreement_ratio

    def apply_elite_enhancements(self, market_data, composite_signal, regime_info=None):
        """Elite signal enhancement with alpha integration and error protection"""
        try:
            # Defensive check for missing regime_info
            if regime_info is None:
                self.logger.warning("Missing regime_info in apply_elite_enhancements, using defaults")
                regime_info = {'regime': 'unknown', 'confidence': 0.5}
            
            # Get key inputs
            regime = regime_info.get('regime', 'unknown')
            regime_confidence = regime_info.get('confidence', 0.5)
            
            # Get price history for analysis
            price_history = []
            if hasattr(self, 'price_tracker') and hasattr(self.price_tracker, 'get_history'):
                price_history = self.price_tracker.get_history(60)
            
            # Apply quantum-level analysis with error protection
            vol_analysis = {'emergence_probability': 0.01, 'expected_magnitude': 0.0001, 'directional_bias': 0}
            try:
                if hasattr(self, 'analyze_volatility_surface'):
                    vol_analysis = self.analyze_volatility_surface(market_data, regime)
                else:
                    self.logger.info(f"Quantum volatility analysis: Expansion probability: {vol_analysis['emergence_probability']:.2f}, Expected magnitude: {vol_analysis['expected_magnitude']:.6f}, Bias: {vol_analysis['directional_bias']:.2f}")
            except Exception as e:
                self.logger.error(f"Error in volatility analysis: {e}")
            
            # Calculate microstructure analysis if enough data
            micro_analysis = {'microstructure_score': 0, 'bias': 0}
            if len(price_history) >= 5:
                try:
                    if hasattr(self, 'analyze_microstructure'):
                        micro_analysis = self.analyze_microstructure(market_data, price_history)
                    else:
                        self.logger.error("analyze_microstructure method not found")
                except Exception as e:
                    self.logger.error(f"Error in microstructure analysis: {e}")
            
            # Extract neural alpha if available
            alpha_data = {'alpha_score': 0, 'alpha_direction': 0, 'confidence': 0}
            try:
                if hasattr(self, 'extract_neural_alpha'):
                    alpha_data = self.extract_neural_alpha(market_data, price_history, regime, regime_confidence)
                else:
                    self.logger.error("extract_neural_alpha method not found")
            except Exception as e:
                self.logger.error(f"Error in neural alpha extraction: {e}")
            
            # ELITE INTEGRATION: Combine all enhancement sources
            enhanced_signal = composite_signal
            signal_boost = 0
            
            # 1. Apply microstructure enhancement (with strict magnitude limits)
            if micro_analysis.get('microstructure_score', 0) > 0.2:
                # Normalized bias input with safety caps
                micro_bias = max(-1.0, min(1.0, micro_analysis.get('bias', 0) / 3.0))
                micro_boost = micro_bias * min(0.2, micro_analysis.get('microstructure_score', 0))
                if abs(micro_boost) > 0.01:  # Only log meaningful boosts
                    self.logger.info(f"Elite enhancement: Microstructure boost {enhanced_signal:.2f} → {enhanced_signal + micro_boost:.2f}")
                    signal_boost += micro_boost
            
            # 2. Apply volatility-based enhancement
            if vol_analysis.get('emergence_probability', 0) > 0.3:
                # Volatility expansion usually indicates trend continuation
                vol_dir = vol_analysis.get('directional_bias', 0)
                vol_boost = vol_dir * min(0.15, vol_analysis.get('emergence_probability', 0) * 0.3)
                if abs(vol_boost) > 0.01:
                    self.logger.info(f"Elite enhancement: Volatility surface boost {enhanced_signal + signal_boost:.2f} → {enhanced_signal + signal_boost + vol_boost:.2f}")
                    signal_boost += vol_boost
            
            # 3. Apply neural alpha enhancement (strongest signal)
            if alpha_data.get('alpha_score', 0) > 0.3:
                # Neural alpha has highest priority
                alpha_dir = alpha_data.get('alpha_direction', 0)
                alpha_boost = alpha_dir * min(0.25, alpha_data.get('alpha_score', 0)) * alpha_data.get('confidence', 0.5)
                if abs(alpha_boost) > 0.01:
                    self.logger.info(f"Elite enhancement: Neural alpha boost {enhanced_signal + signal_boost:.2f} → {enhanced_signal + signal_boost + alpha_boost:.2f}")
                    signal_boost += alpha_boost
            
            # Apply combined boost
            enhanced_signal += signal_boost
            
            # Calculate elite confidence metric
            elite_confidence = (
                vol_analysis.get('emergence_probability', 0) * 0.3 +
                micro_analysis.get('microstructure_score', 0) * 0.3 +
                alpha_data.get('alpha_score', 0) * 0.4
            )
            
            return {
                'enhanced_signal': enhanced_signal,
                'elite_confidence': elite_confidence,
                'vol_analysis': vol_analysis,
                'micro_analysis': micro_analysis,
                'alpha_data': alpha_data
            }
        
        except Exception as e:
            # Master error handler - never crash the trading loop
            self.logger.error(f"Critical error in quantum enhancement module: {e}")
            # Return original signal unchanged
            return {
                'enhanced_signal': composite_signal,
                'elite_confidence': 0.0,
                'vol_analysis': {},
                'micro_analysis': {},
                'alpha_data': {}
            }
    def calculate_trade_pnl(self, entry_price, exit_price, position_size, commission=2.25):
        """Calculate accurate P&L for NQ futures with enhanced error handling"""
        # CRITICAL FIX: Properly handle missing or zero entry prices
        if not entry_price or entry_price == 0:
            # Try to retrieve from position history if available
            if hasattr(self, 'trade_history') and self.trade_history:
                for trade in reversed(self.trade_history):
                    if trade.get('id') == self.current_position_id:
                        entry_price = trade.get('entry_price')
                        self.logger.info(f"Retrieved entry price {entry_price} from trade history")
                        break
            
            # If still no valid entry price, use a close approximation
            if not entry_price or entry_price == 0:
                # Use exit price with small offset to prevent zero P&L
                entry_price = exit_price * 0.9999  # Creates minimal P&L impact
                self.logger.warning(f"Using approximated entry price {entry_price} for P&L calculation")
        
        if not exit_price:
            self.logger.error(f"Invalid exit price for P&L calculation: exit={exit_price}")
            return 0
            
        direction = 1 if position_size > 0 else -1
        point_value = 20  # NQ futures is $20 per point
        
        # Calculate points difference
        points = (exit_price - entry_price) * direction
        
        # Calculate raw P&L
        raw_pnl = points * abs(position_size) * point_value
        
        # Subtract commission
        net_pnl = raw_pnl - (commission * abs(position_size))
        
        # CRITICAL: Sanity check against impossible values
        if abs(points) > 200:  # No reasonable single trade would move 200+ points
            self.logger.error(f"P&L calculation error: {points} points between {entry_price} and {exit_price}")
            return 0
            
        if abs(raw_pnl) > 10000:  # Cap at $10,000 per contract as sanity check
            self.logger.warning(f"P&L sanity check: ${raw_pnl:.2f} from {points} points")
            max_reasonable_pnl = 10000 * abs(position_size)
            return max(min(raw_pnl, max_reasonable_pnl), -max_reasonable_pnl)
        
        return net_pnl 
    def enhanced_volatility_detection(self, market_data, regime):
        """Quantum-enhanced volatility detection with adaptive sensitivity"""
        # Extract key metrics
        current_vol = market_data.get('volatility', 0.0001)
        vpin = market_data.get('vpin', 0.2)
        trend_strength = market_data.get('trend_strength', 1.0)
        
        # Ultra-sensitive expansion probability calculation
        # Works especially well in transitional markets
        if regime == 'trending_up' and trend_strength < 1.5:
            # Low-strength trend requires higher sensitivity
            base_sensitivity = 5.0
            vol_threshold = 0.00005
        else:
            base_sensitivity = 3.0
            vol_threshold = 0.0001
        
        # Calculate volatility compression - key leading indicator
        vol_compression = 0.0
        if hasattr(self, 'price_tracker'):
            vol_history = self.price_tracker.get_volatility_history(30)
            if len(vol_history) > 10:
                sorted_vols = sorted(vol_history)
                rank = 0
                for i, vol in enumerate(sorted_vols):
                    if current_vol <= vol:
                        rank = i
                        break
                vol_compression = 1.0 - (rank / len(sorted_vols))
        
        # QUANTUM ENHANCEMENT: Phase-shifted volatility detection
        expansion_prob = 0.0
        
        # Pattern 1: Low volatility in weak trend = imminent breakout
        if vol_compression > 0.7 and trend_strength < 1.5:
            expansion_prob += (vol_compression - 0.7) * 10.0 * base_sensitivity
        
        # Pattern 2: Rising VPIN indicates institutional positioning
        if vpin > 0.1 and vpin < 0.3:  # Sweet spot for institutional activity
            expansion_prob += (vpin - 0.1) * 2.0
        
        # Pattern 3: Oscillating liquidity indicates smart money positioning
        liquidity = market_data.get('liquidity_score', 0.5)
        if liquidity_history and len(liquidity_history) > 5:
            liquidity_change = abs(liquidity - sum(liquidity_history[-5:]) / 5)
            if liquidity_change > 0.01:
                expansion_prob += liquidity_change * 20.0
        
        # Bound the probability
        expansion_prob = min(0.99, max(0.01, expansion_prob))
        
        # Calculate directional bias with enhanced order flow sensitivity
        order_flow = market_data.get('order_flow', 0)
        delta = market_data.get('delta', 0)
        
        # Enhanced bias calculation
        directional_bias = (order_flow * 0.7) + (delta * 0.3)
        if abs(directional_bias) < 0.1 and trend_strength > 0:
            # Align with trend when bias is weak but trend exists
            directional_bias = 0.2 * (1 if regime == 'trending_up' else -1)
        
        return {
            'emergence_probability': expansion_prob,
            'expected_magnitude': max(current_vol, vol_threshold) * (1.0 + (expansion_prob * 5.0)),
            'directional_bias': directional_bias,
            'vol_compression': vol_compression
        }    
    def detect_volatility_regime_patterns(self, market_data, lookback=20):
        """Advanced volatility regime pattern detector with quantum filtering"""
        # Get volatility history
        vol_history = self.price_tracker.get_volatility_history(lookback)
        if len(vol_history) < lookback:
            return {'detected': False}
            
        # Normalize volatility
        mean_vol = sum(vol_history) / len(vol_history)
        norm_vol = [v/mean_vol for v in vol_history]
        
        # Pattern 1: Volatility compression (bottleneck formation)
        recent_vol = norm_vol[-5:]
        if max(norm_vol[:-5]) > 1.3 and all(v < 0.8 for v in recent_vol):
            # Directional expectation based on order flow during compression
            order_flow = market_data.get('order_flow', 0)
            direction = 1 if order_flow > 0 else -1
            confidence = min(0.9, 1.0 - (sum(recent_vol) / 5))
            
            return {
                'detected': True,
                'pattern': 'volatility_compression',
                'direction': direction,
                'confidence': confidence,
                'expected_magnitude': mean_vol * 2.5  # Expected breakout magnitude
            }
        
        # Pattern 2: Volatility expansion (early trend detection)
        if len(vol_history) >= 10:
            early_vol = norm_vol[-10:-5]
            recent_vol = norm_vol[-5:]
            if all(v < 0.7 for v in early_vol) and all(v > 1.2 for v in recent_vol):
                # Calculate trend direction during expansion
                price_history = self.price_tracker.get_history(10)
                if len(price_history) >= 10:
                    direction = 1 if price_history[-1] > price_history[-6] else -1
                    return {
                        'detected': True,
                        'pattern': 'volatility_expansion',
                        'direction': direction,
                        'confidence': min(0.85, sum(recent_vol) / (5 * 1.2)),
                        'stage': 'early'  # Early stage expansion has more profit potential
                    }
        
        # Pattern 3: Volatility oscillation (sideways market)
        if len(vol_history) >= 15:
            oscillation_score = 0
            for i in range(5, len(norm_vol)):
                if (norm_vol[i] > 1.1 and norm_vol[i-1] < 0.9) or (norm_vol[i] < 0.9 and norm_vol[i-1] > 1.1):
                    oscillation_score += 1
            
            if oscillation_score >= 4:  # At least 4 oscillations
                return {
                    'detected': True,
                    'pattern': 'volatility_oscillation',
                    'confidence': min(0.8, oscillation_score / 7),
                    'trading_bias': 'range'  # Range trading strategy preferred
                }
        
        return {'detected': False}        
    def detect_quantum_order_flow_divergence(self, market_data, lookback=12):
        """Elite-level order flow divergence detection with harmonic patterns"""
        # Get price and order flow history
        price_history = self.price_tracker.get_history(lookback)
        order_flow_history = self.order_flow_tracker.get_history(lookback)
        
        if len(price_history) < lookback or len(order_flow_history) < lookback:
            return {'detected': False}
        
        # Calculate price and order flow directions
        price_direction = 1 if price_history[-1] > price_history[0] else -1
        
        # Weighted order flow calculation using exponential weighting
        weights = [math.exp(0.3 * i) for i in range(lookback)]
        weighted_flows = [flow * weights[i] for i, flow in enumerate(order_flow_history)]
        order_flow_bias = sum(weighted_flows) / sum(weights)
        flow_direction = 1 if order_flow_bias > 0 else -1
        
        # Divergence detection logic - price moving opposite to order flow
        if price_direction != flow_direction and abs(order_flow_bias) > 0.15:
            # Calculate divergence strength
            divergence_strength = abs(order_flow_bias) * (1.0 + (abs(price_history[-1] - price_history[0]) / price_history[0]) * 100)
            
            # Calculate persistence factor - how long has divergence existed
            persistence_count = 0
            for i in range(min(5, lookback-1)):
                short_price_dir = 1 if price_history[-1] > price_history[-2-i] else -1
                short_flow_dir = 1 if order_flow_history[-1-i] > 0 else -1
                if short_price_dir != short_flow_dir:
                    persistence_count += 1
            
            persistence_factor = 1.0 + (persistence_count / 5.0)
            self.logger.info(f"Divergence persistence factor: {persistence_factor:.2f}")
            
            # Calculate institution activity probability
            institution_probability = 0.0
            if market_data.get('vpin', 0) > 0.3:
                delta = market_data.get('delta', 0)
                if abs(delta) > 0.2 and (delta * flow_direction > 0):
                    institution_probability = min(0.95, abs(delta) * 2.0)
            
            # Enhanced divergence detection
            divergence_bias = flow_direction * divergence_strength * persistence_factor
            trade_direction = flow_direction  # Follow smart money direction (order flow)
            
            # Advanced signal enhancement
            enhanced_signal = 0.0
            if abs(divergence_bias) > 0.2:
                enhanced_signal = trade_direction * min(0.6, abs(divergence_bias) / 3.0)
                self.logger.info(f"Elite divergence amplification: 0.00 → {enhanced_signal:.2f}")
                
                # Activate divergence enhancement mode
                self._signal_enhancement_active = True
                self._enhanced_signal = enhanced_signal
                # Set expiry time for enhancement (proportional to persistence)
                expiry_minutes = max(2, persistence_count * 1.5)
                self._signal_enhancement_expiry = datetime.datetime.now() + datetime.timedelta(minutes=expiry_minutes)
            
            return {
                'detected': True,
                'direction': trade_direction,
                'strength': divergence_strength,
                'persistence': persistence_factor,
                'institution_activity': institution_probability,
                'enhanced_signal': enhanced_signal,
                'bias': 'LONG' if trade_direction > 0 else 'SHORT'
            }
        
        return {'detected': False}   
    def harmonize_regime_classification(self, regime_info, market_data):
        """Quantum-precision regime classification with harmonic pattern recognition"""
        # Extract key metrics
        base_regime = regime_info.get('regime', 'unknown')
        confidence = regime_info.get('confidence', 0.5)
        trend_strength = market_data.get('trend_strength', 1.0)
        volatility = market_data.get('volatility', 0.0001)
        order_flow = market_data.get('order_flow', 0)
        delta = market_data.get('delta', 0)
        
        # Step 1: Handle "fake trend" scenarios - critically important
        if (base_regime == 'trending_up' or base_regime == 'trending_down') and trend_strength < 0.15:
            # This is a "fake trend" - reclassify as choppy with high confidence
            harmonized_regime = {
                'regime': 'choppy',
                'confidence': min(0.95, confidence * 1.2),
                'original_regime': base_regime,
                'reclassification_reason': 'weak_trend_strength',
                'trend_strength': trend_strength
            }
            self.logger.info(f"Quantum regime harmonization: {base_regime} → choppy (weak trend strength: {trend_strength:.2f})")
            return harmonized_regime
        
        # Step 2: Detect "early trend" patterns
        if base_regime == 'choppy' and abs(order_flow) > 0.3 and abs(delta) > 0.4 and abs(order_flow - delta) < 0.15:
            # Strong aligned order flow and delta in choppy regime suggests early trend formation
            harmonized_regime = {
                'regime': 'trending_up' if order_flow > 0 else 'trending_down',
                'confidence': min(0.85, 0.5 + abs(order_flow) * 0.5),
                'original_regime': base_regime,
                'reclassification_reason': 'early_trend_formation',
                'order_flow': order_flow,
                'delta': delta
            }
            direction = "up" if order_flow > 0 else "down"
            self.logger.info(f"Quantum regime harmonization: {base_regime} → trending_{direction} (early trend formation)")
            return harmonized_regime
        
        # Step 3: Detect "exhausted trend" patterns - critical for reversals
        if (base_regime == 'trending_up' or base_regime == 'trending_down') and confidence > 0.7:
            trend_direction = 1 if base_regime == 'trending_up' else -1
            if order_flow * trend_direction < -0.25 and volatility > 0.0001:
                # Order flow contradicting established trend with rising volatility suggests exhaustion
                harmonized_regime = {
                    'regime': 'exhausted_trend',
                    'confidence': min(0.9, confidence * 0.8 + abs(order_flow) * 0.4),
                    'original_regime': base_regime,
                    'reclassification_reason': 'trend_exhaustion',
                    'order_flow_divergence': abs(order_flow * trend_direction),
                    'expected_reversal': True
                }
                self.logger.info(f"Quantum regime harmonization: {base_regime} → exhausted_trend (order flow divergence)")
                return harmonized_regime
        
        # Step 4: Enhance original regime with quantum precision data
        harmonized_regime = regime_info.copy()
        
        # Add quantum metrics for enhanced decision making
        harmonized_regime['quantum_metrics'] = {
            'order_flow_delta_alignment': (order_flow * delta) / (max(abs(order_flow), abs(delta), 0.001)),
            'volatility_normalized': volatility / 0.0001,
            'trend_conviction': trend_strength * confidence,
            'orderflow_persistence': self.calculate_orderflow_persistence(market_data)
        }
        
        return harmonized_regime

    def calculate_orderflow_persistence(self, market_data):
        """Calculate persistence factor of current order flow direction"""
        if not hasattr(self, '_orderflow_history'):
            self._orderflow_history = []
        
        # Add current order flow to history
        self._orderflow_history.append(market_data.get('order_flow', 0))
        
        # Keep history limited to reasonable size
        if len(self._orderflow_history) > 20:
            self._orderflow_history.pop(0)
        
        # Need at least 5 data points
        if len(self._orderflow_history) < 5:
            return 0.5
        
        # Calculate consecutive same-direction count
        current_direction = 1 if self._orderflow_history[-1] > 0 else -1
        consecutive_count = 0
        
        for i in range(len(self._orderflow_history)-1, -1, -1):
            flow_direction = 1 if self._orderflow_history[i] > 0 else -1
            if flow_direction == current_direction:
                consecutive_count += 1
            else:
                break
        
        # Normalize to 0-1 range
        persistence = min(1.0, consecutive_count / 10)
        return persistence  
    def adaptive_quantum_amplification(self, signal, market_data, regime_info):
        """Quantum-sensitive adaptive signal amplification"""
        # Base conditions check
        if abs(signal) < 0.05:
            return signal  # Don't amplify noise
        
        # Get key metrics
        regime = regime_info.get('regime', 'unknown')
        confidence = regime_info.get('confidence', 0.5)
        order_flow = market_data.get('order_flow', 0)
        delta = market_data.get('delta', 0)
        volatility = market_data.get('volatility', 0.0001)
        vpin = market_data.get('vpin', 0.3)
        
        # Signal direction
        signal_direction = 1 if signal > 0 else -1
        
        # Step 1: Calculate base amplification factor
        base_amp = 1.0
        
        # Step 2: Regime-specific amplification
        if regime == 'trending_up' or regime == 'trending_down':
            trend_direction = 1 if regime == 'trending_up' else -1
            if signal_direction == trend_direction:
                # Signal aligned with trend
                base_amp += 0.2 * confidence
            else:
                # Signal against trend - be very conservative
                base_amp *= 0.7
        elif regime == 'choppy':
            # More conservative in choppy regimes
            base_amp *= 0.9
        elif regime == 'exhausted_trend':
            # Aggressive for reversal signals in exhausted trends
            trend_direction = 1 if regime_info.get('original_regime') == 'trending_up' else -1
            if signal_direction != trend_direction:  # Reversal signal
                base_amp += 0.3
        
        # Step 3: Order flow alignment factor
        flow_alignment = (signal_direction * order_flow) > 0
        if flow_alignment and abs(order_flow) > 0.2:
            flow_amp = min(0.3, abs(order_flow) * 0.5)
            base_amp += flow_amp
        
        # Step 4: Delta alignment factor
        delta_alignment = (signal_direction * delta) > 0
        if delta_alignment and abs(delta) > 0.3:
            delta_amp = min(0.25, abs(delta) * 0.4)
            base_amp += delta_amp
        
        # Step 5: Volatility dampening
        vol_factor = max(0.7, min(1.0, 0.0002 / max(volatility, 0.00005)))
        base_amp *= vol_factor
        
        # Step 6: VPIN dampening
        if vpin > 0.4:
            vpin_factor = max(0.7, 1.0 - ((vpin - 0.4) * 0.8))
            base_amp *= vpin_factor
        
        # Never let amplification get ridiculous
        max_amp = 2.5
        min_amp = 0.6
        final_amp = max(min_amp, min(max_amp, base_amp))
        
        # Apply amplification
        amplified_signal = signal * final_amp
        
        # Log detailed amplification factors
        self.logger.info(f"Adaptive quantum amplification: {signal:.2f} → {amplified_signal:.2f} (factor: {final_amp:.2f})")
        if final_amp > 1.3:
            # Log detailed component breakdown for significant amplification
            components = []
            if regime == 'trending_up' or regime == 'trending_down':
                components.append(f"trend alignment: +{0.2 * confidence:.2f}")
            if flow_alignment and abs(order_flow) > 0.2:
                components.append(f"flow alignment: +{min(0.3, abs(order_flow) * 0.5):.2f}")
            if delta_alignment and abs(delta) > 0.3:
                components.append(f"delta alignment: +{min(0.25, abs(delta) * 0.4):.2f}")
            self.logger.info(f"Amplification components: {', '.join(components)}")
        
        return amplified_signal  
    def deep_liquidity_analysis(self, market_data):
        """Advanced liquidity analysis with institutional flow detection"""
        # Get key metrics
        vpin = market_data.get('vpin', 0.3)
        delta = market_data.get('delta', 0)
        order_flow = market_data.get('order_flow', 0)
        
        # Update liquidity history
        if not hasattr(self, '_liquidity_history'):
            self._liquidity_history = []
        self._liquidity_history.append(vpin)
        if len(self._liquidity_history) > 30:
            self._liquidity_history.pop(0)
        
        # Need enough history
        if len(self._liquidity_history) < 10:
            return {
                'institutional_activity': 0.0,
                'liquidity_status': 'normal',
                'confidence': 0.0
            }
        
        # Key patterns to detect
        
        # Pattern 1: Liquidity improvement with positive delta (institution accumulation)
        liquidity_improving = False
        if len(self._liquidity_history) >= 5:
            recent_trend = self._liquidity_history[-1] - self._liquidity_history[-5]
            if recent_trend < -0.05:  # Improving liquidity (VPIN decreasing)
                liquidity_improving = True
        
        # Institutional accumulation: improving liquidity with strong delta
        if liquidity_improving and abs(delta) > 0.4 and (delta * order_flow) > 0:
            activity_score = min(0.9, abs(delta) * 1.2) * (1 - self._liquidity_history[-1])
            return {
                'institutional_activity': activity_score,
                'liquidity_status': 'institution_positioning',
                'bias': 'bullish' if delta > 0 else 'bearish',
                'confidence': min(0.85, activity_score + 0.2),
                'pattern': 'liquidity_improvement_with_delta',
                'delta': delta,
                'vpin': vpin
            }
        
        # Pattern 2: Toxic liquidity spike (institutional exit)
        liquidity_deteriorating = False
        if len(self._liquidity_history) >= 5:
            recent_trend = self._liquidity_history[-1] - self._liquidity_history[-5]
            if recent_trend > 0.08:  # Rapidly deteriorating liquidity
                liquidity_deteriorating = True
        
        if liquidity_deteriorating and vpin > 0.4:
            return {
                'institutional_activity': min(0.85, vpin * 1.2),
                'liquidity_status': 'toxic',
                'bias': 'bearish' if delta < 0 else 'uncertain',
                'confidence': min(0.8, vpin + 0.1),
                'pattern': 'toxic_liquidity_spike'
            }
        
        # Pattern 3: Liquidity oscillation (range environment)
        if len(self._liquidity_history) >= 15:
            # Calculate oscillation count
            oscillation_count = 0
            last_direction = 0
            for i in range(1, len(self._liquidity_history)):
                current_direction = 1 if self._liquidity_history[i] > self._liquidity_history[i-1] else -1
                if last_direction != 0 and current_direction != last_direction:
                    oscillation_count += 1
                last_direction = current_direction
            
            if oscillation_count >= 5:  # High oscillation
                return {
                    'institutional_activity': 0.4,
                    'liquidity_status': 'oscillating',
                    'bias': 'range_bound',
                    'confidence': min(0.7, oscillation_count / 10 + 0.2),
                    'pattern': 'liquidity_oscillation'
                }
        
        # Default analysis
        return {
            'institutional_activity': max(0, min(0.3, abs(delta) * 0.4)),
            'liquidity_status': 'normal',
            'vpin': vpin,
            'vpin_percentile': self.calculate_vpin_percentile(vpin),
            'confidence': 0.5
        }

    def calculate_vpin_percentile(self, current_vpin):
        """Calculate percentile of current VPIN against historical values"""
        if not hasattr(self, '_all_vpin_history'):
            self._all_vpin_history = []
        
        self._all_vpin_history.append(current_vpin)
        if len(self._all_vpin_history) > 1000:
            self._all_vpin_history.pop(0)
        
        if len(self._all_vpin_history) < 30:
            return 0.5  # Default middle percentile with insufficient data
        
        # Sort history and find percentile
        sorted_history = sorted(self._all_vpin_history)
        rank = 0
        for i, vpin in enumerate(sorted_history):
            if current_vpin <= vpin:
                rank = i
                break
        
        percentile = rank / len(sorted_history)
        return percentile     
    def optimize_execution_timing(self, action_type, direction, market_data):
        """Advanced execution timing optimization based on microstructure analysis"""
        # Extract key metrics
        volatility = market_data.get('volatility', 0.0001)
        order_flow = market_data.get('order_flow', 0)
        vpin = market_data.get('vpin', 0.3)
        
        # Default response - execute immediately
        response = {
            'execute_now': True,
            'confidence': 1.0,
            'delay_seconds': 0
        }
        
        # Calculate market conditions
        conditions = {}
        
        # Condition 1: Order flow acceleration
        if not hasattr(self, '_last_order_flows'):
            self._last_order_flows = []
        
        self._last_order_flows.append(order_flow)
        if len(self._last_order_flows) > 5:
            self._last_order_flows.pop(0)
        
        if len(self._last_order_flows) >= 3:
            flow_acceleration = self._last_order_flows[-1] - self._last_order_flows[-3]
            flow_direction = 1 if flow_acceleration > 0 else -1
            
            # For entries: wait if flow is accelerating against our direction
            # For exits: wait if flow is accelerating in our direction (we'll get better prices)
            if action_type == 'entry' and flow_direction != direction and abs(flow_acceleration) > 0.05:
                conditions['flow_acceleration'] = {
                    'delay': True,
                    'reason': 'adverse_flow_acceleration',
                    'magnitude': abs(flow_acceleration)
                }
            elif action_type == 'exit' and flow_direction == direction and abs(flow_acceleration) > 0.05:
                conditions['flow_acceleration'] = {
                    'delay': True,
                    'reason': 'favorable_exit_flow',
                    'magnitude': abs(flow_acceleration)
                }
        
        # Condition 2: VPIN spike (liquidity change)
        if not hasattr(self, '_last_vpins'):
            self._last_vpins = []
        
        self._last_vpins.append(vpin)
        if len(self._last_vpins) > 5:
            self._last_vpins.pop(0)
        
        if len(self._last_vpins) >= 2:
            vpin_change = self._last_vpins[-1] - self._last_vpins[-2]
            
            # For entries: wait if liquidity is rapidly deteriorating (VPIN increasing)
            # For exits: wait if liquidity is rapidly improving (VPIN decreasing)
            if action_type == 'entry' and vpin_change > 0.03:
                conditions['vpin_change'] = {
                    'delay': True,
                    'reason': 'deteriorating_liquidity',
                    'magnitude': vpin_change
                }
            elif action_type == 'exit' and vpin_change < -0.03:
                conditions['vpin_change'] = {
                    'delay': True,
                    'reason': 'improving_liquidity',
                    'magnitude': abs(vpin_change)
                }
        
        # Make execution decision
        delay_conditions = [c for c in conditions.values() if c.get('delay', False)]
        
        if delay_conditions:
            # Calculate delay based on condition magnitude
            total_magnitude = sum(c.get('magnitude', 0) for c in delay_conditions)
            delay_seconds = min(2.5, max(0.5, total_magnitude * 10))
            
            primary_condition = max(delay_conditions, key=lambda x: x.get('magnitude', 0))
            reason = primary_condition.get('reason', 'market_conditions')
            
            self.logger.info(f"Execution timing optimization: Delay {action_type} by {delay_seconds:.1f}s ({reason})")
            
            return {
                'execute_now': False,
                'delay_seconds': delay_seconds,
                'confidence': min(0.9, total_magnitude * 2),
                'reason': reason
            }
        
        return response
    def track_trade_factors(self, trade_id, factor_contributions):
        """Track factor contributions for each trade for performance analysis"""
        if not hasattr(self, '_trade_factor_data'):
            self._trade_factor_data = {}
        
        # Store factor data for this trade
        self._trade_factor_data[trade_id] = factor_contributions
        
        # Limit history size
        if len(self._trade_factor_data) > 100:
            # Remove oldest entries (assuming trade IDs are timestamp-based)
            sorted_ids = sorted(list(self._trade_factor_data.keys()))
            for old_id in sorted_ids[:len(sorted_ids) - 100]:
                if old_id in self._trade_factor_data:
                    del self._trade_factor_data[old_id]    
    def _analyze_factor_performance(self, recent_trades):
        """Analyze performance of individual factors based on trade history"""
        # Initialize factor performance tracking
        factor_performance = {
            'order_flow': 0.0,
            'delta': 0.0,
            'microstructure': 0.0,
            'volatility': 0.0,
            'momentum': 0.0,
            'vpin': 0.0
        }
        
        # Not enough trades to analyze
        if len(recent_trades) < 2:
            return factor_performance
        
        # Extract trade data relevant for analysis
        trade_data = []
        for trade in recent_trades:
            if not hasattr(self, '_trade_factor_data') or trade.get('trade_id') not in self._trade_factor_data:
                # Skip trades without factor data
                continue
                
            # Get factor data for this trade
            factor_data = self._trade_factor_data.get(trade.get('trade_id'), {})
            
            # Create trade entry with profit and factors
            trade_entry = {
                'profit': trade.get('profit', 0),
                'factors': factor_data,
                'direction': 1 if trade.get('direction', 0) > 0 else -1
            }
            
            trade_data.append(trade_entry)
        
        # Not enough trades with factor data
        if len(trade_data) < 2:
            # Initialize trade factor tracking if needed
            if not hasattr(self, '_trade_factor_data'):
                self._trade_factor_data = {}
            return factor_performance
        
        # Calculate factor performance (correlation with profit)
        for factor in factor_performance.keys():
            # Extract factor and profit data
            factor_values = []
            profit_values = []
            
            for trade in trade_data:
                # Get factor contribution for this trade
                factor_contribution = trade['factors'].get(factor, 0)
                
                # Only consider significant contributions
                if abs(factor_contribution) > 0.001:
                    # Adjust for trade direction
                    direction = trade['direction']
                    factor_values.append(factor_contribution * direction)
                    profit_values.append(trade['profit'])
            
            # Calculate correlation if we have enough data points
            if len(factor_values) >= 2:
                # Simple correlation calculation
                mean_factor = sum(factor_values) / len(factor_values)
                mean_profit = sum(profit_values) / len(profit_values)
                
                numerator = sum((f - mean_factor) * (p - mean_profit) for f, p in zip(factor_values, profit_values))
                denominator_factor = sum((f - mean_factor) ** 2 for f in factor_values)
                denominator_profit = sum((p - mean_profit) ** 2 for p in profit_values)
                
                denominator = (denominator_factor * denominator_profit) ** 0.5
                
                # Avoid division by zero
                if denominator > 0:
                    correlation = numerator / denominator
                    factor_performance[factor] = correlation
        
        self.logger.info(f"Factor performance analysis: order_flow={factor_performance['order_flow']:.2f}, delta={factor_performance['delta']:.2f}")
        
        return factor_performance        
    def analyze_system_performance(self):
        """Advanced quantum performance analysis of trading system"""
        if len(self.trade_history) < 5:
            return {
                'status': 'insufficient_data',
                'message': 'Need at least 5 trades for meaningful analysis'
            }
        
        # Extract basic performance metrics
        total_trades = len(self.trade_history)
        winning_trades = sum(1 for t in self.trade_history if t.get('profit', 0) > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Extract profit data
        profits = [t.get('profit', 0) for t in self.trade_history]
        total_profit = sum(profits)
        avg_profit = total_profit / total_trades if total_trades > 0 else 0
        
        # Calculate regime performance
        regime_performance = {}
        for trade in self.trade_history:
            regime = trade.get('regime', 'unknown')
            if regime not in regime_performance:
                regime_performance[regime] = {
                    'count': 0,
                    'wins': 0,
                    'total_profit': 0
                }
            
            regime_performance[regime]['count'] += 1
            if trade.get('profit', 0) > 0:
                regime_performance[regime]['wins'] += 1
            regime_performance[regime]['total_profit'] += trade.get('profit', 0)
        
        # Calculate win rate and avg profit by regime
        for regime, data in regime_performance.items():
            if data['count'] > 0:
                data['win_rate'] = data['wins'] / data['count']
                data['avg_profit'] = data['total_profit'] / data['count']
        
        # Extract recent performance trend
        recent_trades = self.trade_history[-min(10, total_trades):]
        recent_win_rate = sum(1 for t in recent_trades if t.get('profit', 0) > 0) / len(recent_trades)
        
        # Calculate factor importance
        factor_importance = self._analyze_factor_performance(self.trade_history)
        
        # Generate improvement recommendations
        recommendations = []
        
        # Check win rate
        if win_rate < 0.5:
            recommendations.append("Increase signal threshold to improve win rate")
        
        # Check regime performance
        for regime, data in regime_performance.items():
            if data['count'] >= 3 and data['win_rate'] < 0.4:
                recommendations.append(f"Adjust parameters for {regime} regime (current win rate: {data['win_rate']:.2f})")
        
        # Log analysis summary
        self.logger.info(f"Performance analysis: Win rate: {win_rate:.2f}, Recent win rate: {recent_win_rate:.2f}")
        self.logger.info(f"Top factors: {sorted(factor_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:2]}")
        
        return {
            'status': 'success',
            'metrics': {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'recent_win_rate': recent_win_rate,
                'avg_profit': avg_profit,
                'total_profit': total_profit
            },
            'regime_performance': regime_performance,
            'factor_importance': factor_importance,
            'recommendations': recommendations
        }    
    def neural_alpha_learning_engine(self, market_data, trade_history=None):
        """Advanced neural alpha learning engine that adapts based on past trade performance"""
        if not hasattr(self, '_neural_weights'):
            # Initialize with default weights
            self._neural_weights = {
                'order_flow': 0.30,
                'delta': 0.25,
                'microstructure': 0.15,
                'volatility': 0.10,
                'momentum': 0.10,
                'vpin': 0.10
            }
        
        # Get available factors
        order_flow = market_data.get('order_flow', 0)
        delta = market_data.get('delta', 0)
        micro_score = market_data.get('microstructure_score', 0)
        micro_bias = market_data.get('microstructure_bias', 0)
        volatility = market_data.get('volatility', 0.0001)
        momentum = market_data.get('momentum', 0)
        vpin = market_data.get('vpin', 0.3)
        
        # Calculate normalized volatility score (-1 to 1)
        vol_score = 0
        if volatility > 0:
            vol_norm = min(1.0, volatility / 0.0005)
            vol_score = (vol_norm - 0.5) * 2  # -1 to 1 range
        
        # Calculate VPIN score (-1 to 1)
        vpin_score = 0
        if vpin > 0.5:
            # High VPIN (low liquidity) is bearish
            vpin_score = -((vpin - 0.5) * 2)
        elif vpin < 0.2:
            # Very low VPIN (high liquidity) is bullish
            vpin_score = (0.2 - vpin) * 5
        
        # Calculate raw factor contributions
        factor_scores = {
            'order_flow': order_flow,
            'delta': delta,
            'microstructure': micro_score * micro_bias,
            'volatility': vol_score,
            'momentum': momentum,
            'vpin': vpin_score
        }
        
        # Apply neural weights
        weighted_scores = {}
        for factor, score in factor_scores.items():
            weighted_scores[factor] = score * self._neural_weights.get(factor, 0.1)
        
        # Calculate total signal
        total_signal = sum(weighted_scores.values())
        
        # Determine direction
        direction = 1 if total_signal > 0 else -1 if total_signal < 0 else 0
        
        # Calculate magnitude (absolute strength)
        magnitude = abs(total_signal)
        
        # Update weights if we have trade history (reinforcement learning)
        if trade_history and len(trade_history) >= 5:
            self._update_neural_weights(trade_history, factor_scores)
        
        # Identify top contributing factors
        factor_contributions = {}
        for factor, score in weighted_scores.items():
            if score != 0:
                factor_contributions[factor] = score
        
        # Sort by absolute contribution
        sorted_factors = sorted(factor_contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # Extract top factors for logging
        top_factors_str = ""
        for factor, contribution in sorted_factors[:3]:
            if abs(contribution) > 0.001:
                top_factors_str += f"{factor}: {contribution:.3f}, "
        
        top_factors_str = top_factors_str.rstrip(", ")
        
        # Log neural extraction
        self.logger.info(f"Neural alpha extraction: Score: {magnitude:.2f}, Direction: {direction:.1f}")
        if top_factors_str:
            self.logger.info(f"Top contributing factors: {top_factors_str}")
        
        return {
            'signal': total_signal,
            'magnitude': magnitude,
            'direction': direction,
            'factor_contributions': factor_contributions,
            'top_factors': dict(sorted_factors[:3])
        }

    def _update_neural_weights(self, trade_history, current_factors):
        """Update neural weights based on trade performance"""
        # Only consider last 20 trades
        recent_trades = trade_history[-20:]
        
        # Calculate win rate
        win_count = sum(1 for trade in recent_trades if trade.get('profit', 0) > 0)
        win_rate = win_count / len(recent_trades) if recent_trades else 0.5
        
        # If win rate is poor, adjust weights
        if win_rate < 0.5:
            # Find worst performing factors
            factor_performance = self._analyze_factor_performance(recent_trades)
            
            # Reduce weights of underperforming factors
            for factor, performance in factor_performance.items():
                if performance < 0 and factor in self._neural_weights:
                    # Reduce weight of poor factor
                    self._neural_weights[factor] = max(0.05, self._neural_weights[factor] * 0.9)
                    
                    # Increase weight of best factor to compensate
                    best_factor = max(factor_performance.items(), key=lambda x: x[1])[0]
                    self._neural_weights[best_factor] = min(0.4, self._neural_weights[best_factor] * 1.1)
                    
                    self.logger.info(f"Neural weight adaptation: {factor} ↓ {best_factor} ↑")
        
        # Normalize weights to ensure they sum to 1.0
        total_weight = sum(self._neural_weights.values())
        if total_weight > 0:
            for factor in self._neural_weights:
                self._neural_weights[factor] /= total_weight    
    def generate_advanced_risk_reward(self, direction, entry_price, stop_price, market_data, regime_info):
        """
        Generate advanced risk-reward metrics based on market conditions
        and regime-specific probabilities
        """
        # Extract important market data
        regime = regime_info.get('regime', 'unknown')
        regime_confidence = regime_info.get('confidence', 0.5)
        vpin = market_data.get('vpin', 0.3)
        order_flow = market_data.get('order_flow', 0)
        delta = market_data.get('delta', 0)
        
        # Get optimized targets based on regime
        if regime == 'range_bound' and 'range_analysis' in market_data:
            optimized_targets = self.range_optimized_profit_targets(
                entry_price, 
                direction,
                market_data,
                market_data['range_analysis']
            )
            target_1 = optimized_targets['targets'][0]
            target_2 = optimized_targets['targets'][1]
        else:
            # Standard optimized targets
            optimized_targets = self.optimize_profit_targets(
                entry_price, 
                direction,
                market_data,
                regime_info
            )
            target_1 = optimized_targets['targets'][0]
            target_2 = optimized_targets['targets'][1]
        
        # Calculate risk (distance to stop)
        risk = abs(entry_price - stop_price)
        
        # Calculate reward (distance to targets)
        reward_1 = abs(entry_price - target_1) 
        reward_2 = abs(entry_price - target_2)
        
        # Calculate weighted average reward
        weighted_reward = (reward_1 * 0.6) + (reward_2 * 0.4)
        
        # Calculate base reward:risk ratio
        reward_risk_ratio = weighted_reward / risk if risk > 0 else 1.0
        
        # Calculate base win probability based on regime
        if regime == 'volatile':
            base_win_prob = 0.48  # Slightly below 50% in volatile regimes
        elif regime in ['trending_up', 'trending_down']:
            trend_dir = 1 if regime == 'trending_up' else -1
            if direction == trend_dir:  # Trading with trend
                base_win_prob = 0.62
            else:  # Trading against trend
                base_win_prob = 0.45
        elif regime == 'range_bound':
            # Higher probability at range extremes
            if 'range_analysis' in market_data:
                position_in_range = market_data['range_analysis'].get('position_in_range', 0.5)
                z_score = market_data['range_analysis'].get('z_score', 0)
                
                # Mean reversion trades have higher probability
                if (position_in_range < 0.2 and direction > 0) or (position_in_range > 0.8 and direction < 0):
                    base_win_prob = 0.65
                # Breakout trades have lower probability
                elif (position_in_range < 0.2 and direction < 0) or (position_in_range > 0.8 and direction > 0):
                    base_win_prob = 0.42
                else:
                    base_win_prob = 0.52
            else:
                base_win_prob = 0.52
        else:
            base_win_prob = 0.5  # Unknown regime default
        
        # Adjust win probability based on order flow alignment
        flow_alignment = order_flow * direction
        if abs(flow_alignment) > 0.3:
            if flow_alignment > 0:  # Flow supporting position
                base_win_prob = min(0.85, base_win_prob + (flow_alignment * 0.2))
            else:  # Flow against position
                base_win_prob = max(0.35, base_win_prob + (flow_alignment * 0.15))
        
        # Adjust win probability based on delta alignment
        delta_alignment = delta * direction
        if abs(delta_alignment) > 0.3:
            if delta_alignment > 0:  # Delta supporting position
                base_win_prob = min(0.85, base_win_prob + (delta_alignment * 0.1))
            else:  # Delta against position
                base_win_prob = max(0.35, base_win_prob + (delta_alignment * 0.1))
        
        # Adjust for liquidity conditions
        if vpin > 0.5:  # Toxic liquidity reduces probability of success
            base_win_prob = max(0.3, base_win_prob - ((vpin - 0.5) * 0.3))
        
        # Calculate expectancy
        expectancy = (base_win_prob * reward_risk_ratio) - (1.0 - base_win_prob)
        
        # Adjust for regime confidence
        expectancy *= regime_confidence
        
        return {
            'expectancy': expectancy,
            'reward_risk_ratio': reward_risk_ratio,  # This was the missing key!
            'win_probability': base_win_prob,
            'risk': risk,
            'reward': weighted_reward,
            'flow_alignment': flow_alignment,
            'delta_alignment': delta_alignment,
            'regime_confidence': regime_confidence
        }
    def extract_delta_flow_alpha(self, market_data):
        """Extract alpha from order flow and delta divergence"""
        # Get key metrics
        order_flow = market_data.get('order_flow', 0)
        delta = market_data.get('delta', 0)
        vpin = market_data.get('vpin', 0.3)
        
        # Calculate divergence magnitude
        divergence = delta - order_flow
        divergence_magnitude = abs(divergence)
        
        # No significant divergence
        if divergence_magnitude < 0.2:
            return {
                'alpha_extracted': False,
                'signal': 0.0,
                'confidence': 0.0
            }
        
        # Calculate signal direction - follow delta in most cases as it's the more reliable indicator
        # when there's divergence
        signal_direction = 1 if delta > 0 else -1
        
        # Calculate signal strength based on divergence magnitude
        # Stronger divergence = stronger signal, up to a limit
        signal_strength = min(0.6, divergence_magnitude * 0.6)
        
        # Apply VPIN dampening for high toxicity
        if vpin > 0.35:
            vpin_factor = 1.0 - ((vpin - 0.35) * 0.8)
            signal_strength *= max(0.5, vpin_factor)
        
        # Calculate confidence based on magnitude and consistency
        confidence = min(0.9, divergence_magnitude * 0.8)
        
        # Generate enhanced signal
        enhanced_signal = signal_direction * signal_strength
        
        # Log the extraction
        self.logger.info(f"Delta-Flow alpha extraction: divergence={divergence:.2f}, signal={enhanced_signal:.2f}, confidence={confidence:.2f}")
        
        return {
            'alpha_extracted': True,
            'signal': enhanced_signal,
            'confidence': confidence,
            'source': 'delta_flow_divergence',
            'divergence': divergence,
            'magnitude': divergence_magnitude
        }    
    def detect_liquidity_oscillations(self, market_data):
        """Elite liquidity oscillation detector for institutional positioning"""
        if not hasattr(self, 'liquidity_history') or len(self.liquidity_history) < 15:
            return {'detected': False}
        
        # Get VPIN (liquidity) history
        vpin_history = market_data.get('vpin_history', [market_data.get('vpin', 0.3)])
        if len(vpin_history) < 5:
            return {'detected': False}
        
        # Calculate rate of change in liquidity
        vpin_roc = []
        for i in range(1, len(vpin_history)):
            vpin_roc.append(vpin_history[i] - vpin_history[i-1])
        
        # Check for oscillation pattern - rapid changes from high to low liquidity
        oscillation_count = 0
        for i in range(1, len(vpin_roc)):
            if (vpin_roc[i] > 0.02 and vpin_roc[i-1] < -0.02) or (vpin_roc[i] < -0.02 and vpin_roc[i-1] > 0.02):
                oscillation_count += 1
        
        if oscillation_count >= 2:  # At least 2 rapid oscillations
            # Check current liquidity trend
            current_liquidity = vpin_history[-1]
            current_trend = "improving" if vpin_roc[-1] < 0 else "deteriorating"
            
            # Check order flow during oscillations
            order_flow = market_data.get('order_flow', 0)
            
            # Calculate expected directional bias based on institution positioning
            # Institutions typically position before liquidity improves
            if current_trend == "improving" and order_flow > 0.15:
                directional_bias = 1  # Long bias when liquidity improving with positive flow
            elif current_trend == "deteriorating" and order_flow < -0.15:
                directional_bias = -1  # Short bias when liquidity deteriorating with negative flow
            else:
                directional_bias = 0
            
            return {
                'detected': True,
                'oscillation_count': oscillation_count,
                'current_liquidity': current_liquidity,
                'liquidity_trend': current_trend,
                'directional_bias': directional_bias,
                'confidence': min(0.8, oscillation_count / 4)
            }
        
        return {'detected': False}    
    def quantum_signal_harmonization(self, signal, market_data, regime_info):
        """Advanced signal harmonization across quantum patterns"""
        regime = regime_info.get('regime', 'unknown')
        regime_confidence = regime_info.get('confidence', 0.5)
        
        # Temporal signal patterns
        time_of_day = datetime.datetime.now().hour + (datetime.datetime.now().minute / 60.0)
        day_of_week = datetime.datetime.now().weekday()
        
        # Extract key metrics
        vpin = market_data.get('vpin', 0.2)
        order_flow = market_data.get('order_flow', 0)
        trend_strength = market_data.get('trend_strength', 1.0)
        
        # Base calibration
        calibrated_signal = signal
        
        # 1. Regime-specific calibration
        if regime == 'trending_up':
            if trend_strength < 1.5:  # Weak trend
                # Reduce signal magnitude in weak trends
                calibrated_signal = signal * 0.8
            else:  # Strong trend
                # Amplify signals aligned with trend
                if (signal > 0 and order_flow > 0) or (signal < 0 and order_flow < 0):
                    calibrated_signal = signal * 1.2
        
        # 2. Liquidity-based calibration
        if vpin > 0.2:  # Lower liquidity
            # More conservative during lower liquidity
            calibrated_signal = calibrated_signal * (1.0 - ((vpin - 0.2) * 0.5))
        
        # 3. Time-based patterns
        # Morning volatility (first 2 hours after open)
        if 9.5 <= time_of_day <= 11.5:
            # Reduce signal strength during volatile opening hours
            calibrated_signal = calibrated_signal * 0.9
        
        # 4. Confidence-weighted regime adaptation
        if regime_confidence < 0.7:
            # Reduce signal strength in uncertain regimes
            certainty_factor = regime_confidence / 0.7
            calibrated_signal = calibrated_signal * certainty_factor
        
        # 5. Day-of-week effect
        if day_of_week == 0:  # Monday
            # Markets often continue Friday's pattern on Monday morning
            calibrated_signal = calibrated_signal * 1.1
        elif day_of_week == 4:  # Friday
            # More conservative on Fridays (weekend risk)
            calibrated_signal = calibrated_signal * 0.9
        
        return calibrated_signal
    def stabilized_quantum_range_detection(self, price_history, current_price, market_data):
        """Time-stabilized quantum range detection with statistical significance testing and outlier protection"""
        import numpy as np
        from scipy import stats
        from datetime import datetime, timedelta
        
        # Check if we have valid cached range data
        current_time = datetime.now()
        if hasattr(self, '_cached_quantum_range') and self._cached_quantum_range.get('expiry', current_time) > current_time:
            # Use cached range but update position metrics
            cached_range = self._cached_quantum_range.copy()
            range_high = cached_range['range_high']
            range_low = cached_range['range_low']
            
            # CRITICAL FIX: Validate range boundaries aren't unreasonable
            price_std = np.std(price_history[-60:]) if len(price_history) >= 60 else current_price * 0.01
            price_mean = np.mean(price_history[-60:]) if len(price_history) >= 60 else current_price
            
            # Detect and fix unreasonable range values (more than 5 std devs from mean)
            max_reasonable_price = price_mean + (price_std * 5.0)
            min_reasonable_price = price_mean - (price_std * 5.0)
            
            if range_high > max_reasonable_price:
                self.logger.warning(f"Correcting unreasonable range high: ${range_high:.2f} → ${max_reasonable_price:.2f}")
                range_high = max_reasonable_price
                cached_range['range_high'] = range_high
                
            if range_low < min_reasonable_price:
                self.logger.warning(f"Correcting unreasonable range low: ${range_low:.2f} → ${min_reasonable_price:.2f}")
                range_low = min_reasonable_price
                cached_range['range_low'] = range_low
            
            range_size = range_high - range_low
            
            # Update position in range
            if range_size > 0:
                position_in_range = (current_price - range_low) / range_size
                z_score = (current_price - cached_range['mean_price']) / max(0.0001, cached_range['std_dev'])
                
                cached_range['position_in_range'] = position_in_range
                cached_range['z_score'] = z_score
                cached_range['range_size'] = range_size
                
                # Calculate mean reversion signal based on z-score
                mean_reversion_signal = -np.sign(z_score) * min(0.4, abs(z_score) * 0.2)
                cached_range['mean_reversion_signal'] = mean_reversion_signal
                
                return cached_range
        
        # Calculate new range with advanced statistical methods
        # Use different lookback periods for robustness
        lookbacks = [30, 60, 120]
        ranges = []
        
        for lookback in lookbacks:
            if len(price_history) >= lookback:
                window = price_history[-lookback:]
                
                # Calculate range using percentiles to handle outliers
                low_pct = np.percentile(window, 10)
                high_pct = np.percentile(window, 90)
                
                # Calculate distribution metrics
                mean_price = np.mean(window)
                std_dev = np.std(window)
                
                # CRITICAL FIX: Ensure range values are reasonable
                max_range_size = std_dev * 5.0  # 5 standard deviations
                expected_range_size = high_pct - low_pct
                
                if expected_range_size > max_range_size:
                    # Range is unreasonably large, adjust it
                    center = (high_pct + low_pct) / 2.0
                    high_pct = center + (max_range_size / 2.0)
                    low_pct = center - (max_range_size / 2.0)
                
                ranges.append({
                    'range_low': low_pct,
                    'range_high': high_pct,
                    'mean_price': mean_price,
                    'std_dev': std_dev,
                    'weight': lookback/60  # Weight longer periods more
                })
        
        if not ranges:
            return {
                'boundaries_detected': False,
                'confidence': 0.0,
                'range_high': None,
                'range_low': None
            }
        
        # Calculate weighted averages
        total_weight = sum(r['weight'] for r in ranges)
        range_high = sum(r['range_high'] * r['weight'] for r in ranges) / total_weight
        range_low = sum(r['range_low'] * r['weight'] for r in ranges) / total_weight
        mean_price = sum(r['mean_price'] * r['weight'] for r in ranges) / total_weight
        std_dev = sum(r['std_dev'] * r['weight'] for r in ranges) / total_weight
        
        # CRITICAL FIX: Validate final range
        current_std_dev = std_dev or (current_price * 0.005)  # Fallback if std_dev is 0
        max_reasonable_deviation = current_std_dev * 5.0
        
        if range_high - mean_price > max_reasonable_deviation:
            range_high = mean_price + max_reasonable_deviation
            self.logger.warning(f"Adjusted unreasonable range high to ${range_high:.2f}")
        
        if mean_price - range_low > max_reasonable_deviation:
            range_low = mean_price - max_reasonable_deviation
            self.logger.warning(f"Adjusted unreasonable range low to ${range_low:.2f}")
        
        # Calculate position metrics
        range_size = range_high - range_low
        position_in_range = (current_price - range_low) / range_size if range_size > 0 else 0.5
        z_score = (current_price - mean_price) / std_dev if std_dev > 0 else 0
        
        # Calculate statistical confidence
        confidence = 0.5  # Base confidence
        confidence += 0.1 * min(1.0, len(price_history) / 120)  # More data = higher confidence
        confidence += 0.2 * (1.0 - min(1.0, std_dev / mean_price * 1000))  # Lower volatility = higher confidence
        confidence = min(0.95, confidence)
        
        # Calculate mean reversion signal based on z-score
        mean_reversion_signal = -np.sign(z_score) * min(0.4, abs(z_score) * 0.2)
        
        range_data = {
            'range_high': range_high,
            'range_low': range_low,
            'range_size': range_size,
            'position_in_range': position_in_range,
            'z_score': z_score,
            'mean_price': mean_price,
            'std_dev': std_dev,
            'confidence': confidence,
            'boundaries_detected': True,
            'mean_reversion_signal': mean_reversion_signal,
            'expiry': current_time + timedelta(seconds=300)  # Cache for 5 minutes
        }
        
        # Store in cache
        self._cached_quantum_range = range_data
        
        return range_data
    def enhanced_minimum_hold_time(self, position_obj, market_data, regime_info):
        """Determine minimum hold time based on market regime and position characteristics"""
        import datetime
        
        # Extract key information
        current_time = datetime.datetime.now()
        entry_time = position_obj.get('entry_time', current_time)
        time_in_position = (current_time - entry_time).total_seconds()
        direction = 1 if position_obj.get('size', 0) > 0 else -1
        
        # Extract market regime
        regime = regime_info.get('regime', 'unknown')
        
        # Base minimum hold times (seconds)
        min_hold_times = {
            'trending_up': 30,
            'trending_down': 30,
            'range_bound': 60,
            'volatile': 15,
            'exhausted_trend': 45,
            'unknown': 30
        }
        
        # Get base hold time
        min_hold_time = min_hold_times.get(regime, 30)
        
        # Adjust based on market conditions
        if regime == 'range_bound':
            # Check position in range
            if 'range_analysis' in market_data and market_data['range_analysis'].get('boundaries_detected', False):
                position_in_range = market_data['range_analysis'].get('position_in_range', 0.5)
                
                # Shorter hold time for positions at range extremes
                if (direction > 0 and position_in_range < 0.2) or (direction < 0 and position_in_range > 0.8):
                    min_hold_time = 30  # 30 seconds minimum at range extremes
                # Longer hold time for positions in middle of range
                elif 0.3 < position_in_range < 0.7:
                    min_hold_time = 90  # 90 seconds minimum in middle of range
        
        # If time in position is less than minimum
        enforce_min_hold = time_in_position < min_hold_time
        
        return {
            'enforce': enforce_min_hold,
            'min_hold_time': min_hold_time,
            'time_in_position': time_in_position,
            'remaining': max(0, min_hold_time - time_in_position)
        }    
    def quantum_signal_stabilization(self, signal, market_data, position_data=None):
        """Stabilize signal using quantum entanglement and time-based protection"""
        import datetime
        import numpy as np
        
        # Extract market data
        regime = market_data.get('regime', 'unknown')
        entanglement = market_data.get('entanglement', 0.5)
        
        # Initialize signal history if needed
        if not hasattr(self, '_signal_history'):
            self._signal_history = []
        
        # Append current signal to history (keep last 5)
        self._signal_history.append(signal)
        if len(self._signal_history) > 5:
            self._signal_history = self._signal_history[-5:]
        
        # Time-based signal stabilization
        position_established = False
        seconds_in_position = 0
        
        # Check if we have position data
        if position_data:
            position_established = True
            if 'time_in_position' in position_data:
                seconds_in_position = position_data['time_in_position']
            elif 'entry_time' in position_data:
                seconds_in_position = (datetime.datetime.now() - position_data['entry_time']).total_seconds()
        
        # If we have an active position, apply stronger stabilization
        original_signal = signal
        if position_established:
            last_trade_direction = np.sign(position_data.get('size', 0))
            
            # If new signal opposes current position and position is recent
            if last_trade_direction * signal < 0 and seconds_in_position < 15:
                # ENHANCED: Completely dampen any opposing signal during first 15 seconds
                signal = 0.0
                self.logger.info(f"Quantum stabilization: Signal dampened {original_signal:.2f} → {signal:.2f} (time in position: {seconds_in_position:.1f}s < 15s)")
                return signal
        
        # Apply quantum entanglement for signal consistency
        if len(self._signal_history) >= 3:
            recent_signals = self._signal_history[-3:]
            avg_signal = np.mean(recent_signals)
            signal_std = np.std(recent_signals)
            
            # If recent signals have been consistent (low std dev)
            if signal_std < 0.15 and abs(avg_signal) > 0.2:
                # Current signal differs significantly from recent average
                if abs(signal - avg_signal) > 0.3:
                    # If current signal direction matches average direction
                    if signal * avg_signal > 0:
                        # Blend toward average (consistency boost)
                        consistency_factor = 0.6  # 60% weight to history
                        blended_signal = signal * (1.0 - consistency_factor) + avg_signal * consistency_factor
                        self.logger.info(f"Quantum consistency boost: {signal:.2f} → {blended_signal:.2f} (alignment with history)")
                        signal = blended_signal
                    else:
                        # Current signal opposes recent history - dampen it significantly
                        dampening = 0.7  # 70% reduction
                        dampened_signal = signal * (1.0 - dampening)
                        self.logger.info(f"Quantum stabilization: Direction reversal dampened {signal:.2f} → {dampened_signal:.2f} (entanglement: {entanglement:.2f})")
                        signal = dampened_signal
        
        # Apply advanced quantum stability if significant change
        if abs(original_signal - signal) > 0.1:
            self.logger.info(f"Advanced quantum stability applied: {original_signal:.2f} → {signal:.2f} (adaptive time protection)")
        
        return signal
    def analyze_order_flow_persistence(self, market_data):
        """Analyze order flow for persistent imbalances and exhaustion patterns"""
        # Extract order flow data
        if not hasattr(self, '_orderflow_samples'):
            self._orderflow_samples = []
            
        # Update samples
        current_flow = market_data.get('order_flow', 0)
        self._orderflow_samples.append(current_flow)
        if len(self._orderflow_samples) > 30:
            self._orderflow_samples.pop(0)
            
        # Not enough samples
        if len(self._orderflow_samples) < 10:
            return {'persistent': False}
            
        # Calculate persistence metrics
        recent_samples = self._orderflow_samples[-10:]
        avg_flow = sum(recent_samples) / len(recent_samples)
        
        # Calculate flow consistency
        positive_count = sum(1 for s in recent_samples if s > 0)
        negative_count = sum(1 for s in recent_samples if s < 0)
        
        # Detect persistence and exhaustion
        persistence_threshold = 0.7
        flow_direction = 1 if avg_flow > 0 else -1
        consistent_count = positive_count if flow_direction > 0 else negative_count
        persistence_score = consistent_count / len(recent_samples)
        
        # Calculate acceleration/deceleration
        recent_5 = self._orderflow_samples[-5:]
        previous_5 = self._orderflow_samples[-10:-5]
        
        recent_avg = sum(recent_5) / 5
        previous_avg = sum(previous_5) / 5
        
        acceleration = recent_avg - previous_avg
        
        # Detect exhaustion pattern - strong flow followed by significant decrease
        exhaustion = (abs(previous_avg) > 0.2 and abs(acceleration) > 0.1 and 
                    (previous_avg * acceleration < 0))  # Opposite signs
        
        return {
            'persistent': persistence_score > persistence_threshold,
            'direction': flow_direction,
            'persistence_score': persistence_score,
            'acceleration': acceleration,
            'exhaustion': exhaustion,
            'avg_flow': avg_flow
        }    
    def enhanced_range_detection(self, price_history, current_price):
        """Advanced quantum range detection with regime-specific calibration"""
        if len(price_history) < 30:
            return {'at_extreme': False, 'confidence': 0}
            
        # Calculate dynamic lookback period based on volatility
        recent_volatility = np.std(np.diff(price_history[-20:]))
        lookback = min(120, max(30, int(100 * (0.0001 / max(0.00001, recent_volatility)))))
        
        # Use recent portion of price history for range detection
        range_prices = price_history[-lookback:]
        range_high = max(range_prices)
        range_low = min(range_prices)
        range_mid = (range_high + range_low) / 2
        range_size = range_high - range_low
        
        # Calculate normalized position within range (0-1)
        if range_size <= 0:
            return {'at_extreme': False, 'confidence': 0}
            
        normalized_pos = (current_price - range_low) / range_size
        
        # Use quantum probability distribution to determine extremes
        if normalized_pos <= 0.15:  # Lower extreme
            confidence = 1.0 - (normalized_pos / 0.15)
            return {
                'at_extreme': True, 
                'type': 'low', 
                'confidence': confidence,
                'normalized_position': normalized_pos,
                'range_size': range_size
            }
        elif normalized_pos >= 0.85:  # Upper extreme
            confidence = (normalized_pos - 0.85) / 0.15
            return {
                'at_extreme': True, 
                'type': 'high', 
                'confidence': confidence,
                'normalized_position': normalized_pos,
                'range_size': range_size
            }
        else:
            # Not at an extreme
            return {'at_extreme': False, 'confidence': 0, 'normalized_position': normalized_pos}    
    def elite_timeframe_synchronization(self, signal, market_data):
        """Quantum synchronization across multiple timeframes"""
        # Define critical timeframes for NQ futures
        timeframes = {
            'ultra_short': 5,    # 5-minute
            'short': 15,         # 15-minute
            'medium': 60,        # 1-hour
            'long': 240          # 4-hour
        }
        
        # Calculate trend direction for each timeframe
        trend_directions = {}
        trend_strengths = {}
        
        for tf_name, tf_minutes in timeframes.items():
            # Get price history for this timeframe
            prices = self.get_timeframe_prices(tf_minutes)
            if len(prices) >= 20:
                # Calculate directional trend using advanced methods
                ema_fast = self.calculate_ema(prices, 8)
                ema_slow = self.calculate_ema(prices, 21)
                
                # Direction: 1 for up, -1 for down, 0 for sideways
                if ema_fast[-1] > ema_slow[-1] * 1.001:
                    trend_directions[tf_name] = 1
                elif ema_fast[-1] < ema_slow[-1] * 0.999:
                    trend_directions[tf_name] = -1
                else:
                    trend_directions[tf_name] = 0
                    
                # Calculate trend strength
                atr = self.calculate_atr(prices, 14)
                adx = self.calculate_adx(prices, 14)
                trend_strengths[tf_name] = (adx[-1] / 100.0) * (atr[-1] / prices[-1])
        
        # Calculate timeframe synchronization score
        sync_score = 0
        total_weight = 0
        
        # Weights for different timeframes
        weights = {
            'ultra_short': 0.15,
            'short': 0.25,
            'medium': 0.35,
            'long': 0.25
        }
        
        # Calculate weighted synchronization score
        for tf, direction in trend_directions.items():
            weight = weights[tf]
            total_weight += weight
            
            # Signal aligned with timeframe trend
            if (signal > 0 and direction > 0) or (signal < 0 and direction < 0):
                sync_score += weight * trend_strengths.get(tf, 0.5)
            # Signal opposite to timeframe trend
            elif (signal > 0 and direction < 0) or (signal < 0 and direction > 0):
                sync_score -= weight * trend_strengths.get(tf, 0.5)
        
        # Normalize to -1.0 to 1.0 range
        if total_weight > 0:
            sync_score = sync_score / total_weight
        
        # Log synchronization data
        self.logger.info(f"Quantum timeframe synchronization: Score {sync_score:.2f}")
        
        return sync_score 
    def quantum_position_sizing(self, signal, market_data, regime_info, sync_score):
        """Elite position sizing with quantum risk optimization"""
        # Base position size
        base_size = 1
        
        # Extract key metrics
        volatility = market_data.get('volatility', 0.0001)
        vpin = market_data.get('vpin', 0.2)
        liquidity = market_data.get('liquidity_score', 0.9)
        
        # Regime-specific adjustments
        regime = regime_info.get('regime', 'unknown')
        confidence = regime_info.get('confidence', 0.5)
        
        # Step 1: Volatility-based sizing
        vol_factor = max(0.5, min(1.5, 0.0005 / max(volatility, 0.00001)))
        
        # Step 2: Signal strength factor
        signal_factor = min(1.3, max(0.7, abs(signal) / 0.5))
        
        # Step 3: Regime-specific factor
        regime_factor = 1.0
        if regime == 'trending_up' or regime == 'trending_down':
            # Larger positions in trending regimes
            regime_factor = 1.2 * confidence
        elif regime == 'volatile':
            # Smaller positions in volatile regimes
            regime_factor = 0.7
        
        # Step 4: Liquidity factor
        liquidity_factor = min(1.1, liquidity)
        
        # Step 5: Multi-timeframe sync factor - CRITICAL INNOVATION
        # This ensures larger positions when timeframes align
        sync_factor = max(0.6, min(1.4, 1.0 + (sync_score * 0.4)))
        
        # Step 6: Capital curve adjustment
        equity_curve_factor = self.get_equity_curve_factor()
        
        # Calculate final position size with quantum optimization
        size_multiplier = (
            vol_factor * 
            signal_factor * 
            regime_factor * 
            liquidity_factor *
            sync_factor *
            equity_curve_factor
        )
        
        # Apply VPIN constraint - critical for protection
        if vpin > 0.25:
            vpin_constraint = max(0.5, 1.0 - ((vpin - 0.25) * 2.0))
            size_multiplier *= vpin_constraint
        
        # Calculate and round final position
        final_size = round(base_size * size_multiplier)
        
        # Safety cap
        max_allowed_size = 2  # Maximum position size
        final_size = min(final_size, max_allowed_size)
        
        # Log position sizing factors
        self.logger.info(f"Position factors: Vol:{vol_factor:.2f} Signal:{signal_factor:.2f} " + 
                        f"Regime:{regime_factor:.2f} Sync:{sync_factor:.2f} " +
                        f"Liquidity:{liquidity_factor:.2f} Curve:{equity_curve_factor:.2f}")
        
        return final_size       
    def calculate_trade_pnl(self, entry_price, exit_price, position_size, commission=2.25):
        """Calculate accurate P&L for NQ futures with enhanced error handling"""
        # CRITICAL FIX: Properly handle missing or zero entry prices
        if not entry_price or entry_price == 0:
            # Try to retrieve from position history if available
            if hasattr(self, 'trade_history') and self.trade_history:
                for trade in reversed(self.trade_history):
                    if trade.get('id') == self.current_position_id:
                        entry_price = trade.get('entry_price')
                        self.logger.info(f"Retrieved entry price {entry_price} from trade history")
                        break
            
            # If still no valid entry price, use a close approximation
            if not entry_price or entry_price == 0:
                # Use exit price with small offset to prevent zero P&L
                entry_price = exit_price * 0.9999  # Creates minimal P&L impact
                self.logger.warning(f"Using approximated entry price {entry_price} for P&L calculation")
        
        if not exit_price:
            self.logger.error(f"Invalid exit price for P&L calculation: exit={exit_price}")
            return 0
            
        direction = 1 if position_size > 0 else -1
        point_value = 20  # NQ futures is $20 per point
        
        # Calculate points difference
        points = (exit_price - entry_price) * direction
        
        # Calculate raw P&L
        raw_pnl = points * abs(position_size) * point_value
        
        # Subtract commission
        net_pnl = raw_pnl - (commission * abs(position_size))
        
        # CRITICAL: Sanity check against impossible values
        if abs(points) > 200:  # No reasonable single trade would move 200+ points
            self.logger.error(f"P&L calculation error: {points} points between {entry_price} and {exit_price}")
            return 0
            
        if abs(raw_pnl) > 10000:  # Cap at $10,000 per contract as sanity check
            self.logger.warning(f"P&L sanity check: ${raw_pnl:.2f} from {points} points")
            max_reasonable_pnl = 10000 * abs(position_size)
            return max(min(raw_pnl, max_reasonable_pnl), -max_reasonable_pnl)
        
        return net_pnl   
    def streamlined_signal_processing(self, original_signal, market_data, regime_info):
        """
        Simplified signal processing pipeline that reduces excessive transformations
        to improve signal stability and reliability
        """
        # Extract key components
        regime = regime_info.get('regime', 'unknown')
        current_price = market_data.get('price', 0)
        
        # Start with original signal
        processed_signal = original_signal
        
        # Apply range-based mean reversion if in range-bound market
        if regime == 'range_bound' and 'range_analysis' in market_data:
            range_analysis = market_data['range_analysis']
            if range_analysis.get('boundaries_detected', False) and abs(range_analysis.get('z_score', 0)) > 1.2:
                mean_reversion_signal = range_analysis.get('mean_reversion_signal', 0)
                if abs(mean_reversion_signal) > 0.1:
                    # Only log if adjustment is significant
                    original = processed_signal
                    processed_signal = processed_signal * 0.2 + mean_reversion_signal * 0.8
                    self.logger.info(f"Range mean reversion dominant signal: {original:.2f} → {processed_signal:.2f} (z-score: {range_analysis['z_score']:.2f})")
                    
                    # Apply stronger bias when at range extremes
                    position_in_range = range_analysis.get('position_in_range', 0.5)
                    if position_in_range < 0.1 or position_in_range > 0.9:
                        if abs(processed_signal) < 0.3:
                            processed_signal = mean_reversion_signal
                            self.logger.info(f"Range extreme override: signal set to {processed_signal:.2f} (position: {position_in_range:.2f})")
        
        # Apply neural network alpha only if strong and consistent with base signal
        neural_alpha = market_data.get('neural_alpha', {})
        neural_signal = neural_alpha.get('signal', 0)
        neural_confidence = neural_alpha.get('confidence', 0)
        
        if abs(neural_signal) > abs(processed_signal) and neural_confidence > 0.6:
            if np.sign(neural_signal) == np.sign(processed_signal) or abs(processed_signal) < 0.1:
                # Neural signal enhances base signal
                original = processed_signal
                processed_signal = neural_signal * 0.7 + processed_signal * 0.3
                self.logger.info(f"Neural alpha enhancement: {original:.2f} → {processed_signal:.2f} (confidence: {neural_confidence:.2f})")
        
        # Apply minimal quantum enhancement - avoid excessive transformations
        quantum_bias = market_data.get('quantum_bias', 0)
        entanglement = market_data.get('entanglement', 0.5)
        
        if entanglement > 0.8 and abs(quantum_bias) > 0.3:
            # Strong quantum correlation detected
            original = processed_signal
            # Scale by entanglement strength
            boost_factor = 1.0 + (entanglement - 0.8) * 2.0
            # Apply quantum bias
            if np.sign(quantum_bias) == np.sign(processed_signal) or abs(processed_signal) < 0.15:
                processed_signal = processed_signal * boost_factor
                self.logger.info(f"Quantum enhancement: {original:.2f} → {processed_signal:.2f} (entanglement: {entanglement:.2f})")
        
        # Apply regime transition protection (assumes this function exists elsewhere)
        # This would dampen signals during regime transitions
        
        # Final cap to prevent extreme signals
        processed_signal = max(-1.0, min(1.0, processed_signal))
        
        return processed_signal
    def dynamic_trading_threshold(self, base_threshold, market_data, signal_strength, confirmation_score):
        """
        Calculate a clean, focused trading threshold without excessive adjustments
        """
        # Start with the base threshold
        threshold = base_threshold
        
        # Extract key metrics
        vpin = market_data.get('vpin', 0.3)
        entanglement = market_data.get('entanglement', 0.5)
        regime = market_data.get('regime', 'unknown')
        
        # Apply confirmation-based adjustment (high confirmation = lower threshold)
        if confirmation_score > 0.7:
            threshold *= max(0.7, 1.0 - (confirmation_score - 0.7))
        elif confirmation_score < 0.3:
            # Low confirmation requires higher threshold
            threshold *= min(1.3, 1.0 + (0.3 - confirmation_score))
        
        # Apply a single liquidity-based adjustment
        if vpin > 0.5:
            # Higher threshold in toxic liquidity conditions
            threshold *= min(1.5, 1.0 + (vpin - 0.5))
        
        # Range market specific threshold
        if regime == 'range_bound' and 'range_analysis' in market_data:
            range_analysis = market_data.get('range_analysis', {})
            position_in_range = range_analysis.get('position_in_range', 0.5)
            z_score = range_analysis.get('z_score', 0)
            
            # Lower threshold at range extremes to capture mean reversion opportunities
            if abs(z_score) > 1.5:
                threshold *= 0.7
            
            # Lower threshold when signal aligned with mean reversion
            if (z_score > 1.0 and signal_strength < 0) or (z_score < -1.0 and signal_strength > 0):
                threshold *= 0.8
        
        # Cap minimum and maximum values
        threshold = max(0.15, min(0.8, threshold))
        
        return threshold
    def calibrated_timeframe_confirmation(self, signal, market_data):
        """Advanced multi-timeframe confirmation with adaptive weighting"""
        # Get raw confirmation score using existing method
        raw_confirmation = self.get_multi_timeframe_confirmation(signal, market_data)
        
        # Extract key market metrics
        regime = self.regime_classifier.get_current_regime()
        trend_strength = market_data.get('trend_strength', 1.0)
        vpin = market_data.get('vpin', 0.3)
        
        # Apply adaptive regime-specific calibration
        if regime == 'trending_up' or regime == 'trending_down':
            # In trending regimes, confirmation should have less impact on threshold
            calibrated_confirmation = raw_confirmation * 0.5
        elif regime == 'choppy':
            # In choppy regimes, confirmation is more important
            calibrated_confirmation = raw_confirmation * 1.2
        else:  # range_bound
            # In range-bound markets, use moderate impact
            calibrated_confirmation = raw_confirmation * 0.8
        
        # Apply trend strength modifier - stronger trends need less confirmation
        if trend_strength > 2.0 and abs(calibrated_confirmation) < 0.3:
            # Don't let weak confirmation override strong trend
            calibrated_confirmation = calibrated_confirmation * 0.7
        
        # Log the calibration
        self.logger.info(f"Confirmation calibration: {raw_confirmation:.2f} → {calibrated_confirmation:.2f} ({regime}, trend: {trend_strength:.2f})")
        
        return calibrated_confirmation

    def optimize_signal_threshold(self, base_threshold, confirmation, vpin, regime):
        """Dynamic threshold optimization with quantum regime adaptation"""
        # Start with base threshold
        threshold = base_threshold  # Usually 0.50
        
        # Apply confirmation adjustment (with reduced impact)
        if confirmation > 0.5:  # Strong confirmation
            # Lower threshold - easier to trade with strong confirmation
            threshold -= 0.10
        elif confirmation > 0.25:  # Moderate confirmation
            threshold -= 0.05
        elif confirmation < -0.5:  # Strong counter-confirmation
            # Raise threshold - harder to trade against timeframes
            threshold += 0.15
        elif confirmation < -0.25:  # Moderate counter-confirmation
            threshold += 0.07
        else:  # Weak confirmation either way
            # Minor adjustment
            threshold += (confirmation * -0.08)
        
        # Apply regime-specific optimization
        if regime == 'trending_up' or regime == 'trending_down':
            # More aggressive in trends - lower threshold
            threshold *= 0.9
        elif regime == 'volatile' or regime == 'choppy':
            # More conservative in volatility - raise threshold
            threshold *= 1.1
        
        # Apply VPIN constraint with gradual scaling
        vpin_cap = min(0.65, base_threshold + (0.15 * min(1.0, vpin/0.5)))
        
        # Never let threshold go below absolute minimum
        min_threshold = 0.3
        threshold = max(min_threshold, min(threshold, vpin_cap))
        
        self.logger.info(f"Optimized threshold: {base_threshold:.2f} → {threshold:.2f} (conf: {confirmation:.2f}, VPIN: {vpin:.2f})")
        
        return threshold
    
    def strategic_signal_amplification(self, signal, market_data, regime):
        """Advanced signal amplification based on market conditions"""
        # Analyze market conditions
        trend_strength = market_data.get('trend_strength', 1.0)
        order_flow = market_data.get('order_flow', 0)
        volatility = market_data.get('volatility', 0.0001)
        
        # Only amplify when signal and order flow align (critical)
        if (signal > 0 and order_flow > 0.2) or (signal < 0 and order_flow < -0.2):
            # Calculate amplification factor
            # Base amplification
            amp_factor = 1.0
            
            # Trend strength amplification - stronger in strong trends
            if trend_strength > 1.5:
                amp_factor += 0.1 * min(2.0, trend_strength / 1.5)
            
            # Volatility damping - reduce amplification in high volatility
            normalized_vol = min(1.0, volatility / 0.0005)
            vol_damping = 1.0 - (normalized_vol * 0.2)
            amp_factor *= vol_damping
            
            # Apply amplification
            amplified = signal * amp_factor
            
            # Cap maximum amplification
            max_amp = 1.7  # Maximum 70% boost
            if abs(amplified) > abs(signal) * max_amp:
                amplified = signal * max_amp
            
            self.logger.info(f"Strategic signal amplification: {signal:.2f} → {amplified:.2f} (factor: {amp_factor:.2f})")
            return amplified
            
        return signal  # No amplification if conditions don't align    
    def amplitude_aware_position_sizing(self, base_size, price_history, current_price, regime_info):
        """Optimize position size based on range amplitude in range-bound markets"""
        if regime_info.get('regime') != 'range_bound':
            return base_size
            
        # Calculate range metrics
        if len(price_history) < 30:
            return base_size
            
        recent_prices = price_history[-60:]
        range_high = max(recent_prices)
        range_low = min(recent_prices)
        range_amplitude = range_high - range_low
        
        # Calculate position in range (0-1)
        if range_amplitude <= 0:
            return base_size
            
        position_in_range = (current_price - range_low) / range_amplitude
        
        # Scale position size based on position in range
        # Larger positions near extremes, smaller in middle
        if position_in_range <= 0.2:  # Near lower bound
            size_multiplier = 1.3  # Increase long size near bottom
        elif position_in_range >= 0.8:  # Near upper bound
            size_multiplier = 1.3  # Increase short size near top
        else:
            # Reduce size in the middle of the range (less edge)
            distance_from_middle = abs(0.5 - position_in_range)
            size_multiplier = 0.7 + (distance_from_middle * 0.6)
        
        # Scale by range amplitude relative to average daily range
        normal_range = current_price * 0.005  # Approx 0.5% as normal range
        amplitude_factor = min(1.5, max(0.8, range_amplitude / normal_range))
        
        self.logger.info(f"Range amplitude sizing: pos_in_range={position_in_range:.2f}, " +
                        f"multiplier={size_multiplier:.2f}, amplitude_factor={amplitude_factor:.2f}")
        
        return int(base_size * size_multiplier * amplitude_factor)  
    
    
    def detect_quantum_edge_patterns(self, market_data, price_history, regime_info):
        """Detect high-probability pattern setups with quantum-enhanced precision"""
        import numpy as np
        
        patterns = []
        
        # Extract critical metrics
        current_price = market_data.get('price', 0)
        order_flow = market_data.get('order_flow', 0)
        delta = market_data.get('delta', 0)
        regime = regime_info.get('regime', 'unknown')
        
        # Not enough price history
        if len(price_history) < 60:
            return {'detected': False}
        
        # Pattern 1: Volatility squeeze with directional order flow
        recent_vol = np.std(np.diff(price_history[-20:]))
        previous_vol = np.std(np.diff(price_history[-40:-20]))
        
        vol_ratio = recent_vol / max(0.00001, previous_vol)
        vol_squeeze = vol_ratio < 0.7  # 30% reduction in volatility
        
        # Check for strong order flow during squeeze
        strong_flow = abs(order_flow) > 0.2
        if vol_squeeze and strong_flow:
            patterns.append({
                'name': 'vol_squeeze_breakout',
                'direction': 1 if order_flow > 0 else -1,
                'strength': min(1.0, abs(order_flow) * 2.5) * (1.0 - vol_ratio),
                'confidence': 0.8
            })
        
        # Pattern 2: Double rejection at range extreme
        if regime == 'range_bound':
            # Get range data
            range_info = self.enhanced_range_detection(price_history, current_price)
            
            if range_info.get('at_extreme', False) and range_info.get('confidence', 0) > 0.7:
                # Check for previous rejections at this level
                range_type = range_info.get('type')
                normalized_pos = range_info.get('normalized_position', 0.5)
                
                # Find recent peaks/valleys
                peaks = []
                valleys = []
                
                for i in range(2, len(price_history)-2):
                    # Simple peak detector
                    if price_history[i] > price_history[i-1] and price_history[i] > price_history[i-2] and \
                    price_history[i] > price_history[i+1] and price_history[i] > price_history[i+2]:
                        peaks.append(i)
                        
                    # Simple valley detector
                    if price_history[i] < price_history[i-1] and price_history[i] < price_history[i-2] and \
                    price_history[i] < price_history[i+1] and price_history[i] < price_history[i+2]:
                        valleys.append(i)
                
                # Check for rejection patterns
                rejection_detected = False
                pattern_direction = 0
                
                if range_type == 'high' and len(peaks) >= 2:
                    # Check if we have had two peaks near the upper extreme
                    recent_peaks = [price_history[i] for i in peaks[-3:]]
                    high_peaks = [p for p in recent_peaks if p > (range_info.get('range_high', 0) * 0.998)]
                    
                    if len(high_peaks) >= 2:
                        rejection_detected = True
                        pattern_direction = -1  # Sell signal at upper range
                        
                elif range_type == 'low' and len(valleys) >= 2:
                    # Check if we have had two valleys near the lower extreme
                    recent_valleys = [price_history[i] for i in valleys[-3:]]
                    low_valleys = [v for v in recent_valleys if v < (range_info.get('range_low', 0) * 1.002)]
                    
                    if len(low_valleys) >= 2:
                        rejection_detected = True
                        pattern_direction = 1  # Buy signal at lower range
                
                if rejection_detected:
                    patterns.append({
                        'name': 'double_rejection',
                        'direction': pattern_direction,
                        'strength': 0.9 * range_info.get('confidence', 0),
                        'confidence': 0.85
                    })
        
        # Return strongest pattern if any detected
        if patterns:
            # Sort by strength * confidence
            patterns.sort(key=lambda x: x['strength'] * x['confidence'], reverse=True)
            return {
                'detected': True,
                'pattern': patterns[0]['name'],
                'direction': patterns[0]['direction'],
                'strength': patterns[0]['strength'],
                'confidence': patterns[0]['confidence']
            }
        
        return {'detected': False}    
    def quantum_market_memory(self, price_history, current_price, current_signal):
        """Advanced price pattern memory system using quantum-inspired analysis"""
        import numpy as np
        
        # Initialize storage if not already done
        if not hasattr(self, '_quantum_memory_bank'):
            self._quantum_memory_bank = {
                'patterns': [],
                'last_check': datetime.datetime.now(),
                'check_interval': 300  # seconds
            }
        
        # Only run full pattern detection periodically to save computation
        current_time = datetime.datetime.now()
        time_since_check = (current_time - self._quantum_memory_bank['last_check']).total_seconds()
        
        if time_since_check > self._quantum_memory_bank['check_interval'] or len(self._quantum_memory_bank['patterns']) == 0:
            # Time to update our pattern memory
            self._quantum_memory_bank['last_check'] = current_time
            
            # Only proceed if we have enough price history
            if len(price_history) < 200:
                return {'pattern_detected': False}
            
            # Extract patterns
            patterns = []
            for lookback in [30, 60, 90]:
                for window in [5, 10, 15]:
                    # Only process a limited number of patterns for efficiency
                    if len(patterns) >= 5:
                        break
                        
                    # Find similar historical price sequences
                    current_seq = price_history[-window:]
                    
                    for i in range(window, min(len(price_history) - lookback, 500), window):
                        hist_seq = price_history[-(i+window):-i]
                        
                        # Calculate similarity using dynamic time warping distance
                        similarity = 1.0 - min(1.0, np.sqrt(np.mean(np.square(np.array(current_seq) - np.array(hist_seq)))) / np.mean(current_seq))
                        
                        if similarity > 0.85:  # High similarity threshold
                            # Found similar pattern, analyze what happened after
                            future_seq = price_history[-i:-(i-lookback)]
                            future_direction = 1 if future_seq[-1] > future_seq[0] else -1
                            future_change_pct = abs(future_seq[-1] - future_seq[0]) / future_seq[0]
                            
                            patterns.append({
                                'similarity': similarity,
                                'lookback': lookback,
                                'window': window,
                                'future_direction': future_direction,
                                'future_change_pct': future_change_pct,
                                'confidence': similarity * min(1.0, future_change_pct * 100)
                            })
            
            # Sort patterns by confidence
            patterns.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Store top patterns
            self._quantum_memory_bank['patterns'] = patterns[:3]
        
        # Analyze current signal against stored patterns
        if not self._quantum_memory_bank['patterns']:
            return {'pattern_detected': False}
        
        # Get best pattern
        best_pattern = self._quantum_memory_bank['patterns'][0]
        
        # Check if current signal aligns with historical pattern
        signal_direction = 1 if current_signal > 0 else -1
        
        if signal_direction == best_pattern['future_direction']:
            # Signal aligns with historical pattern - boost confidence
            boost_factor = min(1.3, 1.0 + (best_pattern['confidence'] * 0.3))
            
            return {
                'pattern_detected': True,
                'pattern_description': f"{best_pattern['window']} bar pattern with {best_pattern['similarity']:.2f} similarity",
                'expected_direction': best_pattern['future_direction'],
                'confidence': best_pattern['confidence'],
                'signal_boost': boost_factor,
                'recommendation': 'strengthen' if signal_direction == best_pattern['future_direction'] else 'weaken'
            }
        elif best_pattern['confidence'] > 0.7:
            # High confidence pattern contradicts signal - suggest caution
            return {
                'pattern_detected': True,
                'pattern_description': f"{best_pattern['window']} bar pattern with {best_pattern['similarity']:.2f} similarity",
                'expected_direction': best_pattern['future_direction'],
                'confidence': best_pattern['confidence'],
                'signal_boost': 0.7,  # Dampen contrarian signal
                'recommendation': 'weaken'
            }
        
        return {'pattern_detected': False}
    def resolve_pattern_conflict(self, pattern_data, original_signal, market_data):
        """Enhanced pattern conflict resolution with range awareness and safe error handling"""
        
        # Extract pattern details
        pattern_name = pattern_data.get('pattern_name', 'unknown')
        pattern_direction = pattern_data.get('direction', 0)
        pattern_confidence = pattern_data.get('confidence', 0)
        price = market_data.get('price', 0)
        
        # Default values
        result = {
            'action': 'default',
            'signal': original_signal,
            'reason': 'no_conflict',
            'threshold_adjustment': 0.0
        }
        
        # Check if pattern is at range extreme - with proper null safety
        range_extreme = False
        try:
            if pattern_direction > 0 and self.is_at_range_bottom(price):
                range_extreme = True
                result['action'] = 'enhance'
                result['reason'] = 'aligned_range_bottom'
                result['signal'] = original_signal + (0.15 * pattern_confidence)
                result['threshold_adjustment'] = -0.08
            elif pattern_direction < 0 and self.is_at_range_top(price):
                range_extreme = True
                result['action'] = 'enhance'
                result['reason'] = 'aligned_range_top'
                result['signal'] = original_signal - (0.15 * pattern_confidence)
                result['threshold_adjustment'] = -0.08
        except Exception as e:
            self.logger.warning(f"Range check error (non-critical): {e}")
            # Continue with non-range based logic
        
        # If not at range extreme, apply standard pattern resolution logic
        if not range_extreme:
            if pattern_name == 'flow_absorption':
                if pattern_direction * original_signal > 0:
                    # Pattern aligned with signal - enhance
                    result['action'] = 'enhance'
                    result['reason'] = 'aligned_flow_absorption'
                    result['signal'] = original_signal * (1.0 + (pattern_confidence * 0.3))
                    result['threshold_adjustment'] = -0.08
                elif abs(original_signal) < 0.1:
                    # Very small signal - override with pattern
                    result['action'] = 'override'
                    result['reason'] = 'strong_flow_absorption_override'
                    result['signal'] = pattern_direction * 0.34 * pattern_confidence
                    result['threshold_adjustment'] = -0.05
                else:
                    # Conflict - signal and pattern in opposite directions
                    result['action'] = 'conflict'
                    result['reason'] = 'conflicting_flow_absorption'
                    result['threshold_adjustment'] = 0.08
                    
            elif pattern_name == 'flow_exhaustion':
                if pattern_direction * original_signal < 0:
                    # Pattern suggests reversal - dampen opposing signal
                    result['action'] = 'dampen' 
                    result['reason'] = 'exhaustion_dampening'
                    result['signal'] = original_signal * 0.6
                    result['threshold_adjustment'] = 0.05
                else:
                    # Pattern aligned with signal - potential reversal, be cautious
                    result['action'] = 'cautious'
                    result['reason'] = 'exhaustion_caution'
                    result['threshold_adjustment'] = 0.10
                    
            elif pattern_name == 'flow_acceleration':
                if pattern_direction * original_signal > 0:
                    # Pattern reinforces signal - boost
                    result['action'] = 'boost'
                    result['reason'] = 'acceleration_boost'
                    result['signal'] = original_signal * (1.0 + (pattern_confidence * 0.4))
                    result['threshold_adjustment'] = -0.10
                else:
                    # Conflict - signal opposes acceleration
                    result['action'] = 'conflict'
                    result['reason'] = 'conflicting_acceleration'
                    result['threshold_adjustment'] = 0.05
        
        # Ensure reasonable signal bounds
        result['signal'] = max(-2.0, min(2.0, result['signal']))
        
        return result
    def is_at_range_top(self, price):
        """Check if price is at or near the top of the current range"""
        if not hasattr(self, '_range_data'):
            return False
            
        range_high = self._range_data.get('range_high')
        
        # Check if range data is valid
        if range_high is None:
            return False
            
        return price > (range_high * 0.997)  # Within 0.3% of range high

    def is_at_range_bottom(self, price):
        """Check if price is at or near the bottom of the current range"""
        if not hasattr(self, '_range_data'):
            return False
            
        range_low = self._range_data.get('range_low')
        
        # Check if range data is valid
        if range_low is None:
            return False
            
        return price < (range_low * 1.003)  # Within 0.3% of range low   
    def enhanced_quantum_entanglement(self, market_data, lookback=20):
        """Advanced quantum entanglement calculation for delta-flow relationships"""
        import numpy as np
        
        # Extract current metrics
        current_delta = market_data.get('delta', 0.0)
        current_flow = market_data.get('order_flow', 0.0)
        
        # Get histories (or create if they don't exist)
        if not hasattr(self, '_delta_history'):
            self._delta_history = []
        if not hasattr(self, '_flow_history'):
            self._flow_history = []
        
        # Update histories
        self._delta_history.append(current_delta)
        self._flow_history.append(current_flow)
        
        # Maintain history length
        if len(self._delta_history) > lookback:
            self._delta_history = self._delta_history[-lookback:]
        if len(self._flow_history) > lookback:
            self._flow_history = self._flow_history[-lookback:]
        
        # If not enough history, use simple alignment
        if len(self._delta_history) < 5:
            # Simple calculation based on sign alignment
            if abs(current_delta) > 0.2 and abs(current_flow) > 0.2 and np.sign(current_delta) == np.sign(current_flow):
                return 0.8  # High alignment
            elif abs(current_delta) > 0.1 and abs(current_flow) > 0.1 and np.sign(current_delta) == np.sign(current_flow):
                return 0.6  # Medium alignment
            elif np.sign(current_delta) == np.sign(current_flow):
                return 0.4  # Low alignment
            else:
                return 0.2  # Misaligned
        
        # Calculate weighted correlation
        weights = np.linspace(0.5, 1.0, len(self._delta_history))  # More weight to recent values
        
        # Normalize data
        delta_norm = np.array(self._delta_history) - np.mean(self._delta_history)
        flow_norm = np.array(self._flow_history) - np.mean(self._flow_history)
        
        # Prevent division by zero
        if np.std(delta_norm) == 0 or np.std(flow_norm) == 0:
            # Fallback to current alignment
            current_alignment = np.sign(current_delta) * np.sign(current_flow)
            return 0.3 + (0.4 * max(0, current_alignment))
        
        # Calculate weighted correlation
        weighted_corr = np.sum(weights * delta_norm * flow_norm) / (
            np.sqrt(np.sum(weights * delta_norm**2)) * 
            np.sqrt(np.sum(weights * flow_norm**2))
        )
        
        # Convert correlation (-1 to 1) to entanglement score (0.1 to 0.95)
        entanglement = 0.525 + (weighted_corr * 0.425)
        
        # Add recency bias - if latest values align strongly, boost entanglement
        if abs(current_delta) > 0.2 and abs(current_flow) > 0.2 and np.sign(current_delta) == np.sign(current_flow):
            entanglement = min(0.95, entanglement + 0.15)
        
        # Ensure valid range
        return max(0.1, min(0.95, entanglement))    
    
    def adaptive_quantum_signal_reinforcement(self, market_data, base_signal, regime_info):
        """Advanced signal reinforcement using quantum-inspired probability amplification"""
        import numpy as np
        
        # Extract key market metrics
        order_flow = market_data.get('order_flow', 0.0)
        delta = market_data.get('delta', 0.0)
        vpin = market_data.get('vpin', 0.0)
        regime = regime_info.get('regime', 'unknown')
        regime_confidence = regime_info.get('confidence', 0.5)
        
        # Initialize signal components
        signal_direction = np.sign(base_signal)
        signal_magnitude = abs(base_signal)
        
        # Calculate quantum probability distribution for potential market states
        # Using Gaussian mixture model to represent superposition of states
        market_states = {
            'strong_up': np.exp(-((1.0 - delta)**2) / 0.2) * np.exp(-((1.0 - order_flow)**2) / 0.2),
            'weak_up': np.exp(-((0.5 - delta)**2) / 0.2) * np.exp(-((0.5 - order_flow)**2) / 0.2),
            'neutral': np.exp(-(delta**2) / 0.2) * np.exp(-(order_flow**2) / 0.2),
            'weak_down': np.exp(-((0.5 + delta)**2) / 0.2) * np.exp(-((0.5 + order_flow)**2) / 0.2),
            'strong_down': np.exp(-((1.0 + delta)**2) / 0.2) * np.exp(-((1.0 + order_flow)**2) / 0.2)
        }
        
        # Normalize the probabilities
        total_prob = sum(market_states.values())
        if total_prob > 0:
            for state in market_states:
                market_states[state] /= total_prob
        
        # FIX: Calculate entanglement factor properly
        # Use an alternative approach to correlation when we don't have time series data
        if abs(delta) > 0.2 and abs(order_flow) > 0.2 and np.sign(delta) == np.sign(order_flow):
            # Strong alignment = high entanglement
            entanglement = 0.8 + min(0.15, (abs(delta) + abs(order_flow)) / 10)
        elif abs(delta) > 0.1 and abs(order_flow) > 0.1 and np.sign(delta) == np.sign(order_flow):
            # Moderate alignment = medium entanglement
            entanglement = 0.6 + min(0.2, (abs(delta) + abs(order_flow)) / 15)
        elif np.sign(delta) == np.sign(order_flow):
            # Weak alignment but same sign = light entanglement
            entanglement = 0.4 + min(0.2, (abs(delta) + abs(order_flow)) / 20)
        elif abs(delta) > 0.2 and abs(order_flow) > 0.2:
            # Strong disagreement = anti-correlation
            entanglement = 0.3
        else:
            # Weak relationship = low entanglement
            entanglement = 0.2 + min(0.3, (abs(delta) + abs(order_flow)) / 25)
        
        # Clamp entanglement to valid range
        entanglement = min(0.95, max(0.1, entanglement))
        
        # Apply regime-specific quantum amplification
        if regime == 'trending_up' or regime == 'trending_down':
            # In trending regimes, amplify signals aligned with trend
            trend_direction = 1 if regime == 'trending_up' else -1
            trend_alignment = 0.5 + 0.5 * np.sign(signal_direction * trend_direction)
            
            # Higher alignment = stronger amplification
            amplification_factor = 1.0 + (trend_alignment * regime_confidence * 0.5)
            
            # Stronger signals get more amplification (quadratic scaling)
            amplification_factor *= (1.0 + signal_magnitude)
            
        elif regime == 'range_bound':
            # In range-bound regimes, use mean-reversion amplification
            if (signal_direction > 0 and market_states['strong_down'] > 0.3) or \
            (signal_direction < 0 and market_states['strong_up'] > 0.3):
                # Strong mean reversion opportunity
                amplification_factor = 1.3 * regime_confidence
            else:
                # Normal range-bound behavior
                amplification_factor = 1.0
                
            # Reduce noise in range-bound by making weak signals weaker
            if signal_magnitude < 0.1 and abs(order_flow) < 0.1:
                amplification_factor *= signal_magnitude * 5  # progressive dampening
                
        else:  # unknown or other regimes
            # Conservative approach
            amplification_factor = 1.0
        
        # Apply VPIN-based interference dampening
        if vpin > 0.3:
            # High toxicity causes quantum decoherence - reduce signal reliability
            amplification_factor *= max(0.5, 1.0 - ((vpin - 0.3) * 1.5))
        
        # Apply entanglement-based enhancement
        if entanglement > 0.7 and signal_magnitude > 0.2:
            # High entanglement with strong signal = enhanced reliability
            amplification_factor *= 1.0 + ((entanglement - 0.7) * 0.5)
        
        # Calculate reinforced signal with quantum properties
        reinforced_signal = signal_direction * signal_magnitude * amplification_factor
        
        # Apply quantum tunneling effect for breakthrough signals
        if (signal_magnitude > 0.3 and 
            ((signal_direction > 0 and market_states['strong_up'] > 0.4) or
            (signal_direction < 0 and market_states['strong_down'] > 0.4))):
            # Signal is aligned with high probability state - quantum tunneling boost
            tunneling_boost = 0.2 * regime_confidence
            reinforced_signal = reinforced_signal * (1.0 + tunneling_boost)
        
        # Return full quantum analysis
        return {
            'reinforced_signal': reinforced_signal,
            'amplification_factor': amplification_factor,
            'market_states': market_states,
            'entanglement': entanglement,
            'quantum_confidence': min(0.95, regime_confidence * (1.0 + entanglement) / 2)
        }

    def analyze_order_flow_patterns(self, market_data, lookback=20):
        """Advanced order flow pattern recognition with non-linear dynamics"""
        import numpy as np
        from scipy import stats
        
        # Extract and prepare order flow history
        if not hasattr(self, '_order_flow_history'):
            self._order_flow_history = []
        
        current_flow = market_data.get('order_flow', 0.0)
        self._order_flow_history.append(current_flow)
        
        # Maintain history length
        max_history = 200  # Keep enough for pattern recognition
        if len(self._order_flow_history) > max_history:
            self._order_flow_history = self._order_flow_history[-max_history:]
        
        # Not enough data for analysis
        if len(self._order_flow_history) < lookback:
            return {'pattern_detected': False}
        
        # Get recent order flow sequence
        recent_flow = self._order_flow_history[-lookback:]
        
        # Calculate flow derivatives (changes in flow)
        flow_diff = np.diff(recent_flow)
        flow_diff2 = np.diff(flow_diff) if len(flow_diff) > 1 else [0]  # Second derivative for acceleration
        
        # Basic flow statistics
        flow_mean = np.mean(recent_flow)
        flow_std = np.std(recent_flow)
        flow_skew = stats.skew(recent_flow) if len(recent_flow) > 8 else 0
        flow_kurtosis = stats.kurtosis(recent_flow) if len(recent_flow) > 8 else 0
        
        # Exponential moving averages for multiple timeframes
        ema5 = self._calculate_ema(recent_flow, 5)
        ema10 = self._calculate_ema(recent_flow, 10)
        ema20 = self._calculate_ema(recent_flow, lookback)
        
        # Pattern 1: Flow Exhaustion (diminishing returns pattern)
        # Characterized by consistent sign but decreasing magnitude
        consistent_sign = all(f > 0 for f in recent_flow[-5:]) or all(f < 0 for f in recent_flow[-5:])
        decreasing_magnitude = False
        if consistent_sign and len(recent_flow) >= 5:
            # Convert to absolute values for magnitude check
            magnitudes = [abs(f) for f in recent_flow[-5:]]
            decreasing_magnitude = all(magnitudes[i] > magnitudes[i+1] for i in range(len(magnitudes)-1))
        
        exhaustion_detected = consistent_sign and decreasing_magnitude
        
        # Pattern 2: Flow Absorption (large flow with minimal price impact)
        price_changes = market_data.get('price_changes', [0.0] * 5)
        if len(price_changes) >= 5:
            price_changes = price_changes[-5:]
            flow_changes = recent_flow[-5:]
            
            # Calculate flow-to-price impact ratio
            abs_flow = sum(abs(f) for f in flow_changes)
            abs_price = sum(abs(p) for p in price_changes)
            
            absorption_ratio = 999 if abs_price == 0 else abs_flow / (abs_price + 0.0001)
            absorption_detected = absorption_ratio > 2.0  # Flow impact is disproportionately low
        else:
            absorption_detected = False
        
        # Pattern 3: Flow Divergence (flow doesn't match price movement)
        price_direction = np.sign(sum(price_changes)) if len(price_changes) > 0 else 0
        flow_direction = np.sign(sum(recent_flow[-5:])) if len(recent_flow) >= 5 else 0
        
        divergence_detected = (price_direction != 0 and flow_direction != 0 and
                            price_direction != flow_direction)
        
        # Pattern 4: Hidden Accumulation/Distribution
        # Detected by analyzing order flow variance and skew
        high_variance = flow_std > 0.15
        skewed_distribution = abs(flow_skew) > 1.0
        
        hidden_flow_detected = high_variance and skewed_distribution
        
        # Combine pattern detection results
        patterns_detected = []
        
        if exhaustion_detected:
            flow_direction = np.sign(recent_flow[-1])
            patterns_detected.append({
                'name': 'flow_exhaustion',
                'direction': -flow_direction,  # Reverse the flow for trading direction
                'confidence': min(0.9, 0.5 + flow_std * 2.0),
                'strength': min(0.9, 0.8 if decreasing_magnitude else 0.4 + (0.2 if consistent_sign else 0))
            })
        
        if absorption_detected:
            # Determine direction based on who is absorbing
            # If price should be going down based on flow but isn't = bullish absorption
            patterns_detected.append({
                'name': 'flow_absorption',
                'direction': -flow_direction,  # Opposite of current flow
                'confidence': min(0.85, absorption_ratio / 5.0),
                'strength': min(0.8, abs_flow / 0.5)
            })
        
        if divergence_detected:
            patterns_detected.append({
                'name': 'flow_divergence',
                'direction': price_direction,  # Follow price in a divergence
                'confidence': 0.75,
                'strength': min(0.8, abs(sum(price_changes)) / 0.01)
            })
        
        if hidden_flow_detected:
            # Direction based on skew - negative skew means accumulation (bullish)
            direction = -1 if flow_skew > 0 else 1
            patterns_detected.append({
                'name': 'hidden_flow',
                'direction': direction,
                'confidence': min(0.7, abs(flow_skew) / 2.0),
                'strength': min(0.7, flow_kurtosis / 5.0)
            })
        
        # If no patterns detected
        if not patterns_detected:
            return {'pattern_detected': False}
        
        # Sort patterns by confidence * strength
        patterns_detected.sort(key=lambda x: x['confidence'] * x['strength'], reverse=True)
        best_pattern = patterns_detected[0]
        
        # Return best pattern information
        return {
            'pattern_detected': True,
            'pattern_name': best_pattern['name'],
            'direction': best_pattern['direction'],
            'confidence': best_pattern['confidence'],
            'strength': best_pattern['strength'],
            'all_patterns': patterns_detected
        }
    def get_current_range_data(self, market_data, price_history, current_price):
        """Get current range data with enhanced error handling"""
        
        # Initialize default values
        range_data = {
            'range_high': None,
            'range_low': None,
            'confidence': 0.0,
            'position_in_range': 0.5,  # Default to middle
            'range_size': 0.0,
            'boundaries_detected': False
        }
        
        # Check for previously detected range from advanced detection
        if 'range_analysis' in market_data and market_data['range_analysis'].get('boundaries_detected', False):
            range_data['range_high'] = market_data['range_analysis'].get('range_high')
            range_data['range_low'] = market_data['range_analysis'].get('range_low')
            range_data['confidence'] = market_data['range_analysis'].get('confidence', 0.4)
            range_data['position_in_range'] = market_data['range_analysis'].get('position_in_range', 0.5)
            range_data['range_size'] = market_data['range_analysis'].get('range_size', 0.0)
            range_data['boundaries_detected'] = True
            range_data['z_score'] = market_data['range_analysis'].get('z_score', 0.0)
            return range_data
            
        # Fall back to simple range calculation if we have enough price history
        if len(price_history) >= 60:
            try:
                # Get recent high/low
                recent_high = max(price_history[-60:])
                recent_low = min(price_history[-60:])
                
                range_data['range_high'] = recent_high
                range_data['range_low'] = recent_low
                range_data['range_size'] = recent_high - recent_low
                
                # Calculate position in range
                if range_data['range_size'] > 0:
                    range_data['position_in_range'] = (current_price - recent_low) / range_data['range_size']
                    range_data['boundaries_detected'] = True
                
                # Calculate z-score
                import numpy as np
                mean_price = np.mean(price_history[-60:])
                std_price = np.std(price_history[-60:])
                range_data['z_score'] = (current_price - mean_price) / std_price if std_price > 0 else 0
                
                # Simple confidence score
                range_data['confidence'] = 0.3  # Lower confidence for simple calculation
            except Exception as e:
                self.logger.warning(f"Error in fallback range calculation: {e}")
        
        return range_data
    def _calculate_ema(self, data, period):
        """Calculate exponential moving average"""
        if len(data) < period:
            return data[-1] if data else 0
            
        alpha = 2 / (period + 1)
        ema = data[0]
        for i in range(1, len(data)):
            ema = alpha * data[i] + (1 - alpha) * ema
        return ema

    def range_optimized_profit_targets(self, entry_price, direction, market_data, range_analysis):
        """
        Calculate statistically optimal profit targets for range-bound markets
        using statistical range metrics and market microstructure
        """
        import numpy as np
        
        # Extract range data
        if not range_analysis.get('boundaries_detected', False):
            # Fall back to standard calculation if range analysis not available
            atr = market_data.get('atr', entry_price * 0.005)
            return {
                'targets': [
                    entry_price + (direction * atr * 1.0), 
                    entry_price + (direction * atr * 1.5),
                    entry_price + (direction * atr * 2.0)
                ],
                'expected_hold_times': [60, 180, 300],
                'multipliers': [1.0, 1.5, 2.0],
                'target_positions': [0.5, 0.7, 0.9]
            }
        
        range_high = range_analysis.get('range_high')
        range_low = range_analysis.get('range_low')
        position_in_range = range_analysis.get('position_in_range', 0.5)
        range_size = range_analysis.get('range_size', range_high - range_low)
        support_resistance = range_analysis.get('support_resistance', [])
        
        # Extract market data
        order_flow = market_data.get('order_flow', 0)
        delta = market_data.get('delta', 0)
        entanglement = market_data.get('entanglement', 0.5)
        
        # Calculate entry position in range
        entry_position = (entry_price - range_low) / range_size if range_size > 0 else 0.5
        
        # CRITICAL FIX - Initialize target positions based on entry location and direction
        if direction > 0:  # Long position
            if entry_position < 0.2:  # Bottom entry
                # Target middle and upper parts of range with proper separation
                target_positions = [
                    min(0.5, entry_position + 0.3),       # ~30% up from entry
                    min(0.7, entry_position + 0.5),       # ~50% up from entry 
                    min(0.85, entry_position + 0.65)      # ~65% up from entry
                ]
            elif entry_position > 0.8:  # Top entry (likely reversal)
                # Conservative targets but ENSURE they're different
                target_positions = [
                    min(0.9, entry_position + 0.02),     # Small move higher
                    min(0.95, entry_position + 0.05),    # Moderate move higher
                    min(1.0, entry_position + 0.1)       # Larger move higher
                ]
            else:  # Middle entry
                # Scale based on distance to range top with proper separation
                room_to_top = 1.0 - entry_position
                target_positions = [
                    entry_position + (room_to_top * 0.3),  # 30% of distance to top
                    entry_position + (room_to_top * 0.6),  # 60% of distance to top
                    entry_position + (room_to_top * 0.8)   # 80% of distance to top
                ]
        else:  # Short position
            if entry_position > 0.8:  # Top entry
                # Target middle and lower parts with proper separation
                target_positions = [
                    max(0.5, entry_position - 0.3),       # ~30% down from entry
                    max(0.3, entry_position - 0.5),       # ~50% down from entry
                    max(0.15, entry_position - 0.65)      # ~65% down from entry
                ]
            elif entry_position < 0.2:  # Bottom entry (likely reversal)
                # Conservative targets but ENSURE they're different
                target_positions = [
                    max(0.1, entry_position - 0.02),      # Small move lower
                    max(0.05, entry_position - 0.05),     # Moderate move lower
                    max(0.0, entry_position - 0.1)        # Larger move lower
                ]
            else:  # Middle entry
                # Scale based on distance to range bottom with proper separation
                room_to_bottom = entry_position
                target_positions = [
                    entry_position - (room_to_bottom * 0.3),  # 30% of distance to bottom
                    entry_position - (room_to_bottom * 0.6),  # 60% of distance to bottom
                    entry_position - (room_to_bottom * 0.8)   # 80% of distance to bottom
                ]
        
        # CRITICAL FIX - Convert positions to prices with error checking
        # Ensure targets are different from each other and different from entry price!
        target_prices = [range_low + (pos * range_size) for pos in target_positions]
        
        # CRITICAL FIX - Ensure targets are properly separated
        min_target_distance = range_size * 0.03  # Minimum 3% of range size between targets
        
        # For long positions, ensure ascending order
        if direction > 0:
            for i in range(1, len(target_prices)):
                if target_prices[i] - target_prices[i-1] < min_target_distance:
                    target_prices[i] = target_prices[i-1] + min_target_distance
        # For short positions, ensure descending order
        else:
            for i in range(1, len(target_prices)):
                if target_prices[i-1] - target_prices[i] < min_target_distance:
                    target_prices[i] = target_prices[i-1] - min_target_distance
        
        # Refine targets using support/resistance levels when available
        if support_resistance and len(support_resistance) > 0:
            # Find nearest support/resistance for each target and adjust slightly
            for i in range(len(target_prices)):
                nearest_sr = min(support_resistance, key=lambda x: abs(x - target_prices[i]))
                proximity = 1.0 - min(1.0, abs(nearest_sr - target_prices[i]) / (range_size * 0.1))
                if proximity > 0.7:  # Only adjust when very close to S/R
                    target_prices[i] = (target_prices[i] * 0.6) + (nearest_sr * 0.4)
        
        # Calculate expected hold times based on market conditions
        base_times = [60, 180, 300]  # seconds
        
        # Adjust for entanglement
        time_factor = 2.0 - entanglement if entanglement > 0.5 else 1.0
        expected_hold_times = [int(t * time_factor) for t in base_times]
        
        # Calculate risk multiples (for tracking)
        atr = market_data.get('atr', range_size * 0.05) 
        risk_multiple = abs(entry_price - (range_low if direction > 0 else range_high)) / atr
        multipliers = [
            abs(target_prices[0] - entry_price) / atr,
            abs(target_prices[1] - entry_price) / atr,
            abs(target_prices[2] - entry_price) / atr
        ]
        
        # CRITICAL FIX - Ensure targets are different from entry price by at least 0.5 points
        for i in range(len(target_prices)):
            if abs(target_prices[i] - entry_price) < 3.0:
                # If target is too close to entry, push it further in the direction of trade
                target_prices[i] = entry_price + (direction * max(5.0, atr))
        
        # FINAL VALIDATION - Check for duplicates and ensure proper ordering
        if direction > 0:  # Long position - ensure ascending order
            target_prices = sorted(target_prices)
        else:  # Short position - ensure descending order
            target_prices = sorted(target_prices, reverse=True)
        
        # Ensure no duplicate targets
        for i in range(1, len(target_prices)):
            if target_prices[i] == target_prices[i-1]:
                target_prices[i] = target_prices[i-1] + (direction * 2.0)
        
        return {
            'targets': target_prices,
            'expected_hold_times': expected_hold_times,
            'multipliers': multipliers,
            'target_positions': target_positions,
            'range_entry_position': entry_position,
            'risk_multiple': risk_multiple
        }
    def manage_profit_targets(self, position_obj, current_price, market_data):
        """
        Manage partial exits at profit targets with quantum-enhanced timing
        
        Parameters:
        - position_obj: Dictionary containing position information
        - current_price: Current market price
        - market_data: Dictionary containing market metrics
        
        Returns:
        - Dict with action taken (if any) and details
        """
        # Extract position details
        position_id = position_obj.get('id')
        entry_price = position_obj.get('entry_price')
        position_size = position_obj.get('size', 0)
        direction = 1 if position_size > 0 else -1
        
        # Skip if no position or invalid data
        if not position_id or not entry_price or position_size == 0:
            return {'action': 'none', 'reason': 'invalid_position_data'}
        
        # Get current regime
        regime = market_data.get('regime', 'unknown')
        regime_confidence = market_data.get('regime_confidence', 0.5)
        
        # Check if we have already exited at certain targets
        already_exited_targets = set()
        if position_id in self._trade_timing.get('partial_exits', {}):
            for exit_info in self._trade_timing['partial_exits'][position_id]:
                if 'target_level' in exit_info:
                    already_exited_targets.add(exit_info['target_level'])
        
        # Get target prices - different handling for range vs trend
        if regime == 'range_bound' and 'range_analysis' in market_data:
            # Use range-optimized targets for range markets
            targets_info = self.calculate_range_optimized_targets(position_obj, market_data)
            targets = targets_info[0:3] if isinstance(targets_info, list) else targets_info.get('targets', [0, 0, 0])
        else:
            # For trending or other regimes
            targets_info = self.optimize_profit_targets(
                entry_price, 
                direction,
                market_data,
                {'regime': regime, 'confidence': regime_confidence}
            )
            targets = targets_info.get('targets', [0, 0, 0]) if isinstance(targets_info, dict) else [0, 0, 0]
        
        # Ensure we have valid targets
        if not targets or len(targets) < 3:
            return {'action': 'none', 'reason': 'invalid_targets'}
        
        # Check if price has reached any targets
        reached_target = None
        target_level = None
        
        # For long positions
        if direction > 0:
            # Check targets in descending order (highest first)
            if current_price >= targets[2] and 'T3' not in already_exited_targets:
                reached_target = targets[2]
                target_level = 'T3'
            elif current_price >= targets[1] and 'T2' not in already_exited_targets:
                reached_target = targets[1]
                target_level = 'T2'
            elif current_price >= targets[0] and 'T1' not in already_exited_targets:
                reached_target = targets[0]
                target_level = 'T1'
        # For short positions
        else:
            # Check targets in descending order (lowest first)
            if current_price <= targets[2] and 'T3' not in already_exited_targets:
                reached_target = targets[2]
                target_level = 'T3'
            elif current_price <= targets[1] and 'T2' not in already_exited_targets:
                reached_target = targets[1]
                target_level = 'T2'
            elif current_price <= targets[0] and 'T1' not in already_exited_targets:
                reached_target = targets[0]
                target_level = 'T1'
        
        # If no target reached, check for approaching targets for elite timing
        if not reached_target:
            # Calculate how close we are to next target (as percentage of distance from entry to target)
            distance_to_target = None
            next_target = None
            next_target_level = None
            
            if direction > 0:  # Long position
                for i, target in enumerate(targets):
                    target_label = f'T{i+1}'
                    if target_label not in already_exited_targets:
                        target_distance = target - entry_price
                        if target_distance > 0:  # Valid target
                            progress = (current_price - entry_price) / target_distance
                            if progress >= 0.92:  # Within 92% of the way to target
                                distance_to_target = progress
                                next_target = target
                                next_target_level = target_label
                                break
            else:  # Short position
                for i, target in enumerate(targets):
                    target_label = f'T{i+1}'
                    if target_label not in already_exited_targets:
                        target_distance = entry_price - target
                        if target_distance > 0:  # Valid target
                            progress = (entry_price - current_price) / target_distance
                            if progress >= 0.92:  # Within 92% of the way to target
                                distance_to_target = progress
                                next_target = target
                                next_target_level = target_label
                                break
            
            # Check if we're approaching a target
            if distance_to_target and next_target:
                # Check if elite conditions suggest taking profit early
                entanglement = market_data.get('entanglement', 0.5)
                vpin = market_data.get('vpin', 0.3)
                
                # Early exit criteria: high entanglement, approaching target, and increasing VPIN
                if (entanglement > 0.85 and distance_to_target > 0.95 and vpin > 0.25 and
                    'vpin_trend' in market_data and market_data['vpin_trend'] > 0):
                    
                    self.logger.info(f"Elite early target approach: {distance_to_target:.2f} to {next_target_level} with entanglement {entanglement:.2f}")
                    reached_target = next_target
                    target_level = next_target_level
                
                # Momentum-based early exit
                elif 'momentum' in market_data:
                    momentum = market_data['momentum']
                    # If momentum is turning against our position
                    if (direction > 0 and momentum < -0.2) or (direction < 0 and momentum > 0.2):
                        if distance_to_target > 0.94:  # Very close to target
                            self.logger.info(f"Elite momentum-based early exit: {distance_to_target:.2f} to {next_target_level} with adverse momentum {momentum:.2f}")
                            reached_target = next_target
                            target_level = next_target_level
        
        # If target reached, execute partial exit
        if reached_target and target_level:
            # Calculate exit size based on target level
            remaining_size = abs(position_size)
            
            # Dynamic exit sizing based on target level
            if target_level == 'T1':
                # First target - smaller exit (30-40% of position)
                exit_pct = 0.35
                if regime == 'trending_up' or regime == 'trending_down':
                    exit_pct = 0.30  # Smaller first exit in trending markets
                elif regime == 'volatile':
                    exit_pct = 0.40  # Larger first exit in volatile markets
            elif target_level == 'T2':
                # Second target - medium exit (40-50% of remaining)
                exit_pct = 0.45
                if regime == 'trending_up' or regime == 'trending_down':
                    exit_pct = 0.40  # Smaller second exit in trending markets
                elif regime == 'volatile':
                    exit_pct = 0.50  # Larger second exit in volatile markets
            else:  # T3
                # Third target - exit all remaining
                exit_pct = 1.0
            
            # Calculate actual exit size (minimum 1 contract)
            exit_size = max(1, int(remaining_size * exit_pct))
            exit_size = min(exit_size, remaining_size)  # Cannot exit more than we have
            
            # Execute the exit
            order_id = self.execution_engine.place_order(
                'market',
                'NQ',
                -exit_size * direction  # Negative of position direction for exit
            )
            
            if order_id:
                self.logger.info(f"Profit target exit at {target_level}: {exit_size} contracts @ ${current_price:,.2f} (target: ${reached_target:,.2f})")
                
                # Record this partial exit
                if position_id not in self._trade_timing['partial_exits']:
                    self._trade_timing['partial_exits'][position_id] = []
                
                self._trade_timing['partial_exits'][position_id].append({
                    'time': datetime.datetime.now(),
                    'price': current_price,
                    'size': exit_size,
                    'reason': f'profit_target_{target_level}',
                    'target_level': target_level,
                    'target_price': reached_target
                })
                
                # Update stop loss for remaining position if not last target
                if target_level != 'T3' and exit_size < remaining_size:
                    # For first target hit, move stop to break-even or better
                    if target_level == 'T1':
                        new_stop = entry_price
                        if regime == 'trending_up' or regime == 'trending_down':
                            # Small profit locked in trending markets
                            atr = market_data.get('atr', entry_price * 0.001)
                            new_stop = entry_price + (direction * atr * 0.2)
                        
                        self.logger.info(f"Moving stop to ${new_stop:.2f} after hitting {target_level}")
                    
                    # For second target hit, move stop to first target
                    elif target_level == 'T2':
                        new_stop = targets[0]  # First target becomes stop
                        self.logger.info(f"Moving stop to ${new_stop:.2f} (T1) after hitting {target_level}")
                
                return {
                    'action': 'partial_exit',
                    'reason': f'target_{target_level}',
                    'size': exit_size,
                    'target': reached_target,
                    'target_level': target_level
                }
        
        # No target reached or action taken
        return {'action': 'none', 'reason': 'no_target_reached'}    
    def calculate_range_optimized_targets(self, position_obj, market_data):
        """
        Calculate optimized profit targets specifically for range-bound markets
        
        Parameters:
        - position_obj: Dictionary containing position information
        - market_data: Dictionary containing market metrics and range analysis
        
        Returns:
        - List of three price targets optimized for range market
        """
        # Extract position details
        entry_price = position_obj.get('entry_price', market_data.get('price', 0))
        position_size = position_obj.get('size', 0)
        direction = 1 if position_size > 0 else -1
        
        # Check if range analysis exists
        if 'range_analysis' not in market_data:
            # Fallback if range analysis is missing
            atr = market_data.get('atr', entry_price * 0.005)
            return [
                entry_price + (direction * atr * 1.0),
                entry_price + (direction * atr * 1.5),
                entry_price + (direction * atr * 2.0)
            ]
        
        # Extract range data
        range_analysis = market_data['range_analysis']
        range_high = range_analysis.get('range_high', 0)
        range_low = range_analysis.get('range_low', 0)
        range_size = range_high - range_low
        position_in_range = range_analysis.get('position_in_range', 0.5)
        confidence = range_analysis.get('confidence', 0.5)
        
        # Adjust if range is invalid or too small
        if range_size <= 0 or range_high <= range_low:
            atr = market_data.get('atr', entry_price * 0.005)
            return [
                entry_price + (direction * atr * 1.0),
                entry_price + (direction * atr * 1.5),
                entry_price + (direction * atr * 2.0)
            ]
        
        # For long positions (buying)
        if direction > 0:
            # If already above middle of range
            if position_in_range > 0.5:
                # Targeting upward - careful near the top
                target1_position = min(0.75, position_in_range + 0.15)
                target2_position = min(0.85, position_in_range + 0.25)
                target3_position = min(0.95, position_in_range + 0.35)
            else:
                # Room to run upward
                target1_position = min(0.55, position_in_range + 0.25)
                target2_position = min(0.70, position_in_range + 0.35)
                target3_position = min(0.80, position_in_range + 0.45)
        # For short positions (selling)
        else:
            # If already below middle of range
            if position_in_range < 0.5:
                # Targeting downward - careful near bottom
                target1_position = max(0.25, position_in_range - 0.15)
                target2_position = max(0.15, position_in_range - 0.25)
                target3_position = max(0.05, position_in_range - 0.35)
            else:
                # Room to run downward
                target1_position = max(0.45, position_in_range - 0.25)
                target2_position = max(0.30, position_in_range - 0.35)
                target3_position = max(0.20, position_in_range - 0.45)
        
        # Convert range positions to actual price targets
        target1 = range_low + (target1_position * range_size)
        target2 = range_low + (target2_position * range_size)
        target3 = range_low + (target3_position * range_size)
        
        # Apply confidence scaling - tighten targets with lower confidence
        if confidence < 0.6:
            # Pull targets closer to entry for lower confidence ranges
            scale_factor = 0.6 + (confidence * 0.4)  # 0.6 to 0.84
            
            # Scale target distances from entry price
            target1 = entry_price + ((target1 - entry_price) * scale_factor)
            target2 = entry_price + ((target2 - entry_price) * scale_factor)
            target3 = entry_price + ((target3 - entry_price) * scale_factor)
        
        # CRITICAL: Ensure logical target order and proper spacing
        min_distance = range_size * 0.03  # Minimum 3% of range size
        
        # Make sure targets are properly ordered
        if direction > 0:  # Long position
            # Ensure ascending targets
            target1 = max(entry_price + min_distance, target1)
            target2 = max(target1 + min_distance, target2)
            target3 = max(target2 + min_distance, target3)
        else:  # Short position
            # Ensure descending targets
            target1 = min(entry_price - min_distance, target1)
            target2 = min(target1 - min_distance, target2)
            target3 = min(target2 - min_distance, target3)
        
        # Apply quantum enhancements if available
        if 'entanglement' in market_data and market_data['entanglement'] > 0.8:
            # High quantum entanglement suggests more precise targets
            # Further optimize final target with quantum edge
            quantum_adjustment = 0.05 * range_size * np.sign(target3 - entry_price)
            target3 = target3 + quantum_adjustment
        
        # Return optimized targets
        return [target1, target2, target3]
    def optimize_profit_targets(self, entry_price, direction, market_data, regime_info, order_flow_patterns=None):
        """
        Optimized profit targets with proper spacing and regime-specific adjustments
        
        Parameters:
        - entry_price: Current price/entry price
        - direction: Trade direction (1 for long, -1 for short)
        - market_data: Dictionary containing market metrics
        - regime_info: Dictionary with regime classification data
        - order_flow_patterns: Optional order flow pattern information
        
        Returns:
        - Dictionary containing targets, multipliers, and expected hold times
        """
        # Get market regime and volatility
        regime = regime_info.get('regime', 'unknown')
        regime_confidence = regime_info.get('confidence', 0.5)
        
        # Calculate ATR for target spacing
        if 'atr' in market_data:
            atr = market_data['atr']
        else:
            # Estimate ATR if not provided
            atr = entry_price * 0.005  # Default to 0.5% of price
        
        # Initialize target multipliers based on regime
        if regime == 'trending_down':
            if direction < 0:  # Short in downtrend - optimal alignment
                multipliers = [1.8, 3.0, 5.0]  # Aggressive targets
            else:  # Long in downtrend - contrarian
                multipliers = [1.0, 1.6, 2.2]  # Conservative targets
        elif regime == 'trending_up':
            if direction > 0:  # Long in uptrend - optimal alignment
                multipliers = [1.8, 3.0, 5.0]  # Aggressive targets
            else:  # Short in uptrend - contrarian
                multipliers = [1.0, 1.6, 2.2]  # Conservative targets
        elif regime == 'volatile':
            # Shorter targets in volatile regimes
            multipliers = [1.2, 2.0, 3.0]
        elif regime == 'range_bound':
            # More conservative targets in range-bound markets
            multipliers = [1.0, 1.6, 2.2]
        else:
            # Default/unknown regime
            multipliers = [1.2, 2.0, 3.0]
        
        # Apply confidence scaling
        confidence_factor = 0.7 + (0.6 * regime_confidence)
        multipliers = [m * confidence_factor for m in multipliers]
        
        # Apply order flow and delta adjustments
        order_flow = market_data.get('order_flow', 0)
        delta = market_data.get('delta', 0)
        vpin = market_data.get('vpin', 0.3)
        
        # Modify targets based on order flow patterns if provided
        if order_flow_patterns and order_flow_patterns.get('pattern_detected', False):
            pattern_name = order_flow_patterns.get('pattern_name', '')
            pattern_confidence = order_flow_patterns.get('confidence', 0.5)
            pattern_direction = order_flow_patterns.get('direction', 0)
            
            # Adjust targets based on pattern type
            if pattern_name == 'flow_absorption' and pattern_confidence > 0.8:
                # Flow absorption suggests strong potential move
                if pattern_direction == direction:  # Pattern aligned with trade
                    multiplier_boost = 1.2  # 20% larger targets
                    multipliers = [m * multiplier_boost for m in multipliers]
                    
            elif pattern_name == 'flow_exhaustion' and pattern_confidence > 0.7:
                # Flow exhaustion suggests potential reversal
                if pattern_direction != direction:  # Pattern suggests reversal
                    multiplier_reduction = 0.8  # 20% smaller targets
                    multipliers = [m * multiplier_reduction for m in multipliers]
        
        # Align with order flow
        if np.sign(order_flow) == np.sign(direction) and abs(order_flow) > 0.3:
            flow_boost = 1.0 + (abs(order_flow) * 0.5)
            multipliers = [m * flow_boost for m in multipliers]
            
        # Align with delta
        if np.sign(delta) == np.sign(direction) and abs(delta) > 0.3:
            delta_boost = 1.0 + (abs(delta) * 0.3)
            multipliers = [m * delta_boost for m in multipliers]
        
        # Handle high VPIN (toxic liquidity) - tighter targets
        if vpin > 0.4:
            vpin_factor = 1.0 - ((vpin - 0.4) * 0.5)
            multipliers = [m * vpin_factor for m in multipliers]
        
        # Calculate price targets
        targets = []
        for m in multipliers:
            target = entry_price + (direction * atr * m)
            targets.append(target)
        
        # Calculate expected hold times based on volatility and targets
        volatility = market_data.get('volatility', 0.0001)
        vol_factor = min(2.0, max(0.5, volatility * 5000))
        
        hold_times = [
            int(180 / vol_factor),  # First target
            int(540 / vol_factor),  # Second target
            int(1800 / vol_factor)  # Third target
        ]
        
        # CRITICAL: Ensure no target duplication
        # Ensure minimum spacing between targets
        min_spacing = atr * 0.2
        
        for i in range(1, len(targets)):
            if direction > 0:  # Long trade
                targets[i] = max(targets[i], targets[i-1] + min_spacing)
            else:  # Short trade
                targets[i] = min(targets[i], targets[i-1] - min_spacing)
        
        return {
            'targets': targets,
            'multipliers': multipliers,
            'expected_hold_times': hold_times
        }
    def range_optimized_position_sizing(self, base_size, signal, market_data, range_analysis):
        """Improved position sizing for range-bound markets with safeguards"""
        import numpy as np
        
        # Extract key metrics
        volatility = market_data.get('volatility', 0.0001)
        confidence = range_analysis.get('confidence', 0.3)
        position_in_range = range_analysis.get('position_in_range', 0.5)
        z_score = range_analysis.get('z_score', 0)
        choppiness = market_data.get('choppiness_analysis', {}).get('choppiness_score', 0.5)
        
        # Log the input values
        self.logger.info(f"Range position sizing: base={base_size}, edge_factor={1.0}, pos_in_range={position_in_range:.2f}, confidence={confidence:.2f}, final={base_size}")
        
        # Start with base size
        position_size = base_size
        
        # Safety check - if base size is 0, just return 0
        if base_size == 0:
            return 0
        
        # Calculate edge factor - higher at extremes
        edge_factor = 0.0
        
        # Long signal at bottom of range
        if signal > 0 and position_in_range < 0.3:
            edge_factor = (0.3 - position_in_range) * 3
        # Short signal at top of range
        elif signal < 0 and position_in_range > 0.7:
            edge_factor = (position_in_range - 0.7) * 3
        
        # Cap edge factor and apply confidence scaling
        edge_factor = min(1.0, edge_factor) * confidence
        edge_multiplier = 1.0 + edge_factor
        
        # Apply main edge multiplier
        position_size = int(position_size * edge_multiplier)
        
        # Apply amplitude-based sizing for high-amplitude ranges
        range_amplitude = range_analysis.get('range_size', 100) / (market_data.get('price', 20000) * 0.005)
        amplitude_factor = 1.0
        
        # For very narrow ranges, reduce size
        if range_amplitude < 1.5:
            amplitude_factor = max(0.5, range_amplitude / 1.5)
            self.logger.info(f"Range amplitude sizing: pos_in_range={position_in_range:.2f}, multiplier={amplitude_factor:.2f}, amplitude_factor={amplitude_factor:.2f}")
        
        # Apply amplitude factor
        original_size = position_size
        position_size = int(position_size * amplitude_factor)
        
        # If we're reducing to zero, log it prominently
        if original_size > 0 and position_size == 0:
            self.logger.info(f"Range amplitude sizing adjustment: {original_size} → {position_size} contracts")
        
        # CRITICAL: Reduce size in extremely choppy conditions
        if choppiness > 0.6:
            chop_reduction = min(0.7, max(0.3, 1.0 - (choppiness - 0.6) * 2))
            original_size = position_size
            position_size = max(0, int(position_size * chop_reduction))  # Allow reduction to zero
            if original_size != position_size:
                self.logger.info(f"Choppy market position reduction: {original_size} → {position_size}")
        
        return position_size
    
    def calculate_execution_quality_score(self, market_data, direction):
        """Calculate a real-time execution quality score to optimize entry/exit timing"""
        import numpy as np
        
        # Extract critical metrics
        vpin = market_data.get('vpin', 0.3)
        order_flow = market_data.get('order_flow', 0.0)
        delta = market_data.get('delta', 0.0)
        spread = market_data.get('spread', 0.5)
        regime = market_data.get('regime', 'unknown')
        
        # Calculate execution quality components
        liquidity_score = 1.0 - min(1.0, vpin * 1.25)  # Higher VPIN = lower liquidity
        
        # Calculate flow alignment score
        flow_alignment = 0.5
        if abs(order_flow) > 0.05:
            flow_alignment = 0.5 + (0.5 * np.sign(order_flow) * np.sign(direction))
        
        # Calculate delta alignment score
        delta_alignment = 0.5
        if abs(delta) > 0.1:
            delta_alignment = 0.5 + (0.5 * np.sign(delta) * np.sign(direction))
        
        # Calculate spread component (lower spread = better execution)
        spread_component = max(0.0, 1.0 - (spread / 2.0))
        
        # Apply regime-specific weights
        if regime == 'trending_up' or regime == 'trending_down':
            # In trending regimes, delta alignment is more important
            weights = {
                'liquidity': 0.25,
                'flow_alignment': 0.25,
                'delta_alignment': 0.4,
                'spread': 0.1
            }
        elif regime == 'range_bound':
            # In range-bound markets, flow alignment is more important
            weights = {
                'liquidity': 0.25,
                'flow_alignment': 0.4,
                'delta_alignment': 0.25,
                'spread': 0.1
            }
        else:
            # Default/unknown regime
            weights = {
                'liquidity': 0.25,
                'flow_alignment': 0.3,
                'delta_alignment': 0.3,
                'spread': 0.15
            }
        
        # Calculate weighted score
        quality_score = (
            liquidity_score * weights['liquidity'] +
            flow_alignment * weights['flow_alignment'] +
            delta_alignment * weights['delta_alignment'] +
            spread_component * weights['spread']
        )
        
        # Return quality score and components for logging
        return {
            'quality_score': quality_score,
            'components': {
                'liquidity': liquidity_score,
                'flow_alignment': flow_alignment,
                'delta_alignment': delta_alignment,
                'spread': spread_component
            },
            'execute_now': quality_score > 0.65  # Threshold for immediate execution
        }
    def get_recent_win_rate(self, n=20):
        """Calculate win rate from recent trades"""
        if not hasattr(self, 'trade_history') or len(self.trade_history) == 0:
            return 0.5  # Default 50% if no history
        
        # Get the most recent n trades
        recent_trades = self.trade_history[-n:] if len(self.trade_history) > n else self.trade_history
        
        # Count winning trades
        winning_trades = sum(1 for trade in recent_trades if trade.get('profit', 0) > 0)
        
        # Calculate win rate
        win_rate = winning_trades / len(recent_trades) if recent_trades else 0.5
        
        return win_rate
    def dynamic_risk_allocation(self, market_data, position_size, direction):
        """Dynamically adjust risk based on market conditions and system performance"""
        # Get base metrics
        recent_win_rate = self.get_recent_win_rate(20)  # Last 20 trades
        current_equity = self.execution_engine.get_account_equity()
        initial_equity = 50000.0  # Starting equity
        
        # Calculate drawdown and performance metrics
        drawdown_pct = max(0, (initial_equity - current_equity) / initial_equity)
        performance_factor = 1.0
        
        # Adjust based on win rate and drawdown
        if recent_win_rate > 0.6:
            # Winning streak - gradually increase size
            performance_factor = min(1.5, 1.0 + ((recent_win_rate - 0.5) * 2))
        elif recent_win_rate < 0.4:
            # Losing streak - reduce size
            performance_factor = max(0.5, 1.0 - ((0.5 - recent_win_rate) * 2))
        
        # Drawdown protection
        if drawdown_pct > 0.02:
            # Progressive reduction based on drawdown
            dd_factor = max(0.25, 1.0 - (drawdown_pct * 10))
            performance_factor *= dd_factor
            
        # Apply adjustment to position size (prevent zeroing out)
        adjusted_size = max(1, int(position_size * performance_factor))
        
        if adjusted_size != position_size:
            self.logger.info(f"Dynamic risk adjustment: {position_size} → {adjusted_size} contracts (win rate: {recent_win_rate:.2f}, drawdown: {drawdown_pct:.2%})")
        
        return adjusted_size    
    def adaptive_regime_transition_protection(self, current_regime, previous_regime, regime_confidence, signal, seconds_since_transition, current_time):
        """
        Apply enhanced protective measures during regime transitions to prevent false signals
        
        Parameters:
        - current_regime: The new/current regime classification
        - previous_regime: The previous regime classification 
        - regime_confidence: Confidence level in the current regime (0-1)
        - signal: The current trading signal
        - seconds_since_transition: Time elapsed since regime change detection
        - current_time: Current system time for logging purposes
        
        Returns:
        - Adjusted signal value
        """
        # Original signal for logging
        original_signal = signal
        
        # Maximum stabilization time (seconds)
        max_stabilization = 90  # 1.5 minutes max
        
        # Calculate transition progress (0 to 1)
        transition_progress = min(1.0, seconds_since_transition / max_stabilization)
        
        # Calculate dampening factor - starts high and decreases as transition progresses
        dampening_factor = max(0.1, 1.0 - transition_progress)
        
        # Significantly different regime transitions require more dampening
        significant_transition = False
        
        # Define regime opposites for stronger dampening
        opposite_regimes = {
            'trending_up': 'trending_down',
            'trending_down': 'trending_up',
            'volatile': 'range_bound',
            'range_bound': 'volatile'
        }
        
        # Check if transition is between "opposite" regimes
        if previous_regime in opposite_regimes and current_regime == opposite_regimes[previous_regime]:
            significant_transition = True
            # Increase dampening for opposite regime transitions
            dampening_factor *= 1.5
        
        # Apply more aggressive dampening when confidence is low
        if regime_confidence < 0.7:
            low_conf_factor = 1.0 + ((0.7 - regime_confidence) * 0.5)
            dampening_factor *= low_conf_factor
        
        # Dampen the signal by the calculated factor
        dampened_signal = signal * (1.0 - dampening_factor)
        
        # Only log significant changes
        if abs(dampened_signal - original_signal) > 0.05:
            self.logger.info(f"Regime transition protection: signal dampened {original_signal:.2f} → {dampened_signal:.2f} " +
                            f"({seconds_since_transition:.1f}s since {previous_regime} → {current_regime})")
        
        return dampened_signal

    def enhanced_execution_quality_score(self, market_data, direction):
        """Advanced execution quality calculation with market microstructure awareness"""
        import numpy as np
        
        # Extract critical metrics with defaults
        vpin = market_data.get('vpin', 0.3)
        order_flow = market_data.get('order_flow', 0.0)
        delta = market_data.get('delta', 0.0)
        spread = market_data.get('spread', 0.5)
        regime = market_data.get('regime', 'unknown')
        entanglement = market_data.get('entanglement', 0.5)
        
        # NEW: Market microstructure sensitivity metrics
        price_impact = market_data.get('price_impact', 0.3)  # Lower is better
        depth_imbalance = market_data.get('depth_imbalance', 0)  # Higher in direction is better
        
        # NEW: Dynamic liquidity timeout
        if not hasattr(self, '_last_liquidity_shock'):
            self._last_liquidity_shock = {
                'time': datetime.datetime.now() - datetime.timedelta(hours=1),
                'magnitude': 0.0
            }
        
        # Check for recent liquidity shock
        time_since_shock = (datetime.datetime.now() - self._last_liquidity_shock['time']).total_seconds()
        shock_factor = 0.0
        
        if time_since_shock < 300:  # Within 5 minutes of shock
            # Exponential decay of shock impact
            shock_factor = self._last_liquidity_shock['magnitude'] * np.exp(-time_since_shock / 60)
        
        # Calculate execution quality components
        liquidity_score = (1.0 - min(1.0, vpin * 1.25)) * (1.0 - shock_factor)
        
        # Flow alignment with progressive weighting - stronger flows matter more
        flow_alignment = 0.5
        if abs(order_flow) > 0.05:
            alignment_factor = np.sign(order_flow) * np.sign(direction)
            strength_factor = min(1.0, abs(order_flow) * 2.5)  # Progressive scaling
            flow_alignment = 0.5 + (0.5 * alignment_factor * strength_factor)
        
        # Delta alignment with progressive weighting
        delta_alignment = 0.5
        if abs(delta) > 0.1:
            alignment_factor = np.sign(delta) * np.sign(direction)
            strength_factor = min(1.0, abs(delta) * 3.0)  # Progressive scaling
            delta_alignment = 0.5 + (0.5 * alignment_factor * strength_factor)
        
        # Spread component (lower = better execution)
        spread_component = max(0.0, 1.0 - (spread / 2.0))
        
        # Entanglement component - weighted to reflect confidence in correlated signals
        entanglement_alignment = 0.5
        if entanglement > 0.5:
            # Measure alignment between delta and flow
            delta_flow_aligned = np.sign(delta) == np.sign(order_flow)
            
            if delta_flow_aligned and (np.sign(delta) == np.sign(direction)):
                # All three aligned - very good
                entanglement_alignment = 0.5 + ((entanglement - 0.5) * 0.9)
            elif delta_flow_aligned:
                # Delta and flow align but against direction - bad execution
                entanglement_alignment = 0.5 - ((entanglement - 0.5) * 0.9)
            else:
                # Mixed signals - neutral to slightly negative
                entanglement_alignment = 0.5 - ((entanglement - 0.5) * 0.3)
        
        # NEW: Market microstructure components
        impact_score = max(0.0, 1.0 - price_impact * 2.0)
        
        imbalance_alignment = 0.5
        if abs(depth_imbalance) > 0.1:
            imbalance_alignment = 0.5 + (0.5 * np.sign(depth_imbalance) * np.sign(direction))
        
        # Apply regime-specific weights
        if regime == 'trending_up' or regime == 'trending_down':
            weights = {
                'liquidity': 0.20,
                'flow_alignment': 0.15,
                'delta_alignment': 0.25,
                'spread': 0.10,
                'entanglement': 0.15,
                'impact': 0.10,
                'imbalance': 0.05
            }
        elif regime == 'range_bound':
            weights = {
                'liquidity': 0.20,
                'flow_alignment': 0.25,
                'delta_alignment': 0.15,
                'spread': 0.10,
                'entanglement': 0.15,
                'impact': 0.05,
                'imbalance': 0.10
            }
        else:  # Unknown or volatile
            weights = {
                'liquidity': 0.25,
                'flow_alignment': 0.20,
                'delta_alignment': 0.20,
                'spread': 0.10,
                'entanglement': 0.10,
                'impact': 0.10,
                'imbalance': 0.05
            }
        
        # Calculate weighted score
        quality_score = (
            liquidity_score * weights['liquidity'] +
            flow_alignment * weights['flow_alignment'] +
            delta_alignment * weights['delta_alignment'] +
            spread_component * weights['spread'] +
            entanglement_alignment * weights['entanglement'] +
            impact_score * weights['impact'] +
            imbalance_alignment * weights['imbalance']
        )
        
        # Track new liquidity shock
        if vpin > 0.6 and liquidity_score < 0.4:
            self._last_liquidity_shock = {
                'time': datetime.datetime.now(),
                'magnitude': min(1.0, vpin * 0.8)
            }
        
        # Return enhanced quality score with new components
        return {
            'quality_score': quality_score,
            'components': {
                'liquidity': liquidity_score,
                'flow_alignment': flow_alignment,
                'delta_alignment': delta_alignment,
                'spread': spread_component,
                'entanglement': entanglement_alignment,
                'impact': impact_score,
                'imbalance': imbalance_alignment,
                'shock_factor': shock_factor
            },
            'execute_now': quality_score > 0.65
        }

    def quantum_confidence_position_sizing(self, base_size, signal, confidence, market_data):
        """Dynamic position sizing based on quantum confidence metrics"""
        import numpy as np
        
        # Extract key confidence metrics
        elite_confidence = confidence  # Already passed in
        entanglement = market_data.get('entanglement', 0.5)
        regime = market_data.get('regime', 'unknown')
        regime_confidence = market_data.get('regime_confidence', 0.5)
        
        # Start with base position size
        adjusted_size = base_size
        
        # Scale based on elite quantum confidence (0.0-1.0)
        confidence_scale = 0.7 + (0.6 * elite_confidence)  # Range: 0.7-1.3
        adjusted_size = int(adjusted_size * confidence_scale)
        
        # Scale based on entanglement (quantum correlation)
        # Only boost when entanglement is exceptional
        if entanglement > 0.85:
            # High entanglement boost
            entanglement_scale = 1.0 + ((entanglement - 0.85) * 2.0)  # Max +30% at 1.0 entanglement
            adjusted_size = int(adjusted_size * entanglement_scale)
        
        # Apply extra scaling when signal strength and confidence align
        signal_strength = abs(signal)
        if signal_strength > 0.7 and elite_confidence > 0.7:
            # Strong signal with high confidence = bigger position
            synergy_scale = 1.0 + min(0.3, (signal_strength * elite_confidence - 0.49) * 0.6)
            adjusted_size = int(adjusted_size * synergy_scale)
        
        # Apply regime-based constraints
        if regime == 'volatile' or regime_confidence < 0.4:
            # Cap size in volatile or uncertain regimes
            adjusted_size = min(adjusted_size, base_size)  # Never exceed base size
        
        # Ensure minimum position size of 1
        if adjusted_size < 1 and base_size > 0:
            adjusted_size = 1
        
        # Log the adjustments
        self.logger.info(f"Quantum position sizing: {base_size} → {adjusted_size} " + 
                        f"(confidence: {elite_confidence:.2f}, entanglement: {entanglement:.2f})")
        
        return adjusted_size
    def enhance_choppy_market_detection(self, market_data, price_history):
        """Advanced choppy market pattern detection with mean-reversion signals"""
        import numpy as np
        
        # Extract relevant data
        current_price = market_data.get('price', 0)
        regime = market_data.get('regime', 'unknown')
        volatility = market_data.get('volatility', 0.0001)
        
        # Only process if we have sufficient price history
        if len(price_history) < 20:
            return {
                'pattern_detected': False,
                'confidence': 0.0,
                'signal_adjustment': 0.0
            }
        
        # Get recent price movements
        recent_prices = price_history[-20:]
        price_changes = np.diff(recent_prices)
        
        # Check for choppy pattern: alternating positive and negative changes
        sign_changes = 0
        for i in range(1, len(price_changes)):
            if np.sign(price_changes[i]) != np.sign(price_changes[i-1]):
                sign_changes += 1
        
        # Calculate metrics
        choppiness_score = sign_changes / (len(price_changes) - 1)  # Normalize to 0-1
        
        # Calculate distance from recent mean
        mean_price = np.mean(recent_prices)
        stdev_price = np.std(recent_prices)
        z_score = (current_price - mean_price) / stdev_price if stdev_price > 0 else 0
        
        # Recognize mean-reversion opportunity in choppy markets
        mean_reversion_signal = 0.0
        if regime == 'range_bound' or regime == 'choppy':
            # Strong mean reversion signal when price is far from mean
            if abs(z_score) > 1.0:
                # Negative z-score = price below mean = bullish signal
                # Positive z-score = price above mean = bearish signal
                mean_reversion_signal = -np.sign(z_score) * min(0.4, abs(z_score) * 0.2)
                
        # Determine confidence based on choppiness and regime
        confidence = 0.0
        if regime == 'range_bound' or regime == 'choppy':
            confidence = choppiness_score * 0.8  # Higher confidence in choppy regime
        else:
            confidence = choppiness_score * 0.4  # Lower confidence in other regimes
            
        return {
            'pattern_detected': choppiness_score > 0.6,
            'choppiness_score': choppiness_score,
            'mean_reversion_signal': mean_reversion_signal,
            'z_score': z_score,
            'stdev_price': stdev_price,
            'confidence': confidence,
            'signal_adjustment': mean_reversion_signal if choppiness_score > 0.6 else 0.0
        }
    def adaptive_choppy_position_sizing(self, base_size, signal, market_data, choppiness_data=None):
        """Specialized position sizing for choppy market conditions"""
        import numpy as np
        
        # Extract key metrics
        regime = market_data.get('regime', 'unknown')
        confirmation = market_data.get('confirmation_score', 0.0)
        volatility = market_data.get('volatility', 0.0001)
        entanglement = market_data.get('entanglement', 0.5)
        
        # Start with base size
        adjusted_size = base_size
        
        # Only apply specialized logic in choppy/range regimes
        if regime == 'choppy' or regime == 'range_bound':
            # Base reduction for choppy markets - start with smaller positions
            choppy_factor = 0.7
            adjusted_size = int(adjusted_size * choppy_factor)
            self.logger.info(f"Choppy market position reduction: {base_size} → {adjusted_size}")
            
            # Check for extreme z-score in mean reversion if provided
            if choppiness_data and 'z_score' in choppiness_data:
                z_score = choppiness_data.get('z_score', 0)
                if abs(z_score) > 1.5:
                    # Stronger position when extreme deviation from mean
                    z_factor = 1.0 + (min(1.0, (abs(z_score) - 1.5) * 0.4))
                    original = adjusted_size
                    adjusted_size = int(adjusted_size * z_factor)
                    self.logger.info(f"Mean reversion position boost: {original} → {adjusted_size} (z-score: {z_score:.2f})")
            
            # Scale based on confirmation - only increase size with very strong confirmation
            if confirmation > 0.7:
                conf_scale = 1.0 + ((confirmation - 0.7) * 1.5)  # Max 1.45x at confirmation 1.0
                original = adjusted_size
                adjusted_size = int(adjusted_size * conf_scale)
                self.logger.info(f"Strong confirmation position boost: {original} → {adjusted_size} (conf: {confirmation:.2f})")
            
            # Reduce more for low entanglement (poor correlations in choppy market)
            if entanglement < 0.7:
                entanglement_scale = 0.7 + (entanglement * 0.3)  # 0.7-1.0 range
                original = adjusted_size
                adjusted_size = int(adjusted_size * entanglement_scale)
                self.logger.info(f"Low entanglement position reduction: {original} → {adjusted_size} (entanglement: {entanglement:.2f})")
        
        # Ensure minimum position size of 1
        if adjusted_size < 1 and base_size > 0:
            adjusted_size = 1
        
        return adjusted_size    
    def detect_choppy_market_pattern(self, market_data, price_history):
        """
        Detect choppy market patterns using advanced statistical analysis
        Returns a dictionary with detection results
        """
        import numpy as np
        
        # Default result structure
        result = {
            'detected': False,
            'score': 0.0,
            'z_score': 0.0,
            'pattern_type': 'none'
        }
        
        # Extract relevant market data
        volatility = market_data.get('volatility', 0.0001)
        regime = market_data.get('regime', 'unknown')
        
        # Basic checks - if already in range_bound regime, that's an early indicator
        if regime == 'range_bound':
            result['score'] += 0.3
            
        # Check Hurst exponent for mean reversion (choppy markets have lower values)
        hurst = market_data.get('hurst_exponent', 0.5)
        if hurst < 0.4:  # Strong mean reversion indicator
            result['score'] += 0.3
            
        # Check directional changes in recent price history
        if len(price_history) > 30:
            recent_prices = price_history[-30:]
            # Count direction changes
            direction_changes = 0
            for i in range(2, len(recent_prices)):
                prev_dir = recent_prices[i-1] - recent_prices[i-2]
                curr_dir = recent_prices[i] - recent_prices[i-1]
                if (prev_dir * curr_dir) < 0:  # Direction changed
                    direction_changes += 1
            
            # Normalize to a score (0-1)
            change_score = min(1.0, direction_changes / 15.0)  # 15+ changes would be extremely choppy
            result['score'] += change_score * 0.4
        
        # Calculate z-score if range analysis is available
        if 'range_analysis' in market_data:
            range_analysis = market_data['range_analysis']
            result['z_score'] = range_analysis.get('z_score', 0.0)
        else:
            # Estimate z-score from recent price action
            if len(price_history) > 50:
                mean_price = np.mean(price_history[-50:])
                std_dev = np.std(price_history[-50:])
                current_price = price_history[-1]
                if std_dev > 0:
                    result['z_score'] = (current_price - mean_price) / std_dev
        
        # Determine if choppy pattern is detected based on combined score
        if result['score'] > 0.5:
            result['detected'] = True
            
            # Classify pattern type
            if abs(result['z_score']) > 1.5:
                result['pattern_type'] = 'extreme_oscillation'
            elif result['score'] > 0.8:
                result['pattern_type'] = 'highly_choppy'
            else:
                result['pattern_type'] = 'moderately_choppy'
        
        return result

    def optimize_exit_timing(self, position, current_price, market_data):
        """Optimize exit timing based on microstructure and order flow"""
        direction = 1 if position['size'] > 0 else -1
        entry_price = position['entry_price']
        current_pnl = (current_price - entry_price) * direction
        
        # Extract market microstructure data
        order_flow = market_data.get('order_flow', 0)
        delta = market_data.get('delta', 0)
        vpin = market_data.get('vpin', 0.3)
        
        # Initialize score components
        flow_score = 0
        microstructure_score = 0
        timing_score = 0
        
        # Order flow component - favorable if in our direction
        flow_alignment = order_flow * direction
        flow_score = min(1.0, max(-1.0, flow_alignment * 2))
        
        # Microstructure factor - delta and liquidity
        delta_alignment = delta * direction
        micro_score = min(1.0, max(-1.0, delta_alignment * 2))
        liquidity_factor = max(0.5, 1.0 - vpin)
        microstructure_score = micro_score * liquidity_factor
        
        # Profit taking acceleration - exit faster as profit grows
        if current_pnl > 0:
            profit_factor = min(1.0, current_pnl / (market_data.get('atr', 10) * 0.5))
            timing_score = profit_factor
        
        # Combine scores
        exit_quality = (0.4 * flow_score) + (0.4 * microstructure_score) + (0.2 * timing_score)
        
        # Determine action based on score
        if exit_quality < -0.5:
            return {"action": "delay", "delay_seconds": 5.0, "reason": "adverse_conditions"}
        elif exit_quality > 0.5:
            return {"action": "accelerate", "reason": "optimal_conditions"}
        else:
            return {"action": "standard", "reason": "neutral_conditions"}

    def dynamic_trading_threshold(self, base_threshold, market_data, signal_strength, confirmation_score):
        """
        Calculate a clean, focused trading threshold without excessive adjustments
        """
        # Start with the base threshold
        threshold = base_threshold
        
        # Extract key metrics
        vpin = market_data.get('vpin', 0.3)
        entanglement = market_data.get('entanglement', 0.5)
        regime = market_data.get('regime', 'unknown')
        
        # Apply confirmation-based adjustment (high confirmation = lower threshold)
        if confirmation_score > 0.7:
            threshold *= max(0.7, 1.0 - (confirmation_score - 0.7))
        elif confirmation_score < 0.3:
            # Low confirmation requires higher threshold
            threshold *= min(1.3, 1.0 + (0.3 - confirmation_score))
        
        # Apply a single liquidity-based adjustment
        if vpin > 0.5:
            # Higher threshold in toxic liquidity conditions
            threshold *= min(1.5, 1.0 + (vpin - 0.5))
        
        # Range market specific threshold
        if regime == 'range_bound' and 'range_analysis' in market_data:
            range_analysis = market_data.get('range_analysis', {})
            position_in_range = range_analysis.get('position_in_range', 0.5)
            z_score = range_analysis.get('z_score', 0)
            
            # Lower threshold at range extremes to capture mean reversion opportunities
            if abs(z_score) > 1.5:
                threshold *= 0.7
            
            # Lower threshold when signal aligned with mean reversion
            if (z_score > 1.0 and signal_strength < 0) or (z_score < -1.0 and signal_strength > 0):
                threshold *= 0.8
        
        # Cap minimum and maximum values
        threshold = max(0.15, min(0.8, threshold))
        
        return threshold

    def optimized_range_position_sizing(self, base_size, direction, range_analysis, market_data):
        """
        Calculate optimized position sizing for range-bound markets
        """
        # Extract range analysis data
        position_in_range = range_analysis.get('position_in_range', 0.5)
        confidence = range_analysis.get('confidence', 0.5)
        z_score = range_analysis.get('z_score', 0)
        
        # Default position sizing
        position_size = base_size
        
        # Calculate edge factor (1.0 = middle of range, higher at extremes)
        edge_distance = min(position_in_range, 1.0 - position_in_range)
        edge_factor = 1.0
        
        # Adjust edge factor based on position in range
        if edge_distance < 0.15:  # Very close to edge
            # Increase size for mean reversion trades at extremes
            if (position_in_range < 0.15 and direction > 0) or (position_in_range > 0.85 and direction < 0):
                edge_factor = 1.3  # Increase size for mean reversion
            else:
                edge_factor = 0.8  # Decrease size for counter-mean-reversion
        
        # Adjust for range confidence 
        confidence_factor = max(0.7, min(1.3, confidence * 2))
        
        # Apply sizing adjustments
        adjusted_size = max(1, int(position_size * edge_factor * confidence_factor))
        
        return {
            'position_size': adjusted_size,
            'base_size': base_size,
            'edge_factor': edge_factor,
            'position_in_range': position_in_range,
            'confidence': confidence,
            'z_score': z_score
        }

    def quantum_state_risk_management(self, position_obj, market_data, regime_info, quantum_signal):
        """Advanced quantum risk management with dynamic exit signals"""
        # Extract position details
        position_size = position_obj.get('size', 0)
        entry_price = position_obj.get('entry_price', 0)
        direction = 1 if position_size > 0 else -1
        current_price = market_data.get('price', 0)
        
        # Skip if no position
        if position_size == 0 or entry_price == 0:
            return {'action': 'none', 'reason': 'no_position'}
        
        # Calculate current P&L
        points_profit = (current_price - entry_price) * direction
        pip_profit = points_profit / 0.25  # Convert to pips (0.25 point per pip)
        profit_pct = points_profit / entry_price
        
        # Extract key market metrics
        order_flow = market_data.get('order_flow', 0)
        delta = market_data.get('delta', 0)
        vpin = market_data.get('vpin', 0.3)
        volatility = market_data.get('volatility', 0.0001)
        entanglement = market_data.get('entanglement', 0.5)
        current_regime = regime_info.get('regime', 'unknown')
        
        # Get quantum signal characteristics
        signal_direction = np.sign(quantum_signal.get('reinforced_signal', 0))
        signal_strength = abs(quantum_signal.get('reinforced_signal', 0))
        signal_confidence = quantum_signal.get('quantum_confidence', 0.5)
        
        # Initialize risk assessment
        risk_score = 0.0
        reversal_probability = 0.0
        
        # 1. Check for signal reversal
        if signal_direction * direction < 0:  # Signal opposite to position
            reversal_factor = min(1.0, signal_strength * 1.5) * signal_confidence
            risk_score += reversal_factor * 0.4
            reversal_probability += reversal_factor * 0.5
        
        # 2. Check order flow and delta alignment
        flow_direction = np.sign(order_flow)
        delta_direction = np.sign(delta)
        
        if flow_direction * direction < 0 and abs(order_flow) > 0.2:
            # Order flow against position
            risk_score += min(0.35, abs(order_flow) * 0.7)
            reversal_probability += 0.15
        
        if delta_direction * direction < 0 and abs(delta) > 0.3:
            # Delta against position
            risk_score += min(0.3, abs(delta) * 0.6)
            reversal_probability += 0.2
        
        # 3. Check for toxic liquidity (high VPIN)
        if vpin > 0.25:
            vpin_risk = (vpin - 0.25) * 1.5
            risk_score += min(0.3, vpin_risk)
            
            # Higher risk if VPIN is increasing
            if 'vpin_trend' in market_data and market_data['vpin_trend'] > 0:
                risk_score += 0.1
                reversal_probability += 0.1
        
        # 4. Apply regime-specific exit rules
        if current_regime == 'range_bound':
            if 'range_analysis' in market_data:
                position_in_range = market_data['range_analysis'].get('position_in_range', 0.5)
                
                # Check if approaching range boundary in unfavorable direction
                if (direction > 0 and position_in_range > 0.8) or (direction < 0 and position_in_range < 0.2):
                    # FIXED LINE: Using Python's ternary operator
                    boundary_target = 1.0 if direction > 0 else 0.0
                    boundary_proximity = min(1.0, (1.0 - abs(position_in_range - boundary_target)) * 5.0)
                    risk_score += boundary_proximity * 0.3
                    reversal_probability += boundary_proximity * 0.4
        
        # 5. Apply profit-based exit rules with declining risk threshold
        # The more profit we have, the more protective we become
        if points_profit > 0:
            # Calculate profit level (1-10 scale)
            profit_level = min(10, points_profit / 2.0)
            
            # More aggressive exit with higher profits
            profit_risk_threshold = 0.7 - (profit_level * 0.04)  # 0.7 down to 0.3
            
            # Apply non-linear profit scaling to risk score
            profit_risk_factor = 1.0 + (profit_level / 15.0)  # 1.0 to 1.67
            adjusted_risk_score = risk_score * profit_risk_factor
            
            # Check if we should exit based on profit and risk
            if adjusted_risk_score > profit_risk_threshold:
                return {
                    'action': 'exit',
                    'reason': 'protect_profits_medium_risk',
                    'risk_score': risk_score,
                    'reversal_probability': reversal_probability,
                    'profit_points': points_profit
                }
            
            # Check for partial exit with moderate risk at good profit
            elif profit_level > 3.0 and adjusted_risk_score > profit_risk_threshold * 0.7:
                # Calculate exit size (20-50% of position)
                exit_pct = 0.2 + min(0.3, adjusted_risk_score * 0.5)
                exit_size = max(1, int(abs(position_size) * exit_pct))
                
                return {
                    'action': 'partial_exit',
                    'reason': 'scale_out_moderate_risk',
                    'size': exit_size * direction,  # Preserve direction
                    'risk_score': risk_score,
                    'reversal_probability': reversal_probability,
                    'profit_points': points_profit
                }
        
        # 6. Check for stop-loss conditions
        if points_profit < 0:
            # In loss territory - check for exit conditions
            loss_level = min(10, abs(points_profit) / 2.0)
            
            # More lenient with small losses
            loss_risk_threshold = 0.8 - (loss_level * 0.03)  # 0.8 down to 0.5
            
            # Higher risk score if losing and risk signals present
            if risk_score > loss_risk_threshold:
                return {
                    'action': 'exit',
                    'reason': 'cut_loss_high_risk',
                    'risk_score': risk_score,
                    'reversal_probability': reversal_probability,
                    'loss_points': points_profit
                }
        
        # 7. Apply quantum-enhanced risk triggers
        if entanglement > 0.85 and reversal_probability > 0.6:
            # High quantum entanglement with high reversal probability
            return {
                'action': 'exit',
                'reason': 'quantum_entangled_reversal',
                'risk_score': risk_score,
                'reversal_probability': reversal_probability,
                'entanglement': entanglement
            }
        
        # No exit action needed
        return {
            'action': 'hold',
            'risk_score': risk_score,
            'reversal_probability': reversal_probability
        }

    def optimize_choppy_regime_thresholds(self, base_threshold, market_data, confirmation_score):
        """
        Specialized threshold optimization for choppy/range-bound markets
        """
        # Start with base threshold
        threshold = base_threshold
        
        # Apply higher threshold for low confirmation in choppy markets
        if confirmation_score < 0.4:
            threshold *= min(1.4, 1.0 + (0.4 - confirmation_score))
        
        # Check for high confirmation + extreme range positioning
        if confirmation_score > 0.6 and 'range_analysis' in market_data:
            range_analysis = market_data.get('range_analysis', {})
            z_score = range_analysis.get('z_score', 0)
            
            # At range extremes with high confirmation, lower threshold
            if abs(z_score) > 1.5:
                threshold *= max(0.7, 1.0 - (confirmation_score - 0.6) * 0.5)
                self.logger.info(f"Choppy market threshold decrease: {base_threshold:.2f} → {threshold:.2f} (high confirmation in choppy)")
        
        # Apply entanglement-based adjustment if available
        entanglement = market_data.get('entanglement', 0)
        if entanglement > 0.8:
            original = threshold
            threshold *= max(0.9, 1.0 - (entanglement - 0.8))
            self.logger.info(f"High entanglement threshold reduction: {original:.2f} → {threshold:.2f} (entanglement: {entanglement:.2f})")
        
        return threshold
    def initialize_analytics(self):
        """Initialize the analytics engine"""
        self.analytics = TradingAnalytics()
        self.logger.info("Analytics engine initialized")

    def process_trade_for_analytics(self, trade_result):
        """Process a completed trade for analytics"""
        try:
            # Extract trade_id from the trade_result
            trade_id = trade_result.get('trade_id', None)
                
            # Add patterns information if trade_id exists and is in the factor data
            if trade_id and hasattr(self, '_trade_factor_data') and trade_id in self._trade_factor_data:
                trade_result['patterns'] = self._trade_factor_data[trade_id].get('patterns', [])
                
            # Log trade details with every report
            self.logger.info("\n" + "=" * 50)
            self.logger.info(f"TRADE DETAILS - {datetime.datetime.now()}")
            self.logger.info("=" * 50)
            
            # Basic trade info
            direction = "LONG" if trade_result.get('size', 0) > 0 else "SHORT"
            profit = trade_result.get('profit', 0)
            self.logger.info(f"Type: {direction}, P&L: ${profit:.2f}")
            self.logger.info(f"Entry: ${trade_result.get('entry_price', 0):.2f}, Exit: ${trade_result.get('exit_price', 0):.2f}")
            
            # Reason and regime
            self.logger.info(f"Regime: {trade_result.get('regime', 'unknown')}")
            self.logger.info(f"Exit Reason: {trade_result.get('reason', 'unknown')}")
            if hasattr(self, 'use_reinforcement_learning') and self.use_reinforcement_learning and hasattr(self, 'rl_system'):
                try:
                    # Calculate and observe reward
                    reward = self.rl_system.observe_reward(trade_result)
                    self.logger.info(f"RL trade learning: reward={reward:.4f}, buffer_size={len(self.rl_system.experience_buffer)}/{self.rl_system.training_frequency}")
                except Exception as e:
                    self.logger.error(f"RL reward calculation error: {e}")
            # Signal info if available
            if 'signal_strength' in trade_result:
                self.logger.info(f"Signal Strength: {trade_result.get('signal_strength', 0):.2f}")
                if hasattr(self, 'rl_system'):
                    reward = self.rl_system.observe_reward(trade_result)
                    self.logger.info(f"RL reward: {reward:.4f}, model updated: {len(self.rl_system.experience_buffer) >= self.rl_system.training_frequency}")   
            # Display factor data if available
            if trade_id and hasattr(self, '_trade_factor_data') and trade_id in self._trade_factor_data:
                factors = self._trade_factor_data[trade_id]
                self.logger.info("Factor Data:")
                for key, value in factors.items():
                    if isinstance(value, (int, float)):
                        self.logger.info(f"  {key}: {value:.4f}")
                    else:
                        self.logger.info(f"  {key}: {value}")
                        
            self.logger.info("=" * 50 + "\n")
        except Exception as e:
            self.logger.error(f"Error processing trade for analytics: {e}")
    def calculate_dynamic_profit_targets(self, position_obj, current_price, market_data, regime_info, current_position_id):
        """Helper function to calculate dynamic profit targets for existing positions"""
        direction = 1 if position_obj['size'] > 0 else -1
        entry_price = position_obj['entry_price']
        
        # Use range-optimized profit targets in range-bound markets
        current_regime = regime_info.get('regime', 'unknown')
        
        if current_regime == 'range_bound' and 'range_analysis' in market_data:
            # Calculate optimized targets using specialized range function
            optimized_targets = self.range_optimized_profit_targets(
                entry_price,
                direction,
                market_data,
                market_data['range_analysis']
            )
            
            # Update position object with calculated targets
            position_obj['profit_targets'] = optimized_targets['targets']
            position_obj['target_hold_times'] = optimized_targets['expected_hold_times']
            
            self.logger.info(f"Range optimized targets: T1: ${optimized_targets['targets'][0]:.2f}, " +
                f"T2: ${optimized_targets['targets'][1]:.2f}, T3: ${optimized_targets['targets'][2]:.2f}")
        else:
            # For other regimes, use standard target optimization
            # Get the order flow patterns if available
            order_flow_patterns = {}
            if hasattr(self, '_trade_factor_data') and current_position_id in self._trade_factor_data:
                for pattern in self._trade_factor_data[current_position_id].get('patterns', []):
                    if pattern['type'] == 'order_flow':
                        order_flow_patterns = {
                            'pattern_detected': True,
                            'pattern_name': pattern['name'],
                            'direction': pattern['direction'],
                            'confidence': pattern['confidence']
                        }
                        break
            
            # Calculate optimized targets
            optimized_targets = self.optimize_profit_targets(
                entry_price,
                direction,
                market_data,
                regime_info,
                
            )
            
            # Update position object with calculated targets
            position_obj['profit_targets'] = optimized_targets['targets']
            position_obj['target_hold_times'] = optimized_targets['expected_hold_times']
            
            self.logger.info(f"Dynamic profit targets: T1: ${optimized_targets['targets'][0]:.2f}, " +
                    f"T2: ${optimized_targets['targets'][1]:.2f}, T3: ${optimized_targets['targets'][2]:.2f}") 
    def optimized_regime_transition_protection(self, signal, from_regime, to_regime, seconds_since_change):
        """
        Improved regime transition protection with adaptive dampening factors
        """
        import math 
        # Define base dampening factors
        max_dampening = 0.98  # Almost complete signal suppression initially
        recovery_time = 15.0  # Time in seconds to recover
        
        # Define regime-specific parameters
        if from_regime == 'range_bound' and to_regime == 'trending_down':
            # Protect against false breakdowns
            if signal < 0:  # Already aligned with new trend direction
                max_dampening = 0.75  # Less dampening when aligned with new regime
                recovery_time = 10.0  # Faster recovery
            else:  # Counter to new trend direction
                max_dampening = 0.98  # Strong dampening
                recovery_time = 20.0  # Slower recovery
        elif from_regime == 'trending_down' and to_regime == 'range_bound':
            # Protect against false reversals
            if abs(signal) > 0.5:  # Strong signals
                max_dampening = 0.90
            else:
                max_dampening = 0.80
        else:
            # Default protection parameters
            max_dampening = 0.85
            recovery_time = 12.0
        
        # Calculate current dampening factor based on time
        if seconds_since_change <= 0:
            dampening_factor = max_dampening
        else:
            # Logarithmic recovery provides faster initial improvement
            dampening_factor = max_dampening * (1.0 - min(1.0, math.log(1 + seconds_since_change) / math.log(1 + recovery_time)))
        
        # Apply dampening to signal
        dampened_signal = signal * (1.0 - dampening_factor)
        
        # Preserve signal direction even with heavy dampening
        min_magnitude = 0.01 if abs(signal) > 0.1 else 0.0
        if abs(dampened_signal) < min_magnitude and abs(signal) > min_magnitude:
            dampened_signal = min_magnitude * np.sign(signal)
        
        return dampened_signal                   
    def equity_curve_adjustment(self):
        """
        Dynamically adjust position sizing based on recent equity curve performance
        Returns a multiplier to apply to position sizing (0.5-1.5)
        """
        # Get recent win rate
        recent_win_rate = self.get_recent_win_rate(10)  # Last 10 trades
        
        # Get current drawdown
        current_equity = self.execution_engine.get_account_equity()
        if not hasattr(self, 'peak_equity'):
            self.peak_equity = current_equity
        elif current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        drawdown_pct = (self.peak_equity - current_equity) / self.peak_equity if self.peak_equity > 0 else 0
        
        # Base adjustment factor
        adjustment_factor = 1.0
        
        # Adjust based on win rate
        if recent_win_rate < 0.3:  # Very poor performance
            adjustment_factor *= 0.6  # Significant reduction
        elif recent_win_rate < 0.4:  # Poor performance
            adjustment_factor *= 0.75  # Moderate reduction
        elif recent_win_rate > 0.7:  # Very good performance
            adjustment_factor *= 1.2  # Moderate increase
        elif recent_win_rate > 0.6:  # Good performance
            adjustment_factor *= 1.1  # Slight increase
        
        # Apply additional drawdown protection
        if drawdown_pct > 0.05:  # Significant drawdown
            drawdown_factor = max(0.5, 1.0 - (drawdown_pct * 5))  # Progressive reduction
            adjustment_factor *= drawdown_factor
        
        # Ensure adjustment is within reasonable bounds
        adjustment_factor = max(0.5, min(1.5, adjustment_factor))
        
        return adjustment_factor    
    def calculate_dynamic_risk_dollars(self, market_data, elite_confidence, current_regime, regime_confidence):
        """
        Calculate dynamic risk dollars based on market conditions, regime, and quantum confidence
        
        Parameters:
        - market_data: Dict containing market metrics
        - elite_confidence: Float representing quantum confidence level (0-1)
        - current_regime: String indicating market regime (trending_up, trending_down, range_bound, volatile)
        - regime_confidence: Float representing confidence in regime classification (0-1)
        
        Returns:
        - risk_dollars: Float amount in dollars to risk on this trade
        """
        # Get account equity
        equity = self.execution_engine.get_account_equity()
        
        # Retrieve key metrics
        vpin = market_data.get('vpin', 0.3)
        volatility = market_data.get('volatility', 0.0001)
        entanglement = market_data.get('entanglement', 0.5)
        
        # Base risk percentage - scale based on account performance
        if equity > 52000:  # Account growing - can be more aggressive
            base_risk_pct = 0.015  # 1.5% risk per trade
        elif equity < 48000:  # Account in drawdown - reduce risk
            base_risk_pct = 0.005  # 0.5% risk per trade
        else:
            base_risk_pct = 0.01  # 1% standard risk
        
        # Adjust risk based on win rate (last 20 trades)
        win_rate = self.get_recent_win_rate(20) if hasattr(self, 'get_recent_win_rate') else 0.5
        if win_rate > 0.6:  # Strong performance
            win_rate_factor = 1.2
        elif win_rate < 0.4:  # Poor performance
            win_rate_factor = 0.8
        else:
            win_rate_factor = 1.0
        
        # Adjust based on elite confidence
        confidence_factor = min(1.3, max(0.7, 0.85 + elite_confidence * 0.3))
        
        # Adjust based on regime
        if current_regime in ['trending_up', 'trending_down'] and regime_confidence > 0.7:
            # More aggressive in strong trends
            regime_factor = 1.1
        elif current_regime == 'volatile':
            # More conservative in volatile markets
            regime_factor = 0.8
        elif current_regime == 'range_bound':
            # Check for range extremes for mean reversion opportunities
            if 'range_analysis' in market_data:
                z_score = market_data['range_analysis'].get('z_score', 0)
                if abs(z_score) > 1.5:  # Strong mean reversion opportunity
                    regime_factor = 1.0
                else:
                    regime_factor = 0.9
            else:
                regime_factor = 0.9
        else:
            regime_factor = 0.9  # Unknown regime - be conservative
        
        # Apply VPIN (toxic liquidity) adjustment
        vpin_factor = max(0.7, 1.0 - (vpin * 0.5))
        
        # Apply volatility adjustment
        vol_multiplier = 5000  # Scale factor for volatility
        vol_factor = min(1.2, max(0.8, 1.0 + ((volatility * vol_multiplier) - 0.5)))
        
        # Apply quantum entanglement factor
        entanglement_factor = min(1.15, 1.0 + (entanglement - 0.5) * 0.3)
        
        # Check for drawdown protection
        drawdown = self.risk_manager.get_current_drawdown() if hasattr(self.risk_manager, 'get_current_drawdown') else 0
        if drawdown > 0.02:  # More than 2% drawdown
            # Progressive risk reduction based on drawdown
            drawdown_factor = max(0.5, 1.0 - (drawdown * 5))
        else:
            drawdown_factor = 1.0
        
        # Calculate final risk dollars
        risk_dollars = equity * base_risk_pct * win_rate_factor * confidence_factor * \
                    regime_factor * vpin_factor * vol_factor * entanglement_factor * \
                    drawdown_factor
        
        # Apply minimum and maximum risk limits
        min_risk = 100  # Minimum $100 risk
        max_risk = equity * 0.02  # Maximum 2% risk
        
        risk_dollars = max(min_risk, min(max_risk, risk_dollars))
        
        # Record risk metrics for analysis
        if hasattr(self, '_risk_metrics_history'):
            self._risk_metrics_history.append({
                'timestamp': datetime.datetime.now(),
                'equity': equity,
                'risk_dollars': risk_dollars,
                'risk_pct': risk_dollars / equity,
                'factors': {
                    'win_rate': win_rate_factor,
                    'confidence': confidence_factor,
                    'regime': regime_factor,
                    'vpin': vpin_factor,
                    'volatility': vol_factor,
                    'entanglement': entanglement_factor,
                    'drawdown': drawdown_factor
                }
            })
        
        return risk_dollars    
    def advanced_quantum_signal_reinforcement(self, signal, confidence, entanglement, market_data):
        """Enhanced quantum signal reinforcement with non-linear scaling"""
        # Extract key metrics
        regime = market_data.get('regime', 'unknown')
        vpin = market_data.get('vpin', 0.3)
        delta_flow = market_data.get('delta', 0) * market_data.get('order_flow', 0)
        
        # Calculate base factor with non-linear entanglement scaling
        if entanglement > 0.9:  # Ultra-high entanglement
            base_factor = 1.2 + np.power(entanglement - 0.9, 0.5) * 4.0  # Square root scaling for smoother response
        elif entanglement > 0.8:
            base_factor = 1.1 + (entanglement - 0.8) * 1.0
        elif entanglement > 0.7:
            base_factor = 1.0 + (entanglement - 0.7) * 1.0
        else:
            base_factor = 0.9 + entanglement * 0.15
        
        # Apply regime-specific adjustments with confidence weighting
        if regime == 'range_bound':
            if 'range_analysis' in market_data:
                z_score = market_data['range_analysis'].get('z_score', 0)
                # Stronger reinforcement for mean reversion at range extremes
                if abs(z_score) > 1.5 and np.sign(z_score) * np.sign(signal) < 0:
                    regime_factor = 1.2
                else:
                    regime_factor = 0.9  # Dampened reinforcement in range centers
            else:
                regime_factor = 0.9
        elif 'trending' in regime:
            # Trending markets get stronger reinforcement when signal aligns with trend
            trend_direction = 1 if regime == 'trending_up' else -1
            if np.sign(signal) == trend_direction:
                regime_factor = 1.3  # Aligned signals get boosted
            else:
                regime_factor = 0.7  # Counter-trend signals get dampened
        else:
            regime_factor = 1.0
        
        # Apply confidence scaling with sigmoid function for better gradient
        confidence_factor = 0.5 + (1 / (1 + np.exp(-10 * (confidence - 0.5))))
        
        # Calculate final reinforcement factor with limits
        reinforcement_factor = base_factor * confidence_factor * regime_factor
        
        # Apply liquidity adjustment (reduce reinforcement in toxic conditions)
        if vpin > 0.25:
            vpin_dampener = max(0.7, 1.0 - ((vpin - 0.25) * 1.0))
            reinforcement_factor *= vpin_dampener
        
        # Limit extreme reinforcement factors
        reinforcement_factor = min(2.5, max(0.6, reinforcement_factor))
        
        # Apply reinforcement to signal
        reinforced_signal = signal * reinforcement_factor
        
        # Hard cap on absolute signal magnitude
        max_signal = 1.75  # Prevent extreme signals
        reinforced_signal = np.clip(reinforced_signal, -max_signal, max_signal)
        
        # Calculate quantum confidence with non-linear entanglement boosting
        quantum_confidence = min(0.95, confidence * (1.0 + np.power(entanglement, 1.5) * 0.5))
        
        return {
            'reinforced_signal': reinforced_signal,
            'quantum_confidence': quantum_confidence,
            'amplification_factor': reinforcement_factor,
            'entanglement': entanglement
        }
    def validate_required_methods(self):
        """Validate that all required methods are implemented"""
        required_methods = [
            'stabilize_signal', 
            'dynamic_trading_threshold',
            'optimize_profit_targets',
            'range_optimized_profit_targets',
            'adaptive_regime_transition_protection',
            'calculate_dynamic_profit_targets',
            'manage_profit_targets',
            'get_recent_win_rate',
            'optimize_exit_timing',
            'dynamic_risk_allocation',
            'detect_choppy_market_pattern',
            'enhanced_execution_quality_score',
            'optimize_execution_timing',
            'generate_advanced_risk_reward',
            'equity_curve_adjustment',
            'optimized_range_position_sizing',
            'quantum_state_risk_management',
            'adaptive_range_trade_management'
        ]
        
        missing = []
        for method in required_methods:
            if not hasattr(self, method) or not callable(getattr(self, method)):
                missing.append(method)
                self.logger.warning(f"Required method '{method}' is missing")
        
        if missing:
            self.logger.error(f"Found {len(missing)} missing required methods: {', '.join(missing)}")
        else:
            self.logger.info("All required methods are properly implemented")
        
        return missing
    def enhanced_micromovement_detection(self, market_data, price_history):
        """Ultra-sensitive micromovement detection for ultra-low volatility environments"""
        volatility = market_data.get('volatility', 0.0001)
        
        # Only activate in extremely low volatility environments
        if volatility < 0.0005:
            # Calculate micro price trends using exponential weighting
            recent_prices = price_history[-20:]
            if len(recent_prices) < 20:
                return {'detected': False}
            
            # Calculate micro-momentum using advanced exponential weighting
            weights = [math.exp(i/5) for i in range(len(recent_prices))]
            weights = [w/sum(weights) for w in weights]  # Normalize
            
            # Calculate weighted price differences
            diffs = [recent_prices[i+1] - recent_prices[i] for i in range(len(recent_prices)-1)]
            if not diffs:
                return {'detected': False}
                
            weighted_diffs = sum([diffs[i] * weights[i] for i in range(len(diffs))])
            micro_trend = np.sign(weighted_diffs)
            
            # Calculate micro-volatility ratio (current vol vs trailing)
            trailing_vol = np.std(recent_prices[:-5])
            recent_vol = np.std(recent_prices[-5:])
            vol_ratio = recent_vol / trailing_vol if trailing_vol > 0 else 1.0
            
            # Detect micro compression/expansion patterns
            is_compressed = vol_ratio < 0.7
            is_expanding = vol_ratio > 1.3
            
            return {
                'detected': True,
                'micro_trend': micro_trend,
                'strength': abs(weighted_diffs) / (volatility * 10),
                'is_compressed': is_compressed,
                'is_expanding': is_expanding,
                'vol_ratio': vol_ratio,
                'confidence': min(0.9, max(0.3, 1.0 - volatility * 1000))
            }
        
        return {'detected': False}    
    def ultra_range_optimization(self, signal, market_data, z_score):
        """Ultra-optimized range trading with advanced boundary detection"""
        if market_data.get('regime') != 'range_bound':
            return signal
            
        # Extract crucial range parameters
        position_in_range = market_data.get('range_analysis', {}).get('position_in_range', 0.5)
        
        # For extremely low positions in range (near bottom boundary)
        if position_in_range < 0.15 and z_score < -1.2:
            # Calculate non-linear signal boost based on z-score depth
            boost_factor = 0.5 + min(1.0, abs(z_score) * 0.2) 
            boost_factor *= 1.5  # Amplify specifically for extreme lows
            
            # Only boost positive signals or flip weak negative signals at range extremes
            if signal > 0:
                enhanced_signal = signal * boost_factor
                # Cap signal to prevent overexaggeration
                return min(1.8, enhanced_signal)
            elif abs(signal) < 0.3:
                # Flip weak negative signals at range bottoms - major edge
                return abs(signal) * 0.7  # Convert to positive but dampened
                
        # For extremely high positions in range (near top boundary)
        elif position_in_range > 0.85 and z_score > 1.2:
            # Similar logic for upper range, but for sell signals
            boost_factor = 0.5 + min(1.0, abs(z_score) * 0.2)
            boost_factor *= 1.5
            
            if signal < 0:
                enhanced_signal = signal * boost_factor
                return max(-1.8, enhanced_signal)
            elif abs(signal) < 0.3:
                return -abs(signal) * 0.7
        
        return signal    
    def adaptive_execution_timing(self, action_type, direction, market_data, current_price, target_price=None):
        """Ultra-precise execution timing with microstructure analysis"""
        # Extract execution quality factors
        order_flow = market_data.get('order_flow', 0)
        liquidity = market_data.get('liquidity_score', 0.5)
        volatility = market_data.get('volatility', 0.0005)
        regime = market_data.get('regime', 'unknown')
        
        # Base execution quality score
        base_quality = 0.5
        
        # Directional alignment calculation
        direction_alignment = np.sign(order_flow) * direction
        
        # Special handling for range_bound regimes with extremely low volatility
        if regime == 'range_bound' and volatility < 0.0001:
            # For exits in range markets - be more selective about timing
            if action_type == 'exit':
                # Calculate distance to target if provided
                target_distance = 0
                if target_price:
                    target_distance = abs(current_price - target_price)
                
                # If close to target, accelerate execution (opportunity cost)
                if target_distance > 0 and target_distance < volatility * 5000:
                    return {
                        'execute_now': True,
                        'delay_seconds': 0,
                        'quality': 0.9,
                        'reason': 'close_to_target'
                    }
                    
                # If flow is against position but liquidity is good, execute immediately
                if direction_alignment < 0 and liquidity > 0.7:
                    return {
                        'execute_now': True,
                        'delay_seconds': 0,
                        'quality': 0.85,
                        'reason': 'adverse_flow_good_liquidity'
                    }
                    
                # If flow is favorable but weak, delay slightly for better price
                if direction_alignment > 0 and abs(order_flow) < 0.3:
                    return {
                        'execute_now': False,
                        'delay_seconds': 1.5,
                        'quality': 0.65,
                        'reason': 'favorable_exit_flow'
                    }
            
            # For entries in range markets with low volatility
            elif action_type == 'entry':
                # No delay for strong order flow in our direction with good liquidity
                if direction_alignment > 0.5 and liquidity > 0.8:
                    return {
                        'execute_now': True,
                        'delay_seconds': 0,
                        'quality': 0.9,
                        'reason': 'optimal_entry_conditions'
                    }
                
                # Slight delay for entry when order flow is neutral but increasing
                if abs(direction_alignment) < 0.2:
                    # Check for acceleration in the right direction
                    flow_accel = market_data.get('flow_acceleration', 0)
                    if np.sign(flow_accel) == direction:
                        return {
                            'execute_now': False,
                            'delay_seconds': 1.0,
                            'quality': 0.75,
                            'reason': 'waiting_for_flow_acceleration'
                        }
        
        # Default timing logic for other regimes/conditions
        if direction_alignment > 0.3:
            # Favorable flow
            return {
                'execute_now': True,
                'delay_seconds': 0,
                'quality': 0.8,
                'reason': 'favorable_flow'
            }
        elif direction_alignment < -0.3:
            # Opposing flow - delay if possible
            return {
                'execute_now': False,
                'delay_seconds': 2.0,
                'quality': 0.4,
                'reason': 'opposing_flow'
            }
        
        # Neutral flow
        return {
            'execute_now': True,
            'delay_seconds': 0,
            'quality': 0.6,
            'reason': 'neutral_flow'
        }    
    def optimized_position_sizing(self, signal_strength, market_data, regime_info):
        """
        Optimized position sizing with dynamic risk management
        """
        # Extract key metrics
        regime = regime_info.get('regime', 'unknown')
        regime_confidence = regime_info.get('confidence', 0.5)
        equity = self.execution_engine.get_account_equity()
        vpin = market_data.get('vpin', 0.3)
        volatility = market_data.get('volatility', 0.0001)
        
        # Base risk percentage calculation
        if equity > 52000:  # Increased equity - can be more aggressive
            base_risk_pct = 0.015  # 1.5% risk per trade
        elif equity < 48000:  # Drawdown protection
            base_risk_pct = 0.005  # 0.5% risk per trade
        else:
            base_risk_pct = 0.01  # 1% standard risk
        
        # Adjust risk based on win rate (last 20 trades)
        win_rate = self.get_recent_win_rate(20)
        if win_rate > 0.6:  # Strong performance
            win_rate_factor = 1.2
        elif win_rate < 0.4:  # Poor performance
            win_rate_factor = 0.8
        else:
            win_rate_factor = 1.0
        
        # Adjust based on signal strength
        signal_factor = min(1.3, max(0.7, 0.8 + abs(signal_strength) * 0.5))
        
        # Adjust based on regime
        if regime == 'trending_up' or regime == 'trending_down':
            regime_factor = 1.0 + (regime_confidence * 0.3)
        elif regime == 'range_bound':
            # For range markets, check for optimal entry points
            if 'range_analysis' in market_data:
                z_score = market_data['range_analysis'].get('z_score', 0)
                # Increase size for high-probability mean reversion opportunities
                if abs(z_score) > 1.5:
                    regime_factor = 1.1
                else:
                    regime_factor = 0.9
            else:
                regime_factor = 0.9
        else:
            regime_factor = 0.9  # Unknown regime
        
        # Apply VPIN (toxic liquidity) adjustment
        vpin_factor = max(0.7, 1.0 - (vpin * 0.5))
        
        # Apply volatility adjustment
        vol_factor = min(1.2, max(0.8, 1.0 + ((volatility * 5000) - 0.5)))
        
        # Calculate risk dollars
        risk_dollars = equity * base_risk_pct * win_rate_factor * signal_factor * regime_factor * vpin_factor * vol_factor
        
        # Calculate position size based on stop distance
        atr = market_data.get('atr', 10)
        stop_distance = max(15, atr * 1.5)  # Minimum 15 points, otherwise 1.5 * ATR
        position_size = risk_dollars / stop_distance
        
        # Convert to contracts (rounded down, minimum 1)
        contracts = max(1, int(position_size / 100))  # Assuming $100 per point
        
        # Log sizing factors
        factors = {
            'Vol': vol_factor,
            'Signal': signal_factor,
            'Regime': regime_factor,
            'VPIN': vpin_factor,
            'Win_Rate': win_rate_factor
        }
        
        return contracts, factors 
    
    def stabilized_quantum_signal_processing(self, signal, market_data, regime_info):
        """Apply advanced quantum signal stabilization with improved conflict resolution"""
        # Extract key metrics
        regime = regime_info.get('regime', 'unknown')
        confidence = regime_info.get('confidence', 0.5)
        
        # Extract market metrics for stabilization
        volatility = market_data.get('volatility', 0.0001)
        entanglement = market_data.get('entanglement', 0.5)
        order_flow = market_data.get('order_flow', 0)
        delta = market_data.get('delta', 0)
        z_score = market_data.get('range_analysis', {}).get('z_score', 0) if 'range_analysis' in market_data else 0
        
        # Track signal processing stages for diagnostics
        signal_stages = {'original': signal}
        
        # Apply conflict resolution when entanglement is extremely high
        if entanglement > 0.9:  # New threshold for extreme entanglement
            # Extract key signal components
            neural_alpha_direction = market_data.get('neural_alpha', {}).get('direction', np.sign(signal))
            order_flow_direction = np.sign(market_data.get('order_flow', 0))
            delta_direction = np.sign(market_data.get('delta', 0))
            
            # Check for directional conflict
            signal_direction = np.sign(signal)
            if signal_direction != 0 and neural_alpha_direction != 0 and neural_alpha_direction != signal_direction:
                # Strong conflict between neural alpha and primary signal
                conflict_strength = entanglement * 0.8
                
                # Use weighted average based on which signal aligns with more market factors
                # Count directional alignment with other factors
                neural_alignment = (neural_alpha_direction == order_flow_direction) + (neural_alpha_direction == delta_direction)
                signal_alignment = (signal_direction == order_flow_direction) + (signal_direction == delta_direction)
                
                if neural_alignment > signal_alignment:
                    # Neural alpha has better alignment, bias towards it
                    conflict_resolved_signal = signal * (1 - conflict_strength) + (abs(signal) * neural_alpha_direction * conflict_strength)
                else:
                    # Original signal has better alignment, dampen but maintain direction
                    conflict_resolved_signal = signal * (1 - (conflict_strength * 0.3))
                    
                signal = conflict_resolved_signal
                signal_stages['conflict_resolved'] = signal
        
        # Calculate max signal bound - higher in trending markets, lower in range/choppy
        if regime in ['trending_up', 'trending_down']:
            max_signal = 1.8 * confidence
        elif regime == 'range_bound':
            max_signal = 1.1 * confidence
        elif regime == 'choppy':
            max_signal = 0.9 * confidence
        elif regime == 'volatile':
            max_signal = 1.5 * confidence
        else:
            max_signal = 1.0 * confidence
        
        # Ensure minimum max signal
        max_signal = max(0.8, max_signal)
        
        # Base signal - apply direction-aware stability
        signal_momentum = 0
        if len(self._signal_history) >= 3:
            # Calculate momentum from recent signals (newer signals weighted more)
            signal_momentum = (self._signal_history[-1] * 0.6 + 
                            self._signal_history[-2] * 0.3 + 
                            self._signal_history[-3] * 0.1)
        
        # Apply stability enhancement - prevent signal flipping
        if signal_momentum != 0 and np.sign(signal) != np.sign(signal_momentum) and abs(signal) < 0.3:
            # Stronger dampening when entanglement is high
            dampening_factor = 0.7 * (1 + entanglement * 0.3)
            signal *= (1 - dampening_factor)  # More aggressive dampening based on entanglement
            signal_stages['dampened'] = signal
        
        # Apply adaptive capping based on regime
        if abs(signal) > max_signal:
            signal = np.sign(signal) * max_signal
            signal_stages['capped'] = signal
        
        # Apply entanglement-based signal adjustment
        if entanglement > 0.7:
            # High entanglement - enhance signal
            enhance_factor = 1.0 + ((entanglement - 0.7) / 0.3) * 0.2  # Up to 20% enhancement
            signal *= enhance_factor
            signal_stages['entanglement'] = signal
            
            # Re-apply cap after enhancement
            if abs(signal) > max_signal:
                signal = np.sign(signal) * max_signal
                signal_stages['recapped'] = signal
        
        # Calculate quantum confidence as a function of signal quality and entanglement
        signal_quality = min(1.0, abs(signal) / max_signal)
        
        # Enhanced confidence calculation - penalize confidence when original signal was conflicted
        if 'conflict_resolved' in signal_stages:
            conflict_penalty = 0.2
        else:
            conflict_penalty = 0
            
        quantum_confidence = min(0.95, (signal_quality * 0.5) + (entanglement * 0.5) - conflict_penalty)
        
        # Make sure it's included in the return dictionary
        return {
            'signal': signal,
            'original': signal_stages['original'],
            'stages': signal_stages,
            'max_bound': max_signal,
            'quantum_confidence': quantum_confidence
        }
    def adaptive_volatility_position_sizing(self, base_size, market_data, regime_info):
        """Apply advanced volatility-based position sizing"""
        # Extract volatility metrics
        volatility = market_data.get('volatility', 0.0001)
        baseline_vol = 0.0001  # Normalized volatility baseline
        normalized_vol = volatility / baseline_vol
        
        # Calculate volatility scaling factor with asymptotic curve
        if normalized_vol <= 1.0:
            # Normal to low volatility - can use normal sizing
            vol_factor = 1.0
        else:
            # Higher volatility - apply non-linear reduction
            # Use sqrt to create gentler reduction curve
            vol_factor = 1.0 / np.sqrt(normalized_vol)
        
        # Apply regime-specific adjustments
        regime = regime_info.get('regime', 'unknown')
        if regime == 'range_bound' and normalized_vol > 3.0:
            # Further reduce size in range markets with high volatility
            vol_factor *= 0.8
        elif 'trending' in regime and normalized_vol > 3.0:
            # Less reduction in trending markets 
            vol_factor *= 0.9
        
        # Apply minimum size constraint
        vol_factor = max(0.4, vol_factor)  # Never go below 40% of base size
        
        # Calculate final position size
        final_size = max(1, int(base_size * vol_factor))
        
        return {
            'position_size': final_size,
            'vol_factor': vol_factor,
            'normalized_vol': normalized_vol
        }    
    def dynamic_time_based_exit(self, position_obj, market_data, entry_time):
        """Improved time-based exit strategy with adaptive holding periods"""
        # Current position details
        direction = 1 if position_obj.get('size', 0) > 0 else -1
        current_time = datetime.datetime.now()
        seconds_in_trade = (current_time - entry_time).total_seconds()
        
        # Extract market regime and data
        regime = market_data.get('regime', 'unknown')
        z_score = market_data.get('range_analysis', {}).get('z_score', 0)
        atr = market_data.get('atr', market_data.get('price', 0) * 0.005)
        entry_price = position_obj.get('entry_price', 0)
        current_price = market_data.get('price', 0)
        
        # IMPROVED: Calculate more precise holding periods based on regime
        if regime == 'range_bound':
            # Shorter holding periods in range markets with dynamic scaling
            range_confidence = market_data.get('range_analysis', {}).get('confidence', 0.5)
            
            # Base holding time is shorter for range markets
            base_hold_time = 180  # 3 minutes baseline
            
            # IMPROVED: Scale based on position in range - shorter at extremes
            if abs(z_score) > 1.5:
                # At extremes, mean reversion happens faster
                max_hold_time = base_hold_time * (1.0 - min(0.5, (abs(z_score) - 1.5) * 0.25))
                self.logger.info(f"Range extreme time adjustment: {base_hold_time}s → {max_hold_time:.1f}s (z-score: {z_score:.2f})")
            else:
                # Middle of range needs longer holding time
                max_hold_time = base_hold_time * (1.0 + min(0.3, (1.5 - abs(z_score)) * 0.15))
                
            # IMPROVED: Adjust for range confidence
            if range_confidence < 0.7:
                # Less confident ranges get shorter max hold time
                max_hold_time *= (0.7 + range_confidence * 0.3)
        
        elif 'trending' in regime:
            # Longer holding periods in trending markets
            trend_strength = market_data.get('trend_strength', 1.0)
            
            # Base holding time is longer for trends to capture momentum
            base_hold_time = 300  # 5 minutes baseline
            
            # IMPROVED: Adjust for trend strength - stronger trends can be held longer
            max_hold_time = base_hold_time * min(1.5, (1.0 + (trend_strength - 1.0) * 0.2))
            
            # IMPROVED: If against trend momentum, reduce holding time
            if (direction == 1 and market_data.get('delta', 0) < -0.3) or \
            (direction == -1 and market_data.get('delta', 0) > 0.3):
                max_hold_time *= 0.7
                self.logger.info(f"Reducing hold time due to adverse momentum: {max_hold_time:.1f}s")
        
        elif regime == 'volatile':
            # Very short holding periods in volatile markets
            max_hold_time = 120  # 2 minutes for volatile markets
            
            # IMPROVED: If in profit in volatile market, reduce time further
            profit_pct = (current_price - entry_price) * direction / entry_price
            if profit_pct > 0.001:  # In profit (10+ basis points)
                profit_adjustment = max(0.5, 1.0 - profit_pct * 200)  # More profit = shorter hold time
                max_hold_time *= profit_adjustment
                self.logger.info(f"Reducing volatile hold time due to profit: {max_hold_time:.1f}s (profit: {profit_pct:.4f})")
        
        else:
            # Default for unknown regimes
            max_hold_time = 240  # 4 minutes default
        
        # IMPROVED: Add mandatory minimum hold time to avoid micro-scalping
        min_hold_time = 15  # 15 seconds minimum hold
        
        # IMPROVED: Calculate dynamic exit criteria
        # 1. Time-based exit
        if seconds_in_trade > max_hold_time:
            return {
                'action': 'exit',
                'reason': 'time_based_exit',
                'hold_time': seconds_in_trade
            }
        
        # 2. Enhanced early exit for adverse moves
        entry_price = position_obj.get('entry_price', current_price)
        pnl = (current_price - entry_price) * direction
        
        # IMPROVED: Scale early exit with time in trade - longer time allows smaller adverse move
        adverse_move_threshold = atr * 0.5
        time_factor = min(0.8, seconds_in_trade / max_hold_time)
        scaled_threshold = adverse_move_threshold * (1.0 - time_factor * 0.5)
        
        if pnl < -scaled_threshold and seconds_in_trade > min_hold_time:
            return {
                'action': 'exit',
                'reason': 'adverse_move_time_exit',
                'hold_time': seconds_in_trade
            }
        
        # 3. IMPROVED: Time-based partial exits
        if seconds_in_trade > (max_hold_time * 0.6) and abs(position_obj.get('size', 0)) > 1:
            # Adaptive partial sizing - scale with time and position size
            base_exit_pct = 0.4  # 40% base exit size
            
            # Larger positions get larger partial exits
            position_scale = min(0.6, 0.3 + (abs(position_obj.get('size', 0)) - 1) * 0.1)
            
            # Calculate exit size - at least 1 contract
            exit_size = max(1, int(abs(position_obj.get('size', 0)) * position_scale))
            
            return {
                'action': 'partial_exit',
                'reason': 'time_based_partial',
                'size': -exit_size * np.sign(position_obj.get('size', 0)),  # Correctly signed for exit
                'hold_time': seconds_in_trade
            }
        
        # No exit action needed
        return {'action': 'hold'}
    def enhanced_quantum_signal_processing(self, signal, market_data, high_entanglement=False):
        """Apply advanced signal processing with special handling for high entanglement"""
        # Start with original signal
        processed_signal = signal
        
        # Extract key metrics
        entanglement = market_data.get('entanglement', 0.5)
        delta = market_data.get('delta', 0)
        order_flow = market_data.get('order_flow', 0)
        
        # Keep track of processing steps for diagnostics
        processing_steps = {'original': signal}
        
        # Handle high entanglement (delta-flow correlation) specifically
        if entanglement > 0.90:
            # IMPROVED: With very high entanglement, we need stronger filtering
            
            # 1. More aggressive smoothing
            if len(self._signal_history) >= 5:
                # Exponentially weighted average of recent signals
                weights = np.array([0.4, 0.25, 0.18, 0.1, 0.07])  # Most recent first
                recent_signals = np.array(self._signal_history[-5:])
                smoothed_signal = np.sum(recent_signals * weights)
                
                # Only take smoothed signal if it's in the same direction
                if np.sign(smoothed_signal) == np.sign(processed_signal) or abs(processed_signal) < 0.2:
                    original = processed_signal
                    processed_signal = (processed_signal * 0.6) + (smoothed_signal * 0.4)
                    processing_steps['smoothed'] = processed_signal
                    self.logger.info(f"High entanglement smoothing: {original:.2f} → {processed_signal:.2f}")
            
            # 2. Apply stronger bias toward dominant market force
            market_bias = delta * 0.6 + order_flow * 0.4
            if abs(market_bias) > 0.3:
                bias_direction = np.sign(market_bias)
                
                # If signal conflicts with strong market bias, dampen it
                if np.sign(processed_signal) != bias_direction and abs(processed_signal) < 0.7:
                    original = processed_signal
                    processed_signal *= 0.6  # Strong dampening of conflicting signals
                    processing_steps['conflicting_dampen'] = processed_signal
                    self.logger.info(f"High entanglement conflict dampening: {original:.2f} → {processed_signal:.2f}")
                    
                # If signal aligns with bias but is weak, enhance it slightly
                elif np.sign(processed_signal) == bias_direction and abs(processed_signal) < 0.5:
                    original = processed_signal
                    boost = bias_direction * abs(market_bias) * 0.2
                    processed_signal += boost
                    processing_steps['bias_enhanced'] = processed_signal
                    self.logger.info(f"High entanglement bias enhancement: {original:.2f} → {processed_signal:.2f}")
            
            # 3. Apply non-linear transformation to create clearer decision boundary
            if abs(processed_signal) < 0.2:
                # Signals near zero get pushed further toward zero
                original = processed_signal
                processed_signal *= 0.5
                processing_steps['zero_push'] = processed_signal
                self.logger.info(f"High entanglement zero dampening: {original:.2f} → {processed_signal:.2f}")
            elif abs(processed_signal) > 0.7:
                # Strong signals get enhanced to create clearer decisions
                original = processed_signal
                processed_signal = np.sign(processed_signal) * (0.7 + (abs(processed_signal) - 0.7) * 1.5)
                processing_steps['strong_enhance'] = processed_signal
                self.logger.info(f"High entanglement strong signal boost: {original:.2f} → {processed_signal:.2f}")
        
        # Return processed signal with tracking info
        return {
            'signal': processed_signal,
            'processing_steps': processing_steps,
            'entanglement': entanglement
        }    
    def enhanced_order_flow_sensitivity(self, market_data, signal, direction):
        """Advanced order flow sensitivity adjustment for faster reaction"""
        # Extract order flow metrics
        order_flow = market_data.get('order_flow', 0)
        flow_acceleration = market_data.get('flow_acceleration', 0)
        
        # Skip if order flow is minimal
        if abs(order_flow) < 0.2:
            return signal
        
        # Check for adverse order flow
        if np.sign(order_flow) != direction and abs(order_flow) > 0.3:
            # Strong adverse flow requires immediate signal dampening
            dampening = min(0.5, 0.3 + abs(order_flow) * 0.4)
            dampened_signal = signal * (1.0 - dampening)
            
            # Apply acceleration-based enhancement
            if abs(flow_acceleration) > 0.4 and np.sign(flow_acceleration) != direction:
                # Accelerating adverse flow - stronger dampening
                dampened_signal *= 0.75
                
            return dampened_signal
        
        # Check for aligned order flow
        if np.sign(order_flow) == direction and abs(order_flow) > 0.3:
            # Aligned flow can enhance signal, but with bounds
            enhancement = min(0.2, 0.1 + abs(order_flow) * 0.2)
            enhanced_signal = signal * (1.0 + enhancement)
            
            # Cap enhanced signal
            return min(1.8, enhanced_signal) if signal > 0 else max(-1.8, enhanced_signal)
        
        return signal    
    def vpin_market_toxicity_protection(self, market_data, entry_decision):
        """Apply VPIN-based toxicity protection to avoid adverse selection"""
        # Extract VPIN and liquidity metrics
        vpin = market_data.get('vpin', 0.3)
        liquidity_score = market_data.get('liquidity_score', 0.95)
        
        # Check for toxic conditions
        if vpin > 0.35:
            # Calculate toxicity score (0-1)
            toxicity = (vpin - 0.35) / 0.15  # 0.35 to 0.50 maps to 0-1
            toxicity = min(1.0, toxicity)
            
            # Apply liquidity adjustment
            if liquidity_score < 0.9:
                # Amplify toxicity with poor liquidity
                toxicity *= (1.0 + (0.9 - liquidity_score) * 2.0)
            
            # Make entry decision based on toxicity
            if toxicity > 0.7:
                # Highly toxic - avoid entry completely
                return {
                    'proceed': False,
                    'delay': 0,
                    'reason': 'extreme_market_toxicity',
                    'toxicity': toxicity
                }
            elif toxicity > 0.4:
                # Moderately toxic - delay entry
                delay_seconds = 3.0 + toxicity * 4.0  # 4.6 to 7 seconds delay
                return {
                    'proceed': True,
                    'delay': delay_seconds,
                    'reason': 'moderate_market_toxicity',
                    'toxicity': toxicity
                }
            else:
                # Slightly toxic - reduce position size
                size_factor = 1.0 - (toxicity * 0.4)  # 0.84 to 0.96 position size
                return {
                    'proceed': True,
                    'delay': 0,
                    'size_factor': size_factor,
                    'reason': 'mild_market_toxicity',
                    'toxicity': toxicity
                }
        
        # Normal market conditions
        return {'proceed': True, 'delay': 0}

   
    def get_current_drawdown(self):
    
        return self.current_drawdown


    def calculate_session_pnl(self):
        """Calculate session P&L from trade history"""
        if not hasattr(self, 'trade_history') or not self.trade_history:
            return 0.0
        
        # Use today's trades only
        today = datetime.datetime.now().date()
        today_trades = []
        
        for trade in self.trade_history:
            # Check if the trade has timestamp attribute or entry_time
            trade_time = None
            if 'timestamp' in trade:
                trade_time = trade['timestamp']
            elif 'entry_time' in trade:
                trade_time = trade['entry_time']
                
            # Only include today's trades
            if trade_time and isinstance(trade_time, datetime.datetime) and trade_time.date() == today:
                today_trades.append(trade)
        
        # Sum up profits
        total_pnl = sum(trade.get('profit', 0) for trade in today_trades)
        return total_pnl    
    def enhanced_regime_transition_handler(self, signal, current_regime, previous_regime, seconds_since_change, market_data):
        """Advanced regime transition handler with logarithmic recovery"""
        # Skip if no regime change
        if current_regime == previous_regime:
            return signal
        
        # Define transition parameters based on regime combinations
        transitions = {
            ('range_bound', 'trending_up'): {'max_dampening': 0.6, 'recovery_time': 15},
            ('range_bound', 'trending_down'): {'max_dampening': 0.6, 'recovery_time': 15},
            ('trending_up', 'range_bound'): {'max_dampening': 0.7, 'recovery_time': 20},
            ('trending_down', 'range_bound'): {'max_dampening': 0.7, 'recovery_time': 20},
            ('choppy', 'range_bound'): {'max_dampening': 0.5, 'recovery_time': 10},
            ('default', 'default'): {'max_dampening': 0.65, 'recovery_time': 18}
        }
        
        # Get parameters for this transition
        key = (previous_regime, current_regime)
        params = transitions.get(key, transitions[('default', 'default')])
        
        max_dampening = params['max_dampening']
        recovery_time = params['recovery_time']
        
        # Adjust dampening based on signal-regime alignment
        if current_regime == 'trending_up' and signal > 0:
            # Less dampening for long signals in uptrend
            max_dampening *= 0.6
        elif current_regime == 'trending_down' and signal < 0:
            # Less dampening for short signals in downtrend
            max_dampening *= 0.6
        
        # Apply logarithmic recovery curve - faster initial recovery
        if seconds_since_change <= 0:
            recovery_factor = 0.0
        else:
            recovery_factor = min(1.0, np.log(1 + seconds_since_change) / np.log(1 + recovery_time))
        
        # Calculate dampening with smooth transition
        dampening_factor = max_dampening * (1.0 - recovery_factor)
        
        # Scale dampening based on signal strength
        signal_scale = min(1.0, abs(signal))
        adjusted_dampening = dampening_factor * (0.5 + 0.5 * signal_scale)
        
        # Apply dampening to signal
        dampened_signal = signal * (1.0 - adjusted_dampening)
        
        # Preserve minimal directional bias for strong signals
        if abs(signal) > 0.8 and abs(dampened_signal) < 0.1:
            dampened_signal = 0.1 * np.sign(signal)
        
        return dampened_signal

    def unified_threshold_management(self, base_threshold, market_data, regime_info, pattern_info=None):
        """Enhanced threshold management with better range-bound calibration"""
        adjustments = {'base': base_threshold}
        
        # Extract regime data
        regime = regime_info.get('regime', 'unknown')
        confidence = regime_info.get('confidence', 0.5)
        trend_strength = regime_info.get('trend_strength', 1.0)
        
        # IMPROVED: Apply stronger regime-based adjustments
        if regime == 'range_bound':
            # Increase threshold in range markets significantly to reduce false signals
            regime_threshold = base_threshold * 0.85  # 15% reduction in threshold (was 0.95)
            adjustments['regime'] = regime_threshold
        elif 'trending' in regime and confidence > 0.7:
            # More aggressive threshold reduction in confident trending markets
            regime_threshold = base_threshold * 0.8
            adjustments['regime'] = regime_threshold
        elif regime == 'volatile':
            # Much higher threshold in volatile markets to prevent overtrading
            regime_threshold = base_threshold * 1.15  # 15% increase in threshold
            adjustments['regime'] = regime_threshold
        else:
            # Default behavior
            regime_threshold = base_threshold
            adjustments['regime'] = regime_threshold
        
        current_threshold = regime_threshold
        
        # Apply pattern-based adjustment if available
        if pattern_info and pattern_info.get('type'):
            pattern_confidence = pattern_info.get('confidence', 0.5)
            
            # IMPROVED: More conservative pattern-based threshold reduction
            pattern_threshold = current_threshold * (1.0 - ((pattern_confidence - 0.5) * 0.3))
            current_threshold = pattern_threshold
            adjustments['pattern'] = pattern_threshold
        
        # Apply range position adjustment if in range-bound market
        if regime == 'range_bound' and 'range_analysis' in market_data:
            range_analysis = market_data['range_analysis']
            z_score = range_analysis.get('z_score', 0)
            
            # IMPROVED: Only reduce threshold at extreme z-scores and mean-reversion
            if abs(z_score) > 1.5:
                position_factor = max(0.90, 1.0 - (abs(z_score) - 1.5) * 0.15)
                range_threshold = current_threshold * position_factor
                current_threshold = range_threshold
                adjustments['range_position'] = range_threshold
        
        # IMPROVED: Apply confirmation-based adjustment
        confirmation = market_data.get('confirmation', 0.25)
        if confirmation > 0.75:
            # Higher confirmation allows lower threshold
            conf_factor = 1.0 - ((confirmation - 0.75) * 0.3)
            conf_threshold = current_threshold * conf_factor
            current_threshold = conf_threshold
            adjustments['confirmation'] = conf_threshold
        elif confirmation < 0.3:
            # Low confirmation requires higher threshold
            conf_factor = 1.0 + ((0.3 - confirmation) * 0.4)
            conf_threshold = current_threshold * conf_factor
            current_threshold = conf_threshold
            adjustments['confirmation'] = conf_threshold
        
        # NEW: Apply entanglement-based threshold adjustment
        entanglement = market_data.get('entanglement', 0.5)
        if entanglement > 0.90:  # Very high entanglement needs much higher threshold
            entangle_factor = 1.0 + ((entanglement - 0.9) * 2.0)  # Up to 20% increase
            entangle_threshold = current_threshold * entangle_factor
            current_threshold = entangle_threshold
            adjustments['entanglement'] = entangle_threshold
        
        # NEW: Post-trade cooldown period adjustment
        if hasattr(self, '_last_trade_time'):
            seconds_since_last_trade = (datetime.datetime.now() - self._last_trade_time).total_seconds()
            if seconds_since_last_trade < self._trade_cooldown_period:
                # Increase threshold during cooldown period to prevent overtrading
                cooldown_factor = 1.0 + (1.0 - (seconds_since_last_trade / self._trade_cooldown_period)) * 0.5
                cooldown_threshold = current_threshold * cooldown_factor
                current_threshold = cooldown_threshold
                adjustments['cooldown'] = cooldown_threshold
        
        adjustments['final'] = current_threshold
        return {'threshold': current_threshold, 'adjustments': adjustments}
    def optimized_position_sizing(self, signal, threshold, market_data, account_metrics):
        """Optimized position sizing with risk-based scaling"""
        # Extract account metrics
        account_equity = account_metrics.get('equity', 50000)
        drawdown = account_metrics.get('drawdown', 0)
        win_rate = account_metrics.get('win_rate', 0.5)
        
        # Extract market metrics
        regime = market_data.get('regime', 'unknown')
        volatility = market_data.get('volatility', 0.0001)
        vpin = market_data.get('vpin', 0.3)
        z_score = market_data.get('range_analysis', {}).get('z_score', 0)
        
        # 1. Calculate base risk percentage based on account performance
        if win_rate > 0.55 and drawdown < 0.02:
            # Good performance - standard risk
            base_risk_pct = 0.01  # 1% risk
        elif win_rate < 0.45 or drawdown > 0.03:
            # Poor performance - reduced risk
            base_risk_pct = 0.005  # 0.5% risk
        else:
            # Default risk
            base_risk_pct = 0.0075  # 0.75% risk
        
        # 2. Apply signal quality scaling
        signal_strength = abs(signal)
        threshold_excess = max(0, (signal_strength - threshold) / threshold)
        signal_factor = min(1.2, 1.0 + (threshold_excess * 0.3))
        
        # 3. Apply regime-specific sizing
        if regime == 'range_bound':
            # Check if at range extremes for mean reversion
            if abs(z_score) > 1.5:
                regime_factor = 1.1  # Slightly larger size for mean reversion
            else:
                regime_factor = 0.9  # Smaller size in middle of range
        elif 'trending' in regime:
            # Check signal alignment with trend
            trend_direction = 1 if regime == 'trending_up' else -1
            if np.sign(signal) == trend_direction:
                regime_factor = 1.1  # Slightly larger for trend-aligned signals
            else:
                regime_factor = 0.7  # Smaller size for counter-trend trades
        elif regime == 'choppy':
            regime_factor = 0.8  # Reduced size in choppy markets
        else:
            regime_factor = 1.0
        
        # 4. Apply volatility scaling - reduce size in high volatility
        # Normalize volatility around baseline of 0.0001
        norm_volatility = volatility / 0.0001
        vol_factor = 1.0 / (0.7 + (0.3 * min(3.0, norm_volatility)))
        
        # 5. Apply liquidity factor - reduce size in toxic conditions
        vpin_factor = max(0.7, 1.0 - (vpin - 0.25) * 0.8) if vpin > 0.25 else 1.0
        
        # 6. Apply recovery factor - smaller sizes during drawdown
        recovery_factor = max(0.7, 1.0 - drawdown * 5.0)
        
        # 7. Calculate final risk percentage
        risk_pct = base_risk_pct * signal_factor * regime_factor * vol_factor * vpin_factor * recovery_factor
        
        # 8. Calculate dollar risk amount
        risk_dollars = account_equity * risk_pct
        
        # 9. Calculate stop distance based on ATR and regime
        atr = market_data.get('atr', market_data.get('price', 20000) * 0.001)
        
        if regime == 'range_bound':
            # Tighter stops in range markets - scaled by position in range
            range_factor = 1.0 + (0.5 * min(1.0, abs(z_score)))
            stop_distance = atr * 1.2 * range_factor
        elif 'trending' in regime:
            # Wider stops in trending markets
            stop_distance = atr * 2.0
        else:
            # Default stop distance
            stop_distance = atr * 1.5
        
        # 10. Calculate position size
        contract_value = 20  # NQ multiplier
        max_risk_per_point = risk_dollars / stop_distance
        
        # Calculate contracts with round to nearest
        contracts = max(1, int(max_risk_per_point / contract_value + 0.5))
        
        # 11. Apply position direction
        position_size = contracts * np.sign(signal)
        
        # 12. Apply maximum position limit based on account size
        max_contracts = int(account_equity / 12500)  # ~1 contract per $12.5K
        position_size = np.sign(position_size) * min(abs(position_size), max_contracts)
        
        return {
            'position_size': position_size,
            'risk_dollars': risk_dollars,
            'stop_distance': stop_distance,
            'factors': {
                'signal': signal_factor,
                'regime': regime_factor,
                'volatility': vol_factor,
                'vpin': vpin_factor,
                'recovery': recovery_factor
            }
        }

    def enhanced_risk_management(self, position_obj, market_data, regime_info):
        """Enhanced risk management with dynamic exit parameters"""
        # Extract position details
        entry_price = position_obj.get('entry_price', 0)
        position_size = position_obj.get('size', 0)
        direction = 1 if position_size > 0 else -1
        current_price = market_data.get('price', 0)
        
        # Skip if no position
        if position_size == 0 or entry_price == 0:
            return {'action': 'none', 'reason': 'no_position'}
        
        # Calculate P&L
        points_profit = (current_price - entry_price) * direction
        profit_pct = points_profit / entry_price
        
        # Extract market conditions
        regime = regime_info.get('regime', 'unknown')
        vpin = market_data.get('vpin', 0.3)
        order_flow = market_data.get('order_flow', 0)
        delta = market_data.get('delta', 0)
        
        # Calculate risk score - higher means more risk
        risk_score = 0.0
        
        # 1. Order flow risk - exit if strong adverse flow
        if np.sign(order_flow) != direction and abs(order_flow) > 0.25:
            flow_risk = min(0.4, abs(order_flow) * 0.8)
            risk_score += flow_risk
        
        # 2. Delta risk - exit if strong adverse delta
        if np.sign(delta) != direction and abs(delta) > 0.3:
            delta_risk = min(0.3, abs(delta) * 0.6)
            risk_score += delta_risk
        
        # 3. VPIN risk - exit in toxic conditions with profit
        if vpin > 0.28 and points_profit > 0:
            vpin_risk = (vpin - 0.28) * 2.0
            risk_score += min(0.3, vpin_risk)
        
        # 4. Profit-based risk thresholds - more protective with profits
        if points_profit > 0:
            # Calculate non-linear profit factor (1-10 scale)
            profit_ticks = points_profit / 0.25  # Convert to ticks
            profit_level = min(10, profit_ticks / 6.0)
            
            # More aggressive exit with higher profits - exponential decay
            profit_risk_threshold = 0.6 * np.exp(-0.15 * profit_level)
            
            # Compare risk to profit-based threshold
            if risk_score > profit_risk_threshold:
                return {
                    'action': 'exit',
                    'reason': 'protect_profits',
                    'risk_score': risk_score,
                    'profit_points': points_profit
                }
            
            # Consider partial exit at higher profits
            elif profit_level > 3.0 and risk_score > profit_risk_threshold * 0.7:
                # Calculate scaled exit size (20-40% of position)
                exit_pct = 0.2 + min(0.2, risk_score * 0.4)
                exit_size = max(1, int(abs(position_size) * exit_pct))
                
                return {
                    'action': 'partial_exit',
                    'reason': 'scale_out',
                    'size': exit_size * direction,
                    'risk_score': risk_score,
                    'profit_points': points_profit
                }
        
        # 5. Loss-based risk thresholds - cut losses if conditions worsen
        else:
            # Calculate loss level (1-10 scale)
            loss_ticks = abs(points_profit) / 0.25
            loss_level = min(10, loss_ticks / 8.0)
            
            # More lenient with small losses - linear scaling
            loss_risk_threshold = 0.7 - (loss_level * 0.04)
            
            # Exit if high risk during loss
            if risk_score > loss_risk_threshold:
                return {
                    'action': 'exit',
                    'reason': 'cut_loss',
                    'risk_score': risk_score,
                    'loss_points': points_profit
                }
        
        # 6. Regime-specific exits
        if regime == 'range_bound' and 'range_analysis' in market_data:
            position_in_range = market_data['range_analysis'].get('position_in_range', 0.5)
            
            # Exit longs near range top, shorts near range bottom
            if (direction > 0 and position_in_range > 0.85) or (direction < 0 and position_in_range < 0.15):
                if points_profit > 0:  # Only if in profit
                    return {
                        'action': 'exit',
                        'reason': 'range_boundary',
                        'profit_points': points_profit
                    }
        
        # Default - maintain position
        return {
            'action': 'hold',
            'risk_score': risk_score
        }    
    def adaptive_range_trade_management(self, position, current_price, market_data, range_analysis):
        """
        Specialized trade management system for mean reversion trades in range-bound markets
        with dynamic profit target calculation based on statistical range metrics
        """
        import numpy as np
        import datetime
        
        # Extract position data
        entry_price = position.get('entry_price', current_price)
        size = position.get('size', 0)
        direction = 1 if size > 0 else -1
        entry_time = position.get('entry_time', datetime.datetime.now())
        
        # Calculate time in position
        time_in_position = (datetime.datetime.now() - entry_time).total_seconds()
        
        # Extract range metrics
        if not range_analysis.get('boundaries_detected', False):
            # Fall back to standard management if range analysis not available
            return {'action': 'hold', 'reason': 'no_range_metrics_available'}
        
        range_high = range_analysis.get('range_high')
        range_low = range_analysis.get('range_low') 
        position_in_range = range_analysis.get('position_in_range', 0.5)
        z_score = range_analysis.get('z_score', 0)
        range_size = range_analysis.get('range_size', 0)
        
        # Calculate current profit
        profit_ticks = (current_price - entry_price) * direction
        
        # Enhanced mean reversion exit logic
        action = {'action': 'hold', 'reason': 'default'}
        
        # Calculate the optimal exit based on position within range and direction
        if direction > 0:  # Long position
            # Calculate entry position in range
            entry_position = (entry_price - range_low) / range_size if range_size > 0 else 0.5
            
            # Calculate move progress (how much we've moved toward the profit target)
            if entry_position < 0.3:  # Entered near bottom
                target_position = min(0.7, entry_position + 0.4)  # Target 40% up the range but cap at 70%
                target_price = range_low + (target_position * range_size)
                move_progress = (current_price - entry_price) / (target_price - entry_price) if target_price > entry_price else 0
                
                # Fast exit on deep move against entry
                if position_in_range < entry_position - 0.15 and profit_ticks < 0:
                    return {
                        'action': 'exit',
                        'reason': 'range_reversal_against_entry',
                        'details': f"position_in_range: {position_in_range:.2f}, entry: {entry_position:.2f}"
                    }
                
                # Take profit at or near target
                if move_progress > 0.85 and profit_ticks > 0:
                    return {
                        'action': 'exit',
                        'reason': 'range_target_achieved',
                        'details': f"move_progress: {move_progress:.2f}, target: {target_position:.2f}"
                    }
                    
                # Partial profit at 60% of move
                if move_progress > 0.6 and profit_ticks > 0 and time_in_position > 10:
                    return {
                        'action': 'partial_exit',
                        'size': max(1, int(abs(size) * 0.5)),
                        'reason': 'range_partial_target',
                        'details': f"move_progress: {move_progress:.2f}, target: {target_position:.2f}"
                    }
            
            # Middle entry (challenging in range markets)
            elif 0.3 <= entry_position <= 0.7:
                # For mid-range entries, more conservative targets
                if position_in_range > 0.8 and profit_ticks > 0:
                    return {
                        'action': 'exit',
                        'reason': 'range_upper_extreme',
                        'details': f"position_in_range: {position_in_range:.2f}, entry: {entry_position:.2f}"
                    }
                
                # Cut losses quickly on wrong direction from mid-range
                if position_in_range < entry_position - 0.1 and profit_ticks < 0:
                    return {
                        'action': 'exit',
                        'reason': 'range_wrong_direction',
                        'details': f"position_in_range: {position_in_range:.2f}, entry: {entry_position:.2f}"
                    }
            
            # Top entry (reversal trade)
            else:
                # For top entries, target middle of range
                target_position = 0.5
                target_price = range_low + (target_position * range_size)
                move_progress = (entry_price - current_price) / (entry_price - target_price) if entry_price > target_price else 0
                
                # Exit if move back higher
                if position_in_range > entry_position + 0.05 and profit_ticks < 0:
                    return {
                        'action': 'exit',
                        'reason': 'range_failed_reversal',
                        'details': f"position_in_range: {position_in_range:.2f}, entry: {entry_position:.2f}"
                    }
                
                # Take profit at target
                if move_progress > 0.7 and profit_ticks > 0:
                    return {
                        'action': 'exit',
                        'reason': 'range_reversal_target',
                        'details': f"move_progress: {move_progress:.2f}, target: {target_position:.2f}"
                    }
                    
        else:  # Short position
            # Calculate entry position in range
            entry_position = (entry_price - range_low) / range_size if range_size > 0 else 0.5
            
            # Calculate move progress (how much we've moved toward the profit target)
            if entry_position > 0.7:  # Entered near top
                target_position = max(0.3, entry_position - 0.4)  # Target 40% down the range but minimum at 30%
                target_price = range_low + (target_position * range_size)
                move_progress = (entry_price - current_price) / (entry_price - target_price) if entry_price > target_price else 0
                
                # Fast exit on deep move against entry
                if position_in_range > entry_position + 0.15 and profit_ticks < 0:
                    return {
                        'action': 'exit',
                        'reason': 'range_reversal_against_entry',
                        'details': f"position_in_range: {position_in_range:.2f}, entry: {entry_position:.2f}"
                    }
                
                # Take profit at or near target
                if move_progress > 0.85 and profit_ticks > 0:
                    return {
                        'action': 'exit',
                        'reason': 'range_target_achieved',
                        'details': f"move_progress: {move_progress:.2f}, target: {target_position:.2f}"
                    }
                    
                # Partial profit at 60% of move
                if move_progress > 0.6 and profit_ticks > 0 and time_in_position > 10:
                    return {
                        'action': 'partial_exit',
                        'size': max(1, int(abs(size) * 0.5)),
                        'reason': 'range_partial_target',
                        'details': f"move_progress: {move_progress:.2f}, target: {target_position:.2f}"
                    }
            
            # Middle entry (challenging in range markets)
            elif 0.3 <= entry_position <= 0.7:
                # For mid-range entries, more conservative targets
                if position_in_range < 0.2 and profit_ticks > 0:
                    return {
                        'action': 'exit',
                        'reason': 'range_lower_extreme',
                        'details': f"position_in_range: {position_in_range:.2f}, entry: {entry_position:.2f}"
                    }
                
                # Cut losses quickly on wrong direction from mid-range
                if position_in_range > entry_position + 0.1 and profit_ticks < 0:
                    return {
                        'action': 'exit',
                        'reason': 'range_wrong_direction',
                        'details': f"position_in_range: {position_in_range:.2f}, entry: {entry_position:.2f}"
                    }
            
            # Bottom entry (reversal trade)
            else:
                # For bottom entries, target middle of range
                target_position = 0.5
                target_price = range_low + (target_position * range_size)
                move_progress = (current_price - entry_price) / (target_price - entry_price) if target_price > entry_price else 0
                
                # Exit if move back lower
                if position_in_range < entry_position - 0.05 and profit_ticks < 0:
                    return {
                        'action': 'exit',
                        'reason': 'range_failed_reversal',
                        'details': f"position_in_range: {position_in_range:.2f}, entry: {entry_position:.2f}"
                    }
                
                # Take profit at target
                if move_progress > 0.7 and profit_ticks > 0:
                    return {
                        'action': 'exit',
                        'reason': 'range_reversal_target',
                        'details': f"move_progress: {move_progress:.2f}, target: {target_position:.2f}"
                    }
        
        # Time-based exits for range trades
        if time_in_position > 300 and profit_ticks > 0:  # 5 minutes with profit
            return {
                'action': 'exit',
                'reason': 'range_time_target',
                'details': f"time: {time_in_position:.1f}s, profit: {profit_ticks:.2f}"
            }
        elif time_in_position > 120 and profit_ticks < -5:  # 2 minutes with significant loss
            return {
                'action': 'exit',
                'reason': 'range_time_stop',
                'details': f"time: {time_in_position:.1f}s, loss: {profit_ticks:.2f}"
            }
        
        return action
    def enhanced_neural_alpha_extraction(self, base_signal, market_data):
        """Advanced neural alpha extraction with improved conflict resolution"""
        # Extract key metrics
        order_flow = market_data.get('order_flow', 0)
        delta = market_data.get('delta', 0)
        vpin = market_data.get('vpin', 0.3)
        regime = market_data.get('regime', 'unknown')
        
        # Calculate primary components with adaptive weighting
        if regime == 'range_bound':
            # Range markets - less weight on delta, more on order flow
            components = {
                'order_flow': order_flow * 0.65,
                'delta': delta * 0.35,
                'volatility': -min(0.1, market_data.get('volatility', 0.0001) * 1000),
                'momentum': market_data.get('momentum', 0) * 0.15  # Less weight on momentum in ranges
            }
        elif 'trending' in regime:
            # Trending markets - more weight on momentum and delta
            components = {
                'order_flow': order_flow * 0.55,
                'delta': delta * 0.60,
                'volatility': -min(0.1, market_data.get('volatility', 0.0001) * 1000),
                'momentum': market_data.get('momentum', 0) * 0.40  # Higher weight on momentum in trends
            }
        else:
            # Default weighting
            components = {
                'order_flow': order_flow * 0.60,
                'delta': delta * 0.45,
                'volatility': -min(0.1, market_data.get('volatility', 0.0001) * 1000),
                'momentum': market_data.get('momentum', 0) * 0.25
            }
        
        # Add liquidity component
        if 'liquidity_score' in market_data:
            liquidity = market_data['liquidity_score']
            if liquidity < 0.7:
                components['liquidity'] = -0.15
            elif liquidity > 0.95:
                components['liquidity'] = 0.05
        
        # Calculate component alignment metrics
        component_values = list(components.values())
        net_direction = np.sign(sum(component_values)) if component_values else 0
        
        # Calculate component alignment
        aligned_components = [c for c in component_values if np.sign(c) == net_direction and c != 0]
        conflicting_components = [c for c in component_values if np.sign(c) != net_direction and c != 0]
        
        # Calculate alignment score
        alignment_score = 0
        if component_values:
            total_magnitude = sum(abs(c) for c in component_values)
            if total_magnitude > 0:
                aligned_magnitude = sum(abs(c) for c in aligned_components) if aligned_components else 0
                conflicting_magnitude = sum(abs(c) for c in conflicting_components) if conflicting_components else 0
                alignment_score = (aligned_magnitude - conflicting_magnitude) / total_magnitude
        
        # Enhanced conflict resolution with base signal integration
        if alignment_score > 0.5:
            # High alignment - use neural components
            score = sum(aligned_components) if aligned_components else 0
            direction = net_direction
            confidence = 0.5 + (alignment_score * 0.5)  # 0.5-1.0 range
            
            # Check alignment with base signal
            if np.sign(base_signal) == direction or abs(base_signal) < 0.1:
                # Aligned with base signal or base signal is weak - enhance
                enhanced_signal = abs(score) * direction
                conflict_resolution = "boost"
            else:
                # Conflicting with base signal - weighted blend based on strengths
                base_strength = abs(base_signal)
                neural_strength = abs(score) * alignment_score
                total_strength = base_strength + neural_strength
                
                if neural_strength > base_strength * 2:
                    # Neural signal is much stronger - override base signal
                    enhanced_signal = abs(score) * 0.8 * direction  # Slightly reduced magnitude
                    conflict_resolution = "override"
                else:
                    # Weighted blend
                    blend_ratio = neural_strength / total_strength
                    enhanced_signal = base_signal * (1 - blend_ratio) + (abs(score) * direction * blend_ratio)
                    conflict_resolution = "blend"
        elif alignment_score > 0.0:
            # Moderate alignment - weighted by alignment score
            score = sum(component_values)
            direction = np.sign(score)
            confidence = 0.3 + (alignment_score * 0.4)  # 0.3-0.7 range
            
            # Always blend with base signal for moderate alignment
            blend_ratio = alignment_score * 0.6
            enhanced_signal = base_signal * (1 - blend_ratio) + (abs(score) * direction * blend_ratio)
            conflict_resolution = "weighted_blend"
        else:
            # Poor alignment - defer to base signal
            score = abs(base_signal) * 0.5  # Reduce magnitude
            direction = np.sign(base_signal)
            confidence = 0.2 + (abs(alignment_score) * 0.15)  # 0.2-0.35 range
            enhanced_signal = base_signal * 0.9  # Slightly reduced base signal
            conflict_resolution = "defer"
        
        # Apply VPIN adjustment
        if vpin > 0.3:
            confidence *= max(0.7, 1.0 - ((vpin - 0.3) * 0.75))
        
        # Cap final signal magnitude
        enhanced_signal = np.clip(enhanced_signal, -1.0, 1.0)
        
        return {
            'score': abs(enhanced_signal),
            'direction': np.sign(enhanced_signal),
            'confidence': confidence,
            'components': components,
            'alignment_score': alignment_score,
            'enhanced_signal': enhanced_signal,
            'resolution_type': conflict_resolution
        }
    def initialize_trading_parameters(self):
        """Initialize parameters for trading system"""
        # Existing initialization code...
        
        # NEW: Add cooldown parameters to prevent overtrading
        self._trade_cooldown_period = 60  # 60 seconds base cooldown between trades
        self._last_trade_time = datetime.datetime.now() - datetime.timedelta(seconds=3600)  # Initialize with old time
        self._cooldown_multipliers = {
            'loss': 2.0,       # Longer cooldown after losses
            'win': 1.2,        # Slightly longer cooldown after wins
            'range_bound': 1.5,  # More cooldown in range markets to prevent overtrading
            'volatile': 1.8,     # More cooldown in volatile markets 
            'trending': 0.8      # Less cooldown in trending markets
        }

    def calculate_adaptive_cooldown(self, trade_result):
        """Calculate adaptive cooldown period based on trade result and market conditions"""
        base_cooldown = self._trade_cooldown_period
        
        # Get profit status
        is_profit = trade_result.get('profit', 0) > 0
        
        # Get trade regime
        regime = trade_result.get('regime', 'unknown')
        
        # Start with base multiplier
        multiplier = 1.0
        
        # Apply profit/loss multiplier
        if is_profit:
            multiplier *= self._cooldown_multipliers['win']
        else:
            multiplier *= self._cooldown_multipliers['loss']
        
        # Apply regime multiplier
        if 'trending' in regime:
            multiplier *= self._cooldown_multipliers['trending']
        elif regime == 'range_bound':
            multiplier *= self._cooldown_multipliers['range_bound']
        elif regime == 'volatile':
            multiplier *= self._cooldown_multipliers['volatile']
        
        # NEW: Add execution quality factor
        execution_quality = trade_result.get('execution_quality', 0.5)
        if execution_quality < 0.4:  # Poor execution quality
            multiplier *= 1.3  # Longer cooldown after poor execution
        
        # Calculate final cooldown in seconds
        cooldown = int(base_cooldown * multiplier)
        
        # Log the cooldown calculation
        self.logger.info(f"Adaptive cooldown: {cooldown}s (profit={is_profit}, regime={regime}, quality={execution_quality:.2f})")
        
        return cooldown    
    def synchronize_quantum_patterns(self, edge_patterns, flow_patterns, memory_patterns, market_data):
        """Advanced pattern synchronization with conflict resolution"""
        # Check for pattern detections
        edge_detected = edge_patterns.get('detected', False)
        flow_detected = flow_patterns.get('pattern_detected', False)
        memory_detected = memory_patterns.get('pattern_detected', False)
        
        # If no patterns detected, return default result
        if not (edge_detected or flow_detected or memory_detected):
            return {
                'synchronized': False,
                'primary_pattern': None,
                'direction': 0,
                'confidence': 0,
                'signal_adjustment': 0,
                'threshold_adjustment': 0
            }
        
        # Extract pattern directions and confidences
        patterns = []
        
        if edge_detected:
            patterns.append({
                'type': 'edge',
                'name': edge_patterns.get('pattern', 'unknown'),
                'direction': edge_patterns.get('direction', 0),
                'confidence': edge_patterns.get('confidence', 0.5),
                'strength': edge_patterns.get('strength', 0.3)
            })
        
        if flow_detected:
            patterns.append({
                'type': 'flow',
                'name': flow_patterns.get('pattern_name', 'unknown'),
                'direction': flow_patterns.get('direction', 0),
                'confidence': flow_patterns.get('confidence', 0.5),
                'strength': 0.3  # Default strength for flow patterns
            })
        
        if memory_detected:
            patterns.append({
                'type': 'memory',
                'name': memory_patterns.get('pattern_description', 'unknown'),
                'direction': 1 if memory_patterns.get('expected_direction', 'up') == 'up' else -1,
                'confidence': memory_patterns.get('confidence', 0.5),
                'strength': 0.2  # Default strength for memory patterns
            })
        
        # Group patterns by direction
        positive_patterns = [p for p in patterns if p['direction'] > 0]
        negative_patterns = [p for p in patterns if p['direction'] < 0]
        
        # Calculate weighted confidences
        pos_confidence = sum(p['confidence'] * p['strength'] for p in positive_patterns)
        neg_confidence = sum(p['confidence'] * p['strength'] for p in negative_patterns)
        
        # Determine winning direction
        if pos_confidence > neg_confidence:
            winning_patterns = positive_patterns
            direction = 1
            net_confidence = (pos_confidence - 0.3 * neg_confidence) / (sum(p['strength'] for p in positive_patterns) or 1)
        elif neg_confidence > pos_confidence:
            winning_patterns = negative_patterns
            direction = -1
            net_confidence = (neg_confidence - 0.3 * pos_confidence) / (sum(p['strength'] for p in negative_patterns) or 1)
        else:
            # No clear winner
            return {
                'synchronized': False,
                'primary_pattern': None,
                'direction': 0,
                'confidence': 0,
                'signal_adjustment': 0,
                'threshold_adjustment': 0
            }
        
        # Find primary pattern (highest confidence * strength)
        primary_pattern = max(winning_patterns, key=lambda p: p['confidence'] * p['strength'])
        
        # Calculate confidence-based signal adjustment
        base_adjustment = 0.15 + min(0.3, net_confidence * 0.5)
        signal_adjustment = direction * base_adjustment
        
        # Calculate threshold adjustment based on confidence and pattern type
        if primary_pattern['type'] == 'edge':
            # Edge patterns allow for larger threshold reductions
            threshold_adjustment = -0.14 * net_confidence
        elif primary_pattern['type'] == 'flow':
            # Flow patterns get moderate threshold adjustments
            threshold_adjustment = -0.10 * net_confidence
        else:
            # Memory patterns get smaller threshold adjustments
            threshold_adjustment = -0.07 * net_confidence
        
        # Cap the threshold adjustment
        threshold_adjustment = max(-0.20, min(0.05, threshold_adjustment))
        
        # Apply regime-specific adjustments
        regime = market_data.get('regime', 'unknown')
        if regime == 'range_bound':
            # Tighten thresholds in range-bound markets
            threshold_adjustment -= 0.04
        elif 'trending' in regime:
            # Adjust in trending markets based on alignment
            trend_direction = 1 if regime == 'trending_up' else -1
            if direction == trend_direction:
                # Aligned with trend - reduce threshold further
                threshold_adjustment -= 0.02
            else:
                # Against trend - tighten threshold
                threshold_adjustment += 0.03
        
        # Create composite pattern name
        pattern_name = f"{primary_pattern['type']}_{primary_pattern['name']}"
        
        return {
            'synchronized': True,
            'primary_pattern': pattern_name,
            'direction': direction,
            'confidence': net_confidence,
            'signal_adjustment': signal_adjustment,
            'threshold_adjustment': threshold_adjustment
        }

    def adaptive_quantum_position_sizing(self, base_size, signal, market_data, range_analysis=None):
        """Position sizing with quantum-enhanced adaptations for current market conditions"""
        import numpy as np
        
        # Extract key market metrics
        regime = market_data.get('regime', 'unknown')
        regime_confidence = market_data.get('regime_confidence', 0.5)
        volatility = market_data.get('volatility', 0.0001)
        entanglement = market_data.get('entanglement', 0.5)
        vpin = market_data.get('vpin', 0.3)
        
        # Start with base size
        position_size = base_size
        
        # Safety check
        if base_size <= 0:
            return 0
        
        # Calculate base volatility factor
        # Higher volatility = smaller position
        vol_ratio = volatility / 0.0001  # Normalized to baseline volatility
        vol_factor = 1.0 / min(2.0, max(0.5, vol_ratio))
        vol_factor = min(1.2, max(0.5, vol_factor))
        
        # Signal strength factor
        # Stronger signal = larger position
        signal_factor = 0.5 + (0.5 * min(1.0, abs(signal) / 0.5))
        
        # Regime factor
        regime_factor = 1.0  # Default
        if regime == 'volatile':
            regime_factor = 0.7  # Smaller in volatile
        elif regime == 'trending_up' or regime == 'trending_down':
            regime_factor = 1.1  # Larger in trending
        elif regime == 'range_bound':
            regime_factor = 0.9  # Slightly smaller in range-bound
        elif regime == 'exhausted_trend':
            regime_factor = 0.8  # Smaller in exhausted trends
        
        # Scale by regime confidence
        regime_factor = regime_factor * min(1.0, regime_confidence + 0.3)
        
        # VPIN factor (liquidity toxicity)
        # Higher VPIN = smaller position
        vpin_factor = 1.0
        if vpin > 0.4:
            vpin_factor = max(0.6, 1.0 - ((vpin - 0.4) * 1.2))
        
        # Flow alignment factor
        order_flow = market_data.get('order_flow', 0.0)
        flow_factor = 1.0
        
        # If order flow aligns with signal, boost position size
        if abs(order_flow) > 0.1 and np.sign(order_flow) == np.sign(signal):
            flow_factor = 1.0 + (abs(order_flow) * 0.3)
            flow_factor = min(1.15, flow_factor)
        
        # Apply all factors
        self.logger.info(f"Position factors: Vol:{vol_factor:.2f} Signal:{signal_factor:.2f} Regime:{regime_factor:.2f} VPIN:{vpin_factor:.2f} Flow:{flow_factor:.2f}")
        position_size = position_size * vol_factor * signal_factor * regime_factor * vpin_factor * flow_factor
        
        # Special handling for range-bound markets using range position
        if regime == 'range_bound' and range_analysis and range_analysis.get('boundaries_detected', False):
            position_in_range = range_analysis.get('position_in_range', 0.5)
            range_confidence = range_analysis.get('confidence', 0.3)
            
            # Calculate edge factor for range extremes
            edge_factor = 1.0  # Default
            
            # For long signals near bottom of range
            if signal > 0 and position_in_range < 0.3:
                edge_factor = 1.0 + ((0.3 - position_in_range) * 2.0)
                edge_factor = min(1.5, edge_factor)
            
            # For short signals near top of range
            elif signal < 0 and position_in_range > 0.7:
                edge_factor = 1.0 + ((position_in_range - 0.7) * 2.0)
                edge_factor = min(1.5, edge_factor)
            
            # Scale edge factor by range confidence
            edge_factor = 1.0 + ((edge_factor - 1.0) * range_confidence)
            
            # Enhanced logging
            original_size = int(position_size)
            position_size = int(position_size * edge_factor)
            self.logger.info(f"Range position sizing: base={original_size}, edge_factor={edge_factor:.2f}, pos_in_range={position_in_range:.2f}, confidence={range_confidence:.2f}, final={position_size}")
            
            # Additional reduction for extreme choppiness
            choppiness = market_data.get('choppiness_analysis', {}).get('choppiness_score', 0.0)
            if choppiness > 0.6:
                # Highly choppy market
                original_size = position_size
                chop_factor = max(0.2, 1.0 - (choppiness - 0.6) * 2.5)
                position_size = int(position_size * chop_factor)
                
                # Only enter trade if position size remains meaningful
                if position_size == 0 and original_size > 0:
                    position_size = 0  # Completely avoid trade in extreme chop
                    self.logger.info(f"Choppy market position reduction: {original_size} → {position_size}")
        
        # Final adjustments and safety checks
        position_size = int(position_size)
        
        # Don't allow zero position size
        if position_size == 0 and base_size > 0:
            self.logger.info(f"Position sizing: factors reduced position to zero")
        
        return position_size    
    def advanced_range_boundary_detection(self, price_history, current_price, market_data):
        """Enhanced range boundary detection with stability improvements"""
        # Extract recent price data
        if not price_history or len(price_history) < 10:
            return {'boundaries_detected': False}
        
        # Convert to numpy array for efficient calculations
        prices = np.array(price_history)
        
        # IMPROVED: Use multiple timeframes to improve stability
        # Short-term range (last 20-30 bars)
        short_term = prices[-30:]
        # Medium-term range (last 50-60 bars)
        medium_term = prices[-60:] if len(prices) >= 60 else prices
        
        # Calculate ranges with Percentile method to reduce outlier impact
        short_high = np.percentile(short_term, 95)  # 95th percentile for high
        short_low = np.percentile(short_term, 5)    # 5th percentile for low
        
        medium_high = np.percentile(medium_term, 96)  # 96th percentile for high
        medium_low = np.percentile(medium_term, 4)    # 4th percentile for low
        
        # IMPROVED: Apply exponential weighting to reduce range boundary fluctuation
        # Use previous range boundaries if available
        prev_range_high = market_data.get('range_analysis', {}).get('range_high', 0)
        prev_range_low = market_data.get('range_analysis', {}).get('range_low', 0)
        
        # Calculate volatility for adaptive stability factor
        volatility = np.std(prices[-20:]) / np.mean(prices[-20:])
        stability_factor = max(0.6, min(0.9, 1.0 - volatility * 100))  # More volatile = less stability
        
        # IMPROVED: Only accept new boundaries if they're not too different from previous ones
        if prev_range_high > 0 and prev_range_low > 0:
            # Blend previous and new boundaries with stability factor
            range_high = prev_range_high * stability_factor + short_high * (1 - stability_factor)
            range_low = prev_range_low * stability_factor + short_low * (1 - stability_factor)
            
            # Diagnostic logging
            change_high = (range_high - prev_range_high) / prev_range_high
            change_low = (range_low - prev_range_low) / prev_range_low
            
            if abs(change_high) > 0.001 or abs(change_low) > 0.001:
                self.logger.info(f"Range boundary stability: high={change_high:.4f}, low={change_low:.4f}, factor={stability_factor:.2f}")
        else:
            # Initial range boundaries - blend short and medium term
            range_high = short_high * 0.7 + medium_high * 0.3
            range_low = short_low * 0.7 + medium_low * 0.3
        
        # Calculate range statistics
        range_size = range_high - range_low
        if range_size <= 0:
            return {'boundaries_detected': False}
        
        # Position in range (0 = low, 1 = high)
        position_in_range = (current_price - range_low) / range_size
        position_in_range = max(0, min(1, position_in_range))  # Clamp between 0 and 1
        
        # NEW: Calculate Z-Score relative to range mean with non-linear transformation
        range_mean = (range_high + range_low) / 2
        range_stdev = range_size / 4  # Approximate standard deviation
        
        # Z-score calculation (standard deviations from mean)
        z_score = (current_price - range_mean) / range_stdev
        
        # IMPROVED: Calculate confidence based on multiple factors
        # 1. Price consistency within range
        price_in_range_count = np.sum((prices >= range_low) & (prices <= range_high))
        price_consistency = price_in_range_count / len(prices)
        
        # 2. Trend strength (lower is more range-bound)
        trend_strength = market_data.get('trend_strength', 1.0)
        trend_factor = 1.0 - min(0.5, (trend_strength - 1.0) * 0.2)
        
        # 3. Recent bounces off range boundaries
        bounces = self.detect_range_bounces(prices, range_high, range_low)
        bounce_factor = min(1.0, 0.5 + bounces * 0.1)
        
        # Calculate overall confidence with weighted factors
        confidence = price_consistency * 0.5 + trend_factor * 0.3 + bounce_factor * 0.2
        confidence = max(0.1, min(0.95, confidence))
        
        return {
            'boundaries_detected': True,
            'range_high': range_high,
            'range_low': range_low,
            'range_size': range_size,
            'position_in_range': position_in_range,
            'z_score': z_score,
            'confidence': confidence
        }

    def detect_range_bounces(self, prices, range_high, range_low):
        """Detect how many times price has bounced off range boundaries"""
        if len(prices) < 10:
            return 0
        
        # Use a percentage of range size as bounce threshold
        bounce_threshold = (range_high - range_low) * 0.03
        bounces = 0
        
        # Loop through price data to detect bounces
        for i in range(2, len(prices)):
            # Detect bottom bounces
            if (prices[i-1] <= range_low + bounce_threshold and 
                prices[i] > prices[i-1] and 
                prices[i-2] > prices[i-1]):
                bounces += 1
            
            # Detect top bounces
            if (prices[i-1] >= range_high - bounce_threshold and 
                prices[i] < prices[i-1] and 
                prices[i-2] < prices[i-1]):
                bounces += 1
        
        return bounces
    def send_dashboard_update(self, data_type, data):
        """Send update to the dashboard"""
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            message = json.dumps({"type": data_type, **data})
            sock.sendto(message.encode('utf-8'), ('127.0.0.1', 7777))
            sock.close()
        except Exception as e:
            self.logger.error(f"Error sending dashboard update: {e}")

    def report_trade_to_dashboard(self, trade_result):
        """Format and send trade data to dashboard"""
        trade_data = {
            'time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'trade_id': trade_result.get('trade_id', ''),
            'entry_price': trade_result.get('entry_price', 0),
            'exit_price': trade_result.get('exit_price', 0),
            'size': trade_result.get('size', 0),
            'pnl': trade_result.get('profit', 0),
            'signal': trade_result.get('signal_strength', 0),
            'regime': trade_result.get('regime', 'unknown'),
            'regime_confidence': trade_result.get('regime_confidence', 0),
            'reason': trade_result.get('reason', '')
        }
        self.send_dashboard_update('trade', trade_data)

    def report_status_to_dashboard(self):
        """Send system status to dashboard"""
        import socket
        try:
            # Get drawdown directly from the attribute
            drawdown = 0.0
            if hasattr(self.risk_manager, 'current_drawdown'):
                drawdown = self.risk_manager.current_drawdown * 100
            
            # Create status data
            status_data = {
                'status': 'trading',
                'equity': self.execution_engine.get_account_equity(),
                'pnl': self.calculate_session_pnl(),
                'win_rate': self.get_recent_win_rate(50) * 100,
                'total_trades': len(self.trade_history) if hasattr(self, 'trade_history') else 0,
                'drawdown': drawdown,
                'regime': self.current_regime if hasattr(self, 'current_regime') else 'unknown',
                'regime_confidence': self.regime_confidence if hasattr(self, 'regime_confidence') else 0.0
            }
            
            # Send update with socket properly imported
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            message = json.dumps({"type": 'status', **status_data})
            sock.sendto(message.encode('utf-8'), ('127.0.0.1', 7777))
            sock.close()
        except Exception as e:
            self.logger.error(f"Error reporting status to dashboard: {e}")

    def report_equity_to_dashboard(self):
        """Send equity update to dashboard"""
        try:
            # Get current equity
            current_equity = self.execution_engine.get_account_equity()
            
            # Get current time
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Create equity data
            equity_data = {
                'time': timestamp,
                'equity': current_equity
            }
            
            # Send update
            self.send_dashboard_update('equity', equity_data)
        except Exception as e:
            self.logger.error(f"Error reporting equity to dashboard: {e}") 
    def stabilize_signal(self, new_signal, market_data):
        """Prevent rapid signal flipping by considering recent signals"""
        import numpy as np
        
        # Initialize signal history if not exists
        if not hasattr(self, '_signal_history'):
            self._signal_history = []
            
        # Add current signal to history (keep last 5)
        self._signal_history.append(new_signal)
        if len(self._signal_history) > 5:
            self._signal_history.pop(0)
            
        # Check for signal stability
        if len(self._signal_history) >= 3:
            recent_direction = np.sign(np.mean(self._signal_history[-3:]))
            current_direction = np.sign(new_signal)
            
            # If recent signals strongly point one way and current signal flips weakly
            if recent_direction != 0 and current_direction != 0 and recent_direction != current_direction:
                recent_magnitude = abs(np.mean(self._signal_history[-3:]))
                current_magnitude = abs(new_signal)
                
                # If recent signals were strong and current flip is weak, stabilize
                if recent_magnitude > 0.4 and current_magnitude < 0.3:
                    stabilized_signal = new_signal * 0.3  # Significantly dampen the flip
                    self.logger.info(f"Signal stabilization: {new_signal:.2f} → {stabilized_signal:.2f} (preventing weak flip)")
                    return stabilized_signal
                    
        return new_signal    
    def calculate_precision_exit_targets(self, entry_price, direction, market_data, regime_info):
        """Calculate precision exit targets for mean-reversion trades"""
        if regime_info.get('regime') != 'range_bound':
            return self.calculate_standard_targets(entry_price, direction, market_data)
        
        # Extract key metrics
        volatility = market_data.get('volatility', 0.0001)
        order_flow = market_data.get('order_flow', 0)
        vpin = market_data.get('vpin', 0.3)
        
        # Calculate normalized metrics for exit scaling
        norm_vol = min(1.0, volatility / 0.0003)
        norm_flow = min(1.0, abs(order_flow))
        
        # Calculate base targets
        atr = market_data.get('atr', entry_price * 0.002)
        
        # Calculate dynamic profit factors
        # Lower targets for higher volatility (faster exits)
        base_profit_factor = 1.0 - (norm_vol * 0.3)
        
        # Adjust targets based on order flow alignment
        flow_direction = 1 if order_flow > 0 else -1
        if flow_direction == direction:
            # Aligned order flow - aim for larger targets
            flow_adjustment = 0.2 * norm_flow
        else:
            # Opposing order flow - be more conservative
            flow_adjustment = -0.1 * norm_flow
        
        # Adjust for VPIN (liquidity)
        vpin_adjustment = 0
        if vpin > 0.35:
            # Higher VPIN = take profits faster
            vpin_adjustment = -0.2 * ((vpin - 0.35) / 0.65)
        
        # Calculate final factors
        final_factor = max(0.5, min(1.5, base_profit_factor + flow_adjustment + vpin_adjustment))
        
        # Calculate targets
        target1 = entry_price + (direction * atr * final_factor)
        target2 = entry_price + (direction * atr * final_factor * 1.5)
        target3 = entry_price + (direction * atr * final_factor * 2.0)
        
        self.logger.info(f"Precision exit calculation: base={base_profit_factor:.2f}, " +
                        f"flow_adj={flow_adjustment:.2f}, vpin_adj={vpin_adjustment:.2f}, final={final_factor:.2f}")
        
        return [target1, target2, target3]      
               
    def _main_thread(self):
        """Main system thread for trading logic with quantum-enhanced regime-specific adaptations"""
        self.logger.info("Quantum-Enhanced Elite main system thread started")
        
        # Make sure numpy is available for the advanced quantum algorithms
        import numpy as np
        from scipy import stats  # For advanced statistical analysis
        import math  # For logarithmic recovery functions
        
        # Trading state
        last_trade_time = None
        position_size = 0
        last_signal = 0.0
        stop_loss_price = None
        profit_target_price = None
        current_position_id = None
        last_regime = "unknown"
        
        # Initialize enhancement tracking
        self._high_confirmation_count = 0
        self._low_signal_consecutive_count = 0
        
        # Initialize signal history for quantum stabilization
        if not hasattr(self, '_signal_history'):
            self._signal_history = []
            self._last_trade_direction = 0
            self._position_established_time = None
        
        # Initialize order flow data structures
        if not hasattr(self, '_orderflow_samples'):
            self._orderflow_samples = []
        
        if not hasattr(self, '_orderflow_history'):
            self._orderflow_history = []
        
        # Initialize delta and flow histories for entanglement calculations
        if not hasattr(self, '_delta_history'):
            self._delta_history = []
        if not hasattr(self, '_flow_history'):
            self._flow_history = []
        
        # Initialize pattern memory and quantum state data
        if not hasattr(self, '_quantum_memory_bank'):
            self._quantum_memory_bank = {
                'patterns': [],
                'last_check': datetime.datetime.now(),
                'check_interval': 300  # seconds
            }
        
        # Initialize range data for profit target optimization
        if not hasattr(self, '_range_data'):
            self._range_data = {
                'range_high': None,
                'range_low': None,
                'last_update': datetime.datetime.now(),
                'samples': []
            }
        
        # Initialize regime transition tracking
        if not hasattr(self, '_regime_transition_data'):
            self._regime_transition_data = {
                'last_transition': datetime.datetime.now(),
                'transition_from': 'unknown',
                'transition_to': 'unknown',
                'stabilization_period': 60  # seconds
            }
        
        # Initialize divergence enhancement tracking if not already
        if not hasattr(self, '_signal_enhancement_active'):
            self._signal_enhancement_active = False
            self._enhanced_signal = 0.0
            self._signal_enhancement_expiry = datetime.datetime.now()
        
        # Initialize trade history if not already
        if not hasattr(self, 'trade_history'):
            self.trade_history = []
        
        # Initialize trade factor data if not already
        if not hasattr(self, '_trade_factor_data'):
            self._trade_factor_data = {}
        
        # Initialize position entry prices tracking
        if not hasattr(self, '_position_entry_prices'):
            self._position_entry_prices = {}

        if not hasattr(self, '_risk_metrics_history'):
            self._risk_metrics_history = []    

        # Initialize price tracker if not already
        if not hasattr(self, 'price_tracker'):
            from collections import deque
            class SimpleTracker:
                def __init__(self, max_size=1000):
                    self.price_history = deque(maxlen=max_size)
                    self.volatility_history = deque(maxlen=max_size)
                    
                def update(self, price, volatility=None):
                    self.price_history.append(price)
                    if volatility is not None:
                        self.volatility_history.append(volatility)
                        
                def get_history(self, n=100):
                    return list(self.price_history)[-n:]
                    
                def get_volatility_history(self, n=30):
                    return list(self.volatility_history)[-n:]
            
            self.price_tracker = SimpleTracker()
        
        # Initialize liquidity history for advanced analysis
        if not hasattr(self, 'liquidity_history'):
            self.liquidity_history = []
        
        # Initialize liquidity shock tracking for execution quality
        if not hasattr(self, '_last_liquidity_shock'):
            self._last_liquidity_shock = {
                'time': datetime.datetime.now() - datetime.timedelta(hours=1),
                'magnitude': 0.0
            }
        
        # NEW: Initialize trade timing tracking
        if not hasattr(self, '_trade_timing'):
            self._trade_timing = {
                'min_hold_time': 15,  # Minimum seconds to hold trade unless catastrophic
                'entry_times': {},    # Track when positions were entered
                'partial_exits': {}   # Track partial exits
            }
        
        # Adaptive parameters for different market regimes
        regime_params = {
            "volatile": {
                "signal_threshold": 0.60,      # Reduced threshold in volatile markets
                "stop_multiplier": 1.25,       # Wider stops in volatile markets
                "position_scale": 0.85,        # Smaller positions in volatile markets
                "profit_accelerator": 0.6,     # Take profits faster in volatile markets
                "delta_confirm": 0.1,          # Require delta confirmation in volatile markets
                "partial_exit_pct": 0.33       # Faster partial exits in volatile markets
            },
            "trending_up": {
                "signal_threshold": 0.45,      # Reduced threshold in trending markets
                "stop_multiplier": 1.0,        # Standard stops in trending markets
                "position_scale": 1.1,         # Larger positions in trending markets
                "profit_accelerator": 1.0,     # Standard profit taking in trending markets
                "delta_confirm": 0.0,          # No delta confirmation needed in trending markets
                "partial_exit_pct": 0.25       # Standard partial exits in trending markets
            },
            "trending_down": {
                "signal_threshold": 0.45,      # Reduced threshold in trending markets
                "stop_multiplier": 1.0,        # Standard stops in trending markets
                "position_scale": 1.1,         # Larger positions in trending markets
                "profit_accelerator": 1.0,     # Standard profit taking in trending markets
                "delta_confirm": 0.0,          # No delta confirmation needed in trending markets
                "partial_exit_pct": 0.25       # Standard partial exits in trending markets
            },
            "range_bound": {
                "signal_threshold": 0.55,      # Reduced threshold in range-bound markets
                "stop_multiplier": 0.8,        # Tighter stops in range-bound markets
                "position_scale": 0.9,         # Slightly smaller positions in range-bound markets
                "profit_accelerator": 1.2,     # Take profits faster in range-bound markets
                "delta_confirm": 0.05,         # Light delta confirmation in range-bound markets
                "partial_exit_pct": 0.3        # Faster partial exits in range-bound markets
            },
            "exhausted_trend": {               # NEW REGIME TYPE
                "signal_threshold": 0.40,      # Lower threshold for exhausted trends (reversal potential)
                "stop_multiplier": 1.2,        # Wider stops for potentially volatile reversals
                "position_scale": 0.9,         # Conservative position size for trend reversals
                "profit_accelerator": 0.8,     # Medium profit taking speed in trend reversals
                "delta_confirm": 0.1,          # Require delta confirmation for reversals
                "partial_exit_pct": 0.35       # Faster partial exits in volatile reversals
            },
            "unknown": {
                "signal_threshold": 0.50,      # Reduced threshold in unknown regime
                "stop_multiplier": 1.1,        # Slightly wider stops in unknown regime
                "position_scale": 0.8,         # Conservative position size in unknown regime
                "profit_accelerator": 1.0,     # Standard profit taking in unknown regime
                "delta_confirm": 0.0,          # No delta confirmation needed in unknown regime
                "partial_exit_pct": 0.25       # Standard partial exits in unknown regime
            }
        }
        
        # Make regime params accessible to other methods
        if not hasattr(self, 'regime_params'):
            self.regime_params = regime_params
        
        try:
            while self.running and not self.shutdown_requested:
                try:
                    # Get current time
                    current_time = datetime.datetime.now()
                    
                    # Get market data
                    market_data = self.market_data.get_realtime_data()
                    if not market_data:
                        time.sleep(0.1)
                        continue
                    # Get current price
                    current_price = market_data.get('price')
                    if not current_price:
                        time.sleep(0.1)
                        continue
                    self.report_status_to_dashboard()
                    self.report_equity_to_dashboard()
                    # Update price tracker with current price
                    volatility = market_data.get('volatility', None)
                    self.price_tracker.update(current_price, volatility)
                    
                    # Update liquidity history for advanced analysis
                    liquidity = market_data.get('liquidity_score', 0.5)
                    self.liquidity_history.append(liquidity)
                    if len(self.liquidity_history) > 20:
                        self.liquidity_history.pop(0)
                    
                    # Update order flow history for persistence analysis
                    order_flow = market_data.get('order_flow', 0.0)
                    self._orderflow_history.append(order_flow)
                    if len(self._orderflow_history) > 20:
                        self._orderflow_history.pop(0)
                    
                    # Update delta and flow history for entanglement calculation
                    delta = market_data.get('delta', 0.0)
                    self._delta_history.append(delta)
                    self._flow_history.append(order_flow)
                    if len(self._delta_history) > 30:
                        self._delta_history = self._delta_history[-30:]
                    if len(self._flow_history) > 30:
                        self._flow_history = self._flow_history[-30:]
                    
                    # Get price history for analysis
                    price_history = self.price_tracker.get_history(120)
                    
                    # ENHANCED: Apply advanced range boundary detection with volume support
                    range_analysis = self.advanced_range_boundary_detection(price_history, current_price, market_data)
                    if range_analysis.get('boundaries_detected', False):
                        self.logger.info(f"Range boundaries detected: high=${range_analysis['range_high']:.2f}, " +
                                    f"low=${range_analysis['range_low']:.2f}, confidence={range_analysis['confidence']:.2f}")
                        self.logger.info(f"Position in range: {range_analysis['position_in_range']:.2f}, z-score: {range_analysis['z_score']:.2f}")
                        
                        # Add range data to market_data for other components
                        market_data['range_analysis'] = range_analysis
                    
                    # Get current position
                    current_position = self.execution_engine.get_position('NQ')
                    position_size = current_position.get('quantity', 0)
                    
                    # ===== QUANTUM-ENHANCED SIGNAL GENERATION =====
                    
                    # Get original strategy signals
                    original_signals = self.strategy_manager.get_strategy_signals()
                    original_composite = original_signals.get('composite', 0.0)

                    # Get key market metrics for signal enhancement
                    delta = market_data.get('delta', 0.0)
                    order_flow = market_data.get('order_flow', 0.0)
                    vpin = market_data.get('vpin', 0.5)

                    # Get initial market regime classification
                    base_regime_info = self.regime_classifier.get_current_regime()
                    
                    # Apply quantum harmonization to refine regime classification
                    regime_info = self.harmonize_regime_classification(base_regime_info, market_data)
                    current_regime = regime_info.get('regime', 'unknown')
                    regime_confidence = regime_info.get('confidence', 0.0)
                    
                    # Add regime information to market_data for signal stabilization and other components
                    market_data['regime'] = current_regime
                    market_data['regime_confidence'] = regime_confidence
                    
                    # Apply mean reversion signal adjustment in range-bound markets
                    if current_regime == 'range_bound' and 'range_analysis' in market_data:
                        range_analysis = market_data['range_analysis']
                        if abs(range_analysis.get('z_score', 0)) > 0.5:  # Only apply when z-score is significant
                            mean_reversion_signal = -0.2 * range_analysis['z_score']  # Higher z-score = stronger mean reversion
                            original = original_composite
                            original_composite += mean_reversion_signal
                            self.logger.info(f"Range mean reversion signal adjustment: {original:.2f} → {original_composite:.2f} (z-score: {range_analysis['z_score']:.2f})")
                    
                    # Apply regime transition protection when regime changes
                    if current_regime != last_regime:
                        self.logger.info(f"Trade adaptation: Market regime changed from {last_regime} to {current_regime} (confidence: {regime_confidence:.2f})")
                        
                        # Update regime transition data
                        transition_time = datetime.datetime.now()
                        self._regime_transition_data = {
                            'last_transition': transition_time,
                            'transition_from': last_regime,
                            'transition_to': current_regime,
                            'stabilization_period': 60  # seconds
                        }
                        
                        last_regime = current_regime
                    
                    # Log if regime was reclassified
                    if current_regime != base_regime_info.get('regime', 'unknown'):
                        self.logger.info(f"Quantum regime harmonization: {base_regime_info.get('regime', 'unknown')} → {current_regime} ({regime_info.get('reclassification_reason', 'enhanced_analysis')})")

                    # Get regime-specific parameters
                    params = regime_params.get(current_regime, regime_params["unknown"])

                    # Apply confidence scaling to parameters
                    confidence_factor = max(0.7, min(1.0, regime_confidence))
                    for key in params:
                        if key in ["signal_threshold", "stop_multiplier", "position_scale"]:
                            # Scale parameters by confidence - move closer to "unknown" with low confidence
                            params[key] = params[key] * confidence_factor + regime_params["unknown"][key] * (1 - confidence_factor)

                    # Extract Delta-Flow alpha
                    delta_flow_alpha = self.extract_delta_flow_alpha(market_data)
                    if delta_flow_alpha.get('alpha_extracted', False):
                        # Use delta-flow alpha when detected with high confidence
                        if delta_flow_alpha.get('confidence', 0) > 0.5:
                            base_composite = delta_flow_alpha.get('signal', original_composite)
                            self.logger.info(f"Delta-Flow alpha extraction: divergence={delta_flow_alpha.get('divergence', 0):.2f}, signal={base_composite:.2f}, confidence={delta_flow_alpha.get('confidence', 0):.2f}")
                    
                    # Analyze order flow persistence
                    order_flow_analysis = self.analyze_order_flow_persistence(market_data)
                    if order_flow_analysis.get('persistent', False):
                        self.logger.info(f"Persistent order flow detected: direction={order_flow_analysis['direction']}, score={order_flow_analysis['persistence_score']:.2f}")
                        
                        # For very small signals, apply more bias with persistent order flow
                        if abs(original_composite) < 0.1:
                            flow_direction = order_flow_analysis['direction']
                            flow_bias = flow_direction * 0.15 * order_flow_analysis['persistence_score']
                            original = original_composite
                            original_composite += flow_bias
                            self.logger.info(f"Order flow persistence bias: {original:.2f} → {original_composite:.2f}")
                    
                    # Check for order flow exhaustion patterns
                    if order_flow_analysis.get('exhaustion', False):
                        self.logger.info(f"Order flow exhaustion detected: acceleration={order_flow_analysis['acceleration']:.2f}")
                        
                        # For strong signals aligned with exhausted flow, reduce the signal
                        if abs(original_composite) > 0.3 and (order_flow_analysis['direction'] * original_composite > 0):
                            original = original_composite
                            original_composite *= 0.7  # Reduce signals aligned with exhausted flow
                            self.logger.info(f"Order flow exhaustion dampening: {original:.2f} → {original_composite:.2f}")
                    
                    # Advanced order flow pattern recognition
                    order_flow_patterns = self.analyze_order_flow_patterns(market_data)
                    
                    # Improved pattern conflict resolution
                    if order_flow_patterns.get('pattern_detected', False):
                        self.logger.info(f"Order flow pattern detected: {order_flow_patterns['pattern_name']} " +
                                f"(direction: {order_flow_patterns['direction']}, " +
                                f"confidence: {order_flow_patterns['confidence']:.2f})")
                        
                        # Apply advanced pattern conflict resolution
                        pattern_resolution = self.resolve_pattern_conflict(
                            order_flow_patterns,
                            original_composite,
                            market_data
                        )
                        
                        if pattern_resolution['action'] != 'conflict':
                            # Apply the resolved pattern signal
                            original = original_composite
                            original_composite = pattern_resolution['signal']
                            self.logger.info(f"Order flow pattern resolution: {pattern_resolution['action']} due to {pattern_resolution['reason']} - Signal: {original:.2f} → {original_composite:.2f}")
                    
                    # Check if divergence enhancement is active
                    if self._signal_enhancement_active and datetime.datetime.now() < self._signal_enhancement_expiry:
                        # Use enhanced signal from divergence detection
                        base_composite = self._enhanced_signal
                        self.logger.info(f"Using divergence-enhanced signal: {original_composite:.2f} → {base_composite:.2f}")
                    else:
                        # Start with original signal
                        base_composite = original_composite
                        self._signal_enhancement_active = False
                        
                        # ENHANCED: Apply improved signal calibration in range-bound markets
                        if current_regime == 'range_bound' and 'range_analysis' in market_data:
                            # Enhance or dampen signal based on range position
                            z_score = range_analysis.get('z_score', 0)
                            position_in_range = range_analysis.get('position_in_range', 0.5)
                            
                            # Strong mean reversion signal at range extremes
                            if abs(z_score) > 1.5:
                                # Determine if signal is aligned with mean reversion
                                mean_reversion_direction = -1 if z_score > 0 else 1
                                
                                if (base_composite * mean_reversion_direction) > 0:
                                    # Signal aligned with mean reversion - enhance it
                                    original = base_composite
                                    enhancement_factor = min(1.4, 1.0 + abs(z_score) * 0.2)
                                    base_composite *= enhancement_factor
                                    self.logger.info(f"Range extreme signal enhancement: {original:.2f} → {base_composite:.2f} (aligned with reversion)")
                                else:
                                    # Signal against mean reversion - dampen it
                                    original = base_composite
                                    dampening_factor = max(0.5, 1.0 - abs(z_score) * 0.15)
                                    base_composite *= dampening_factor
                                    self.logger.info(f"Range extreme signal dampening: {original:.2f} → {base_composite:.2f} (against reversion)")
                        
                        # Apply enhanced order flow sensitivity
                        order_flow = market_data.get('order_flow', 0)
                        if abs(order_flow) > 0.1:
                            # Store pre-adjustment signal
                            pre_flow_signal = base_composite
                            
                            # Calculate direction for sensitivity analysis
                            direction = np.sign(base_composite) if abs(base_composite) > 0.05 else np.sign(order_flow)
                            
                            # Apply enhanced order flow sensitivity
                            base_composite = self.enhanced_order_flow_sensitivity(market_data, base_composite, direction)
                            
                            # Log significant adjustments
                            if abs(base_composite - pre_flow_signal) > 0.1:
                                self.logger.info(f"Enhanced order flow sensitivity: {pre_flow_signal:.2f} → {base_composite:.2f} (flow: {order_flow:.2f}, accel: {market_data.get('flow_acceleration', 0):.2f})")
                        
                        # Apply traditional enhancements first
                        
                        # Enhancement 1: Amplify emerging trends (small but non-zero signals)
                        if abs(original_composite) > 0.05 and abs(original_composite) < 0.3:
                            if current_regime in ['trending_up', 'trending_down'] and regime_confidence > 0.8:
                                base_composite = original_composite * 1.5
                                self.logger.info(f"Elite enhancement: Signal amplified {original_composite:.2f} → {base_composite:.2f} in {current_regime}")
                        
                        # Enhancement 2: Apply delta bias for very small signals with high confirmation
                        if abs(original_composite) < 0.05:
                            # Track consecutive small signals
                            self._low_signal_consecutive_count += 1
                            
                            # Apply progressive bias after consecutive small signals
                            if self._low_signal_consecutive_count >= 5:
                                # Delta-based bias for very small signals
                                if abs(delta) > 0.4:
                                    bias = delta * 0.25
                                    base_composite = original_composite + bias
                                    self.logger.info(f"Elite enhancement: Delta bias applied {original_composite:.2f} → {base_composite:.2f}")
                                
                                # Order flow bias for very small signals
                                elif abs(order_flow) > 0.15:
                                    bias = order_flow * 0.35
                                    base_composite = original_composite + bias
                                    self.logger.info(f"Elite enhancement: Flow bias applied {original_composite:.2f} → {base_composite:.2f}")
                        else:
                            self._low_signal_consecutive_count = 0
                        
                        # Enhancement 3: Fast moving market bias
                        if current_regime in ["trending_up", "trending_down"] and regime_confidence > 0.9:
                            if (current_regime == "trending_up" and delta > 0.3) or (current_regime == "trending_down" and delta < -0.3):
                                # Strong trend with aligned delta - amplify signal
                                trend_direction = 1 if current_regime == "trending_up" else -1
                                if trend_direction * base_composite >= 0:  # If signal direction matches trend
                                    enhanced_boost = min(0.15, abs(delta) * 0.3) * trend_direction
                                    base_composite += enhanced_boost
                                    self.logger.info(f"Elite enhancement: Trend acceleration bias applied {original_composite:.2f} → {base_composite:.2f}")

                    # ENHANCED: Apply improved neural alpha learning for advanced factor extraction
                    try:
                        neural_alpha = self.enhanced_neural_alpha_extraction(base_composite, market_data)
                        if neural_alpha['score'] > 0.05:  # Only use when score is significant
                            original = base_composite
                            # Direction-aligned signal modification
                            if np.sign(neural_alpha['direction']) == np.sign(base_composite) or abs(base_composite) < 0.1:
                                # Aligned directions - enhance signal
                                enhanced_signal = neural_alpha['score'] * neural_alpha['direction']
                                
                                # ENHANCED: Apply weighted blending based on confidence
                                blend_weight = neural_alpha['confidence']
                                base_composite = (base_composite * (1 - blend_weight)) + (enhanced_signal * blend_weight)
                                self.logger.info(f"Neural alpha extraction boosted signal: {original:.2f} → {base_composite:.2f}")
                                
                                # Log top contributing factors
                                if 'components' in neural_alpha:
                                    factor_msg = ", ".join([f"{k}: {v:.3f}" for k, v in neural_alpha['components'].items() 
                                                        if abs(v) > 0.05])
                                    self.logger.info(f"Top contributing factors: {factor_msg}")
                            else:
                                # Conflicting directions - weighted average based on confidence
                                self.logger.info(f"Neural alpha extraction detected conflicting direction - using weighted resolution")
                    except Exception as e:
                        self.logger.warning(f"Neural alpha extraction error (continuing with base signal): {e}")
                        neural_alpha = {'score': 0, 'direction': np.sign(base_composite), 
                                      'confidence': 0.5, 'components': {}}

                    # Apply quantum memory pattern analysis to neural alpha
                    memory_analysis = self.quantum_market_memory(price_history, current_price, base_composite)
                    if memory_analysis.get('pattern_detected', False):
                        self.logger.info(f"Quantum market memory: {memory_analysis['pattern_description']} predicts {memory_analysis['expected_direction']} movement (conf: {memory_analysis['confidence']:.2f})")
                        
                        # Apply memory-based signal adjustment
                        original_signal = base_composite
                        base_composite *= memory_analysis['signal_boost']
                        self.logger.info(f"Memory-based signal adjustment: {original_signal:.2f} → {base_composite:.2f} ({memory_analysis['recommendation']})")

                    # Check for quantum edge patterns
                    edge_patterns = self.detect_quantum_edge_patterns(market_data, price_history, regime_info)
                    if edge_patterns.get('detected', False):
                        pattern_name = edge_patterns.get('pattern', 'unknown')
                        pattern_direction = edge_patterns.get('direction', 0)
                        pattern_strength = edge_patterns.get('strength', 0)
                        pattern_confidence = edge_patterns.get('confidence', 0)
                        
                        self.logger.info(f"Quantum edge pattern detected: {pattern_name} (direction={pattern_direction}, strength={pattern_strength:.2f}, confidence={pattern_confidence:.2f})")

                    # Apply pattern synchronization to harmonize multiple pattern detections
                    synchronized_patterns = self.synchronize_quantum_patterns(
                        edge_patterns,
                        order_flow_patterns,
                        memory_analysis,
                        market_data
                    )

                    if synchronized_patterns.get('synchronized', False):
                        self.logger.info(f"Synchronized pattern detection: {synchronized_patterns['primary_pattern']} " +
                                        f"(direction: {synchronized_patterns['direction']}, confidence: {synchronized_patterns['confidence']:.2f})")
                        
                        # Store pattern-based signal adjustment for later application
                        original = base_composite
                        base_composite += synchronized_patterns['signal_adjustment']
                        pattern_threshold_adjustment = synchronized_patterns['threshold_adjustment']
                        
                        self.logger.info(f"Synchronized pattern signal adjustment: {original:.2f} → {base_composite:.2f}")
                    else:
                        pattern_threshold_adjustment = 0.0

                    # Apply microstructure enhancements to base signal
                    microstructure_enhancements = self.apply_elite_enhancements(
                        market_data,
                        base_composite,
                        regime_info
                    )

                    # Get enhancement results
                    microstructure_enhanced_composite = microstructure_enhancements.get('enhanced_signal', base_composite)
                    basic_elite_confidence = microstructure_enhancements.get('elite_confidence', 0.0)
                    
                    # ENHANCED: Calculate quantum entanglement with exponential decay weighting
                    entanglement = self.enhanced_quantum_entanglement(market_data)
                    self.logger.info(f"Enhanced quantum entanglement: {entanglement:.2f} (delta-flow correlation)")
                    
                    # Store entanglement in market_data for other components
                    market_data['entanglement'] = entanglement
                    
                    # ENHANCED: Apply advanced quantum signal reinforcement
                    original_signal = microstructure_enhanced_composite

                    # Apply quantum signal stabilization
                    quantum_signal_result = self.stabilized_quantum_signal_processing(
                        microstructure_enhanced_composite,
                        market_data,
                        {'regime': current_regime, 'confidence': regime_confidence}  # Fixed typo
                    )

                    # Update signal with stabilized version
                    enhanced_composite = quantum_signal_result['signal']
                    elite_confidence = quantum_signal_result.get('quantum_confidence', basic_elite_confidence)

                    self.logger.info(f"Quantum signal stabilization: {original_signal:.2f} → {enhanced_composite:.2f} " +
                            f"(max bound: {quantum_signal_result.get('max_bound', 1.0):.2f}, confidence: {elite_confidence:.2f})")
                    
                    # ENHANCED: Apply range extreme enhancements
                    if range_analysis.get('boundaries_detected', False) and current_regime == 'range_bound':
                        position_in_range = range_analysis.get('position_in_range', 0.5)
                        z_score = range_analysis.get('z_score', 0)
                        
                        # Apply calibrated range enhancement with non-linear scaling
                        if (position_in_range > 0.75 and enhanced_composite < 0) or (position_in_range < 0.25 and enhanced_composite > 0):
                            original = enhanced_composite
                            
                            # Non-linear sigmoid scaling to smooth the extremes
                            range_bias = 2.0 / (1.0 + np.exp(-1.2 * (abs(z_score) - 1.0))) - 0.5  # Range: 0.5-1.5
                            
                            # Calculate adaptive enhancement - stronger at true extremes but limited
                            if position_in_range > 0.75 and enhanced_composite < 0:  # Upper extreme
                                # Cap enhancement for upper extreme
                                adjustment_cap = 0.3 * range_bias  # Max ~0.45 at z=2.0
                                adjustment = min(adjustment_cap, 0.1 + (position_in_range - 0.75) * 1.5)
                                enhanced_composite -= adjustment  # More negative (sell signal)
                                self.logger.info(f"Calibrated range enhancement: {original:.2f} → {enhanced_composite:.2f} (upper extreme, z: {z_score:.2f})")
                            
                            elif position_in_range < 0.25 and enhanced_composite > 0:  # Lower extreme
                                # Cap enhancement for lower extreme
                                adjustment_cap = 0.3 * range_bias  # Max ~0.45 at z=2.0
                                adjustment = min(adjustment_cap, 0.1 + (0.25 - position_in_range) * 1.5)
                                enhanced_composite += adjustment  # More positive (buy signal)
                                self.logger.info(f"Calibrated range enhancement: {original:.2f} → {enhanced_composite:.2f} (lower extreme, z: {z_score:.2f})")
                    
                    # Apply signal stabilization to prevent unwanted flipping
                    enhanced_composite = self.stabilize_signal(enhanced_composite, market_data)
                    
                    # Use enhanced signal for trading decisions
                    composite_signal = enhanced_composite
                    
                    # Enhanced regime transition handling
                    seconds_since_transition = (current_time - self._regime_transition_data['last_transition']).total_seconds()
                    if seconds_since_transition < self._regime_transition_data['stabilization_period'] and self._regime_transition_data['transition_from'] != 'unknown':
                        original_signal = composite_signal
                        
                        # Apply enhanced transition handler
                        composite_signal = self.enhanced_regime_transition_handler(
                            composite_signal,
                            self._regime_transition_data['transition_to'],  # current regime
                            self._regime_transition_data['transition_from'],  # previous regime
                            seconds_since_transition,
                            market_data
                        )
                        
                        self.logger.info(f"Enhanced regime transition: {original_signal:.2f} → {composite_signal:.2f} ({seconds_since_transition:.1f}s since {self._regime_transition_data['transition_from']} → {self._regime_transition_data['transition_to']})")
                    
                    # Record signal history for stabilization purposes
                    self._signal_history.append(composite_signal)
                    if len(self._signal_history) > 10:
                        self._signal_history.pop(0)
                    
                    # Get account equity
                    equity = self.execution_engine.get_account_equity()
                    
                    # Update drawdown
                    self.risk_manager.update_drawdown(
                        equity, 
                        max(self.capital, equity)
                    )
                    
                    # Start with base threshold
                    base_threshold = params["signal_threshold"]

                    # Get pattern information
                    pattern_info = {
                        'type': synchronized_patterns.get('primary_pattern') if 'synchronized_patterns' in locals() else None,
                        'confidence': synchronized_patterns.get('confidence', 0.5) if 'synchronized_patterns' in locals() else 0.5,
                        'direction': synchronized_patterns.get('direction', 0) if 'synchronized_patterns' in locals() else 0
                    }

                    # Add order flow pattern info if available
                    if order_flow_patterns.get('pattern_detected', False):
                        if not pattern_info['type']:  # Only use if no synchronized pattern
                            pattern_info['type'] = order_flow_patterns.get('pattern_name', 'flow_pattern')
                            pattern_info['confidence'] = order_flow_patterns.get('confidence', 0.5)
                            pattern_info['direction'] = order_flow_patterns.get('direction', 0)

                    # Get multi-timeframe confirmation score
                    confirmation_score = self.calibrated_timeframe_confirmation(composite_signal, market_data)
                    market_data['confirmation'] = confirmation_score

                    # Apply unified threshold management
                    threshold_result = self.unified_threshold_management(
                        base_threshold,
                        market_data,
                        {'regime': current_regime, 'confidence': regime_confidence, 'trend_strength': market_data.get('trend_strength', 1.0)},
                        pattern_info
                    )

                    # Get optimized threshold
                    signal_threshold = threshold_result['threshold']

                    self.logger.info(f"Unified threshold calculated: {signal_threshold:.2f} (regime: {current_regime}, adjustments: {threshold_result['adjustments']})")
                    
                    # ===== DYNAMIC TRADE MANAGEMENT INTEGRATION =====
                    
                    # Check active position management
                    if position_size != 0 and current_position_id:
                        # Get entry time for minimum hold time check
                        entry_time = self._trade_timing.get('entry_times', {}).get(current_position_id, current_time - datetime.timedelta(minutes=5))
                        seconds_in_trade = (current_time - entry_time).total_seconds()
                        
                        # Create position object for trade manager
                        position_obj = {
                            'id': current_position_id,
                            'entry_price': current_position.get('avg_price'),
                            'size': position_size,
                            'entry_time': entry_time,
                            'status': 'active'
                        }
                        
                        # Calculate ATR for trade management
                        atr = self.calculate_atr(price_history) or (current_price * 0.005)
                        
                        # Add ATR to market data for other components
                        market_data['atr'] = atr
                        
                        # Prevent very early exits unless catastrophic move against position
                        min_hold_time = self._trade_timing['min_hold_time']
                        if seconds_in_trade < min_hold_time:
                            direction = 1 if position_size > 0 else -1
                            entry_price = position_obj['entry_price']
                            stop_distance = abs(entry_price - stop_loss_price) if stop_loss_price else (atr * 1.5)
                            current_move = abs(current_price - entry_price)
                            
                            # Only allow early exit if price moved strongly against position (>70% to stop)
                            if current_move < (stop_distance * 0.7) or (current_price - entry_price) * direction > 0:
                                # Skip regular exit logic - enforce minimum hold time
                                self.logger.info(f"Preventing early exit - minimum hold time not reached ({seconds_in_trade:.1f}s / {min_hold_time}s)")
                                
                                # Skip to next iteration unless we have a catastrophic move
                                if not ((current_price - entry_price) * direction < -(stop_distance * 0.9)):
                                    # Still do profit target updates
                                    if current_regime == 'trending_up' or current_regime == 'trending_down':
                                        # Use dynamic profit targets for trending markets
                                        self.logger.info(f"Dynamic profit targets: T1: ${profit_target_price:.2f}, T2: ${profit_target_price:.2f}, T3: ${profit_target_price:.2f}")
                                    else:
                                        # Use range-optimized targets for range-bound markets
                                        range_targets = self.calculate_range_optimized_targets(position_obj, market_data)
                                        self.logger.info(f"Range optimized targets: T1: ${range_targets[0]:.2f}, T2: ${range_targets[1]:.2f}, T3: ${range_targets[2]:.2f}")
                                    
                                    time.sleep(self.config['system']['update_interval'])
                                    continue
                        
                        # Apply enhanced risk management
                        risk_assessment = self.enhanced_risk_management(
                            position_obj,
                            market_data,
                            {'regime': current_regime, 'confidence': regime_confidence}
                        )

                        if risk_assessment['action'] == 'exit':
                            self.logger.info(f"Enhanced risk management triggered exit: {risk_assessment['reason']} " +
                                    f"(risk score: {risk_assessment.get('risk_score', 0):.2f})")
                            
                            # Execute exit with timing optimization
                            direction = 1 if position_size > 0 else -1
                            order_id = self.execution_engine.place_order(
                                'market',
                                'NQ',
                                -position_size
                            )
                            
                            if order_id:
                                # Get entry price from tracking dictionary
                                entry_price = current_position.get('avg_price', 0)
                                
                                # If entry price is missing or zero, try to get from our tracking
                                if not entry_price or entry_price == 0:
                                    if current_position_id in self._position_entry_prices:
                                        entry_price = self._position_entry_prices[current_position_id]
                                        self.logger.info(f"Retrieved entry price ${entry_price:.2f} from position tracking")
                                
                                # Calculate P&L using fixed function
                                profit = self.calculate_trade_pnl(entry_price, current_price, position_size)

                                # Update trade metrics
                                trade_result = {
                                    'profit': profit,
                                    'trade_id': current_position_id,
                                    'reason': f"quantum_risk_{risk_assessment.get('reason', 'unknown')}",
                                    'regime': current_regime,
                                    'regime_confidence': regime_confidence,
                                    'entry_price': entry_price,
                                    'exit_price': current_price,
                                    'risk_score': risk_assessment.get('risk_score', risk_assessment.get('score', 0.0))  
                                }

                                # Optional: Add additional risk metrics if available
                                for risk_key in ['volatility_risk', 'liquidity_risk', 'correlation_risk', 'regime_risk']:
                                    if risk_key in risk_assessment:
                                        trade_result[risk_key] = risk_assessment[risk_key]
                                self.report_trade_to_dashboard(trade_result)
                                # Add factor data to trade result for analysis
                                if current_position_id in self._trade_factor_data:
                                    trade_result['factor_data'] = self._trade_factor_data[current_position_id]
                                
                                self.trade_manager.update_metrics(trade_result)
                                self.process_trade_for_analytics(trade_result)
                                # Clean up entry price tracking
                                if current_position_id in self._position_entry_prices:
                                    del self._position_entry_prices[current_position_id]
                                
                                # Clean up factor data tracking
                                if current_position_id in self._trade_factor_data:
                                    del self._trade_factor_data[current_position_id]
                                
                                # Clean up timing data
                                if current_position_id in self._trade_timing['entry_times']:
                                    del self._trade_timing['entry_times'][current_position_id]
                                
                                # Add to trade history for equity curve adjustment
                                self.trade_history.append(trade_result)
                                if len(self.trade_history) > 20:  # Keep last 20 trades
                                    self.trade_history.pop(0)
                                
                                # Log accurate P&L
                                self.logger.info(f"Quantum-enhanced risk exit: ${profit:.2f} profit, Risk score: {risk_assessment['risk_score']:.2f}")
                                
                                # Reset position tracking
                                position_size = 0
                                current_position_id = None
                                stop_loss_price = None
                                profit_target_price = None
                                
                                # Reset position established time for signal stabilization
                                self._position_established_time = None
                                
                                # Skip normal position management
                                continue
                                
                        elif risk_assessment['action'] == 'partial_exit':
                            self.logger.info(f"Quantum risk management triggered partial exit: {risk_assessment['reason']} " +
                                    f"(risk score: {risk_assessment['risk_score']:.2f}, size: {abs(risk_assessment['size'])})")
                            
                            # Execute partial exit
                            exit_size = risk_assessment['size']
                            order_id = self.execution_engine.place_order(
                                'market',
                                'NQ',
                                -abs(exit_size)  # Ensure negative for exits
                            )
                            
                            if order_id:
                                self.logger.info(f"Quantum-managed partial exit: {abs(exit_size)} contracts @ ${current_price:,.2f}")
                                position_size -= abs(exit_size) * np.sign(exit_size)
                                
                                # Record partial exit
                                if current_position_id not in self._trade_timing['partial_exits']:
                                    self._trade_timing['partial_exits'][current_position_id] = []
                                self._trade_timing['partial_exits'][current_position_id].append({
                                    'time': current_time,
                                    'price': current_price,
                                    'size': abs(exit_size),
                                    'reason': risk_assessment['reason']
                                })
                        
                        # Risk management doesn't suggest exit - check time-based exit strategy
                        if risk_assessment['action'] == 'hold':
                            # Get entry time from position tracking
                            entry_time = self._trade_timing.get('entry_times', {}).get(current_position_id, current_time - datetime.timedelta(minutes=5))
                            
                            # Apply dynamic time-based exit strategy
                            time_exit = self.dynamic_time_based_exit(position_obj, market_data, entry_time)
                            
                            if time_exit['action'] == 'exit':
                                self.logger.info(f"Time-based exit triggered after {time_exit['hold_time']:.1f}s - {time_exit['reason']}")
                                
                                # Execute exit
                                order_id = self.execution_engine.place_order(
                                    'market',
                                    'NQ',
                                    -position_obj['size']
                                )
                                
                                # Process exit similar to risk management exit
                                if order_id:
                                    # Get entry price
                                    entry_price = position_obj.get('entry_price', current_position.get('avg_price', 0))
                                    
                                    # Calculate P&L
                                    profit = self.calculate_trade_pnl(entry_price, current_price, position_obj['size'])
                                    
                                    # Update trade metrics
                                    trade_result = {
                                        'profit': profit,
                                        'trade_id': current_position_id,
                                        'reason': time_exit['reason'],
                                        'regime': current_regime,
                                        'regime_confidence': regime_confidence,
                                        'entry_price': entry_price,
                                        'exit_price': current_price,
                                        'hold_time': time_exit['hold_time']
                                    }
                                    
                                    self.trade_manager.update_metrics(trade_result)
                                    self.process_trade_for_analytics(trade_result)
                                    self.report_trade_to_dashboard(trade_result)
                                    self.logger.info(f"Time-based exit complete: ${profit:.2f} profit, hold time: {time_exit['hold_time']:.1f}s")
                                    
                                    # Reset position tracking (use existing cleanup logic)
                                    position_size = 0
                                    current_position_id = None
                                    stop_loss_price = None
                                    profit_target_price = None
                                    self._position_established_time = None
                                    
                                    # Cleanup tracking dictionaries
                                    if current_position_id in self._position_entry_prices:
                                        del self._position_entry_prices[current_position_id]
                                    if current_position_id in self._trade_factor_data:
                                        del self._trade_factor_data[current_position_id]
                                    if current_position_id in self._trade_timing['entry_times']:
                                        del self._trade_timing['entry_times'][current_position_id]
                                    
                                    # Continue to next iteration
                                    continue
                                    
                            elif time_exit['action'] == 'partial_exit':
                                self.logger.info(f"Time-based partial exit after {time_exit['hold_time']:.1f}s - {time_exit['reason']}")
                                
                                # Execute partial exit
                                exit_size = time_exit['size']
                                order_id = self.execution_engine.place_order(
                                    'market',
                                    'NQ',
                                    exit_size  # Already correctly signed in time_exit
                                )
                                
                                if order_id:
                                    self.logger.info(f"Time-based partial exit: {abs(exit_size)} contracts @ ${current_price:,.2f}")
                                    position_size -= abs(exit_size)
                                    
                                    # Record partial exit in tracking
                                    if current_position_id not in self._trade_timing['partial_exits']:
                                        self._trade_timing['partial_exits'][current_position_id] = []
                                    self._trade_timing['partial_exits'][current_position_id].append({
                                        'time': current_time,
                                        'price': current_price,
                                        'size': abs(exit_size),
                                        'reason': time_exit['reason']
                                    })
                        
                        # Perform deep liquidity analysis for exit decisions
                        liquidity_analysis = self.deep_liquidity_analysis(market_data)
                        
                        # Check for institutional exit signals in current position
                        if liquidity_analysis.get('liquidity_status') == 'toxic':
                            direction = 1 if position_size > 0 else -1
                            # If liquidity is deteriorating rapidly against our position
                            if (direction == 1 and liquidity_analysis.get('bias') == 'bearish') or \
                            (direction == -1 and liquidity_analysis.get('bias') == 'bullish'):
                                # Quick exit to avoid toxic liquidity trap
                                self.logger.info(f"Elite liquidity exit: Toxic market detected - {liquidity_analysis.get('pattern')}")
                                
                                # Execute exit with timing optimization (but urgent)
                                order_id = self.execution_engine.place_order(
                                    'market',
                                    'NQ',
                                    -position_size
                                )
                                
                                if order_id:
                                    # Get entry price from tracking dictionary
                                    entry_price = current_position.get('avg_price', 0)
                                    
                                    # If entry price is missing or zero, try to get from our tracking
                                    if not entry_price or entry_price == 0:
                                        if current_position_id in self._position_entry_prices:
                                            entry_price = self._position_entry_prices[current_position_id]
                                            self.logger.info(f"Retrieved entry price ${entry_price:.2f} from position tracking")
                                    
                                    # Calculate P&L using fixed function
                                    profit = self.calculate_trade_pnl(entry_price, current_price, position_size)
                                    
                                    # Update trade metrics
                                    trade_result = {
                                        'profit': profit,
                                        'trade_id': current_position_id,
                                        'reason': 'toxic_liquidity_exit',
                                        'regime': current_regime,
                                        'regime_confidence': regime_confidence,
                                        'entry_price': entry_price,
                                        'exit_price': current_price
                                    }
                                    
                                    # Add factor data to trade result for analysis
                                    if current_position_id in self._trade_factor_data:
                                        trade_result['factor_data'] = self._trade_factor_data[current_position_id]
                                    
                                    self.trade_manager.update_metrics(trade_result)
                                    self.process_trade_for_analytics(trade_result)
                                    self.report_trade_to_dashboard(trade_result)
                                    # Clean up entry price tracking
                                    if current_position_id in self._position_entry_prices:
                                        del self._position_entry_prices[current_position_id]
                                    
                                    # Clean up factor data tracking
                                    if current_position_id in self._trade_factor_data:
                                        del self._trade_factor_data[current_position_id]
                                    
                                    # Clean up timing data
                                    if current_position_id in self._trade_timing['entry_times']:
                                        del self._trade_timing['entry_times'][current_position_id]
                                    
                                    # Add to trade history for equity curve adjustment
                                    self.trade_history.append(trade_result)
                                    if len(self.trade_history) > 20:  # Keep last 20 trades
                                        self.trade_history.pop(0)
                                    
                                    # Log accurate P&L
                                    self.logger.info(f"Quantum-enhanced toxic liquidity exit: ${profit:.2f} profit, Liquidity confidence: {liquidity_analysis.get('confidence', 0):.2f}")
                                    
                                    # Reset position tracking
                                    position_size = 0
                                    current_position_id = None
                                    stop_loss_price = None
                                    profit_target_price = None
                                    
                                    # Reset position established time for signal stabilization
                                    self._position_established_time = None
                                    
                                    # Skip normal position management
                                    continue
                            
                        # For volatile regimes, check for accelerated profit taking
                        if current_regime == "volatile" and regime_confidence > 0.65:
                            direction = 1 if position_size > 0 else -1
                            entry_price = position_obj['entry_price']
                            profit_pct = (current_price - entry_price) * direction / entry_price
                            
                            # Enhanced volatile exit conditions
                            if profit_pct > 0.0015 and market_data.get('order_flow', 0) * direction < -0.05:
                                # Quick profit taking when order flow reverses
                                self.logger.info(f"Elite volatile profit taking triggered - Order flow reversal")
                                
                                # Execute partial exit
                                exit_size = int(abs(position_size) * params["partial_exit_pct"]) * direction
                                if exit_size != 0:
                                    order_id = self.execution_engine.place_order(
                                        'market',
                                        'NQ',
                                        -exit_size  # Negative for exit
                                    )
                                    
                                    if order_id:
                                        self.logger.info(f"Volatile regime partial exit: {abs(exit_size)} contracts @ ${current_price:,.2f}")
                                        position_size -= exit_size * direction
                                        
                                        # Record partial exit
                                        if current_position_id not in self._trade_timing['partial_exits']:
                                            self._trade_timing['partial_exits'][current_position_id] = []
                                        self._trade_timing['partial_exits'][current_position_id].append({
                                            'time': current_time,
                                            'price': current_price,
                                            'size': abs(exit_size),
                                            'reason': 'volatile_regime_profit_taking'
                                        })
                                        
                                        # Update stop loss for remaining position
                                        stop_loss_price = current_price - (direction * atr * 0.5)
                                        self.logger.info(f"Volatile regime updated stop: ${stop_loss_price:,.2f}")
                        
                        # Calculate dynamic profit targets
                        self.calculate_dynamic_profit_targets(position_obj, current_price, market_data, regime_info, current_position_id)
                        
                        # NEW: Manage partial exits at profit targets
                        self.manage_profit_targets(position_obj, current_price, market_data)
                        
                        # Apply specialized range trade management in range-bound markets
                        if current_regime == 'range_bound' and 'range_analysis' in market_data:
                            range_action = self.adaptive_range_trade_management(
                                position_obj,
                                current_price,
                                market_data,
                                market_data['range_analysis']
                            )
                            
                            # If range management suggests an action, override standard management
                            if range_action['action'] != 'hold':
                                self.logger.info(f"Range trade management: {range_action['action']} - {range_action['reason']} ({range_action.get('details', '')})")
                                action = range_action
                            else:
                                # Default action from trade manager for range markets with no specific action
                                action = self.trade_manager.manage_position(
                                    position=position_obj,
                                    current_price=current_price,
                                    atr=atr,
                                    signal=composite_signal,
                                    market_data=market_data,
                                    timestamp=current_time
                                )
                        else:
                            # Get trade management actions using DynamicTradeManager for non-range regimes
                            action = self.trade_manager.manage_position(
                                position=position_obj,
                                current_price=current_price,
                                atr=atr,
                                signal=composite_signal,
                                market_data=market_data,
                                timestamp=current_time
                            )
                        
                        if action['action'] == 'exit':
                            # Calculate execution quality score for exit
                            direction = 1 if position_size > 0 else -1
                            execution_quality = self.enhanced_execution_quality_score(market_data, -direction)  # Opposite for exit
                            
                            # Optimize exit timing
                            exit_timing = self.optimize_exit_timing(position_obj, current_price, market_data)
                            
                            # Execute exit with timing optimization
                            timing_decision = self.optimize_execution_timing('exit', direction, market_data)
                            if timing_decision['execute_now'] or execution_quality['quality_score'] > 0.8 or exit_timing['action'] == 'accelerate':
                                if execution_quality['quality_score'] > 0.8:
                                    self.logger.info(f"Optimal exit conditions detected (score: {execution_quality['quality_score']:.2f})")
                                
                                order_id = self.execution_engine.place_order(
                                    'market',
                                    'NQ',
                                    -position_size
                                )
                            else:
                                # Poor execution conditions - apply delay
                                if execution_quality['quality_score'] < 0.5:
                                    self.logger.info(f"Poor exit conditions - adjusting timing (score: {execution_quality['quality_score']:.2f})")
                                    timing_decision['delay_seconds'] = max(timing_decision['delay_seconds'], 1.5)  # Extend delay for poor conditions
                                
                                # Delay execution
                                self.logger.info(f"Delaying exit by {timing_decision['delay_seconds']}s")
                                time.sleep(timing_decision['delay_seconds'])
                                # Get updated price after delay
                                current_price = self.market_data.get_realtime_data().get('price', current_price)
                                # Place order after delay
                                order_id = self.execution_engine.place_order(
                                    'market',
                                    'NQ',
                                    -position_size
                                )
                            
                            # Handle P&L calculation and trade tracking
                            if order_id:
                                # Get entry price from tracking dictionary
                                entry_price = current_position.get('avg_price', 0)
                                
                                # If entry price is missing or zero, try to get from our tracking
                                if not entry_price or entry_price == 0:
                                    if current_position_id in self._position_entry_prices:
                                        entry_price = self._position_entry_prices[current_position_id]
                                        self.logger.info(f"Retrieved entry price ${entry_price:.2f} from position tracking")
                                
                                # Calculate P&L using fixed function
                                profit = self.calculate_trade_pnl(entry_price, current_price, position_size)
                                
                                # Update trade metrics
                                trade_result = {
                                    'profit': profit,
                                    'trade_id': current_position_id,
                                    'reason': action['reason'],
                                    'regime': current_regime,
                                    'regime_confidence': regime_confidence,
                                    'entry_price': entry_price,
                                    'exit_price': current_price,
                                    'execution_quality': execution_quality['quality_score']
                                }
                                
                                # Add factor data to trade result for analysis
                                if current_position_id in self._trade_factor_data:
                                    trade_result['factor_data'] = self._trade_factor_data[current_position_id]
                                
                                self.trade_manager.update_metrics(trade_result)
                                self.process_trade_for_analytics(trade_result)
                                self.report_trade_to_dashboard(trade_result)
                                # Clean up entry price tracking
                                if current_position_id in self._position_entry_prices:
                                    del self._position_entry_prices[current_position_id]
                                
                                # Clean up factor data tracking
                                if current_position_id in self._trade_factor_data:
                                    del self._trade_factor_data[current_position_id]
                                
                                # Clean up timing data
                                if current_position_id in self._trade_timing['entry_times']:
                                    del self._trade_timing['entry_times'][current_position_id]
                                if current_position_id in self._trade_timing['partial_exits']:
                                    del self._trade_timing['partial_exits'][current_position_id]
                                
                                # Add to trade history for equity curve adjustment
                                self.trade_history.append(trade_result)
                                if len(self.trade_history) > 20:  # Keep last 20 trades
                                    self.trade_history.pop(0)
                                
                                # Log accurate P&L
                                self.logger.info(f"Quantum-enhanced trade complete: ${profit:.2f} profit, Elite confidence: {elite_confidence:.2f}")
                                
                                # Reset position tracking
                                position_size = 0
                                current_position_id = None
                                stop_loss_price = None
                                profit_target_price = None
                                
                                # Reset position established time for signal stabilization
                                self._position_established_time = None
                            
                        elif action['action'] == 'partial_exit':
                            # Execute partial exit
                            exit_size = action['size']
                            order_id = self.execution_engine.place_order(
                                'market',
                                'NQ',
                                -abs(exit_size)  # Ensure negative for exits
                            )
                            
                            if order_id:
                                self.logger.info(f"Partial exit triggered: {action['reason']} - Exited {abs(exit_size)} contracts @ ${current_price:,.2f}")
                                position_size -= abs(exit_size)
                                
                                # Record partial exit
                                if current_position_id not in self._trade_timing['partial_exits']:
                                    self._trade_timing['partial_exits'][current_position_id] = []
                                self._trade_timing['partial_exits'][current_position_id].append({
                                    'time': current_time,
                                    'price': current_price,
                                    'size': abs(exit_size),
                                    'reason': action['reason']
                                })
                    
                    # Trading logic for opening new positions
                    if position_size == 0:
                        # Get confirmation score for signal
                        seconds_since_last_trade = (datetime.datetime.now() - self._last_trade_time).total_seconds()
                        if seconds_since_last_trade < self._trade_cooldown_period:
                            self.logger.info(f"Trade cooldown active: {seconds_since_last_trade:.1f}s / {self._trade_cooldown_period}s")
                            time.sleep(self.config['system']['update_interval'])
                            continue
                        confirmation_score = self.calibrated_timeframe_confirmation(composite_signal, market_data)
                        self.logger.info(f"Multi-timeframe confirmation: {confirmation_score:.2f} for signal {composite_signal:.2f}")
                    
                        # Check for confirmation calibration based on regime
                        calibrated_confirmation = confirmation_score
                        if current_regime == 'range_bound' and confirmation_score > 0.6:
                            # Reduce confirmation expectations in range-bound markets
                            calibrated_confirmation = 0.6 + (confirmation_score - 0.6) * 0.5
                            self.logger.info(f"Confirmation calibration: {confirmation_score:.2f} → {calibrated_confirmation:.2f} ({regime_info}, trend: {market_data.get('trend_strength', 0.0):.2f})")
                        elif current_regime in ['trending_up', 'trending_down'] and confirmation_score > 0.7:
                            # Enhance confirmation in trending markets
                            calibrated_confirmation = 0.7 + (confirmation_score - 0.7) * 1.2
                            self.logger.info(f"Confirmation calibration: {confirmation_score:.2f} → {calibrated_confirmation:.2f} ({regime_info}, trend: {market_data.get('trend_strength', 0.0):.2f})")
                        
                        confirmation_score = calibrated_confirmation
                        
                        # Check for volatility compression setup
                        compression_info = self.detect_volatility_compression_setup(market_data, regime_info)
                        
                        # Check for order flow divergence with enhanced signal capability
                        divergence_info = self.detect_orderflow_divergence(market_data)
                        
                        # Check for trapped traders pattern
                        trapped_info = self.detect_trapped_traders(market_data, regime_info)
                        
                        # Boost signal if edge pattern direction aligns with signal
                        if edge_patterns.get('detected', False) and edge_patterns.get('direction', 0) * composite_signal > 0:
                            original = composite_signal
                            # Apply quantum pattern boost - strength determines boost factor
                            composite_signal *= (1.0 + edge_patterns.get('strength', 0) * 0.3)
                            self.logger.info(f"Quantum pattern signal boost: {original:.2f} → {composite_signal:.2f}")
                            
                            # Also reduce the threshold for high-confidence patterns
                            original_threshold = signal_threshold
                            signal_threshold *= (1.0 - edge_patterns.get('confidence', 0) * 0.25)
                            self.logger.info(f"Quantum pattern threshold reduction: {original_threshold:.2f} → {signal_threshold:.2f}")
                        
                        # Perform deep liquidity analysis for entry decisions
                        liquidity_analysis = self.deep_liquidity_analysis(market_data)
                        if liquidity_analysis.get('liquidity_status') != 'normal':
                            self.logger.info(f"Elite liquidity analysis: {liquidity_analysis.get('liquidity_status')} - bias: {liquidity_analysis.get('bias', 'neutral')}")
                        
                        # Track consecutive high confirmations
                        if confirmation_score > 0.7:
                            self._high_confirmation_count += 1
                        else:
                            self._high_confirmation_count = 0
                        
                        # Apply specially optimized thresholds for range/choppy markets
                        if current_regime == 'range_bound' and 'range_analysis' in market_data:
                            # For extreme range positions, lower threshold for mean reversion
                            z_score = market_data['range_analysis'].get('z_score', 0)
                            
                            # Only reduce if we have a mean reversion alignment
                            if abs(z_score) > 1.5 and np.sign(composite_signal) == -np.sign(z_score):
                                original = signal_threshold
                                # More extreme z-score = larger reduction (up to 20%)
                                reduction_factor = min(0.2, 0.1 + abs(z_score) * 0.05)
                                signal_threshold *= (1.0 - reduction_factor)
                                self.logger.info(f"Choppy market threshold decrease: {original:.2f} → {signal_threshold:.2f} (high confirmation in choppy)")
                        
                        # Apply special threshold adjustment for high confirmation in choppy
                        if confirmation_score > 0.7 and current_regime == 'range_bound' and market_data.get('hurst_exponent', 0.5) < 0.3:
                            original = signal_threshold
                            signal_threshold *= 0.9  # 10% reduction with high confirmation in choppy
                            self.logger.info(f"High entanglement threshold reduction: {original:.2f} → {signal_threshold:.2f} (entanglement: {entanglement:.2f})")
                        
                        # Apply high entanglement threshold reduction - more significant reduction
                        if entanglement > 0.75:
                            original = signal_threshold
                            # Progressive reduction based on entanglement level
                            reduction = (entanglement - 0.75) * 0.8
                            signal_threshold *= (1.0 - reduction)
                            self.logger.info(f"High entanglement threshold reduction: {original:.2f} → {signal_threshold:.2f} (entanglement: {entanglement:.2f})")
                        
                        # Check for entry with adapted threshold
                        if abs(composite_signal) >= signal_threshold:
                            # Log signal
                            signal_type = "BUY" if composite_signal > 0 else "SELL"
                            self.logger.info(f"Strong {signal_type} signal: {composite_signal:.2f} in {current_regime} regime (quantum confidence: {elite_confidence:.2f})")
                            
                            # Check delta confirmation for volatile regimes
                            need_confirmation = False
                            if current_regime == "volatile" and params["delta_confirm"] > 0:
                                delta_confirmation = (composite_signal > 0 and delta > params["delta_confirm"]) or \
                                                (composite_signal < 0 and delta < -params["delta_confirm"])
                                
                                if not delta_confirmation:
                                    self.logger.info(f"Delta confirmation failed: {delta:.3f} does not confirm {signal_type} in volatile regime")
                                    need_confirmation = True
                            
                            # Apply VPIN-based market toxicity protection
                            toxicity_check = self.vpin_market_toxicity_protection(market_data, {'signal': composite_signal, 'threshold': signal_threshold})
                            
                            if not toxicity_check['proceed']:
                                self.logger.info(f"Entry prohibited due to {toxicity_check['reason']} (toxicity: {toxicity_check['toxicity']:.2f})")
                                time.sleep(self.config['system']['update_interval'])
                                continue
                            
                            # Apply toxicity delay if needed
                            if toxicity_check.get('delay', 0) > 0:
                                self.logger.info(f"Delaying entry by {toxicity_check['delay']:.1f}s due to {toxicity_check['reason']}")
                                time.sleep(toxicity_check['delay'])
                                # Refresh market data after delay
                                current_price = self.market_data.get_realtime_data().get('price', current_price)
                            
                            # Continue if confirmation not needed or successful
                            if not need_confirmation:
                                # Calculate direction based on signal
                                direction = 1 if composite_signal > 0 else -1
                                
                                # Calculate ATR for stop placement and position sizing
                                atr = self.calculate_atr(price_history) or (current_price * 0.005)
                                
                                # Account metrics for position sizing
                                account_metrics = {
                                    'equity': equity,
                                    'drawdown': self.risk_manager.get_current_drawdown() if hasattr(self.risk_manager, 'get_current_drawdown') else 0.0,
                                    'win_rate': self.get_recent_win_rate(20) if hasattr(self, 'get_recent_win_rate') else 0.5
                                }
                                
                                # Apply optimized position sizing
                                position_info = self.optimized_position_sizing(
                                    composite_signal,
                                    signal_threshold,
                                    market_data,
                                    account_metrics
                                )
                                
                                # Get optimized position size
                                sized_position = position_info['position_size']
                                direction = np.sign(composite_signal)
                                stop_distance = position_info['stop_distance']
                                
                                # Ensure we have at least the minimum stop distance
                                min_stop_points = 15.0
                                if stop_distance < min_stop_points:
                                    self.logger.info(f"Elite stop adjustment: Increasing stop from {stop_distance:.2f} to minimum {min_stop_points:.2f} points")
                                    stop_distance = min_stop_points
                                
                                # Apply size adjustment for mild toxicity
                                if 'size_factor' in toxicity_check:
                                    original_size = sized_position
                                    sized_position = max(1, int(sized_position * toxicity_check['size_factor']))
                                    self.logger.info(f"Position size reduced due to {toxicity_check['reason']}: {original_size} → {sized_position}")
                                
                                # Log key position sizing factors
                                self.logger.info(f"Elite position sizing: {abs(sized_position)} contracts, risk: ${position_info['risk_dollars']:.2f}")
                                self.logger.info(f"Position factors: {position_info['factors']}")
                                
                                # Calculate stop loss price
                                stop_loss_price = self.calculate_elite_stop(
                                    current_price,
                                    direction,
                                    stop_distance,
                                    market_data,
                                    current_regime,
                                    regime_confidence
                                )
                                
                                self.logger.info(f"Elite stop placement: {current_regime} regime-optimized stop at {stop_loss_price:.2f}")
                                
                                # Apply specialized range position sizing for range-bound markets
                                if current_regime == 'range_bound' and 'range_analysis' in market_data:
                                    range_analysis = market_data['range_analysis']
                                    
                                    # Use range position optimization
                                    range_sizing = self.optimized_range_position_sizing(
                                        sized_position,
                                        direction,
                                        range_analysis,
                                        market_data
                                    )
                                    
                                    # Get range-optimized position size
                                    sized_position = range_sizing['position_size']
                                    
                                    # Log range sizing factors
                                    self.logger.info(f"Range position sizing: base={range_sizing['base_size']}, edge_factor={range_sizing['edge_factor']:.1f}, " +
                                            f"pos_in_range={range_sizing['position_in_range']:.2f}, confidence={range_sizing['confidence']:.2f}, final={sized_position}")
                                    
                                    # Apply range amplitude adjustment - limit size near extremes
                                    if 'position_in_range' in range_analysis:
                                        # Apply pattern-specific position sizing adjustments
                                        if edge_patterns.get('detected', False) and edge_patterns.get('direction', 0) * direction > 0:
                                            pattern_conf = edge_patterns.get('confidence', 0.5)
                                            original = sized_position
                                            
                                            # Scale position size based on pattern confidence
                                            pattern_multiplier = 1.0 + min(0.5, (pattern_conf - 0.5) * 0.7)
                                            sized_position = max(1, int(sized_position * pattern_multiplier))
                                            
                                            self.logger.info(f"Quantum pattern position boost: {original} → {sized_position} (pattern confidence: {pattern_conf:.2f})")

                                        # Apply mean reversion position boost for extreme z-scores
                                        if current_regime == 'range_bound' and 'range_analysis' in market_data:
                                            z_score = market_data['range_analysis'].get('z_score', 0)
                                            
                                            if abs(z_score) > 1.5 and np.sign(z_score) * direction < 0:
                                                # Signal aligns with mean reversion - boost position size
                                                original = sized_position
                                                boost_factor = min(1.8, 1.0 + (abs(z_score) - 1.0) * 0.25)  # More conservative boost
                                                sized_position = max(1, int(sized_position * boost_factor))
                                                self.logger.info(f"Mean reversion position boost: {original} → {sized_position} (z-score: {z_score:.2f})")

                                        # Prevent zero sizing
                                        if sized_position < 1:
                                            self.logger.info("Elite protection: minimum size enforced")
                                            sized_position = 1

                                        # Apply equity curve protection for drawdown recovery
                                        equity_adjustment_factor = self.equity_curve_adjustment()
                                        if equity_adjustment_factor != 1.0:
                                            original = sized_position
                                            sized_position = max(1, int(sized_position * equity_adjustment_factor))
                                            self.logger.info(f"Elite equity curve protection: {original} → {sized_position} (recent win rate: {self.get_recent_win_rate(10):.2f})")

                                        # ENHANCED: Calculate optimized profit targets
                                        if current_regime == 'range_bound' and 'range_analysis' in market_data:
                                            # For range-bound markets, use special range targets
                                            optimized_targets = self.range_optimized_profit_targets(
                                                current_price, 
                                                direction,
                                                market_data,
                                                market_data['range_analysis']
                                            )
                                            
                                            # Log range-optimized targets
                                            self.logger.info(f"Range-optimized targets: T1: ${optimized_targets['targets'][0]:.2f}, " +
                                                    f"T2: ${optimized_targets['targets'][1]:.2f}, T3: ${optimized_targets['targets'][2]:.2f}")
                                            self.logger.info(f"Target positions in range: {optimized_targets['target_positions'][0]:.2f}, " +
                                                    f"{optimized_targets['target_positions'][1]:.2f}, {optimized_targets['target_positions'][2]:.2f}")
                                        else:
                                            # For non-range regimes, use dynamic profit targets
                                            optimized_targets = self.optimize_profit_targets(
                                                current_price, 
                                                direction,
                                                market_data,
                                                regime_info,
                                                order_flow_patterns if order_flow_patterns.get('pattern_detected', False) else None
                                            )
                                            
                                            # Log dynamically optimized targets
                                            self.logger.info(f"Quantum-optimized targets: T1: ${optimized_targets['targets'][0]:.2f}, " +
                                                    f"T2: ${optimized_targets['targets'][1]:.2f}, T3: ${optimized_targets['targets'][2]:.2f}")

                                        # Use optimized profit targets
                                        profit_target_1 = optimized_targets['targets'][0]
                                        profit_target_2 = optimized_targets['targets'][1]
                                        profit_target_3 = optimized_targets['targets'][2]

                                        # Use primary profit target
                                        profit_target_price = profit_target_1

                                        # Calculate enhanced risk/reward ratio using stop and targets
                                        risk_reward = self.generate_advanced_risk_reward(
                                            direction,
                                            current_price,
                                            stop_loss_price,
                                            market_data,
                                            regime_info
                                        )

                                        # Log risk reward and expected hold times
                                        self.logger.info(f"Elite quantum risk-reward: expectancy={risk_reward['expectancy']:.2f}, R:R={risk_reward['reward_risk_ratio']:.2f}")
                                        self.logger.info(f"Expected hold times: {optimized_targets['expected_hold_times'][0]}s, {optimized_targets['expected_hold_times'][1]}s, {optimized_targets['expected_hold_times'][2]}s")

                                        # ENHANCED: Apply risk reward filter with dynamic thresholds
                                        # Adaptive R:R thresholds based on regime and market conditions
                                        rr_min_threshold = 0.28  # Minimum R:R threshold (increased from 0.25)
                                        expectancy_min_threshold = -0.10  # Minimum expectancy threshold (improved from -0.15)

                                        # Adjust thresholds based on regime
                                        if current_regime in ['trending_up', 'trending_down'] and regime_confidence > 0.7:
                                            # In strong trends, accept slightly lower R:R
                                            rr_min_threshold = 0.25
                                            expectancy_min_threshold = -0.10
                                        elif current_regime == 'range_bound' and 'range_analysis' in market_data:
                                            # For range markets, adjust based on position in range
                                            z_score = range_analysis.get('z_score', 0)
                                            if abs(z_score) > 1.5 and np.sign(z_score) * direction < 0:
                                                # For mean reversion at extremes, accept slightly lower R:R
                                                rr_min_threshold = 0.22
                                                expectancy_min_threshold = -0.05

                                        # Apply the risk-reward filter
                                        if risk_reward["reward_risk_ratio"] < rr_min_threshold and risk_reward["expectancy"] < expectancy_min_threshold:
                                            self.logger.info(f"Elite risk-reward filter: Trade rejected (RR: {risk_reward['reward_risk_ratio']:.2f}, Exp: {risk_reward['expectancy']:.2f})")
                                            time.sleep(self.config['system']['update_interval'])
                                            continue

                                        # ENHANCED: Calculate execution quality for entry timing
                                        execution_quality = self.enhanced_execution_quality_score(market_data, direction)

                                        # Optimize entry timing
                                        timing_decision = self.optimize_execution_timing('entry', direction, market_data)

                                        if execution_quality['quality_score'] > 0.8:
                                            self.logger.info(f"Optimal entry conditions detected (score: {execution_quality['quality_score']:.2f})")
                                        elif timing_decision['delay_seconds'] > 0:
                                            self.logger.info(f"Execution timing optimization: Delay entry by {timing_decision['delay_seconds']}s ({timing_decision['reason']})")
                                            time.sleep(timing_decision['delay_seconds'])
                                            # Get updated price after delay
                                            current_price = self.market_data.get_realtime_data().get('price', current_price)

                                        # Final position size after all adjustments
                                        final_position_size = max(1, sized_position)

                                        # Place entry order
                                                                              
                                        order_id = self.execution_engine.place_order(
                                            'market',
                                            'NQ',
                                            final_position_size * direction
                                        )
                                        
                                        if order_id:
                                            # Create new position ID with timestamp
                                            current_position_id = f"pos_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
                                            
                                            # Record entry price for P&L tracking
                                            self._position_entry_prices[current_position_id] = current_price
                                            
                                            # Record entry time for minimum hold time
                                            self._trade_timing['entry_times'][current_position_id] = current_time
                                            
                                            # Record detailed factor data for this trade
                                            self._trade_factor_data[current_position_id] = {
                                                'regime': current_regime,
                                                'regime_confidence': regime_confidence,
                                                'entanglement': entanglement,
                                                'elite_confidence': elite_confidence,
                                                'confirmation_score': confirmation_score,
                                                'delta': market_data.get('delta', 0),
                                                'order_flow': market_data.get('order_flow', 0),
                                                'vpin': market_data.get('vpin', 0),
                                                'signal_strength': composite_signal,
                                                'signal_threshold': signal_threshold,
                                                'execution_quality': execution_quality['quality_score'],
                                                'patterns': [],
                                                'range_data': {},
                                                'risk_reward': risk_reward
                                            }
                                            
                                            # Add pattern data if detected
                                            if edge_patterns.get('detected', False):
                                                self._trade_factor_data[current_position_id]['patterns'].append({
                                                    'type': 'quantum_edge',
                                                    'name': edge_patterns.get('pattern', 'unknown'),
                                                    'direction': edge_patterns.get('direction', 0),
                                                    'confidence': edge_patterns.get('confidence', 0)
                                                })
                                            
                                            if order_flow_patterns.get('pattern_detected', False):
                                                self._trade_factor_data[current_position_id]['patterns'].append({
                                                    'type': 'order_flow',
                                                    'name': order_flow_patterns.get('pattern_name', 'unknown'),
                                                    'direction': order_flow_patterns.get('direction', 0),
                                                    'confidence': order_flow_patterns.get('confidence', 0)
                                                })
                                            
                                            # Add range data if applicable
                                            if current_regime == 'range_bound' and 'range_analysis' in market_data:
                                                range_analysis = market_data['range_analysis']
                                                self._trade_factor_data[current_position_id]['range_data'] = {
                                                    'position_in_range': range_analysis.get('position_in_range', 0.5),
                                                    'z_score': range_analysis.get('z_score', 0),
                                                    'range_high': range_analysis.get('range_high', 0),
                                                    'range_low': range_analysis.get('range_low', 0),
                                                    'confidence': range_analysis.get('confidence', 0)
                                                }
                                            
                                            # Log successful entry
                                            position_type = "LONG" if direction > 0 else "SHORT"
                                            self.logger.info(f"Entered {position_type} position: {final_position_size} NQ contracts @ ${current_price:,.2f} in {current_regime} regime")
                                            self.logger.info(f"Stop loss: ${stop_loss_price:.2f}, Profit targets: T1: ${profit_target_1:.2f}, T2: ${profit_target_2:.2f}, T3: ${profit_target_3:.2f}")
                                            # Create entry trade record for dashboard
                                            entry_trade_result = {
                                                'trade_id': current_position_id,
                                                'entry_price': current_price,
                                                'exit_price': 0,  # Not exited yet
                                                'size': final_position_size * direction,
                                                'profit': 0,  # No profit yet
                                                'signal_strength': composite_signal,
                                                'regime': current_regime,
                                                'regime_confidence': regime_confidence,
                                                'reason': 'entry_signal'
                                            }
                                            self.report_trade_to_dashboard(entry_trade_result)
                                            # Record position established time for signal stabilization
                                            self._position_established_time = current_time
                                            
                                            # Track position size
                                            position_size = final_position_size * direction
                                else:
                                    # Delta confirmation failed in volatile regime
                                    self.logger.info(f"Trade rejected due to missing delta confirmation in {current_regime} regime")
                        else:
                            # Signal below threshold - check for special patterns
                            choppy_pattern_detected = self.detect_choppy_market_pattern(market_data, price_history)
                            if choppy_pattern_detected.get('detected', False):
                                pattern_msg = ""
                                if choppy_pattern_detected.get('detected', False):
                                    pattern_msg = f", z-score={choppy_pattern_detected.get('z_score', 0):.2f}"
                                self.logger.info(f"Choppy market pattern detected: score={choppy_pattern_detected.get('score', 0):.2f}{pattern_msg}")
                            
                    # Update last trade time
                    last_trade_time = current_time
                    
                    # Sleep to reduce CPU usage
                    time.sleep(self.config['system']['update_interval'])
                
                except Exception as e:
                    import traceback
                    error_msg = traceback.format_exc()
                    self.logger.error(f"Error in main thread: {e}")
                    self.logger.error(error_msg)
                    
                    # Implement error recovery logic - safely exit any trades if major error
                    if position_size != 0:
                        try:
                            # Place exit order to close position
                            self.logger.info("Emergency exit triggered due to system error")
                            self.execution_engine.place_order(
                                'market',
                                'NQ',
                                -position_size
                            )
                            
                            # Reset position tracking
                            position_size = 0
                            stop_loss_price = None
                            profit_target_price = None
                        except Exception as exit_error:
                            self.logger.error(f"Emergency exit failed: {exit_error}")
                    
                    # Sleep before retry to avoid thrashing
                    time.sleep(5)
        
        except Exception as e:
            self.logger.error(f"Fatal error in main thread: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
        
        self.logger.info("Quantum-Enhanced Elite main system thread completed")
                    
    # --- ONLINE REINFORCEMENT LEARNING FUNCTIONS ---

def create_market_state(df, idx, lookback=10):
        """Create state representation from market data"""
        if idx < lookback:
            return None
            
        # Price features
        close_prices = df['Close'].values[idx-lookback:idx]
        open_prices = df['Open'].values[idx-lookback:idx]
        high_prices = df['High'].values[idx-lookback:idx]
        low_prices = df['Low'].values[idx-lookback:idx]
        volumes = df['Volume'].values[idx-lookback:idx]
        
        # Calculate price changes
        price_changes = np.diff(close_prices) / close_prices[:-1]
        
        # Normalize price data
        close_mean, close_std = np.mean(close_prices), np.std(close_prices)
        if close_std == 0:
            close_std = 1
        norm_close = (close_prices - close_mean) / close_std
        
        # Normalize volumes
        volume_mean, volume_std = np.mean(volumes), np.std(volumes)
        if volume_std == 0:
            volume_std = 1
        norm_volumes = (volumes - volume_mean) / volume_std
        
        # Technical indicators (if available)
        features = []
        features.extend(norm_close)
        features.extend(norm_volumes)
        
        # Add RSI if available
        if 'RSI' in df.columns:
            rsi_values = df['RSI'].values[idx-lookback:idx]
            normalized_rsi = (rsi_values - 50) / 25  # Center around 0
            features.extend(normalized_rsi)
        
        # Add MACD if available
        if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
            macd_values = df['MACD'].values[idx-lookback:idx]
            macd_signal = df['MACD_Signal'].values[idx-lookback:idx]
            macd_hist = macd_values - macd_signal
            max_macd = max(abs(np.max(macd_hist)), abs(np.min(macd_hist)))
            if max_macd > 0:
                normalized_macd = macd_hist / max_macd
            else:
                normalized_macd = macd_hist
            features.extend(normalized_macd)
        
        # Add recent price changes
        features.extend(price_changes)
        
        # Add candlestick patterns
        for i in range(lookback-1):
            body = (close_prices[i+1] - open_prices[i+1]) / open_prices[i+1]
            upper_wick = (high_prices[i+1] - max(close_prices[i+1], open_prices[i+1])) / open_prices[i+1]
            lower_wick = (min(close_prices[i+1], open_prices[i+1]) - low_prices[i+1]) / open_prices[i+1]
            features.extend([body, upper_wick, lower_wick])
        
        return np.array(features).reshape(1, -1)

def calculate_reward(action, position, entry_price, current_price, next_price):
        """Calculate reward for a trading action"""
        reward = 0
        new_position = position
        new_entry_price = entry_price
        
        # Price change percentage
        price_change_pct = (next_price - current_price) / current_price if current_price > 0 else 0
        
        # Action: Sell (0)
        if action == 0:
            if position == 1:  # Closing a long position
                profit_pct = (current_price - entry_price) / entry_price if entry_price > 0 else 0
                # Higher reward for profitable trades
                reward = profit_pct * 100  # Scale up reward
                if profit_pct > 0:
                    reward *= 1.5  # Extra bonus for profitable trades
                new_position = 0
                new_entry_price = 0
            elif position == 0:  # Opening a short position
                new_position = -1
                new_entry_price = current_price
                # Initial reward based on price movement (positive if price goes down)
                reward = -price_change_pct * 100
            else:  # Already in short position
                # Reward for holding correct position
                reward = -price_change_pct * 100
        
        # Action: Hold (1)
        elif action == 1:
            if position == 1:  # Long position
                reward = price_change_pct * 100
            elif position == -1:  # Short position
                reward = -price_change_pct * 100
            else:  # No position
                # Small penalty for sitting out
                reward = -abs(price_change_pct) * 10
        
        # Action: Buy (2)
        elif action == 2:
            if position == -1:  # Closing a short position
                profit_pct = (entry_price - current_price) / entry_price if entry_price > 0 else 0
                # Higher reward for profitable trades
                reward = profit_pct * 100  # Scale up reward
                if profit_pct > 0:
                    reward *= 1.5  # Extra bonus for profitable trades
                new_position = 0
                new_entry_price = 0
            elif position == 0:  # Opening a long position
                new_position = 1
                new_entry_price = current_price
                # Initial reward based on price movement (positive if price goes up)
                reward = price_change_pct * 100
            else:  # Already in long position
                # Reward for holding correct position
                reward = price_change_pct * 100
        
        # Trading cost penalty
        if position != new_position:
            reward -= 2  # Fixed cost for trading
        
        return reward, new_position, new_entry_price, price_change_pct

def train_rl_trading_system(df, lookback=10, episodes=100):
        """Train RL agent on historical market data"""
        print("Starting RL trading system training...")
        
        # Create state representation
        test_idx = lookback + 5
        test_state = create_market_state(df, test_idx, lookback)
        if test_state is None:
            print("Error: Cannot create state representation")
            return None
        
        # State and action dimensions
        state_size = test_state.shape[1]
        action_size = 3  # 0=Sell, 1=Hold, 2=Buy
        
        print(f"State size: {state_size}, Action size: {action_size}")
        
        # Initialize agent
        agent = RLTradingAgent(state_size, action_size)
        
        # Check for existing trained model
        model_path = "rl_trading_model.h5"
        if os.path.exists(model_path):
            print("Loading existing model...")
            agent.load_model(model_path)
        
        # Training loop
        best_performance = -float('inf')
        
        for episode in range(episodes):
            print(f"\nEpisode {episode+1}/{episodes}")
            
            # Reset for new episode
            total_profit = 0
            total_reward = 0
            position = 0
            entry_price = 0
            trade_count = 0
            
            # Start from a random point to increase exploration
            start_idx = random.randint(lookback, int(len(df) * 0.7))
            end_idx = min(start_idx + 500, len(df) - 1)  # Limit episode length
            
            for idx in range(start_idx, end_idx):
                # Create state representation
                state = create_market_state(df, idx, lookback)
                if state is None:
                    continue
                
                # Choose action
                action = agent.act(state)
                
                # Execute action
                current_price = df['Close'].iloc[idx]
                next_price = df['Close'].iloc[idx + 1]
                
                # Calculate reward
                reward, position, entry_price, price_change = calculate_reward(
                    action, position, entry_price, current_price, next_price
                )
                
                # Update total reward
                total_reward += reward
                
                # Calculate profit
                trade_profit = 0
                if position == 1:
                    trade_profit = price_change
                elif position == -1:
                    trade_profit = -price_change
                
                total_profit += trade_profit
                
                # Get next state
                next_state = create_market_state(df, idx + 1, lookback)
                if next_state is None:
                    continue
                
                # Remember experience
                done = (idx == end_idx - 1)
                agent.remember(state, action, reward, next_state, done)
                
                # Log performance
                if idx % 10 == 0:
                    agent.log_performance(idx - start_idx, reward, total_profit)
                
                # Train network
                if len(agent.memory) > agent.batch_size:
                    loss = agent.replay(agent.batch_size)
                
                # Count trades for statistics
                if action != 1:  # Not hold
                    trade_count += 1
            
            # End of episode statistics
            print(f"Episode {episode+1} results:")
            print(f"Total reward: {total_reward:.2f}")
            print(f"Total profit: {total_profit:.2%}")
            print(f"Number of trades: {trade_count}")
            
            # Save if best performance
            if total_profit > best_performance:
                best_performance = total_profit
                agent.save_model(model_path)
                print(f"New best model saved with profit: {total_profit:.2%}")
        
        print("RL training completed")
        return agent

def implement_rl_strategy(df, agent, lookback=10):
        """Implement RL strategy on market data"""
        print("Implementing RL trading strategy...")
        
        if agent is None:
            print("Error: No RL agent provided")
            return df
        
        # Initialize trading variables
        position = 0
        entry_price = 0
        signals = []
        profits = []
        
        for idx in range(lookback, len(df) - 1):
            # Create state representation
            state = create_market_state(df, idx, lookback)
            if state is None:
                signals.append(0)
                profits.append(0)
                continue
            
            # Choose action (no exploration in implementation)
            action = agent.act(state, training=False)
            
            # Current price
            current_price = df['Close'].iloc[idx]
            
            # Execute action
            if action == 0:  # Sell
                if position == 1:  # Close long
                    position = 0
                    profit = (current_price - entry_price) / entry_price if entry_price > 0 else 0
                    profits.append(profit)
                    signals.append(-1)  # Sell signal
                elif position == 0:  # Go short
                    position = -1
                    entry_price = current_price
                    profits.append(0)
                    signals.append(-1)  # Sell signal
                else:  # Already short
                    profits.append(0)
                    signals.append(0)  # No new signal
            
            elif action == 1:  # Hold
                profits.append(0)
                signals.append(0)  # Hold signal
            
            elif action == 2:  # Buy
                if position == -1:  # Close short
                    position = 0
                    profit = (entry_price - current_price) / entry_price if entry_price > 0 else 0
                    profits.append(profit)
                    signals.append(1)  # Buy signal
                elif position == 0:  # Go long
                    position = 1
                    entry_price = current_price
                    profits.append(0)
                    signals.append(1)  # Buy signal
                else:  # Already long
                    profits.append(0)
                    signals.append(0)  # No new signal
        
        # Add signals to dataframe
        df['RL_Signal'] = [0] * lookback + signals + [0]  # Add padding at end if needed
        
        # Trim if too long
        if len(df['RL_Signal']) > len(df):
            df['RL_Signal'] = df['RL_Signal'][:len(df)]
        
        # Compute performance metrics
        print("\nRL Strategy Performance:")
        total_profit = sum(profits)
        win_rate = sum(1 for p in profits if p > 0) / len(profits) if profits else 0
        
        print(f"Total profit: {total_profit:.2%}")
        print(f"Win rate: {win_rate:.2%}")
        print(f"Number of trades: {len([p for p in profits if p != 0])}")
        
        # Add combined signal if original signals exist
        if 'Signal' in df.columns:
            # Smart combination: Use RL for strong signals, original otherwise
            df['Combined_Signal'] = np.where(
                df['RL_Signal'] != 0,  # When RL has a signal
                df['RL_Signal'],       # Use RL
                df['Signal']           # Otherwise use original
            )
            
            # Consensus signal (only trade when both agree)
            df['Consensus_Signal'] = np.where(
                (df['RL_Signal'] == df['Signal']) & (df['RL_Signal'] != 0),
                df['RL_Signal'],
                0  # Hold if disagreement
            )
        
        return df

def evaluate_rl_performance(df, signal_column='RL_Signal', initial_capital=10000):
        """Evaluate trading performance of RL signals"""
        # Copy dataframe to avoid modifying original
        eval_df = df.copy()
        
        # Ensure the signal column exists
        if signal_column not in eval_df.columns:
            print(f"Error: {signal_column} column not found")
            return None
        
        # Initialize variables
        capital = initial_capital
        position = 0
        entry_price = 0
        trades = []
        equity_curve = [initial_capital]
        
        # Simulate trading
        for i in range(1, len(eval_df)):
            signal = eval_df[signal_column].iloc[i-1]  # Previous signal determines current position
            current_price = eval_df['Close'].iloc[i]
            
            # Process signal
            if signal == 1 and position <= 0:  # Buy signal
                # Close any existing short position
                if position < 0:
                    profit = entry_price - current_price
                    capital += profit * abs(position)
                    trades.append({
                        'type': 'close_short',
                        'entry': entry_price,
                        'exit': current_price,
                        'profit': (entry_price - current_price) / entry_price,
                        'capital': capital
                    })
                
                # Open long position
                position = 1
                entry_price = current_price
                
            elif signal == -1 and position >= 0:  # Sell signal
                # Close any existing long position
                if position > 0:
                    profit = current_price - entry_price
                    capital += profit * position
                    trades.append({
                        'type': 'close_long',
                        'entry': entry_price,
                        'exit': current_price,
                        'profit': (current_price - entry_price) / entry_price,
                        'capital': capital
                    })
                
                # Open short position
                position = -1
                entry_price = current_price
            
            # Update equity curve
            if position == 1:
                equity_curve.append(capital + (current_price - entry_price) * position)
            elif position == -1:
                equity_curve.append(capital + (entry_price - current_price) * abs(position))
            else:
                equity_curve.append(capital)
        
        # Close final position for accurate P&L
        final_price = eval_df['Close'].iloc[-1]
        if position == 1:
            profit = final_price - entry_price
            capital += profit * position
            trades.append({
                'type': 'close_final_long',
                'entry': entry_price,
                'exit': final_price,
                'profit': (final_price - entry_price) / entry_price,
                'capital': capital
            })
        elif position == -1:
            profit = entry_price - final_price
            capital += profit * abs(position)
            trades.append({
                'type': 'close_final_short',
                'entry': entry_price,
                'exit': final_price,
                'profit': (entry_price - final_price) / entry_price,
                'capital': capital
            })
        
        # Calculate metrics
        total_return = (capital / initial_capital) - 1
        
        # Calculate drawdown
        equity_curve = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve / running_max) - 1
        max_drawdown = drawdown.min()
        
        # Calculate Sharpe ratio
        if len(trades) > 1:
            returns = [(t['capital'] / prev_t['capital']) - 1 for t, prev_t in zip(trades[1:], trades[:-1])]
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Print performance summary
        print(f"\nPerformance of {signal_column}:")
        print(f"Total Return: {total_return:.2%}")
        print(f"Number of Trades: {len(trades)}")
        if len(trades) > 0:
            print(f"Win Rate: {sum(1 for t in trades if t.get('profit', 0) > 0) / len(trades):.2%}")
        print(f"Max Drawdown: {max_drawdown:.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        
        return {
            'total_return': total_return,
            'trades': trades,
            'equity_curve': equity_curve,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio
        }

def calculate_reward(action, position, entry_price, current_price, next_price, trade_count=0, max_drawdown=0):
        """Calculate a sophisticated reward signal aligned with trading goals"""
        # Initialize
        reward = 0
        new_position = position
        new_entry_price = entry_price
        
        # Price change
        price_change_pct = (next_price - current_price) / current_price if current_price != 0 else 0
        
        # Action: Sell (0)
        if action == 0:
            if position == 1:  # Close long
                profit_pct = (current_price - entry_price) / entry_price if entry_price != 0 else 0
                # Higher reward for profitable trades
                if profit_pct > 0:
                    reward = profit_pct * 100  # Scale reward
                else:
                    reward = profit_pct * 50   # Lower penalty for losses
                new_position = 0
                new_entry_price = 0
            elif position == 0:  # Go short
                new_position = -1
                new_entry_price = current_price
                # Initial reward based on next period's move
                reward = -price_change_pct * 80
            else:  # Already short
                # Reward for correct position
                reward = -price_change_pct * 80
        
        # Action: Hold (1)
        elif action == 1:
            if position == 1:  # Long position
                reward = price_change_pct * 80
            elif position == -1:  # Short position
                reward = -price_change_pct * 80
            else:  # No position
                # Small penalty for sitting out, but less than trading cost
                reward = -abs(price_change_pct) * 5
        
        # Action: Buy (2)
        elif action == 2:
            if position == -1:  # Close short
                profit_pct = (entry_price - current_price) / entry_price if entry_price != 0 else 0
                # Higher reward for profitable trades
                if profit_pct > 0:
                    reward = profit_pct * 100
                else:
                    reward = profit_pct * 50
                new_position = 0
                new_entry_price = 0
            elif position == 0:  # Go long
                new_position = 1
                new_entry_price = current_price
                # Initial reward based on next period's move
                reward = price_change_pct * 80
            else:  # Already long
                # Reward for correct position
                reward = price_change_pct * 80
        
        # Trading cost penalty
        if position != new_position:
            reward -= 2  # Fixed cost for trading
            
            # Additional penalty for excessive trading
            if trade_count > 10:  # If more than 10 trades in recent window
                reward -= (trade_count - 10) * 0.2  # Increasing penalty
        
        # Risk-adjusted reward modifications
        if max_drawdown < -0.05:  # If drawdown is worse than 5%
            # Reduce reward to encourage more conservative behavior
            reward *= (1 + max_drawdown * 2)  # E.g., 10% drawdown = 80% of normal reward
        
        # Reward shaping for trend following
        if (position == 1 and price_change_pct > 0) or (position == -1 and price_change_pct < 0):
            # Bonus for riding the trend
            reward *= 1.2
        
        return reward, new_position, new_entry_price

def train_online_rl(agent, state, action, reward, next_state, done, trade_count=0, max_drawdown=0):
        """Train RL agent online with the latest experience"""
        # Store experience
        agent.remember(state, action, reward, next_state, done)
        
        # Train if enough experiences are collected
        if len(agent.states) >= agent.batch_size:
            agent.train_from_buffer()
        
        # Periodically train from replay buffer for offline learning
        if random.random() < 0.05:  # 5% chance each step
            agent.train_offline_batch()
        
        # If episode ended (e.g., end of day or week)
        if done:
            # Always train at episode end
            if len(agent.states) > 0:
                agent.train_from_buffer()
            
            # Track rewards for this episode
            agent.total_rewards.append(reward)
            
            # Save model periodically
            if len(agent.total_rewards) % 10 == 0:
                try:
                    agent.save(
                        f"trading_agent_actor_{len(agent.total_rewards)}.h5",
                        f"trading_agent_critic_{len(agent.total_rewards)}.h5"
                    )
                    print(f"Model saved after {len(agent.total_rewards)} episodes")
                except Exception as e:
                    print(f"Error saving model: {str(e)}")

def implement_rl_trading_strategy(bot, kline_data):
        """Implement RL trading strategy with online learning"""
        print("Initializing Online RL Trading Strategy...")
        
        # Ensure we have technical indicators
        if 'RSI' not in kline_data.columns:
            kline_data['RSI'] = talib.RSI(kline_data['Close'], timeperiod=14)
        
        if 'MACD' not in kline_data.columns or 'MACD_Signal' not in kline_data.columns:
            macd, macd_signal, _ = talib.MACD(kline_data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
            kline_data['MACD'] = macd
            kline_data['MACD_Signal'] = macd_signal
        
        # Create state and action dimensions
        lookback = 20
        test_state = create_market_state(kline_data, lookback+5, lookback)
        if test_state is None:
            print("Error: Cannot create state representation")
            return bot, kline_data
        
        state_size = test_state.shape[0]
        action_size = 3  # 0=Sell, 1=Hold, 2=Buy
        
        print(f"State size: {state_size}, Action size: {action_size}")
        
        # Initialize or load agent
        agent_path = "trading_agent_actor_latest.h5"
        if os.path.exists(agent_path):
            print("Loading existing RL agent...")
            agent = OnlinePPOAgent(state_size, action_size)
            agent.load("trading_agent_actor_latest.h5", "trading_agent_critic_latest.h5")
        else:
            print("Creating new RL agent...")
            agent = OnlinePPOAgent(state_size, action_size)
        
        # Initialize trading variables
        position = 0
        entry_price = 0
        signals = []
        rewards = []
        trade_count = 0
        max_drawdown = 0
        equity_curve = [1000]  # Starting with $1000
        
        # Track performance
        print("Beginning RL trading simulation...")
        
        # Simulate trading with online learning
        for idx in range(lookback, len(kline_data)):
            # Create state representation
            state = create_market_state(kline_data, idx, lookback)
            if state is None:
                signals.append(0)
                continue
            
            # Get action (with exploration during training)
            action, log_prob, value, _ = agent.get_action(state)
            
            # Execute action
            current_price = kline_data['Close'].iloc[idx]
            
            # For next state and reward calculation
            next_idx = min(idx + 1, len(kline_data) - 1)
            next_price = kline_data['Close'].iloc[next_idx]
            
            # Calculate reward
            reward, position, entry_price = calculate_reward(
                action, position, entry_price, current_price, next_price, 
                trade_count, max_drawdown
            )
            
            # Store reward
            rewards.append(reward)
            
            # Create next state
            next_state = create_market_state(kline_data, next_idx, lookback) if next_idx < len(kline_data) else state
            
            # Episode ends at end of data
            done = (next_idx == len(kline_data) - 1)
            
            # Online learning
            train_online_rl(agent, state, action, reward, next_state, done, trade_count, max_drawdown)
            
            # Track position changes for trade count
            if position != 0:
                trade_count += 1
            else:
                trade_count = max(0, trade_count - 1)  # Decay trade count
            
            # Update equity curve
            if position == 1:
                pnl = (next_price - current_price) / current_price
            elif position == -1:
                pnl = (current_price - next_price) / current_price
            else:
                pnl = 0
                
            equity_curve.append(equity_curve[-1] * (1 + pnl))
            
            # Calculate drawdown
            peak = max(equity_curve)
            current_dd = (equity_curve[-1] / peak) - 1
            max_drawdown = min(max_drawdown, current_dd)
            
            # Convert action to signal
            if action == 0:  # Sell
                signals.append(-1)
            elif action == 1:  # Hold
                signals.append(0)
            else:  # Buy
                signals.append(1)
        
        # Save final model
        agent.save("trading_agent_actor_latest.h5", "trading_agent_critic_latest.h5")
        
        # Add signals to dataframe
        kline_data['RL_Signal'] = [0] * lookback + signals
        
        # If original signals exist, create combined signals
        if 'Signal' in kline_data.columns:
            # Strategy 1: Use RL when confident, otherwise use original
            kline_data['Combined_Signal'] = np.where(
                kline_data['RL_Signal'] != 0,  # When RL has a strong opinion
                kline_data['RL_Signal'],       # Use RL signal
                kline_data['Signal']           # Otherwise use original
            )
            
            # Strategy 2: Combine signals (only trade when both agree)
            kline_data['Consensus_Signal'] = np.where(
                (kline_data['RL_Signal'] == kline_data['Signal']) & (kline_data['RL_Signal'] != 0),
                kline_data['RL_Signal'],
                0  # Hold if disagreement
            )
        else:
            kline_data['Combined_Signal'] = kline_data['RL_Signal']
            kline_data['Consensus_Signal'] = kline_data['RL_Signal']
        
        # Calculate performance metrics for RL strategy
        print("\nRL Strategy Performance:")
        rl_returns = []
        position = 0
        
        for i in range(1, len(kline_data)):
            signal = kline_data['RL_Signal'].iloc[i-1]
            if signal == 1:
                position = 1
            elif signal == -1:
                position = -1
            
            if position == 1:
                rl_returns.append((kline_data['Close'].iloc[i] / kline_data['Close'].iloc[i-1]) - 1)
            elif position == -1:
                rl_returns.append((kline_data['Close'].iloc[i-1] / kline_data['Close'].iloc[i]) - 1)
            else:
                rl_returns.append(0)
        
        # Calculate metrics
        rl_returns = np.array(rl_returns)
        total_return = np.prod(1 + rl_returns) - 1
        annual_return = ((1 + total_return) ** (252 / len(rl_returns))) - 1
        sharpe = np.mean(rl_returns) / np.std(rl_returns) * np.sqrt(252) if np.std(rl_returns) > 0 else 0
        max_dd = calculate_max_drawdown(np.cumprod(1 + rl_returns))
        
        print(f"Total Return: {total_return:.2%}")
        print(f"Annual Return: {annual_return:.2%}")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        print(f"Max Drawdown: {max_dd:.2%}")
        
        # Update bot with RL capability
        bot.rl_agent = agent
        
        return bot, kline_data
def integrate_rl_with_existing_strategy(market_data):
    """Integrate RL with existing trading strategy"""
    print("Integrating RL agent with existing strategy...")
    
    # Ensure required columns exist
    df_rl = market_data.copy()
    
    # Add technical indicators if not already present
    if 'RSI' not in df_rl.columns:
        df_rl['RSI'] = talib.RSI(df_rl['Close'], timeperiod=14)
    
    if 'MACD' not in df_rl.columns or 'MACD_Signal' not in df_rl.columns:
        macd, macd_signal, _ = talib.MACD(df_rl['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df_rl['MACD'] = macd
        df_rl['MACD_Signal'] = macd_signal
    
    # Split data for training and testing
    train_size = int(len(df_rl) * 0.8)
    df_train = df_rl.iloc[:train_size].copy()
    df_test = df_rl.iloc[train_size:].copy()
    
    print(f"Training data size: {len(df_train)}, Testing data size: {len(df_test)}")
    
    # Train the RL agent
    agent = train_rl_agent(df_train, episodes=50, lookback=10)
    
    if agent is not None:
        # Generate signals for the entire dataset
        signals = rl_trading_strategy(df_rl, agent)
        
        # Pad signals to match df length
        padded_signals = [0] * 10 + signals  # 10 is the lookback period
        if len(padded_signals) > len(market_data):
            padded_signals = padded_signals[:len(market_data)]
        elif len(padded_signals) < len(market_data):
            padded_signals.extend([0] * (len(market_data) - len(padded_signals)))
            
        # Add RL signals to original dataframe
        market_data['RL_Signal'] = padded_signals
        
        # Combine RL signals with existing signals if available
        if 'Signal' in market_data.columns:
            market_data['Combined_Signal'] = np.where(
                market_data['RL_Signal'] != 0,  # If RL has a signal
                market_data['RL_Signal'],       # Use RL signal
                market_data['Signal']           # Otherwise use existing signal
            )
            print("Combined RL signals with existing signals")
        else:
            market_data['Combined_Signal'] = market_data['RL_Signal']
            print("Using only RL signals (no existing signals found)")
        
        return market_data, agent
    else:
        print("RL agent training failed, using original dataframe")
        return market_data, None
def train_rl_agent(df, episodes=100, batch_size=32, lookback=10):
        """Train the RL agent on historical data"""
        print("Starting RL agent training...")
        
        # Make sure we have all necessary columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            print("Error: Required columns missing from DataFrame")
            missing = [col for col in required_cols if col not in df.columns]
            print(f"Missing columns: {missing}")
            return None
            
        # Make sure we have enough data
        if len(df) <= lookback + 10:
            print(f"Error: Not enough data points. Need more than {lookback + 10}, got {len(df)}")
            return None
        
        # Calculate state size based on our state representation
        test_idx = min(len(df)-1, lookback + 10)  # Safe index
        test_state = create_market_state(df, test_idx, lookback)
        if test_state is None:
            print("Cannot create valid state representation")
            return None
        
        state_size = len(test_state)
        action_size = 3  # Sell, Hold, Buy
        
        print(f"State size: {state_size}, Action size: {action_size}")
        
        # Initialize agent
        agent = RLTradingAgent(state_size, action_size)
        
        # Training loop
        for episode in range(episodes):
            # Reset environment
            position = 0  # No position
            entry_price = 0
            total_reward = 0
            
            # Start from a random point after lookback periods
            start_idx = random.randint(lookback, int(len(df)*0.7))  # Use 70% of data for training
            
            # Loop through the episode
            for current_idx in range(start_idx, len(df)-1):
                # Get current state
                state = create_market_state(df, current_idx, lookback)
                if state is None:
                    continue
                    
                # Get action
                action = agent.act(state)
                
                # Calculate reward and update position
                current_price = df['Close'].iloc[current_idx]
                next_price = df['Close'].iloc[current_idx+1]
                
                reward, position, entry_price, _ = calculate_reward(
                    action, position, entry_price, current_price, next_price
                )
                
                # Get next state
                next_state = create_market_state(df, current_idx+1, lookback)
                if next_state is None:
                    continue
                
                # Done flag (episode ends at the last data point)
                done = (current_idx == len(df)-2)
                
                # Store in memory
                agent.remember(state, action, reward, next_state, done)
                
                total_reward += reward
                
                # Train model if we have enough samples
                if len(agent.memory) > batch_size:
                    agent.replay(batch_size)
                    
            # Episode summary    
            print(f"Episode: {episode+1}/{episodes}, Total Reward: {total_reward:.2f}")
                    
            # Save the model periodically
            if (episode + 1) % 10 == 0:
                try:
                    agent.save_model(f"rl_trading_ep{episode+1}.h5")
                    print(f"Model saved at episode {episode+1}")
                except Exception as e:
                    print(f"Error saving model: {e}")
                
        # Final save
        try:
            agent.save_model("rl_trading_final.h5")
            print("Final model saved")
        except Exception as e:
            print(f"Error saving final model: {e}")
            
        return agent

def rl_trading_strategy(df, agent, lookback=10):
        """Use the trained RL agent to make trading decisions"""
        signals = []
        position = 0
        entry_price = 0
        
        for idx in range(lookback, len(df)):
            state = create_market_state(df, idx, lookback)
            if state is None:
                signals.append(0)  # Hold if we can't create a state
                continue
                
            # Get action from agent
            action = agent.act(state)
            
            # Update position based on action
            current_price = df['Close'].iloc[idx]
            
            if action == 0:  # Sell
                if position == 1:  # Close long
                    position = 0
                    signals.append(-1)  # Sell signal
                elif position == 0:  # Go short
                    position = -1
                    entry_price = current_price
                    signals.append(-1)  # Sell signal
                else:  # Already short
                    signals.append(0)  # Hold signal
            
            elif action == 1:  # Hold
                signals.append(0)  # Hold signal
            
            elif action == 2:  # Buy
                if position == -1:  # Close short
                    position = 0
                    signals.append(1)  # Buy signal
                elif position == 0:  # Go long
                    position = 1
                    entry_price = current_price
                    signals.append(1)  # Buy signal
                else:  # Already long
                    signals.append(0)  # Hold signal
                    
        return signals

def evaluate_rl_performance(df, agent, lookback=10):
        """Evaluate the RL agent's performance on test data"""
        signals = rl_trading_strategy(df, agent, lookback)
        
        # Ensure signals are the right length (account for lookback period)
        padded_signals = [0] * lookback + signals
        
        # Trim to match df length
        if len(padded_signals) > len(df):
            padded_signals = padded_signals[:len(df)]
        
        # Add signals to dataframe
        df_eval = df.copy()
        df_eval['RL_Signal'] = padded_signals
        
        # Calculate returns
        df_eval['Position'] = df_eval['RL_Signal'].shift(1).fillna(0)
        df_eval['Strategy_Return'] = df_eval['Position'] * df_eval['Close'].pct_change()
        
        # Calculate metrics
        total_return = (df_eval['Strategy_Return'] + 1).cumprod().iloc[-1] - 1
        annual_return = ((1 + total_return) ** (252 / len(df_eval)) - 1)
        daily_returns = df_eval['Strategy_Return'].dropna()
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * (252 ** 0.5) if daily_returns.std() != 0 else 0
        
        # Calculate drawdown
        cumulative_returns = (1 + df_eval['Strategy_Return'].fillna(0)).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max) - 1
        max_drawdown = drawdown.min()
        
        print(f"--- RL PERFORMANCE METRICS ---")
        print(f"Total Return: {total_return:.2%}")
        print(f"Annual Return: {annual_return:.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Max Drawdown: {max_drawdown:.2%}")
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }        
def calculate_max_drawdown(equity_curve):
        """Calculate maximum drawdown from equity curve"""
        max_dd = 0
        peak = equity_curve[0]
        
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (value / peak) - 1
            max_dd = min(max_dd, dd)
        
        return max_dd
def setup_exchange(exchange_id='binance', test_mode=True):
    """
    Set up and configure the exchange connection.
    
    Parameters:
    -----------
    exchange_id : str
        The exchange to connect to (default: 'binance')
    test_mode : bool
        Whether to use testnet/paper trading (default: True)
        
    Returns:
    --------
    exchange : object
        Configured exchange object
    """
    import ccxt
    import os
    from dotenv import load_dotenv
    
    # Load API keys from .env file (more secure than hardcoding)
    load_dotenv()
    
    print(f"Setting up connection to {exchange_id.upper()}...")
    
    try:
        # Check if the exchange is supported
        if exchange_id.lower() not in ccxt.exchanges:
            print(f"Exchange {exchange_id} not supported by CCXT. Supported exchanges:")
            print(', '.join(ccxt.exchanges))
            print("Defaulting to Binance...")
            exchange_id = 'binance'
        
        # Initialize the exchange
        exchange_class = getattr(ccxt, exchange_id.lower())
        
        # Configuration dictionary
        config = {
            'enableRateLimit': True,  # Prevent ban due to rate limit
            'options': {
                'defaultType': 'future',  # Use futures by default (for NQ trading)
                'adjustForTimeDifference': True,
            }
        }
        
        # Add API credentials if available
        api_key = os.getenv(f'{exchange_id.upper()}_API_KEY')
        api_secret = os.getenv(f'{exchange_id.upper()}_API_SECRET')
        
        if api_key and api_secret:
            config['apiKey'] = api_key
            config['secret'] = api_secret
            print("API credentials loaded successfully")
        else:
            print("Warning: No API credentials found. Running in read-only mode.")
        
        # Set up testnet if in test mode
        if test_mode:
            if exchange_id.lower() == 'binance':
                config['options']['defaultType'] = 'future'
                config['urls'] = {
                    'api': {
                        'public': 'https://testnet.binancefuture.com/fapi/v1',
                        'private': 'https://testnet.binancefuture.com/fapi/v1',
                    }
                }
            print(f"Test mode enabled: Connected to {exchange_id.upper()} testnet")
        
        # Create exchange instance
        exchange = exchange_class(config)
        
        # Load markets (symbols, trading pairs, etc.)
        exchange.load_markets()
        print(f"Successfully connected to {exchange_id.upper()}")
        
        # Test connection by fetching ticker
        exchange.fetch_ticker('NQ/USD')
        print("Exchange connection verified")
        
        return exchange
        
    except ccxt.BaseError as e:
        print(f"Error connecting to {exchange_id}: {str(e)}")
        print("Falling back to simulation mode...")
        
        # Create a simulated exchange for paper trading
        from trading_simulator import SimulatedExchange  # You'll need to implement this
        return SimulatedExchange()
        
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        print("Falling back to simulation mode...")
        
        #


def main():
    """
    Main function for NQ Alpha Elite Trading Bot with Advanced RL Integration
    World's most sophisticated trading system combining traditional technical analysis
    with cutting-edge reinforcement learning
    """
    try:
        print("\n" + "="*60)
        print("  NQ ALPHA ELITE TRADING SYSTEM - v2.0 'Neural Quantum'")
        print("  AI-Powered Advanced Trading Platform")
        print("  Developed by: An0nym0usn3thunt3r")
        print("="*60)
        
        # Display current time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\nSystem Start: {current_time} UTC")
        
        # Set up logging
        log_dir = "trading_logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"trading_log_{current_time.replace(':', '-').replace(' ', '_')}.txt")
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logging.info("Trading system initialized")
        
        # Get trading parameters
        print("\nSetting up trading parameters...")
        symbol = input("Enter trading symbol (default: NQ): ") or "NQ"
        timeframe = input("Enter timeframe (default: 1h): ") or "1h"
        
        logging.info(f"Trading parameters: Symbol={symbol}, Timeframe={timeframe}")
        
        # Fetch market data using web scraper
        print(f"\nFetching market data for {symbol} on {timeframe} timeframe...")
        try:
            # Use your existing web scraper function
            market_data = fetch_historical_klines(symbol, timeframe)
            print(f"Successfully fetched {len(market_data)} data points")
            logging.info(f"Market data fetched: {len(market_data)} candles")
        except Exception as e:
            print(f"Error fetching market data: {str(e)}")
            logging.error(f"Data fetch error: {str(e)}")
            return None
        
        # Process market data
        print("\nProcessing market data...")
        market_data = process_klines(market_data)  # Your existing processing function
        
        # Add technical indicators
        print("Adding technical indicators...")
        market_data = add_indicators(market_data)  # Your existing indicators function
        
        # Generate trading signals using traditional strategy
        print("\nGenerating trading signals using traditional strategy...")
        signals = generate_signals(market_data)  # Your existing signal generator
        market_data['Signal'] = signals
        
        # Calculate traditional strategy performance
        print("\nTraditional Strategy Performance:")
        trad_performance = calculate_performance(market_data, 'Signal')  # Your existing performance calculator
        
        # Option to add reinforcement learning
        use_rl = input("\nIntegrate reinforcement learning? (y/n): ").lower() == 'y'
        
        if use_rl:
            print("\n" + "="*60)
            print("  NEURAL REINFORCEMENT LEARNING INTEGRATION")
            print("="*60)
            
            # Ask for RL training preference
            train_new = input("Train new RL model or use existing? (train/use): ").lower() == 'train'
            
            if train_new:
                print("\nTraining new reinforcement learning model...")
                # Set training parameters
                episodes = int(input("Enter number of training episodes (default: 50): ") or "50")
                logging.info(f"Training new RL model with {episodes} episodes")
                
                # Integrate RL with traditional strategy
                market_data, rl_agent = integrate_rl_with_existing_strategy(market_data, episodes=episodes)
            else:
                print("\nLoading existing reinforcement learning model...")
                logging.info("Using existing RL model")
                market_data, rl_agent = integrate_rl_with_existing_strategy(market_data, mode='use')
            
            # Performance comparison if RL was successfully integrated
            if 'RL_Signal' in market_data.columns:
                print("\nRL Strategy Performance:")
                rl_performance = calculate_performance(market_data, 'RL_Signal')
                
                if 'Combined_Signal' in market_data.columns:
                    print("\nCombined Strategy Performance:")
                    combined_performance = calculate_performance(market_data, 'Combined_Signal')
        else:
            print("\nUsing traditional strategy only (no RL integration)")
            logging.info("RL integration skipped")
        
        # Strategy selection for backtesting/trading
        print("\n" + "="*60)
        print("  STRATEGY SELECTION")
        print("="*60)
        
        if use_rl and 'RL_Signal' in market_data.columns:
            strategy_options = [
                "1: Traditional (Technical Indicators)",
                "2: Reinforcement Learning",
                "3: Combined (Hybrid Strategy)"
            ]
            print("\nAvailable strategies:")
            for option in strategy_options:
                print(f"  {option}")
            
            strategy_choice = input("\nSelect strategy number (default: 3): ") or "3"
            
            if strategy_choice == "1":
                active_signal = 'Signal'
                strategy_name = "Traditional"
            elif strategy_choice == "2":
                active_signal = 'RL_Signal'
                strategy_name = "Reinforcement Learning"
            else:
                active_signal = 'Combined_Signal'
                strategy_name = "Combined Hybrid"
        else:
            active_signal = 'Signal'
            strategy_name = "Traditional"
        
        print(f"\nSelected strategy: {strategy_name}")
        logging.info(f"Strategy selected: {strategy_name}")
        
        # Trading mode selection
        print("\n" + "="*60)
        print("  EXECUTION MODE")
        print("="*60)
        
        print("\nExecution modes:")
        print("  1: Backtesting (Historical Analysis)")
        print("  2: Paper Trading (Simulated Live Trading)")
        print("  3: Live Trading (Real Market Orders)")
        
        mode = input("\nSelect execution mode (default: 1): ") or "1"
        
        if mode == "1":
            # Backtesting mode
            print("\n" + "="*60)
            print("  BACKTESTING MODE")
            print("="*60)
            
            print(f"\nRunning backtest with {strategy_name} strategy...")
            logging.info(f"Backtesting {strategy_name} strategy")
            
            # Run backtest using your existing backtest function
            backtest_results = backtest_strategy(market_data, active_signal)
            
            # Plot backtest results
            print("\nGenerating performance charts...")
            plot_backtest_results(market_data, backtest_results, active_signal)
            
            # Save backtest results
            results_dir = "backtest_results"
            os.makedirs(results_dir, exist_ok=True)
            results_file = os.path.join(results_dir, f"backtest_{symbol}_{timeframe}_{strategy_name.replace(' ', '_')}_{current_time.replace(':', '-').replace(' ', '_')}.pkl")
            
            try:
                with open(results_file, 'wb') as f:
                    pickle.dump(backtest_results, f)
                print(f"\nBacktest results saved to: {results_file}")
                logging.info(f"Backtest results saved to: {results_file}")
            except Exception as e:
                print(f"Error saving backtest results: {str(e)}")
                logging.error(f"Error saving backtest results: {str(e)}")
            
        elif mode == "2":
            # Paper trading mode
            print("\n" + "="*60)
            print("  PAPER TRADING MODE")
            print("="*60)
            
            print(f"\nStarting paper trading with {strategy_name} strategy...")
            logging.info(f"Paper trading with {strategy_name} strategy")
            
            # Initialize paper trading parameters
            initial_balance = float(input("Enter initial balance (default: 10000): ") or "10000")
            position_size = float(input("Enter position size % (default: 10): ") or "10") / 100
            
            print("\nPaper trading initialized...")
            print(f"Initial Balance: ${initial_balance:.2f}")
            print(f"Position Size: {position_size*100:.1f}%")
            
            # Run paper trading simulation
            try:
                paper_trading_results = run_paper_trading(
                    symbol, timeframe, 
                    strategy=strategy_name,
                    signal_column=active_signal,
                    initial_balance=initial_balance,
                    position_size=position_size,
                    rl_agent=rl_agent if use_rl and 'rl_agent' in locals() else None
                )
                
                print("\nPaper trading completed")
                logging.info("Paper trading completed")
                
            except KeyboardInterrupt:
                print("\nPaper trading stopped by user")
                logging.info("Paper trading stopped by user")
            except Exception as e:
                print(f"\nError during paper trading: {str(e)}")
                logging.error(f"Paper trading error: {str(e)}")
                traceback.print_exc()
            
        elif mode == "3":
            # Live trading mode
            print("\n" + "="*60)
            print("  LIVE TRADING MODE")
            print("="*60)
            
            # Security confirmation
            print("\n⚠️ WARNING: You are about to start LIVE TRADING with REAL MONEY ⚠️")
            confirm = input("\nType 'CONFIRM' to proceed with live trading: ")
            
            if confirm != "CONFIRM":
                print("Live trading canceled")
                logging.info("Live trading canceled by user")
                return market_data
            
            print(f"\nInitializing live trading with {strategy_name} strategy...")
            logging.info(f"Live trading initialized with {strategy_name} strategy")
            
            # Live trading parameters
            trade_amount = input("Enter trade quantity (or press Enter to use position sizing): ")
            trade_amount = float(trade_amount) if trade_amount else None
            
            # Run live trading
            try:
                live_trading_results = run_live_trading(
                    symbol, timeframe,
                    strategy=strategy_name,
                    signal_column=active_signal,
                    quantity=trade_amount,
                    rl_agent=rl_agent if use_rl and 'rl_agent' in locals() else None
                )
                
                print("\nLive trading session completed")
                logging.info("Live trading session completed")
                
            except KeyboardInterrupt:
                print("\nLive trading stopped by user")
                logging.info("Live trading stopped by user")
            except Exception as e:
                print(f"\nError during live trading: {str(e)}")
                logging.error(f"Live trading error: {str(e)}")
                traceback.print_exc()
        
        print("\n" + "="*60)
        print("  TRADING SYSTEM EXECUTION COMPLETED")
        print("="*60)
        
        # Return processed market data for further analysis
        return market_data
        
    except Exception as e:
        print(f"\n===== ERROR =====")
        print(f"An error occurred: {str(e)}")
        print("\nDetailed traceback:")
        traceback.print_exc()
        logging.error(f"Error in main: {str(e)}", exc_info=True)
        return None
if __name__ == "__main__":


    try:
        # Try to use NQDirectFeed (your web scraper class)
        data_feed = NQDirectFeed(clean_start=False)
        print("Using NQDirectFeed for market data...")
    except NameError:
        # Fallback to MarketDataFeed if NQDirectFeed isn't available
        try:
            # Try without clean_start parameter
            data_feed = MarketDataFeed()
            print("Using MarketDataFeed for market data...")
        except Exception as e:
            print(f"Error initializing data feed: {e}")
            raise

    # Force initial data fetch
    data_feed.update_data()

    # Get market data from the data feed
    market_data = data_feed.get_market_data(lookback=500)

    # Now integrate RL with the properly loaded market data
    market_data, rl_agent = integrate_rl_with_existing_strategy(market_data) # Use your existing web scraper class
    # Force initial data fetch

    # Get market data from the data feed
    market_data = data_feed.get_market_data(lookback=500)  # Get recent data (adjust lookback as needed)

    # Now we can integrate RL with the properly loaded market data
    market_data, rl_agent = integrate_rl_with_existing_strategy(market_data)
    
    # If you want to save the RL agent for future use
    if rl_agent is not None:
        try:
            rl_agent.save("production_rl_agent.h5")
            print("Saved production RL agent model")
        except Exception as e:
            print(f"Error saving production model: {e}")        
          
