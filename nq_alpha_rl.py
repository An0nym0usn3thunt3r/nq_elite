"""
NQ Alpha Elite Reinforcement Learning Module
Advanced PPO-based RL trading agent with regime-specific optimization
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, optimizers, losses
import random
from collections import deque
import datetime
import os
import logging

# Configure TensorFlow to use memory growth to avoid allocating all GPU memory
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except Exception as e:
    print(f"TensorFlow GPU configuration error: {e}")

# Memory buffer for PPO algorithm
class PPOMemory:
    def __init__(self, batch_size=64):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size

    def store(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []

    def sample_batch(self):
        batch_step = np.arange(0, len(self.states), self.batch_size)
        indices = np.arange(len(self.states), dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_step]
        return batches

    def get_arrays(self):
        return np.array(self.states), np.array(self.actions), np.array(self.probs), \
               np.array(self.vals), np.array(self.rewards), np.array(self.dones)
    
    def __len__(self):
        return len(self.states)


# Actor network for PPO, outputs action distribution parameters
class ActorNetwork(tf.keras.Model):
    def __init__(self, action_dim, name='actor'):
        super(ActorNetwork, self).__init__()
        self.action_dim = action_dim
        
        # Trading systems benefit from deeper networks to capture market complexities
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(128, activation='relu')
        self.dense3 = layers.Dense(64, activation='relu')
        
        # For continuous action space in trading
        self.mu = layers.Dense(action_dim, activation='tanh')  # bounded actions
        self.sigma = layers.Dense(action_dim, activation='softplus')  # positive std dev
        
    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        x = self.dense3(x)
        mu = self.mu(x)
        sigma = self.sigma(x) + 1e-5  # Add small constant for numerical stability
        return mu, sigma


# Critic network for PPO, estimates value function
class CriticNetwork(tf.keras.Model):
    def __init__(self, name='critic'):
        super(CriticNetwork, self).__init__()
        
        # Similar architecture to actor, but outputs scalar value
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(128, activation='relu')
        self.dense3 = layers.Dense(64, activation='relu')
        self.value = layers.Dense(1, activation=None)
        
    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        x = self.dense3(x)
        value = self.value(x)
        return value


# PPO Agent optimized for trading applications
class PPOAgent:
    def __init__(self, state_dim, action_dim, learning_rate=1e-5, gamma=0.98, 
                 gae_lambda=0.95, clip_ratio=0.2, batch_size=64, memory_size=2000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.batch_size = batch_size
        
        # Initialize actor and critic networks
        self.actor = ActorNetwork(action_dim)
        self.critic = CriticNetwork()
        
        # Build models (important to initialize weights)
        dummy_state = np.zeros((1, state_dim), dtype=np.float32)
        _, _ = self.actor(dummy_state)
        _ = self.critic(dummy_state)
        
        # Initialize optimizers
        self.actor_optimizer = optimizers.Adam(learning_rate=learning_rate)
        self.critic_optimizer = optimizers.Adam(learning_rate=learning_rate*2)
        
        # Initialize memory
        self.memory = PPOMemory(batch_size)
        
        # Experience replay buffer (additional to standard PPO for improved stability)
        self.experience_buffer = deque(maxlen=memory_size)
        
        # Exploration rate (more important in trading than standard PPO)
        self.exploration_rate = 0.2
        self.exploration_decay = 0.9995
        self.min_exploration_rate = 0.05
        
        # Trading-specific parameters
        self.market_regime_weights = {
            'trending_up': 1.2,
            'trending_down': 1.2,
            'range_bound': 0.8,
            'choppy': 0.5,
            'volatile': 1.0
        }
        
        # Training stats
        self.train_count = 0
        self.training_stats = {
            'actor_loss': [],
            'critic_loss': [],
            'exploration_rate': []
        }
        
    def predict(self, state):
        """Get action from actor network deterministically for exploitation"""
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        mu, _ = self.actor(state_tensor)
        return mu[0].numpy()
    
    def explore(self, state):
        """Get action from actor network with exploration noise"""
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        mu, sigma = self.actor(state_tensor)
        mu = mu[0].numpy()
        sigma = sigma[0].numpy()
        
        # Sample from normal distribution for continuous action space
        action = mu + np.random.normal(0, sigma)
        
        # Clip actions to appropriate range for trading (-1 to 1)
        return np.clip(action, -1, 1)
    
    def choose_action(self, state, deterministic=False):
        """Choose action based on current policy with option for deterministic mode"""
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        mu, sigma = self.actor(state_tensor)
        value = self.critic(state_tensor)
        
        # Convert to numpy arrays
        mu = mu[0].numpy()
        sigma = sigma[0].numpy()
        value = value[0, 0].numpy()
        
        if deterministic:
            # Return mean action for deterministic policy
            action = mu
            log_prob = np.zeros_like(mu)  # Placeholder for consistent return format
        else:
            # Sample from normal distribution for stochastic policy
            distribution = tf.compat.v1.distributions.Normal(mu, sigma)
            action = distribution.sample().numpy()
            
            # Compute log probability of action (needed for PPO updates)
            log_prob = distribution.log_prob(action).numpy()
            
            # Apply exploration rate for additional randomness in trading
            if np.random.random() < self.exploration_rate:
                action = np.random.uniform(-1, 1, size=self.action_dim)
        
        # Clip actions to appropriate range for trading
        action = np.clip(action, -1, 1)
        
        return action, log_prob, value
    
    def store_transition(self, state, action, probs, val, reward, done):
        """Store transition in memory for PPO update"""
        self.memory.store(state, action, probs, val, reward, done)
        
        # Also store in experience replay for additional training
        self.experience_buffer.append({
            'state': state, 
            'action': action, 
            'reward': reward, 
            'next_state': None,  # Will be filled by next sample
            'done': done
        })
        
        # Update next_state for previous experience
        if len(self.experience_buffer) > 1:
            self.experience_buffer[-2]['next_state'] = state
    
    def update_policy(self, n_epochs=5):
        """Update policy using PPO algorithm"""
        if len(self.memory) < self.batch_size:
            return None, None  # Not enough samples
            
        # Get data from memory
        states, actions, old_probs, vals, rewards, dones = self.memory.get_arrays()
        
        # Convert to tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        old_probs = tf.convert_to_tensor(old_probs, dtype=tf.float32)
        vals = tf.convert_to_tensor(vals, dtype=tf.float32)
        
        # Calculate advantages
        advantages = self._calculate_advantages(rewards, vals, dones)
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
        
        # Normalize advantages (important for trading data with varying magnitudes)
        advantages = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-10)
        
        # Calculate returns for critic training
        returns = advantages + vals
        
        actor_losses = []
        critic_losses = []
        
        # Train for n_epochs
        for _ in range(n_epochs):
            batches = self.memory.sample_batch()
            for batch in batches:
                with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
                    # Get batch data
                    batch_states = tf.gather(states, batch)
                    batch_actions = tf.gather(actions, batch)
                    batch_old_probs = tf.gather(old_probs, batch)
                    batch_advantages = tf.gather(advantages, batch)
                    batch_returns = tf.gather(returns, batch)
                    
                    # Actor loss calculation
                    mu, sigma = self.actor(batch_states)
                    distribution = tf.compat.v1.distributions.Normal(mu, sigma)
                    new_probs = distribution.log_prob(batch_actions)
                    
                    # Compute ratio and clip
                    ratio = tf.exp(new_probs - batch_old_probs)
                    clipped_ratio = tf.clip_by_value(ratio, 1-self.clip_ratio, 1+self.clip_ratio)
                    
                    # Actor losses: standard PPO objective
                    actor_loss1 = -batch_advantages * ratio
                    actor_loss2 = -batch_advantages * clipped_ratio
                    actor_loss = tf.reduce_mean(tf.maximum(actor_loss1, actor_loss2))
                    
                    # Add entropy bonus for exploration (essential for trading)
                    entropy = tf.reduce_mean(distribution.entropy())
                    actor_loss -= 0.01 * entropy
                    
                    # Critic loss calculation
                    value_pred = self.critic(batch_states)
                    critic_loss = tf.reduce_mean(losses.MSE(batch_returns, value_pred))
                
                # Compute gradients and apply updates
                actor_grads = tape1.gradient(actor_loss, self.actor.trainable_variables)
                critic_grads = tape2.gradient(critic_loss, self.critic.trainable_variables)
                
                # Apply gradient clipping to prevent explosive updates (crucial for volatile market data)
                actor_grads = [tf.clip_by_norm(g, 1.0) for g in actor_grads]
                critic_grads = [tf.clip_by_norm(g, 1.0) for g in critic_grads]
                
                self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
                self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
                
                # Track losses
                actor_losses.append(actor_loss.numpy())
                critic_losses.append(critic_loss.numpy())
        
        # Clear memory after update
        self.memory.clear()
        
        # Decay exploration rate
        self.exploration_rate = max(self.min_exploration_rate, 
                                    self.exploration_rate * self.exploration_decay)
        
        # Track training stats
        self.train_count += 1
        self.training_stats['actor_loss'].append(np.mean(actor_losses))
        self.training_stats['critic_loss'].append(np.mean(critic_losses))
        self.training_stats['exploration_rate'].append(self.exploration_rate)
        
        return np.mean(actor_losses), np.mean(critic_losses)
    
    def train_batch(self, experience_buffer):
        """Process a batch of experiences for training"""
        for exp in experience_buffer:
            state = exp['state']
            action = exp['action']
            reward = exp['reward']
            next_state = exp['next_state']
            done = exp['done']
            
            if next_state is None:
                continue  # Skip incomplete experiences
                
            # Compute value
            value = self.critic(tf.convert_to_tensor([state], dtype=tf.float32))[0, 0].numpy()
            
            # Compute action probabilities (for PPO ratio)
            mu, sigma = self.actor(tf.convert_to_tensor([state], dtype=tf.float32))
            mu = mu[0].numpy()
            sigma = sigma[0].numpy()
            distribution = tf.compat.v1.distributions.Normal(mu, sigma)
            log_prob = distribution.log_prob(action).numpy()
            
            # Store transition
            self.store_transition(state, action, log_prob, value, reward, done)
        
        # Perform policy update
        return self.update_policy()
    
    def _calculate_advantages(self, rewards, values, dones):
        """Calculate advantages using Generalized Advantage Estimation (GAE)"""
        advantages = np.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards) - 1)):
            # Calculate TD error 
            if dones[t]:
                # For terminal states
                delta = rewards[t] - values[t]
                gae = delta
            else:
                # For non-terminal states
                delta = rewards[t] + self.gamma * values[t+1] - values[t]
                gae = delta + self.gamma * self.gae_lambda * gae
                
            advantages[t] = gae
            
        return advantages
    
    def save_models(self, directory='rl_models'):
        """Save both actor and critic models"""
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        actor_path = os.path.join(directory, f"actor_{timestamp}.h5")
        critic_path = os.path.join(directory, f"critic_{timestamp}.h5")
        
        self.actor.save_weights(actor_path)
        self.critic.save_weights(critic_path)
        
        return actor_path, critic_path
    
    def load_models(self, actor_path, critic_path):
        """Load both actor and critic models"""
        self.actor.load_weights(actor_path)
        self.critic.load_weights(critic_path)


# Main RL interface for NQ Alpha Elite
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
        state = np.zeros(self.state_dim)
        
        # Extract and normalize key features from market data
        if market_data is not None:
            # 1-5: Price action features
            state[0] = self._normalize(market_data.get('price', 0), 10000, 30000)
            state[1] = self._normalize(market_data.get('delta', 0), -1, 1)
            state[2] = self._normalize(market_data.get('order_flow', 0), -1, 1)
            state[3] = self._normalize(market_data.get('volatility', 0), 0, 0.001)
            state[4] = self._normalize(market_data.get('trend_strength', 0), 0, 3)
            
            # 6-10: Liquidity and flow features
            state[5] = self._normalize(market_data.get('vpin', 0), 0, 1)
            state[6] = self._normalize(market_data.get('liquidity_score', 0), 0, 1)
            state[7] = market_data.get('institutional_flow', 0)
            state[8] = market_data.get('retail_flow', 0)
            state[9] = market_data.get('market_depth', 0)
        
        # Extract system state features
        if system_state is not None:
            # 11-15: Regime and signal features
            state[10] = self._one_hot_regime(system_state.get('regime', 'unknown'))
            state[11] = system_state.get('regime_confidence', 0)
            state[12] = system_state.get('entanglement', 0)
            state[13] = system_state.get('signal_strength', 0)
            state[14] = system_state.get('confirmation_score', 0)
            
            # 16-20: Position and range features
            state[15] = self._normalize(system_state.get('z_score', 0), -3, 3)
            state[16] = system_state.get('position_in_range', 0)
            state[17] = system_state.get('range_confidence', 0)
            state[18] = system_state.get('hurst_exponent', 0)
            state[19] = system_state.get('trend_direction', 0)
            
            # 21-25: Pattern and edge features
            state[20] = system_state.get('has_flow_pattern', 0)
            state[21] = system_state.get('has_edge_pattern', 0)
            state[22] = system_state.get('pattern_confidence', 0)
            state[23] = system_state.get('pattern_direction', 0)
            state[24] = system_state.get('edge_strength', 0)
            
            # 26-32: Performance and risk features
            state[25] = system_state.get('win_rate', 0)
            state[26] = system_state.get('last_trade_profit', 0)
            state[27] = system_state.get('drawdown', 0)
            state[28] = system_state.get('sharpe', 0)
            state[29] = system_state.get('risk_score', 0)
            state[30] = system_state.get('expectancy', 0)
            state[31] = system_state.get('reward_risk_ratio', 0)
        
        return state
    
    def _normalize(self, value, min_val, max_val):
        """Min-max normalization to [0,1] range"""
        if max_val == min_val:
            return 0.5  # Default to middle value
        return max(0, min(1, (value - min_val) / (max_val - min_val)))
    
    def _one_hot_regime(self, regime):
        """Convert regime to a numerical value"""
        regimes = {'trending_up': 1.0, 'trending_down': -1.0, 
                   'range_bound': 0.5, 'choppy': 0.0, 'volatile': 0.75}
        return regimes.get(regime, 0.25)  # Default to 0.25 for unknown regimes
    
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
        # Action[0]: Signal adjustment factor (-1 to 1)
        signal_adjustment = action[0] * 0.5  # Scale to +/- 50% adjustment
        adjusted_signal = base_signal * (1.0 + signal_adjustment)
        
        # Action[1]: Position sizing adjustment (-1 to 1)
        position_sizing_factor = 1.0 + (action[1] * 0.3)  # Scale to +/- 30% adjustment
        
        # Action[2]: Exit timing adjustment (0-1 range mapped to 0.5-2.0)
        exit_timing_factor = 0.5 + (action[2] + 1.0) * 0.75  # [0.5, 2.0] range
        
        return {
            'adjusted_signal': adjusted_signal,
            'position_sizing_factor': position_sizing_factor,
            'exit_timing_factor': exit_timing_factor
        }
    
    def observe_reward(self, trade_result):
        """Calculate reward based on trade outcome"""
        # Primary reward: P&L (normalized)
        pnl = trade_result.get('profit', 0)
        risk = trade_result.get('risk', 100) + 1e-5  # Avoid division by zero
        normalized_pnl = pnl / risk
        
        # Risk-adjusted reward components
        drawdown_penalty = -0.5 * trade_result.get('max_drawdown_pct', 0) 
        time_efficiency = min(1.0, trade_result.get('hold_time', 0) / trade_result.get('optimal_hold_time', 60))
        
        # Regime alignment bonus
        regime = trade_result.get('regime', 'unknown')
        regime_aligned = 0.2 if (
            (regime == 'trending_up' and pnl > 0 and trade_result.get('direction', 0) > 0) or
            (regime == 'trending_down' and pnl > 0 and trade_result.get('direction', 0) < 0) or
            (regime == 'range_bound' and abs(trade_result.get('z_score', 0)) > 1.5 and pnl > 0)
        ) else 0.0
        
        # Composite reward function
        reward = normalized_pnl + 0.2 * time_efficiency + drawdown_penalty + regime_aligned
        
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
        
        # Update metrics
        self.cumulative_reward += reward
        self.rl_performance['rewards'].append(reward)
        
        # Check if it's time to train the model
        training_occurred = self._try_training()
        
        # Log reward and training status
        if hasattr(self.base_system, 'logger'):
            self.base_system.logger.info(f"RL reward: {reward:.4f}, training occurred: {training_occurred}")
        
        return reward
    
    def _try_training(self):
        """Train the RL model incrementally"""
        if len(self.experience_buffer) >= self.training_frequency:
            try:
                # Train on the buffer
                actor_loss, critic_loss = self.model.train_batch(self.experience_buffer)
                
                # Keep the most recent experiences to prevent catastrophic forgetting
                self.experience_buffer = self.experience_buffer[-int(self.training_frequency/2):]
                
                # Log training results
                if hasattr(self.base_system, 'logger'):
                    self.base_system.logger.info(f"RL model trained: actor_loss={actor_loss:.6f}, critic_loss={critic_loss:.6f}, exploration={self.model.exploration_rate:.4f}")
                
                # Track improvement
                self.episode_count += 1
                self.rl_performance['improvements'].append({
                    'episode': self.episode_count,
                    'actor_loss': actor_loss,
                    'critic_loss': critic_loss,
                    'exploration_rate': self.model.exploration_rate
                })
                
                # Save models periodically (every 5 training sessions)
                if self.episode_count % 5 == 0:
                    self.model.save_models(directory='rl_models')
                    
                return True
            except Exception as e:
                if hasattr(self.base_system, 'logger'):
                    self.base_system.logger.error(f"RL training error: {e}")
                return False
        return False