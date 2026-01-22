from replay_buffer import PriorityExperienceReplay
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import wandb
from datetime import datetime

from actor import Actor
from critic import Critic
from embedding import UserMovieEmbedding
from state_representation import DRRAveStateRepresentation

class DRRAgent:
    
    def __init__(self, env, users_num, items_num, state_size, is_test=False, use_wandb=False,
                 std=1.0, use_per=True):
        
        self.env = env
        self.users_num = users_num
        self.items_num = items_num
        
        self.embedding_dim = 100
        self.actor_hidden_dim = 128
        self.actor_learning_rate = 0.001
        self.critic_hidden_dim = 128
        self.critic_learning_rate = 0.001
        self.discount_factor = 0.9
        self.tau = 0.001

        self.replay_memory_size = 1000000
        self.batch_size = 32
        
        self.std = std          # Độ lệch chuẩn cho exploration (1.0 là best case)
        self.use_per = use_per  # Có dùng Priority Replay hay không

        self.actor = Actor(self.embedding_dim, self.actor_hidden_dim, self.actor_learning_rate, state_size, self.tau)
        self.critic = Critic(self.critic_hidden_dim, self.critic_learning_rate, self.embedding_dim, self.tau)
        
        # Embedding network
        self.embedding_network = UserMovieEmbedding(users_num, items_num, self.embedding_dim)
        self.embedding_network([np.zeros((1,)), np.zeros((1,))]) # Build
        
        # Save directory
        self.save_model_weight_dir = f"./save_model/trail-{datetime.now().strftime('%Y-%m-%d-%H')}"
        os.makedirs(self.save_model_weight_dir, exist_ok=True)
        os.makedirs(os.path.join(self.save_model_weight_dir, 'images'), exist_ok=True)
            
        embedding_save_file_dir = './save_weights/user_movie_embedding_case4.h5'
        if os.path.exists(embedding_save_file_dir):
            self.embedding_network.load_weights(embedding_save_file_dir)
            print(f"Loaded embedding weights from {embedding_save_file_dir}")
        else:
            print(f"Warning: Embedding weight file not found. Using random initialization.")

        self.srm_ave = DRRAveStateRepresentation(self.embedding_dim)
        self.srm_ave([np.zeros((1, 100,)), np.zeros((1, state_size, 100))])

        self.buffer = PriorityExperienceReplay(self.replay_memory_size, self.embedding_dim)
        self.epsilon_for_priority = 1e-6

        # Exploration
        self.epsilon = 1.
        self.epsilon_decay = (self.epsilon - 0.1)/500000
        self.is_test = is_test

        # Logger structure
        self.logs = {
            "step": [],
            "precision": [],
            "reward": [],
            "mean_action": [],
            "q_loss": []
        }

        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(project="drr", entity="diominor", config={'std': std, 'per': use_per})

    def calculate_td_target(self, rewards, q_values, dones):
        y_t = np.copy(q_values)
        for i in range(q_values.shape[0]):
            y_t[i] = rewards[i] + (1 - dones[i])*(self.discount_factor * q_values[i])
        return y_t

    def recommend_item(self, action, recommended_items, top_k=False, items_ids=None):
        if items_ids is None:
            items_ids = np.array(list(set(i for i in range(self.items_num)) - recommended_items))

        items_ebs = self.embedding_network.get_layer('movie_embedding')(items_ids)
        action = tf.transpose(action, perm=(1,0))
        if top_k:
            item_indice = np.argsort(tf.transpose(tf.keras.backend.dot(items_ebs, action), perm=(1,0)))[0][-top_k:]
            return items_ids[item_indice]
        else:
            item_idx = np.argmax(tf.keras.backend.dot(items_ebs, action))
            return items_ids[item_idx]

    def smooth_curve(self, data, window_size=50):
        if len(data) < window_size: return data
        kernel = np.ones(window_size) / window_size
        return np.convolve(data, kernel, mode='valid')

    def plot_recommender_metrics(self, logs, save_path=None):
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        
        def plot_sub(ax, data, title, ylabel):
            smoothed = self.smooth_curve(data, 50)
            x = range(len(data) - len(smoothed), len(data))
            ax.plot(x, smoothed)
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)

        plot_sub(axs[0, 0], self.logs["precision"], "Precision (Smooth)", "Precision (%)")
        plot_sub(axs[0, 1], self.logs["reward"], "Total Reward (Smooth)", "Reward")
        plot_sub(axs[1, 0], self.logs["mean_action"], "Mean Action (Smooth)", "Value")
        plot_sub(axs[1, 1], self.logs["q_loss"], "Q Loss (Smooth)", "Loss")

        for ax in axs.flat: ax.set_xlabel("Episode")
        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=300)
        plt.close()

    def train(self, max_episode_num, top_k=False, load_model=False):
        self.actor.update_target_network()
        self.critic.update_target_network()

        if load_model:
            actor_path = "./save_weights/actor_best.weights.h5" 
            critic_path = "./save_weights/critic_best.weights.h5"
            if os.path.exists(actor_path) and os.path.exists(critic_path):
                self.load_model(actor_path, critic_path)
                print('Completely loaded weights!')
            else:
                print('Pre-trained weights not found, starting from scratch.')

        for episode in range(max_episode_num):
            episode_reward = 0
            correct_count = 0
            steps = 0
            q_loss_sum = 0
            mean_action_sum = 0
            
            user_id, items_ids, done = self.env.reset()
            
            while not done:
                user_eb = self.embedding_network.get_layer('user_embedding')(np.array(user_id))
                items_eb = self.embedding_network.get_layer('movie_embedding')(np.array(items_ids))
                state = self.srm_ave([np.expand_dims(user_eb, axis=0), np.expand_dims(items_eb, axis=0)])

                action = self.actor.network(state)

                if self.epsilon > np.random.uniform() and not self.is_test:
                    self.epsilon -= self.epsilon_decay
                    # Sử dụng self.std được truyền vào từ config
                    action += np.random.normal(0, self.std, size=action.shape)

                recommended_item = self.recommend_item(action, self.env.recommended_items, top_k=top_k)
                next_items_ids, reward, done, _ = self.env.step(recommended_item, top_k=top_k)
                if top_k: reward = np.sum(reward)

                next_items_eb = self.embedding_network.get_layer('movie_embedding')(np.array(next_items_ids))
                next_state = self.srm_ave([np.expand_dims(user_eb, axis=0), np.expand_dims(next_items_eb, axis=0)])

                self.buffer.append(state, action, reward, next_state, done)
                
                if self.buffer.crt_idx > 1 or self.buffer.is_full:
                    batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones, weight_batch, index_batch = self.buffer.sample(self.batch_size)

                    target_next_action= self.actor.target_network(batch_next_states)
                    qs = self.critic.network([target_next_action, batch_next_states])
                    target_qs = self.critic.target_network([target_next_action, batch_next_states])
                    min_qs = tf.raw_ops.Min(input=tf.concat([target_qs, qs], axis=1), axis=1, keep_dims=True)
                    td_targets = self.calculate_td_target(batch_rewards, min_qs, batch_dones)
        
                    if self.use_per:
                        for (p, i) in zip(td_targets, index_batch):
                            self.buffer.update_priority(abs(p[0]) + self.epsilon_for_priority, i)

                    loss = self.critic.train([batch_actions, batch_states], td_targets, weight_batch)
                    q_loss_sum += loss
                    
                    s_grads = self.critic.dq_da([batch_actions, batch_states])
                    self.actor.train(batch_states, s_grads)
                    self.actor.update_target_network()
                    self.critic.update_target_network()

                items_ids = next_items_ids
                episode_reward += reward
                mean_action_sum += np.sum(action[0])/(len(action[0]))
                steps += 1
                if reward > 0: correct_count += 1

            precision = (correct_count / steps) * 100
            avg_q_loss = q_loss_sum / steps if steps > 0 else 0
            avg_mean_action = mean_action_sum / steps if steps > 0 else 0

            self.logs["step"].append(episode)
            self.logs["precision"].append(precision)
            self.logs["reward"].append(episode_reward)
            self.logs["mean_action"].append(avg_mean_action)
            self.logs["q_loss"].append(avg_q_loss)

            print(f'Ep {episode}/{max_episode_num} | Prec: {precision:.2f}% | Rew: {episode_reward:.2f} | Loss: {avg_q_loss:.4f} | Eps: {self.epsilon:.3f}')

            if (episode+1) % 100 == 0:
                self.plot_recommender_metrics(self.logs, save_path=os.path.join(self.save_model_weight_dir, "images", "progress.png"))

            if (episode+1) % 1000 == 0:
                self.save_model(os.path.join(self.save_model_weight_dir, f'actor_{episode+1}.weights.h5'),
                                os.path.join(self.save_model_weight_dir, f'critic_{episode+1}.weights.h5'))
        
        return self.logs

    def save_model(self, actor_path, critic_path):
        self.actor.save_weights(actor_path)
        self.critic.save_weights(critic_path)
        
    def load_model(self, actor_path, critic_path):
        self.actor.load_weights(actor_path)
        self.critic.load_weights(critic_path)