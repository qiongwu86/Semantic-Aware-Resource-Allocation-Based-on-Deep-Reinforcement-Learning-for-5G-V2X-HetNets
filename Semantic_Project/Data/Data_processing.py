import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'Times New Roman'

def moving_average(data, window_size):
    weights = np.repeat(1.0, window_size) / window_size
    smoothed_data = np.convolve(data, weights, 'valid')
    return smoothed_data

n_episode = 1000
window_size1 = 2
window_size2 = 5
#phase_reward_ddpg = moving_average(np.load('Phase_Reward_DDPG_1000.npy'), window_size1)
#reward_maddpg = moving_average(np.load('3-BCD_Reward_MADDPG_1000.npy'), window_size1)
reward_ddpg = moving_average(np.load('reward_ddpg1000.npy'), window_size1)
reward_sac = moving_average(np.load('reward_td3_1000.npy'), window_size1)
reward_ppo = moving_average(np.load('reward_ppo_1000.npy'), window_size1)
x1 =np.linspace(0,n_episode, n_episode-1, dtype=int)
x2 =np.linspace(0,n_episode, n_episode-4, dtype=int)

plt.figure(1)
plt.plot(x1, reward_ppo, label='SARADC')
plt.plot(x1, reward_sac, label='TD3')
plt.plot(x1, reward_ddpg, label='DDPG')
plt.grid(True, linestyle='-', linewidth=0.5)
# plt.yticks(y)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend(loc='lower right', fontsize=12)
plt.savefig('D:\QKW\Reward.pdf', dpi=300, format='pdf')
plt.show()