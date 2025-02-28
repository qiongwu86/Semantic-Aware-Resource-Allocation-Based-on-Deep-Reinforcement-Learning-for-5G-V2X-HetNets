from Environment_SC import environment
from ppo import PPO
from ppo import Memory
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

mat_data = scipy.io.loadmat('sem_table.mat')
# 加载 'new_data_i_need.csv' 文件
table_data = mat_data['sem_table']

#-----------------------------------------------Training---------------------------------------------------------------------------
n_episode = 50
n_step = 100
n_agent = 5

n_Macro = 1  # large station
n_Micro = 2  # small station

n_RB = 12
n_state = 5 #多一个大状态
n_action = 1 + (2*n_RB) + 1+ 1
n_actions = n_agent * (1 + (2*n_RB) + 1+ 1) #多一个动作
max_power_Macro = 30 # Vehicle maximum power is 1 watt
max_power_Micro = 30
n_mode = 2 # Macro/Micro mode
n_BS = n_Macro + n_Micro
n_RB_Macro = n_RB
n_RB_Micro = n_RB
size_packet = 1000
u = 10
semantic_size_packet = size_packet / u
BW = 15 #KHz

##################################################################
update_timestep = 5 # update policy every n timesteps
action_std = 0.5  # constant std for action distribution (Multivariate Normal)
K_epochs = 1  # update policy for K epochs
eps_clip = 0.2  # clip parameter for PPO
gamma = 0.99  # discount factor

lr = 0.0001  # parameters for Adam optimizer
betas = (0.9, 0.999)

# --------------------------------------------------------------
memory = Memory()
agent = PPO(n_agent*n_state, n_actions, action_std, lr, betas, gamma, K_epochs, eps_clip)

label = 'model/ppo'
model_path = label + '/agent'

env = environment(n_state=n_state,n_agent=n_agent,n_RB=n_RB)

i_episode_matrix = np.zeros ([n_episode], dtype=np.int16)
reward_per_episode = np.zeros ([n_episode], dtype=np.float16)
reward_mean_all_episode = np.zeros([n_episode], dtype=np.float16)
#----------bit----------------------------------------------------
# rate_mean_all_episode = np.zeros([n_episode], dtype=np.float16)
# rate_level_mean_all = np.zeros([n_episode], dtype=np.float16)
# WiFi_level_mean_all = np.zeros([n_episode], dtype=np.float16)
#----------suit---------------------------------------------------
semantic_rate_mean_all_episode=np.zeros([n_episode], dtype=np.float16)
semantic_rate_level_mean_all=np.zeros([n_episode], dtype=np.float16)
semantic_WiFi_level_mean_all=np.zeros([n_episode], dtype=np.float16)
semantic_HSSE_mean_all=np.zeros([n_episode], dtype=np.float16)
semantic_WiFi_HSSE_mean_all=np.zeros([n_episode], dtype=np.float16)
#----------bit----------------------------------------------------
# rate_per_episode = np.zeros ([n_agent,n_episode], dtype=np.float16)
# rate_level_per_episode = np.zeros ([n_agent,n_episode], dtype=np.float16)
# WiFi_level_per_episode = np.zeros ([n_agent,n_episode], dtype=np.float16)
#----------suit---------------------------------------------------
semantic_rate_per_episode = np.zeros ([n_agent,n_episode], dtype=np.float16)
semantic_rate_level_per_episode = np.zeros ([n_agent,n_episode], dtype=np.float16)
semantic_WiFi_level_per_episode = np.zeros ([n_agent,n_episode], dtype=np.float16)
semantic_HSSE_per_episode = np.zeros ([n_agent,n_episode], dtype=np.float16)
semantic_WiFi_HSSE_per_episode = np.zeros ([n_agent,n_episode], dtype=np.float16)

agent.load_model(model_path)
#reward_mem = []
#lets go(Start episode)---------------------------------------------------------------------------------------------------
for i_episode in range(n_episode):
    # packet_done = np.zeros([n_agent,n_step], dtype=np.int32)
    # ----------suit---------------------------------------------------
    semantic_packet_done = np.zeros([n_agent,n_step], dtype=np.int32)
    i_episode_matrix[i_episode] = i_episode
    #initialize parameters------------------------------------------------------------------------------
    state = np.zeros([n_agent,n_state], dtype=np.float16)
    new_state = np.zeros([n_agent,n_state], dtype=np.float16)
    RB_Micro = np.zeros([n_Micro,n_RB_Micro], dtype=np.int16)
    RB_Macro = np.zeros([n_RB_Macro], dtype=np.int16)
    veh_Micro = np.zeros([n_agent,n_step], dtype=np.int32)
    veh_Macro = np.zeros([n_agent,n_step], dtype=np.int32)
    veh_RB_power = np.zeros([n_agent,n_RB,n_step])
    veh_RB = np.zeros([n_agent,n_RB,n_step],dtype=np.int16)
    veh_num_BS = np.zeros([n_agent,2], dtype=np.int32)
    duty = np.zeros([n_agent], dtype=np.float16)
    n_duty = np.zeros([n_agent,n_step], dtype=np.int32)
    n_duty_chu = np.zeros([n_agent, n_step], dtype=np.float16)

    semantic_similarity= np.zeros([n_agent], dtype=np.float64)

    symbol_p_word = np.zeros([n_agent], dtype=np.int32)
    n_symbol_p_word = np.zeros([n_agent,n_step], dtype=np.int32)

    veh_sinr = np.zeros([n_agent,n_step], dtype=np.int32)
    symbol_lenghth= np.zeros([n_agent], dtype=np.int32) #不知道要不要加

    i_step_matrix = np.zeros ([n_step], dtype=np.int16)
    reward_per_step = np.zeros ([n_step], dtype=np.float16)
    # rate_per_step = np.zeros ([n_agent,n_step], dtype=np.float16)
    semantic_rate_per_step = np.zeros ([n_agent,n_step], dtype=np.float16)
    AoI_veh = np.ones([n_agent], dtype=np.int64)*100
    AoI_WiFi = np.ones([n_agent], dtype=np.int64)*100
    veh_BS_allocate = np.zeros([n_agent], dtype=np.int32)
    veh_gain = np.zeros([n_agent,n_step], dtype = np.float16)
    veh_BS = np.zeros([n_agent,n_step], dtype=np.int32)
    # veh_flag = np.zeros([n_agent], dtype=np.int32)
    semantic_veh_flag = np.zeros([n_agent], dtype=np.int32)
    # ----------bit----------------------------------------------------
    # veh_data = np.zeros([n_agent,n_step], dtype = np.float16)
    # veh_data[:,0] = size_packet
    # ----------suit---------------------------------------------------
    semantic_veh_data = np.zeros([n_agent, n_step], dtype=np.float16)
    semantic_veh_data[:, 0] = semantic_size_packet
    # ----------bit----------------------------------------------------
    # WiFi_level = np.zeros([n_agent,n_step])
    # WiFi_rate = np.zeros([n_agent,n_step])
    # rate_level = np.zeros([n_agent,n_step])
    # ----------suit---------------------------------------------------
    semantic_WiFi_level = np.zeros([n_agent, n_step])
    semantic_WiFi_rate = np.zeros([n_agent, n_step])
    semantic_rate_level = np.zeros([n_agent, n_step])
    semantic_HSSE = np.zeros([n_agent, n_step])
    semantic_WiFi_HSSE = np.zeros([n_agent, n_step])

    #make start Environment------------------------------------------------------------------------------

    veh_BS_start = np.zeros(n_agent)
    for i in range(n_agent):
        veh_BS_start[i] = np.random.randint(0, 2)
    veh_RB_start = np.zeros([n_agent, n_RB])
    veh_RB_power_start = np.zeros([n_agent, n_RB])
    for i in range(n_agent):
        for j in range(n_RB):
            veh_RB_start[i, j] = np.random.randint(0, 2)
            veh_RB_power_start[i, j] = np.random.randint(1, 30)
    for i_agent in range(n_agent):
        state[i_agent], veh_num_BS[i_agent], duty[i_agent], symbol_p_word[i_agent] = env.make_start(i_agent, veh_BS_start, veh_RB_start, veh_RB_power_start)
    #Start step-----------------------------------------------------------------------------------------
    for i_step in range(n_step):
        action = np.zeros([n_agent,n_action], dtype=np.float16)
        reward = np.zeros ([0], dtype=np.float16)
        veh_RB_BS = np.zeros([n_agent,n_RB,n_BS],dtype=np.int16)
        i_step_matrix[i_step] = i_step
        env.mobility_veh()
        i_RB_Macro = 0
        i_RB_Micro = np.zeros([n_Micro], dtype=np.int32)

        #记录每个代理一共传输完成了多少个包
        for i_agent in range(n_agent):
            # packet_done[i_agent,i_step] = 0
            semantic_packet_done[i_agent,i_step] = 0
            veh_gain[i_agent,i_step] = state[i_agent,0]
            veh_sinr[i_agent,i_step] = state[i_agent, 4]
            # WiFi_rate[i_agent,i_step] = np.random.uniform(6,12)
            semantic_WiFi_rate[i_agent,i_step] = np.random.uniform(6,12) / u
            # symbol_p_word[i_agent,i_step] = np.random.randint(1, 20)
            if i_step !=0 :
                # veh_data[i_agent,i_step] = veh_data[i_agent,i_step-1] - (n_duty[i_agent,i_step] * (rate_per_step[i_agent,i_step-1]))
                semantic_veh_data[i_agent, i_step] = semantic_veh_data[i_agent, i_step - 1] - (n_duty[i_agent, i_step] * (semantic_rate_per_step[i_agent, i_step - 1]))
                # if veh_data[i_agent,i_step] <= 0 :
                #     veh_data[i_agent,i_step] = size_packet
                #     veh_flag[i_agent] += 1
                #     packet_done[i_agent,i_step] = 1
                if semantic_veh_data[i_agent,i_step] <= 0 :
                    semantic_veh_data[i_agent, i_step] = semantic_size_packet
                    semantic_veh_flag[i_agent] += 1
                    semantic_packet_done[i_agent, i_step] = 1
        #state process and reshape------------------------------------------------------------
        state_shape = np.reshape(state,(1,n_state*n_agent))
        if np.round(np.ndarray.max(state_shape)) == 0:
            state_shape = np.zeros([1,n_state*n_agent])
        else:
            state_shape = state_shape / np.ndarray.max(state_shape)
        #action process-----------------------------------------------------------------
        action_choose = agent.select_action(np.asarray(state_shape).flatten(), memory)
        action_choose = np.clip(action_choose, 0.000, 0.999)
        for i_agent in range(n_agent):
            #Allocation-------------------------------------------------------------------
            if veh_num_BS[i_agent,1] == -1:
                veh_Micro[i_agent,i_step] = veh_num_BS[i_agent,0] # 如果车辆没有连接到微基站，那么将 veh_Micro[i_agent,i_step] 的值设置为 veh_num_BS[i_agent,0]，即车辆连接的宏蜂窝基站的数量但其实是0，表示车辆没有连接到微蜂窝基站
            if veh_num_BS[i_agent,0] == -1:
                veh_Macro[i_agent,i_step] = veh_num_BS[i_agent,1] #
            #BS & RB & duty-cycle & symbol/word allocation --------------------------------------------------------
            action[i_agent,0] = int((action_choose[0+i_agent*n_action]) * n_mode) # chosen type of BS
            if action[i_agent,0] == 0 : #Allocate to Micro
                veh_BS[i_agent,i_step] = 0 #1.（状态） 记录跟哪一个基站相连
                for i in range(1,n_RB+1): #Allocation RB
                    action[i_agent,i] = int((action_choose[i+i_agent*n_action]) * 2)
                    veh_RB[i_agent,i-1,i_step] = action[i_agent,i] #2.（状态） 记录RB的状态值
                for i in range(n_RB+1,n_RB+n_RB+1): #Allocation Power
                    action[i_agent,i] = np.round(np.clip(action_choose[i+(i_agent*n_action)] * max_power_Micro, 1, max_power_Micro))  # power selected by veh
                    veh_RB_power[i_agent,i-(n_RB+1),i_step] =  action[i_agent,i] # 3.（状态） 记录功率值

                action[i_agent,n_RB+n_RB+1] = action_choose[n_RB+n_RB+1+(i_agent*n_action)] #Duty-cycle
                duty[i_agent] = action[i_agent,n_RB+n_RB+1] #4.（状态） 记录占空比
                # WiFi_level[i_agent,i_step] = (1 - duty[i_agent]) * (WiFi_rate[i_agent,i_step])
                semantic_WiFi_level[i_agent, i_step] = (1 - duty[i_agent]) * (semantic_WiFi_rate[i_agent, i_step])
                i_RB_Micro[veh_Micro[i_agent]] += 1

                action[i_agent, n_RB + n_RB + 1 + 1] = np.round(np.clip(action_choose[n_RB + n_RB + 1 + 1 + (i_agent * n_action)] * 20, 1, 20))
                symbol_lenghth[i_agent]= action[i_agent, n_RB + n_RB + 1 + 1]
                # 边界检查
                if veh_sinr[i_agent, i_step] > 20:
                    veh_sinr[i_agent, i_step] = 20
                elif veh_sinr[i_agent, i_step] < -10:
                    veh_sinr[i_agent, i_step] = -10

                semantic_similarity[i_agent] = table_data[symbol_lenghth[i_agent] - 1, veh_sinr[i_agent, i_step] + 10]


            elif action[i_agent,0] == 1 : #Allocate to Macro
                veh_BS[i_agent,i_step] = 1
                for i in range(1,n_RB+1): #Allocation RB
                    action[i_agent,i] = int((action_choose[i+i_agent*n_action]) * 2)
                    veh_RB[i_agent,i-1,i_step] = action[i_agent,i]
                for i in range(n_RB+1,n_RB+n_RB+1): #Allocation Power
                    action[i_agent,i] = np.round(np.clip(action_choose[i+i_agent*n_action] * max_power_Macro, 1, max_power_Macro))  # power selected by veh
                    veh_RB_power[i_agent,i-(n_RB+1),i_step] =  action[i_agent,i]

                action[i_agent,n_RB+n_RB+1] = 1 #Duty-cycle
                duty[i_agent] = action[i_agent,n_RB+n_RB+1]
                # WiFi_level[i_agent,i_step] = WiFi_rate[i_agent,i_step]
                semantic_WiFi_level[i_agent, i_step] = semantic_WiFi_rate[i_agent, i_step]
                i_RB_Macro +=1

                action[i_agent, n_RB + n_RB + 1 + 1] = np.round(np.clip(action_choose[n_RB + n_RB + 1 + 1 + (i_agent * n_action)] * 20, 1, 20))
                symbol_lenghth[i_agent] = action[i_agent, n_RB + n_RB + 1 + 1]
                # 边界检查
                if veh_sinr[i_agent, i_step] > 20:
                    veh_sinr[i_agent, i_step] = 20
                elif veh_sinr[i_agent, i_step] < -10:
                    veh_sinr[i_agent, i_step] = -10

                semantic_similarity[i_agent] = table_data[symbol_lenghth[i_agent] - 1, veh_sinr[i_agent, i_step] + 10]

        #Check Constrain---------------------------------------------------------------------------------------------
        veh_RB_BS = env.RB_BS_allocate(veh_RB,veh_RB_BS,veh_BS,i_step)
        #veh_RB（车辆的资源块分布）、veh_RB_BS（车辆连接的基站信息）、veh_BS（车辆与基站的连接状态）、i_step（当前时间步）
        #veh_RB = env.check_constrain(veh_RB,veh_RB_BS,i_step)

        #Calculate parameters---------------------------------------------------------------------------------------------
        for i_agent in range(n_agent):
            # rate_per_step[i_agent,i_step] = env.compute_rate(veh_RB_power,veh_BS,veh_RB,WiFi_level,i_agent,i_step)
            # rate_level[i_agent,i_step] = duty[i_agent] * (rate_per_step[i_agent,i_step])
            semantic_rate_level[i_agent, i_step] = (semantic_similarity[i_agent] / symbol_lenghth[i_agent]) * BW *duty[i_agent]*1000#khz HSSE
            # AoI_veh[i_agent], AoI_WiFi[i_agent] = env.Age_of_information(WiFi_level[i_agent,i_step],rate_level[i_agent,i_step])
            n_duty[i_agent,i_step] = np.round(duty[i_agent] * n_step)
            semantic_HSSE[i_agent, i_step] = (semantic_similarity[i_agent] / symbol_lenghth[i_agent])
            semantic_WiFi_HSSE = semantic_WiFi_rate[i_agent, i_step]/ symbol_lenghth[i_agent]


        reward = env.get_reward_sc(np.mean(semantic_WiFi_level[:, i_step]), np.mean(semantic_rate_level[:, i_step]),
                                   veh_RB, veh_RB_BS, veh_Micro, i_step)
        #new state process and reshape-------------------------------------------------------------------------
        for i_agent in range (n_agent):
            new_state[i_agent], veh_num_BS[i_agent] = env.get_state(veh_RB_power,veh_BS,veh_RB,semantic_WiFi_rate,i_agent,i_step)
        new_state_shape = np.reshape(new_state,(1,n_state*n_agent))
        if np.round(np.ndarray.max(new_state_shape)) == 0:
            new_state_shape = np.zeros([1,n_state*n_agent])
        else:
            new_state_shape = new_state_shape / np.ndarray.max(new_state_shape)

        reward_per_step[i_step] = reward

        state = new_state.copy()
#plot process-------------------------------------------------------------------------------------
    reward_per_episode[i_episode] = np.mean(reward_per_step[:])

    print('episode:', i_episode, ' reward:', np.mean(reward_per_step))

    for i_agent in range(n_agent):
        semantic_WiFi_level_per_episode[i_agent, i_episode] = np.mean(semantic_WiFi_level[i_agent, :])
        semantic_rate_level_per_episode[i_agent, i_episode] = np.mean(semantic_rate_level[i_agent, :])
        semantic_HSSE_per_episode[i_agent, i_episode] = np.mean(semantic_HSSE[i_agent, :])
        semantic_WiFi_HSSE_per_episode[i_agent, i_episode] = np.mean(semantic_WiFi_HSSE[i_agent, :])

for i_episode in range(n_episode) :
    semantic_rate_level_mean_all[i_episode] = np.mean(semantic_rate_level_per_episode[:, i_episode])
    semantic_WiFi_level_mean_all[i_episode] = np.mean(semantic_WiFi_level_per_episode[:, i_episode])
    semantic_HSSE_mean_all[i_episode] = np.mean(semantic_WiFi_level_per_episode[:, i_episode])
    semantic_WiFi_HSSE_mean_all[i_episode] = np.mean(semantic_WiFi_level_per_episode[:, i_episode])



plt.plot(i_episode_matrix,reward_per_episode)
plt.show()