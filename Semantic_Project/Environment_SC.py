import numpy as np
import math

import scipy.io

# ------------------------------------------------------------------------------------------------------------------------
np.random.seed(123)

class environment:
    def __init__(self, n_state, n_agent, n_RB):
        self.n_state = n_state
        self.n_agent = n_agent
        self.n_RB = n_RB
        self.Macro_position = [[500, 500]]  # center of Fig
        #self.Micro_position = [[250, 250]]
        self.Micro_position = [[250, 250],[750, 750]]
        self.n_macro = len(self.Macro_position)
        self.n_micro = len(self.Micro_position)
        self.n_BS = self.n_macro + self.n_micro
        self.h_bs = 25
        self.h_ms = 1.5
        self.shadow_std = 8
        self.Decorrelation_distance = 50
        self.time_slow = 0.1
        self.velocity = 36  # km/h
        self.V2I_Shadowing = np.random.normal(0, 8, 1)
        self.delta_distance = self.velocity * self.time_slow
        self.veh_power = np.zeros([n_agent])
        self.sig2_dbm = -84
        self.sig2 = 10 ** (self.sig2_dbm - 30 / 10)
        self.BW = 15  # KHz

        self.bsAntGain = 8
        self.bsNoiseFigure = 5
        self.vehAntGain = 3
        self.vehNoiseFigure = 9
        self.u = 10
        self.WiFi_level_min = 6  # 6MB/s
        self.semantic_WiFi_level_min = 6 / self.u
        self.rate_level_min = 100  # 100KB/s
        self.semantic_rate_level_min = 100 / self.u
        self.V2I_signal = np.zeros([n_agent])
        self.interference = np.zeros([n_agent])
        self.pos_veh = np.zeros([n_agent, 2], dtype=np.float16)  # position_vehicle[x,y]
        self.dir_veh = np.zeros([n_agent])  # direction of vehicle
        self.veh_BS_allocate = np.zeros([n_agent], dtype=np.int32)
        self.veh_RB = np.zeros([n_agent, n_RB], dtype=np.int32)
        self.veh_num_BS = np.zeros([n_agent, 2], dtype=np.int32)
        self.state = np.zeros([self.n_agent, self.n_state], dtype=np.float16)
        self.new_state = np.zeros([self.n_agent, self.n_state], dtype=np.float16)
        self.duty = np.zeros([n_agent], dtype=np.float16)

    def macro_allocate(self, position_veh):
        n_BS = self.n_macro
        dis_all = np.zeros([n_BS])
        for i_BS in range(n_BS):
            Macro_position = self.Macro_position[i_BS]
            d1 = abs(position_veh[0] - Macro_position[0])
            d2 = abs(position_veh[1] - Macro_position[1])
            dis_all[i_BS] = math.hypot(d1, d2)  # 两个数的平方和的平方根

        return np.argmin(dis_all)

    def micro_allocate(self, position_veh):
        n_BS = self.n_micro
        dis_all = np.zeros([n_BS])
        for i_BS in range(n_BS):
            Micro_position = self.Micro_position[i_BS]
            d1 = abs(position_veh[0] - Micro_position[0])
            d2 = abs(position_veh[1] - Micro_position[1])
            dis_all[i_BS] = math.hypot(d1, d2)

        return np.argmin(dis_all)

    def get_path_loss_Macro(self, position_veh, i_macro):
        Macro_position = self.Macro_position[i_macro]
        d1 = abs(position_veh[0] - Macro_position[0])
        d2 = abs(position_veh[1] - Macro_position[1])
        distance = math.hypot(d1, d2)
        r = math.sqrt((distance ** 2) + ((self.h_bs - self.h_ms) ** 2)) / 1000  # 估算信号在空间中传播的衰减情况 km
        if r < 25: r = 25
        Loss = 128.1 + 37.6 * np.log10(r)
        return Loss

    def get_path_loss_Micro(self, position_veh, i_micro):
        Micro_position = self.Micro_position[i_micro]
        d1 = abs(position_veh[0] - Micro_position[0])
        d2 = abs(position_veh[1] - Micro_position[1])
        distance = math.hypot(d1, d2)
        distance = math.hypot(d1, d2)
        r = math.sqrt((distance ** 2) + ((self.h_bs - self.h_ms) ** 2)) / 1000
        if r < 25: r = 25
        Loss = 128.1 + 37.6 * np.log10(r)
        return Loss

    def get_shadowing(self, delta_distance, shadowing):
        self.R = np.sqrt(0.5 * np.ones([1, 1]) + 0.5 * np.identity(1))
        return np.multiply(np.exp(-1 * (delta_distance / self.Decorrelation_distance)), shadowing) \
               + np.sqrt(1 - np.exp(-2 * (delta_distance / self.Decorrelation_distance))) * np.random.normal(0, 8,
                                                                                                             1)  # فرshadowings

    def get_reward(self, WiFi_level, rate_level, veh_RB, veh_RB_BS, veh_micro, i_step):
        reward = 0
        for i_BS in range(self.n_BS):
            for i_RB in range(self.n_RB):
                if np.sum(veh_RB_BS[:, i_RB, i_BS]) > 1:
                    reward += -10
        if WiFi_level > self.WiFi_level_min: reward += (rate_level / self.rate_level_min)
        return reward

    def get_reward_sc(self, semantic_WiFi_level, semantic_rate_level, veh_RB, veh_RB_BS, veh_micro, i_step):
        reward = 0
        for i_BS in range(self.n_BS):
            for i_RB in range(self.n_RB):
                if np.sum(veh_RB_BS[:, i_RB, i_BS]) > 1:  # 0检查每个资源块是否被多个车辆共享。如果某个资源块被多个车辆共享，将 reward 减少 10
                    reward += -10

        if semantic_WiFi_level/(self.BW*1000) > self.semantic_WiFi_level_min:
            reward += (semantic_rate_level / (self.semantic_rate_level_min * 10))
        return reward

    def get_state(self, veh_RB_power, veh_BS, veh_RB, WiFi_rate, i_agent, i_step):
        if veh_BS[i_agent, i_step] == 1:
            i_macro = environment.macro_allocate(self, self.pos_veh[i_agent])
            i_micro = 0
            self.state[i_agent, 0] = environment.get_path_loss_Macro(self, self.pos_veh[i_agent], i_macro)
            self.veh_num_BS[i_agent, 0] = -1
            self.veh_num_BS[i_agent, 1] = i_macro
        else:
            i_micro = environment.micro_allocate(self, self.pos_veh[i_agent])
            i_macro = 0
            self.state[i_agent, 0] = environment.get_path_loss_Micro(self, self.pos_veh[i_agent], i_micro)
            self.veh_num_BS[i_agent, 0] = i_micro
            self.veh_num_BS[i_agent, 1] = -1

        self.state[i_agent, 1] = environment.compute_rate(self, veh_RB_power, veh_BS, veh_RB,
                                                          i_agent, i_step)
        self.state[i_agent, 2] = environment.get_interference(self, veh_RB_power, veh_BS, veh_RB, i_macro, i_micro,
                                                              i_agent, i_step) * (10 ** 16)
        self.state[i_agent, 3] = WiFi_rate[i_agent, i_step]
        self.state[i_agent, 4] = environment.compute_sinr(self, veh_RB_power, veh_BS, veh_RB,
                                                          i_agent, i_step)

        return self.state[i_agent], self.veh_num_BS[i_agent]

    def compute_rate(self, veh_RB_power, veh_BS, veh_RB, i_agent, i_step):
        self.V2I_signal[i_agent] = 0
        if veh_BS[i_agent, i_step] == 1:
            i_macro = environment.macro_allocate(self, self.pos_veh[i_agent])
            i_micro = 0
            veh_gain = environment.get_path_loss_Macro(self, self.pos_veh[i_agent], i_macro)
            for i_RB in range(self.n_RB):
                self.V2I_signal[i_agent] += (veh_RB[i_agent, i_RB, i_step]) * (10 ** ((veh_RB_power[
                                                                                           i_agent, i_RB, i_step] - veh_gain + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10))
            veh_rate = self.BW * np.log2(1 + np.divide(self.V2I_signal[i_agent], (
                        environment.get_interference(self, veh_RB_power, veh_BS, veh_RB, i_macro, i_micro, i_agent,
                                                     i_step) + self.sig2)))

        elif veh_BS[i_agent, i_step] == 0:
            i_micro = environment.micro_allocate(self, self.pos_veh[i_agent])
            i_macro = 0
            veh_gain = environment.get_path_loss_Micro(self, self.pos_veh[i_agent], i_micro)
            for i_RB in range(self.n_RB):
                self.V2I_signal[i_agent] += (veh_RB[i_agent, i_RB, i_step]) * (10 ** ((veh_RB_power[
                                                                                           i_agent, i_RB, i_step] - veh_gain + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10))
            veh_rate = self.BW * np.log2(1 + np.divide(self.V2I_signal[i_agent], (
                        environment.get_interference(self, veh_RB_power, veh_BS, veh_RB, i_macro, i_micro, i_agent,
                                                     i_step) + self.sig2)))

        return veh_rate

    def get_interference(self, veh_RB_power, veh_BS, veh_RB, i_macro, i_micro, i_agent, i_step):
        self.interference[i_agent] = 0
        if veh_BS[i_agent, i_step] == 1:
            for i_agent_plus in range(self.n_agent):
                if i_agent_plus == i_agent: continue
                if veh_BS[i_agent_plus, i_step - 1] == 1:
                    for i_RB in range(self.n_RB):
                        self.interference[i_agent] += veh_RB[i_agent_plus, i_RB, i_step - 1] * veh_RB[
                            i_agent, i_RB, i_step - 1] * (10 ** ((veh_RB_power[
                                                                      i_agent_plus, i_RB, i_step - 1] - environment.get_path_loss_Macro(
                            self, self.pos_veh[i_agent_plus],
                            i_macro) + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10))
        if veh_BS[i_agent, i_step] == 0:
            for i_agent_plus in range(self.n_agent):
                if i_agent_plus == i_agent: continue
                if veh_BS[i_agent_plus, i_step - 1] == 0:
                    for i_RB in range(self.n_RB):
                        self.interference[i_agent] += veh_RB[i_agent_plus, i_RB, i_step - 1] * veh_RB[
                            i_agent, i_RB, i_step - 1] * (10 ** ((veh_RB_power[
                                                                      i_agent_plus, i_RB, i_step - 1] - environment.get_path_loss_Micro(
                            self, self.pos_veh[i_agent_plus],
                            i_micro) + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10))

        return self.interference[i_agent]

    def get_interference_macro(self, i_agent, veh_power_start, veh_BS_start, veh_RB_start, i_macro):
        self.interference[i_agent] = 0
        for i_agent_plus in range(self.n_agent):
            if i_agent_plus == i_agent: continue
            if veh_BS_start[i_agent_plus] == 1:
                for i_RB in range(self.n_RB):
                    self.interference[i_agent] += veh_RB_start[i_agent_plus, i_RB] * veh_RB_start[
                        i_agent, i_RB] * (10 ** ((veh_power_start[i_agent_plus, i_RB] - environment.get_path_loss_Macro(
                        self, self.pos_veh[i_agent_plus], i_macro) + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10))
        return self.interference[i_agent]

    def get_interference_micro(self, i_agent, veh_power_start, veh_BS_start, veh_RB_start, i_micro):
        self.interference[i_agent] = 0
        for i_agent_plus in range(self.n_agent):
            if i_agent_plus == i_agent: continue
            if veh_BS_start[i_agent_plus] == 0:
                for i_RB in range(self.n_RB):
                    self.interference[i_agent] += veh_RB_start[i_agent_plus, i_RB] * veh_RB_start[
                        i_agent, i_RB] * (10 ** ((veh_power_start[i_agent_plus, i_RB] - environment.get_path_loss_Micro(
                        self, self.pos_veh[i_agent_plus], i_micro) + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10))
        return self.interference[i_agent]

    def compute_sinr(self, veh_RB_power, veh_BS, veh_RB, i_agent, i_step):
        self.V2I_signal[i_agent] = 0
        if veh_BS[i_agent, i_step] == 1:
            i_macro = environment.macro_allocate(self, self.pos_veh[i_agent])
            i_micro = 0
            veh_gain = environment.get_path_loss_Macro(self, self.pos_veh[i_agent], i_macro)
            for i_RB in range(self.n_RB):
                self.V2I_signal[i_agent] += (veh_RB[i_agent, i_RB, i_step]) * (10 ** ((veh_RB_power[
                                                                                           i_agent, i_RB, i_step] - veh_gain + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10))
            # 输入信号、干扰和噪声功率，以 dBm 为单位
            signal_power_dBm = self.V2I_signal[i_agent]  # 信号功率
            interference_power_dBm = environment.get_interference(self, veh_RB_power, veh_BS, veh_RB, i_macro, i_micro,
                                                                  i_agent, i_step)  # 干扰功率
            noise_power_dBm = self.sig2  # 噪声功率
            # 计算信干噪比（SINR）
            sinr = environment.calculate_sinr(self, signal_power_dBm, interference_power_dBm, noise_power_dBm)
            # 将 SINR 转换为分贝单位
            sinr_dB = environment.linear_to_db(self, sinr)
            # print("信干噪比（SINR）为：", sinr_dB, "dB")

        elif veh_BS[i_agent, i_step] == 0:
            i_micro = environment.micro_allocate(self, self.pos_veh[i_agent])
            i_macro = 0
            veh_gain = environment.get_path_loss_Micro(self, self.pos_veh[i_agent], i_micro)
            for i_RB in range(self.n_RB):
                self.V2I_signal[i_agent] += (veh_RB[i_agent, i_RB, i_step]) * (10 ** ((veh_RB_power[
                                                                                           i_agent, i_RB, i_step] - veh_gain + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10))
            # 输入信号、干扰和噪声功率，以 dBm 为单位
            signal_power_dBm = self.V2I_signal[i_agent]  # 信号功率
            interference_power_dBm = environment.get_interference(self, veh_RB_power, veh_BS, veh_RB, i_macro, i_micro,
                                                                  i_agent, i_step)  # 干扰功率
            noise_power_dBm = self.sig2  # 噪声功率
            veh_rate = self.BW * np.log2(1 + np.divide(signal_power_dBm, (interference_power_dBm + noise_power_dBm)))
            # 计算信干噪比（SINR）
            sinr = environment.calculate_sinr(self, signal_power_dBm, interference_power_dBm, noise_power_dBm)
            # 将 SINR 转换为分贝单位
            sinr_dB = environment.linear_to_db(self, sinr)
            # print("信干噪比（SINR）为：", sinr_dB, "dB")

        return sinr_dB

    def compute_sinr_macro(self, i_agent, veh_RB_start, veh_power_start, veh_BS_start):
        self.V2I_signal[i_agent] = 0
        i_macro = environment.macro_allocate(self, self.pos_veh[i_agent])
        i_micro = 0
        veh_gain = environment.get_path_loss_Macro(self, self.pos_veh[i_agent], i_macro)
        for i_RB in range(self.n_RB):
            self.V2I_signal[i_agent] += (veh_RB_start[i_agent, i_RB]) * (10 ** ((veh_power_start[i_agent, i_RB] - veh_gain + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10))
        # 输入信号、干扰和噪声功率，以 dBm 为单位
        signal_power_dBm = self.V2I_signal[i_agent]  # 信号功率
        interference_power_dBm = environment.get_interference_macro(self, i_agent, veh_power_start, veh_BS_start, veh_RB_start, i_macro)  # 干扰功率
        noise_power_dBm = self.sig2  # 噪声功率
        # 计算信干噪比（SINR）
        sinr = np.divide(signal_power_dBm, np.sum(interference_power_dBm+noise_power_dBm))

        rate = self.BW * np.log2(1 + sinr)
        # 将 SINR 转换为分贝单位
        sinr_dB = environment.linear_to_db(self, sinr)

        return rate, sinr_dB
        # print("信干噪比（SINR）为：", sinr_dB, "dB")

    def compute_sinr_micro(self, i_agent, veh_RB_start, veh_power_start, veh_BS_start):
        self.V2I_signal[i_agent] = 0
        i_micro = environment.micro_allocate(self, self.pos_veh[i_agent])
        i_macro = 0
        veh_gain = environment.get_path_loss_Micro(self, self.pos_veh[i_agent], i_micro)
        for i_RB in range(self.n_RB):
            self.V2I_signal[i_agent] += (veh_RB_start[i_agent, i_RB]) * (10 ** ((veh_power_start[i_agent, i_RB] - veh_gain + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10))
        # 输入信号、干扰和噪声功率，以 dBm 为单位
        signal_power_dBm = self.V2I_signal[i_agent]  # 信号功率
        interference_power_dBm = environment.get_interference_micro(self, i_agent, veh_power_start, veh_BS_start, veh_RB_start, i_micro)  # 干扰功率
        noise_power_dBm = self.sig2  # 噪声功率
        # 计算信干噪比（SINR）
        sinr = np.divide(signal_power_dBm, np.sum(interference_power_dBm+noise_power_dBm))

        rate = self.BW * np.log2(1 + sinr)
        # 将 SINR 转换为分贝单位
        sinr_dB = environment.linear_to_db(self, sinr)

        return rate, sinr_dB

    def make_start(self, i_agent, veh_BS_start, veh_RB_start, veh_RB_power_start):
        self.AoI_veh = np.ones([self.n_agent], dtype=np.int64) * 100
        self.AoI_WiFi = np.ones([self.n_agent], dtype=np.int64) * 100
        self.pos_veh[i_agent, 0] = np.round(np.random.uniform(0, 1000), 2)
        self.pos_veh[i_agent, 1] = np.round(np.random.uniform(0, 1000), 2)
        self.dir_veh[i_agent] = np.random.choice((1, 2, 3, 4), 1)  # It means: (1='up',2='right',3='down',4='left')

        if veh_BS_start[i_agent] == 1:
            i_Macro = environment.macro_allocate(self, self.pos_veh[i_agent])
            self.state[i_agent, 0] = environment.get_path_loss_Macro(self, self.pos_veh[i_agent], i_Macro)
            self.state[i_agent, 2] = self.get_interference_macro(i_agent, veh_RB_power_start, veh_BS_start, veh_RB_start, i_Macro) * (10 ** 16)
            self.state[i_agent, 3] = np.round(np.random.uniform(6, 12))
            self.state[i_agent, 1], self.state[i_agent, 4] = self.compute_sinr_macro(i_agent, veh_RB_start, veh_RB_power_start, veh_BS_start)
            self.veh_num_BS[i_agent, 0] = -1  # It means vehicle no connected to Micro
            self.veh_num_BS[i_agent, 1] = i_Macro
            self.duty[i_agent] = 1
            self.symbol_p_word = np.round(np.random.uniform(1, 20))  # symbol/word


        else:
            i_Micro = environment.micro_allocate(self, self.pos_veh[i_agent])
            self.state[i_agent, 0] = environment.get_path_loss_Micro(self, self.pos_veh[i_agent], i_Micro)
            self.state[i_agent, 2] = self.get_interference_micro(i_agent, veh_RB_power_start, veh_BS_start, veh_RB_start, i_Micro) * (10 ** 16)
            self.state[i_agent, 3] = np.round(np.random.uniform(6, 12))
            self.state[i_agent, 1], self.state[i_agent, 4] = self.compute_sinr_micro(i_agent, veh_RB_start, veh_RB_power_start, veh_BS_start)

            self.veh_num_BS[i_agent, 0] = i_Micro
            self.veh_num_BS[i_agent, 1] = -1  # It means vehicle no connected to Macro
            self.duty[i_agent] = np.random.uniform(0, 1)
            self.symbol_p_word = np.round(np.random.uniform(1, 20))  # symbol/word

        return self.state[i_agent], self.veh_num_BS[i_agent], self.duty[i_agent], self.symbol_p_word

    def mobility_veh(self):
        for i_agent in range(self.n_agent):
            if (self.dir_veh[i_agent]) == 1:
                self.pos_veh[i_agent, 0] = self.pos_veh[i_agent, 0]
                self.pos_veh[i_agent, 1] = self.pos_veh[i_agent, 1] + 0.1
            if (self.dir_veh[i_agent]) == 2:
                self.pos_veh[i_agent, 0] = self.pos_veh[i_agent, 0]
                self.pos_veh[i_agent, 1] = self.pos_veh[i_agent, 1] - 0.1
            if (self.dir_veh[i_agent]) == 3:
                self.pos_veh[i_agent, 0] = self.pos_veh[i_agent, 0] + 0.1
                self.pos_veh[i_agent, 1] = self.pos_veh[i_agent, 1]
            if (self.dir_veh[i_agent]) == 4:
                self.pos_veh[i_agent, 0] = self.pos_veh[i_agent, 0] - 0.1
                self.pos_veh[i_agent, 1] = self.pos_veh[i_agent, 1]

    def Age_of_information(self, i_agent, rate_level, WiFi_rate):
        if rate_level > self.rate_level_min:
            self.AoI_veh[i_agent] = 1
        else:
            self.AoI_veh[i_agent] += 1
            if self.AoI_veh[i_agent] >= 100:
                self.AoI_veh[i_agent] = 100

        if WiFi_rate > self.WiFi_level_min:
            self.AoI_WiFi[i_agent] = 1
        else:
            self.AoI_WiFi[i_agent] += 1
            if self.AoI_WiFi[i_agent] >= 100:
                self.AoI_WiFi[i_agent] = 100

        return self.AoI_veh[i_agent], self.AoI_WiFi[i_agent]

    def G_functionself(self, G_input):
        if G_input >= 0:
            G_output = 1
        else:
            G_output = 0
        return G_output

    def RB_BS_allocate(self, veh_RB, veh_RB_BS, veh_BS, i_step):
        for i_agent in range(self.n_agent):
            if veh_BS[i_agent, i_step] == 0:
                i_micro = environment.micro_allocate(self, self.pos_veh[i_agent])  # 判断和哪一个小基站距离最小
                # 如果车辆未连接到宏基站（veh_BS[i_agent, i_step] == 0），则将资源块分配给距离车辆最近的微基站（i_micro）[veh_RB_BS的第二个元素]
                for i_RB in range(self.n_RB):
                    veh_RB_BS[i_agent, i_RB, i_micro + self.n_macro] = veh_RB[i_agent, i_RB, i_step]
            if veh_BS[i_agent, i_step] == 1:
                i_macro = environment.macro_allocate(self, self.pos_veh[i_agent])
                # 如果车辆连接到了宏基站（veh_BS[i_agent, i_step] == 1），则将资源块分配给相应的宏基站[veh_RB_BS的第一个元素]
                for i_RB in range(self.n_RB):
                    veh_RB_BS[i_agent, i_RB, i_macro] = veh_RB[i_agent, i_RB, i_step]
        return veh_RB_BS

    def check_constrain(self, veh_RB, veh_RB_BS, i_step):
        for i_BS in range(self.n_BS):
            for i_RB in range(self.n_RB):
                if np.sum(veh_RB_BS[:, i_RB, i_BS]) > 1:
                    for i_agent in range(self.n_agent):
                        if np.sum(veh_RB_BS[i_agent, :, i_BS]) > 1:
                            veh_RB_BS[i_agent, i_RB, i_BS] = 0
                            veh_RB[i_agent, i_RB, i_step] = 0
                if np.sum(veh_RB_BS[:, i_RB, i_BS]) > 1:
                    veh_RB_BS[:, i_RB, i_BS] = 0
                    veh_RB[:, i_RB, i_step] = 0
        return veh_RB

    # 计算信干噪比（SINR）
    def calculate_sinr(self, signal_power, interference_power, noise_power):
        return signal_power / (interference_power + noise_power)

    # 将线性值转换为分贝值
    def linear_to_db(self, value):
        if value <= 0:
            # 当 value <= 0 时，返回一个特定的值，可以根据你的需求来设定
            return -10
        else:
            # 当 value > 0 时，计算以 10 为底的对数
            return 10 * np.log10(value)

