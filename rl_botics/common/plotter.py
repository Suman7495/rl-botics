import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot(title,
         filename,
         xlabel=None,
         ylabel=None,
         n_obj=None,
         algo=None,
         noise=None,
         new_fig=False
         ):
    with open(filename, 'r') as f:
        y = f.readlines()

    y = [(e.strip()) for e in y]
    y = y[1:]
    y = np.asarray([float(e) for e in y])
    x = np.arange(y.shape[0])
    ystd = np.std(y)
    error = ystd / np.sqrt(len(y))
    y = pd.Series(y).rolling(20, min_periods=1).mean()
    legend = ""
    if algo:
        legend = algo + " - "
    if n_obj:
        legend = legend + str(n_obj) + ' objects'
    if new_fig:
        fig, ax = plt.subplots(nrows=1, ncols=1)
    else:
        ax = plt.gca()
    plt.plot(x, y, '-', label=legend)
    plt.fill_between(x, y-error, y+error, alpha=0.5)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)

    plt.title(title)
    plt.tight_layout()
    plt.legend()
    plt.grid(True, alpha=1, c='white')

    # Beautify plot
    ax.set_facecolor((176/256, 224/256, 230/256))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    # plt.show()
    print("Completed")

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# # 1. Scalability - Success Percentages - COMPLETE
# plot("Success Percentage", '../copos/results/final/POMDP_Occ_noise_1_2_1/copos_success_1_2_1.txt', 'Iterations', 'Success Percentage', 4, new_fig=True)
# plot("Success Percentage", '../copos/results/final/POMDP_Occ_Noise_2_3_1/copos_success_2_3_1.txt', 'Iterations', 'Success Percentage', 6)
# plot("Success Percentage", '../copos/results/final/POMDP_Occ_Noise_3_4_1/copos_success_3_4_1.txt', 'Iterations', 'Success Percentage', 8)
# plot("Success Percentage", '../copos/results/final/POMDP_Occ_Noise_4_5_1/copos_success_4_5_1.txt', 'Iterations', 'Success Percentage', 10)
#
# # 2. Scalability - Rewards
# plot("Average Rewards", '../copos/results/final/POMDP_Occ_noise_1_2_1/copos_rew_1_2_1.txt', 'Iterations', 'Average Rewards', 4, new_fig=True)
# plot("Average Rewards", '../copos/results/final/POMDP_Occ_Noise_2_3_1/copos_rew_2_3_1.txt', 'Iterations', 'Average Rewards', 6)
# plot("Average Rewards", '../copos/results/final/POMDP_Occ_Noise_3_4_1/copos_rew_3_4_1.txt', 'Iterations', 'Average Rewards', 8)
# plot("Average Rewards", '../copos/results/final/POMDP_Occ_Noise_4_5_1/copos_rew_4_5_1.txt', 'Iterations', 'Average Rewards', 10)
#
# # 3. Scalability - Entropy
# plot("Entropy", '../copos/results/final/POMDP_Occ_noise_1_2_1/copos_ent_1_2_1.txt', 'Iterations', 'Entropy', 4, new_fig=True)
# plot("Entropy", '../copos/results/final/POMDP_Occ_Noise_2_3_1/copos_ent_2_3_1.txt', 'Iterations', 'Entropy', 6)
# plot("Entropy", '../copos/results/final/POMDP_Occ_Noise_3_4_1/copos_ent_3_4_1.txt', 'Iterations', 'Entropy', 8)
# plot("Entropy", '../copos/results/final/POMDP_Occ_Noise_4_5_1/copos_ent_4_5_1.txt', 'Iterations', 'Entropy', 10)

#-------------------------------------------------------------------------------------------------------------------------------------------------------

# 4. MDP - Success Percentages - 10 - 10 - 1 - COMPLETE
# plot("MDP - Success Percentage", '../copos/results/final/MDP_COPOS_10_10_1/copos_success_10_10_1.txt', 'Iterations', 'Success Percentage', n_obj=21, algo="COPOS", new_fig=True)
# plot("MDP - Success Percentage", '../copos/results/final/MDP_TRPO_10_10_1/trpo_success_10_10_1.txt', 'Iterations', 'Success Percentage', n_obj=21, algo="TRPO")
# plot("MDP - Success Percentage", '../ppo/results/final/MDP_PPO_10_10_1/ppo_success_10_10_1.txt', 'Iterations', 'Success Percentage', n_obj=21, algo="PPO")

# # # 5. MDP - Rewards
# plot("MDP - Average Rewards", '../copos/results/final/MDP_COPOS_10_10_1/copos_rew_10_10_1.txt', 'Iterations', 'Average Rewards', n_obj=21, algo="COPOS", new_fig=True)
# plot("MDP - Average Rewards", '../copos/results/final/MDP_TRPO_10_10_1/trpo_rew_10_10_1.txt', 'Iterations', 'Average Rewards', n_obj=21, algo="TRPO")
# plot("MDP - Average Rewards", '../ppo/results/final/MDP_PPO_10_10_1/ppo_rew_10_10_1.txt', 'Iterations', 'Average Rewards', n_obj=21, algo="PPO")
#
# # # 6. MDP - Entropy
# plot("MDP - Entropy", '../copos/results/final/MDP_COPOS_10_10_1/copos_ent_10_10_1.txt', 'Iterations', 'Entropy', n_obj=21, algo="COPOS", new_fig=True)
# plot("MDP - Entropy", '../copos/results/final/MDP_TRPO_10_10_1/trpo_ent_10_10_1.txt', 'Iterations', 'Entropy', n_obj=21, algo="TRPO")
# plot("MDP - Entropy", '../ppo/results/final/MDP_PPO_10_10_1/ppo_ent_10_10_1.txt', 'Iterations', 'Entropy', n_obj=21, algo="PPO")

# --------------------------------------------------------------------------------------------------------------------------------------------------------

# # 7. POMDP - Occ - Success Percentages - 3 - 4 - 1
# plot("POMDP Occlusion Only - Success Percentage", '../copos/results/final/POMDP_COPOS_Occ_3_4_1/copos_success_3_4_1.txt', 'Iterations', 'Success Percentage', n_obj=8, algo="COPOS", new_fig=True)
# plot("POMDP Occlusion Only - Success Percentage", '../copos/results/final/POMDP_TRPO_Occ_3_4_1/trpo_success_3_4_1.txt', 'Iterations', 'Success Percentage', n_obj=8, algo="TRPO")
# plot("POMDP Occlusion Only - Success Percentage", '../ppo/results/final/POMDP_PPO_Occ_3_4_1/ppo_success_3_4_1.txt', 'Iterations', 'Success Percentage', n_obj=8, algo="PPO")

# # 8. POMDP - Occ - Rewards
# plot("POMDP Occlusion Only - Average Rewards", '../copos/results/final/POMDP_COPOS_Occ_3_4_1/copos_rew_3_4_1.txt', 'Iterations', 'Average Rewards', n_obj=8, algo="COPOS", new_fig=True)
# plot("POMDP Occlusion Only - Average Rewards", '../copos/results/final/POMDP_TRPO_Occ_3_4_1/trpo_rew_3_4_1.txt', 'Iterations', 'Average Rewards', n_obj=8, algo="TRPO")
# plot("POMDP Occlusion Only - Average Rewards", '../ppo/results/final/POMDP_PPO_Occ_3_4_1/ppo_rew_3_4_1.txt', 'Iterations', 'Average Rewards', n_obj=8, algo="PPO")

# # 9. POMDP - Occ - Entropy
# plot("POMDP Occlusion Only - Entropy", '../copos/results/final/POMDP_COPOS_Occ_3_4_1/copos_ent_3_4_1.txt', 'Iterations', 'Entropy', n_obj=8, algo="COPOS", new_fig=True)
# plot("POMDP Occlusion Only - Entropy", '../copos/results/final/POMDP_TRPO_Occ_3_4_1/trpo_ent_3_4_1.txt', 'Iterations', 'Entropy', n_obj=8, algo="TRPO")
# plot("POMDP Occlusion Only - Entropy", '../ppo/results/final/POMDP_PPO_Occ_3_4_1/ppo_ent_3_4_1.txt', 'Iterations', 'Entropy', n_obj=8, algo="PPO")

# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# 10. POMDP - Occ - Noise - Success Percentages - 3 - 4 - 1
plot("POMDP with Occlusion and Noise - Success Percentage", '../copos/results/final/POMDP_COPOS_Occ_Noise_3_3_1/copos_success_3_3_1.txt', 'Iterations', 'Success Percentage', n_obj=8, algo="COPOS", new_fig=True)
plot("POMDP with Occlusion and Noise - Success Percentage", '../copos/results/final/POMDP_TRPO_Occ_Noise_3_4_1/trpo_success_3_4_1.txt', 'Iterations', 'Success Percentage', n_obj=8, algo="TRPO")

# 11. POMDP - Occ - Noise - Avg Rewards
plot("POMPD with Occlusion and Noise - Average Rewards", '../copos/results/final/POMDP_COPOS_Occ_Noise_3_3_1/copos_rew_3_3_1.txt', 'Iterations', 'Average Rewards', n_obj=8, algo="COPOS", new_fig=True)
plot("POMDP with Occlusion and Noise - Average Rewards", '../copos/results/final/POMDP_TRPO_Occ_Noise_3_4_1/trpo_rew_3_4_1.txt', 'Iterations', 'Success Percentage', n_obj=8, algo="TRPO")

# 12. POMDP - Occ - Noise - Entropy
plot("POMDP with Occlusion and Noise - Entropy", '../copos/results/final/POMDP_COPOS_Occ_Noise_3_3_1/copos_ent_3_3_1.txt', 'Iterations', 'Entropy', n_obj=8, algo="COPOS", new_fig=True)
plot("POMDP with Occlusion and Noise - Entropy", '../copos/results/final/POMDP_TRPO_Occ_Noise_3_4_1/trpo_ent_3_4_1.txt', 'Iterations', 'Success Percentage', n_obj=8, algo="TRPO")


# -----------------------------------------------------------------------------------------------------------------------------------------------
#
# # 13. POMDP - Occ - Noise - Success Percentages - 2 - 3 - 1
# plot("POMDP with Occlusion and Noise - Success Percentage", '../copos/results/final/POMDP_COPOS_Occ_Noise_2_3_1/copos_success_2_3_1.txt', 'Iterations', 'Success Percentage', n_obj=6, algo="COPOS", new_fig=True)
# plot("POMDP with Occlusion and Noise - Success Percentage", '../copos/results/final/POMDP_TRPO_Occ_Noise_2_3_1/trpo_success_2_3_1.txt', 'Iterations', 'Success Percentage', n_obj=6, algo="TRPO")
#
# # 14. POMDP - Occ - Noise - Avg Rewards
# plot("POMPD with Occlusion and Noise - Average Rewards", '../copos/results/final/POMDP_COPOS_Occ_Noise_2_3_1/copos_rew_2_3_1.txt', 'Iterations', 'Average Rewards', n_obj=6, algo="COPOS", new_fig=True)
# plot("POMDP with Occlusion and Noise - Average Rewards", '../copos/results/final/POMDP_TRPO_Occ_Noise_2_3_1/trpo_rew_2_3_1.txt', 'Iterations', 'Success Percentage', n_obj=6, algo="TRPO")
#
# # 15. POMDP - Occ - Noise - Entropy
# plot("POMDP with Occlusion and Noise - Entropy", '../copos/results/final/POMDP_COPOS_Occ_Noise_2_3_1/copos_ent_2_3_1.txt', 'Iterations', 'Entropy', n_obj=6, algo="COPOS", new_fig=True)
# plot("POMDP with Occlusion and Noise - Entropy", '../copos/results/final/POMDP_TRPO_Occ_Noise_2_3_1/trpo_ent_2_3_1.txt', 'Iterations', 'Success Percentage', n_obj=6, algo="TRPO")

# -------------------------------------------------------------------------------------------------------------------------------------------------


# 16 - POMDP - Occ - Noise - Seed Plots
# for i in range(10):



# Temp Test
#
# x = np.linspace(0, 300, 300)
# y = np.sin(x/6*np.pi)
# # error = np.random.normal(0.1, 0.02, size=y.shape)
# # error = np.random.rand(len(y)) * 0.1
# ystd = np.std(y)
# error = ystd / np.sqrt(len(y))
# y += np.random.normal(0, 0.1, size=y.shape)
#
# plt.plot(x, y, 'g-')
# plt.fill_between(x, y-error, y+error, facecolor='g')
plt.show()