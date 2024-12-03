from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
import matplotlib.pyplot as plt

data_file = "CarRacing-obstaclesV2_1129_Ts0ApSteer2"

event_file = "logs/" + data_file + '/SAC_1/'
event_acc = EventAccumulator(event_file)
event_acc.Reload()

keys = event_acc.scalars.Keys()
print(keys)

need_info_key = ["rollout/ep_rew_mean", "eval/mean_reward", "train/actor_loss", "train/critic_loss"] #
info = {}

for key in need_info_key:
    key_data_g = event_acc.scalars.Items(key)
    steps = [i.step/1000 for i in key_data_g]  # Extract steps
    values = [i.value for i in key_data_g]  # Extract values
    info[key] = {'steps': steps, 'values': values}

# save to csv
# info_df = pd.DataFrame(info)
# print(info_df)
# save_path = "../../train_log/" + data_file + ".csv"
# info_df.to_csv(save_path)
plt.style.use('seaborn')
fig,ax = plt.subplots(1,len(need_info_key),figsize=(20,5), dpi=120)
fig.subplots_adjust(wspace=0.3)

for i, key in enumerate(need_info_key):
    plt.sca(ax[i])
    total_len  = len(info[key]['steps'])
    plot_len = int( 6/6*total_len)
    plt.plot(info[key]['steps'][:plot_len], info[key]['values'][:plot_len])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("k step", fontsize=14)
    plt.ylabel(key, fontsize=14)
plt.suptitle("Training logs", fontsize=16)
fig.tight_layout()
plt.show()

