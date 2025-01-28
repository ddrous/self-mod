#%% [markdown]
# # MixER Plot Script
""" A script for plots highlithing the benefits of MixER """


#%%
from selfmod import *
from matplotlib import animation



#%%

## We plot the MixER as they compare to naive MoE and task-specific experts

## Data in the form (relative L2 norm - SQRT)
# [1Expert, 2Experts-GD, 2Experts-T0, 2Experts-T2]
# ncf = [0.634, 0.658, 0.204, 0.228]
# coda = [0.659, 0.624, 0.093, 0.114]
# geps = [0.636, 0.647, 0.240, 0.176]

ncf = [0.987, 1.004, 0.147, 0.186]
coda = [1.906, 1.273, 0.029, 0.045]
geps = [1.176, 1.259, 0.590, 0.142]
mixer_id = 3

# Plotting with barplots (4x3 bars in total, 3 groups, 4 bars each). The bars are colored according to the group, and he 4 in each group are next to each other.
fig, ax = plt.subplots(1, 1, figsize=(8, 4))

## Plot the 3 bars corresponding to 1Expert first
barWidth = 0.2
r1 = np.arange(3) + 0.0
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]

ax.bar(r1, [ncf[0], coda[0], geps[0]], color='b', width=barWidth, edgecolor='grey', label='Task-Specific')

## Add the numbers on top of the bars
for i, v in enumerate([ncf[0], coda[0], geps[0]]):
    ax.text(i-0.08, v+0.02, str(round(v, 3)), color='black', fontsize=8)

## Plot the 3 bars corresponding to 2Experts-GD
ax.bar(r2, [ncf[1], coda[1], geps[1]], color='r', width=barWidth, edgecolor='grey', label='Naive MoE')

## Add the numbers on top of the bars
for i, v in enumerate([ncf[1], coda[1], geps[1]]):
    ax.text(i+0.125, v+0.02, str(round(v, 3)), color='black', fontsize=8)

## Plot the 3 bars corresponding to 2Experts-T0
ax.bar(r3, [ncf[mixer_id], coda[mixer_id], geps[mixer_id]], color='g', width=barWidth, edgecolor='grey', label='MixER')

## Add the numbers on top of the bars
for i, v in enumerate([ncf[mixer_id], coda[mixer_id], geps[mixer_id]]):
    ax.text(i+0.33, v+0.02, str(round(v, 3)), color='black', fontsize=8)

## Make the legend horizontal
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.17), shadow=True, ncol=3, fontsize=14)

## Set the x lalebs as the group names
ax.set_xticks([r + barWidth for r in range(3)])
ax.set_xticklabels(['NCF', 'CoDA', 'GEPS'], fontsize=20)

## Set the y label as the relative L2 norm
ax.set_ylabel('Relative $L^2$', fontsize=14)

## Remove the vertical tick marks
ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)

## Set ylim
# ax.set_ylim(0, 0.72)

## Make the left and bottom spines thicker and end with an arrow
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)

## Save as a PDF
plt.savefig('mixer_barplots.pdf', bbox_inches='tight', dpi=300)
















# %%

## Increase the linewidth to 3
plt.rcParams['lines.linewidth'] = 4

## Plot the losses
folders_1_expert = ["_02_NCFRuns/250120-210843-New1Expert/", "_03_CoDARuns/250119-232909-1Expert/", "_04_GEPSRuns/250120-193035-1Expert/"]
folders_2_experts_gd = ["_02_NCFRuns/250120-002029-NewT2-GradientDescnet/", "_03_CoDARuns/250119-231018-2Experts-T0-GD/", "_04_GEPSRuns/250120-191238-2Experts-GD/"]
folders_2_experts = ["_02_NCFRuns/250120-012332-NewT2/", "_03_CoDARuns/250119-224921-2Experts-T0/", "_04_GEPSRuns/250120-152120-2Experts-T0/"]

## Make three colors for the three groups
colors = ['b', 'r', 'g']
markers = ['-', '-.', ':']

## Load the train_history.npz files and plot the losses
fig, ax = plt.subplots(1, 1, figsize=(12, 5))
loss_key = 'losses_ctx'
for i, folder in enumerate(folders_1_expert):
    # history = np.load("../runs/"+folder + 'train_histories.npz')
    # ax.plot(history[loss_key], label='Task-Specific ' + ['NCF', 'CoDA', 'GEPS'][i])
    history = np.load("../runs/"+folder + 'val_losses.npy')
    print(history.shape)
    ax.plot(history[:, 0], history[:, 1], f"{colors[0]}{markers[i]}", label='Task-Specific ' + ['NCF', 'CoDA', 'GEPS'][i])

## Do the same thing for the 2 experts GD
for i, folder in enumerate(folders_2_experts_gd):
    history = np.load("../runs/"+folder + 'val_losses.npy')
    ax.plot(history[:, 0], history[:, 1], f"{colors[1]}{markers[i]}", label='Naive MoE ' + ['NCF', 'CoDA', 'GEPS'][i])

## Do the same thing for the 2 experts
for i, folder in enumerate(folders_2_experts):
    history = np.load("../runs/"+folder + 'val_losses.npy')
    ax.plot(history[:, 0], history[:, 1], f"{colors[2]}{markers[i]}", label='MixER ' + ['NCF', 'CoDA', 'GEPS'][i])

ax.set_xlabel('Train Step', fontsize=22)
ax.set_ylabel('MSE', fontsize=20)
ax.set_yscale('log')

## Increase size of ticks
ax.tick_params(axis='both', which='major', labelsize=18)

# ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.97), shadow=True, ncol=3, fontsize=14)
ax.legend(loc='upper right', shadow=True, ncol=3, fontsize=12)

## Save as a PDF
plt.savefig('mixer_losses.pdf', bbox_inches='tight', dpi=300)
