#%%
""" Analysis of the results of the sine regression experiment. """

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scienceplots

sns.set_theme(context='talk', style='ticks',
        font='sans-serif', font_scale=1, color_codes=True, rc={"lines.linewidth": 2})

# plt.style.use(['science', 'no-latex'])

## Set the following matplotlib parameters 
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

## Columnms are: 
#       ['method', 'num_envs', 'taylor_order', 'context_size',
#       'gradient_updates', 'outer_steps', 'mse_ind', 'ci_ind', 'mse_ood',
#       'ci_ood']

## ## For CAVIA, get all the rows respecting the condition
condition = {'method': 'CAVIA', 'taylor_order': 0, 'context_size': 2, 'gradient_updates': 1}
## Open the zintgraff dataset
results_zg = pd.read_csv("results_zintgraff.csv", header=0)

## Add a column taylor order with 0
results_zg['taylor_order'] = 0

print(results_zg.head())

#%%

## Three paramters are varying: num_envs, taylor_order, context_size
# hps = ['num_envs', 'taylor_order', 'context_size']
hps = ['num_envs', 'context_size', 'taylor_order']
hps_range = {'num_envs': [250, 1000, 12500], 'taylor_order': [0, 2], 'context_size': [2, 3, 50]}
ylim = [5e-5, 2e+1]
plot_orig = True
error_mult = 2.0
gradient_updates = [100, 5, 1]

metric = 'walltime'
ci_metric = 'ci_ind'
extension = 'svg'
save_folder = "Analysis/TrainTimes"

## Make Analysis and IND folders
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

for j, param in enumerate(hps[:2]):
    print(f"Parameter: {param}")
    # print(results[param].unique())

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    other_params = hps[:j] + hps[j+1:]
    for other_val1 in hps_range[other_params[0]]:
        for other_val2 in hps_range[other_params[1]]:

            print("Param is:", param)
            print(f"Other values: {other_params[0]}={other_val1}, {other_params[1]}={other_val2}")

            ## Cavia sintgraff results
            cavia_results_zg = results_zg[(results_zg['method'] == 'CAVIA') & (results_zg[other_params[0]] == other_val1) & (results_zg[other_params[1]] == other_val2)]

            ## MAML zintgraff results
            maml_results_zg = results_zg[(results_zg['method'] == 'MAML') & (results_zg[other_params[0]] == other_val1) & (results_zg[other_params[1]] == other_val2)]

            ## Plot the 3 rows of CAVIA and the single row of NCF: plot the mse_ind, and ci_ind as error bars
            colors_ood = ['r', 'g', 'b']
            colors = ['darkred', 'darkgreen', 'darkblue']
            colors_orig = ['crimson', 'teal', 'royalblue']

            ## Extract 3 sub tables, for each gradient_updates value
            # for gu_id, gu in enumerate(cavia_results['gradient_updates'].unique()):
            for gu_id, gu in enumerate(gradient_updates):
                cavia_results_zg_gu = cavia_results_zg[cavia_results_zg['gradient_updates'] == gu]
                maml_results_zg_gu = maml_results_zg[maml_results_zg['gradient_updates'] == gu]

                ## order according to the param
                cavia_results_gu = cavia_results_gu.sort_values(by=param)
                cavia_results_zg_gu = cavia_results_zg_gu.sort_values(by=param)
                print("MAML results before:", maml_results_zg_gu)
                maml_results_zg_gu = maml_results_zg_gu.sort_values(by=param)
                print("MAML results after:", maml_results_zg_gu)

                ## Do the same thing with seaborn pointplot and striplot
                sns.pointplot(data=cavia_results_gu, x=param, y=metric, errorbar="se", ax=ax, markers="o", markersize=12, legend=False, log_scale=True, label=None, alpha=0.2, color=colors[gu_id], linestyles="-")

                if plot_orig and not cavia_results_zg_gu.empty and not maml_results_zg_gu.empty:
                    sns.pointplot(data=cavia_results_zg_gu, x=param, y=metric, errorbar="se", ax=ax, markers="s", markersize=12, legend=False, log_scale=True, label=None, alpha=0.3, color=colors_orig[gu_id], linestyles="--")

                    sns.pointplot(data=maml_results_zg_gu, x=param, y=metric, errorbar="se", ax=ax, markers="^", markersize=12, legend=False, log_scale=True, label=None, alpha=0.3, color=colors_orig[gu_id], linestyles="-.")

                ## Get the axis ticks and labels
                xticks = ax.get_xticks()
                xticklabels = ax.get_xticklabels()

                # ## Sort the values
                # xticks, xticklabels = zip(*sorted(zip(xticks, xticklabels), key=lambda x: x[0]))

                print("These are the famous xticks:", xticklabels)

            ## Plot the NCF results with pointplot
            ## order according to the param
            ncf_results = ncf_results.sort_values(by=param)

            sns.pointplot(data=ncf_results, x=param, y=metric, errorbar="se", ax=ax, markers="o", markersize=12, legend=False, log_scale=True, color='k', alpha=0.5)

            ## Add the error bars for NCF
            for i, x in enumerate(xticks):
                label = 'NCF' if i==0 else None

                if ci_metric == "ci_ind":
                    yerr = ncf_results[ci_metric].iloc[i]*error_mult
                else:
                    yerr = None

                ax.errorbar(x, ncf_results[metric].iloc[i], yerr=yerr, marker="o", label=label, color='k', capsize=4)

                yi = ncf_results['mse_ind'].iloc[i]
                ci = ncf_results['ci_ind'].iloc[i]*error_mult
                log_yerr_upper = yi+ci
                log_yerr_lower = np.maximum(yi-ci, 1e-3)
                log_yerr_lower = np.minimum(log_yerr_lower, 1e-6)
                yerr = np.array((log_yerr_lower, log_yerr_upper))[:, None]

            ax.set_ylim(ylim)
            # ax.legend(fontsize='x-small')
            ax.set_yscale('log')

            ## Convert param into 
            if param == 'num_envs':
                # x_label = "# environments"
                x_label = r"$N$"
            elif param == 'taylor_order':
                # x_label = 'Taylor Order'
                x_label = r"$k$"
            elif param == 'context_size':
                # x_label = 'Context Size'
                x_label = r"$d_{\xi}$"
            ax.set_xlabel(x_label)

            ## No need for y label
            ax.set_ylabel("")
            if not maml_results_zg_gu.empty:
                old_handles, old_labels = ax.get_legend_handles_labels()

            ## Set the title as the other two hps
            title_names = []
            for i, name in enumerate(other_params):
                if name == 'num_envs':
                    title_names.append(r"$N$")
                elif name == 'taylor_order':
                    title_names.append(r"$k$")
                elif name == 'context_size':
                    title_names.append(r"$d_{\xi}$")

            # title = f"{other_params[0]}={other_val1}, {other_params[1]}={other_val2}"
            title = f"{title_names[0]}={other_val1}, {title_names[1]}={other_val2}"
            # ax.set_title(title)
            ax.set_title(title, fontsize='large', y=0.90)

            ## Save the figure in the folder Analysis. The name should reflect the title
            fig.savefig(f"{save_folder}/{other_params[0]}_{other_val1}_{other_params[1]}_{other_val2}.{extension}", bbox_inches='tight')

            # plt.show()
            plt.draw()

        #     ## Clear the axis for the next plot
            ax.clear()
        #     break
        # break

    # break


#%%

## Extract the legend from the existing ax and plot and plot it in a separate figure

old_ax = ax
old_fig = fig
# old_handles, old_labels = old_ax.get_legend_handles_labels()

fig, ax_new = plt.subplots(1, 1, figsize=(6, 5))
ax_new.legend(old_handles, old_labels, loc='center', fontsize='x-small')
ax_new.axis('off')
plt.draw()

fig.savefig(f"{save_folder}/TrainTimes.{extension}", bbox_inches='tight')

plt.show()


