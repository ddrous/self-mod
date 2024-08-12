#%%

import matplotlib.pyplot as plt
from numpy import NaN

ps = [0, 1, 2, 4, 6, 8]

## 2024
ind_1 = [5.37e-04, 6.64e-04, 1.88e-03, 2.82e-03, 2.88e-03, 2.88e-03]
ood_1 = [1.45e-03, 1.41e-03, 3.08e-03, 4.29e-03, 4.59e-03, 4.51e-03]

## 2026
ind_2 = [4.43e-04, 4.37e-04, 1.59e-03, 3.22e-03, 3.49e-03, 3.28e-03]
ood_2 = [1.56e-03, 1.59e-03, 2.48e-03, 4.97e-03, 5.44e-03, 5.04e-03]

## 2028
ind_3 = [8.33e-04, 9.23e-04, 2.27e-03, 4.56e-03, 4.57e-03, 4.57e-03]
ood_3 = [1.66e-03, 1.68e-03, 2.85e-03, 6.31e-03, 6.36e-03, 6.36e-03]


#%%
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

ax[0].plot(ps, ind_1, "o-", label='In-distribution')
ax[0].plot(ps, ood_1, "o-", label='Out-of-distribution')

ax[1].plot(ps, ind_2, "o-", label='In-distribution')
ax[1].plot(ps, ood_2, "o-", label='Out-of-distribution')

ax[2].plot(ps, ind_3, "o-", label='In-distribution')
ax[2].plot(ps, ood_3, "o-", label='Out-of-distribution')

ax[0].set_title('2024')
ax[1].set_title('2026')
ax[2].set_title('2028')

for i in range(3):
    ax[i].set_xlabel('Context Pool Size')
    ax[i].set_ylabel('MSE')
    ax[i].set_yscale('log')
    ax[i].legend()

plt.show()


## Save the figure
fig.savefig('Plots.png')