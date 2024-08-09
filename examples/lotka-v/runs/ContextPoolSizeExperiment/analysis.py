#%%

import matplotlib.pyplot as plt
from numpy import NaN

ps = [0, 1, 2, 4, 6, 8]

## 2024
ind_1 = [0.15126760, 0.02513400, 0.02397918, 0.02946594, 0.02697295, 0.03468854]
ood_1 = [0.11133826, 0.03829543, 0.03151549, 0.020635363, 0.021617047, 0.02017834]

## 2026
ind_2 = [0.02578021, 0.02852705, 0.02823534, 0.02846532, 0.03042384, 0.02745084]
ood_2 = [0.018909449, 0.037040416, 0.04256853, 0.0442202, 0.04642554, 0.041729372]

## 2028
ind_3 = [0.14994149, NaN, 0.26912639, 0.02546359, 0.02412433, NaN]
ood_3 = [0.11577063, NaN, 0.6005156, 0.015153124, 0.0151267145, NaN]


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
fig.savefig('Pots.png')