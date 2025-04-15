#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from selfmod import *


# Read data from CSV
data = pd.read_csv('memory-time-data.csv')

# Convert model number to numeric (extract the digit after the model name)
data['Model_Num'] = data['Model'].str.extract(r'-(\d+)').astype(int)

# Replace NaN with 0 for plotting (for FlashCAVIA-3 with K=100)
data['Time_Step_sec'] = data['Time_Step_sec'].fillna(0)

# # Set up figure and gridspec for better control of subplot spacing
# fig = plt.figure(figsize=(14, 10))
# gs = fig.add_gridspec(1, 2, height_ratios=[1, 1], hspace=0.3)

# # Create subplots
# ax1 = fig.add_subplot(gs[0])
# ax2 = fig.add_subplot(gs[1])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(26, 6), gridspec_kw={'hspace': 0.3})

# Create twin axes for NCF data
# ax1_twin = ax1.twinx()
# ax2_twin = ax2.twinx()

ax1_twin = ax1
ax2_twin = ax2

# Color palettes
flashcavia_colors = ['#1f77b4', '#2ca02c']  # Blue for K=10, Green for K=100
ncf_colors = ['#ff7f0e', '#d62728']  # Orange for K=10, Red for K=100

# Filter data for different models and K values
flash_k10 = data[(data['Type'] == 'FlashCAVIA') & (data['K'] == 10)]
flash_k100 = data[(data['Type'] == 'FlashCAVIA') & (data['K'] == 100)]
ncf_k10 = data[(data['Type'] == 'NCF') & (data['K'] == 10)]
ncf_k100 = data[(data['Type'] == 'NCF') & (data['K'] == 100)]

# Plot Memory Usage (ax1)
flash_k10_mem = ax1.plot(flash_k10['Model_Num'], flash_k10['Memory_MB'], 'o-', 
                          color=flashcavia_colors[0], label=r'FlashCAVIA ($K$=10)', linewidth=4, markersize=16)
flash_k100_mem = ax1.plot(flash_k100['Model_Num'], flash_k100['Memory_MB'], 's-', 
                           color=flashcavia_colors[1], label=r'FlashCAVIA ($K$=100)', linewidth=4, markersize=16)

ncf_k10_mem = ax1_twin.plot(ncf_k10['Model_Num'], ncf_k10['Memory_MB'], 'o--', 
                             color=ncf_colors[0], label=r'NCF ($K$=10)', linewidth=4, markersize=16)
ncf_k100_mem = ax1_twin.plot(ncf_k100['Model_Num'], ncf_k100['Memory_MB'], 's--', 
                              color=ncf_colors[1], label=r'NCF ($K$=100)', linewidth=4, markersize=16)

# Plot Time per Step (ax2)
flash_k10_time = ax2.plot(flash_k10['Model_Num'], flash_k10['Time_Step_sec'], 'o-', 
                           color=flashcavia_colors[0], label=r'FlashCAVIA ($K$=10)', linewidth=4, markersize=16)
to_plot = flash_k100['Time_Step_sec']
# Filter nans
# Replace thte last toplot value with a NaN
to_plot.iloc[-1] = np.nan
flash_k100_time = ax2.plot(flash_k100['Model_Num'], to_plot, 's-', 
                            color=flashcavia_colors[1], label=r'FlashCAVIA ($K$=100)', linewidth=4, markersize=16)

ncf_k10_time = ax2_twin.plot(ncf_k10['Model_Num'], ncf_k10['Time_Step_sec'], 'o--', 
                              color=ncf_colors[0], label=r'NCF ($K$=10)', linewidth=4, markersize=16)
ncf_k100_time = ax2_twin.plot(ncf_k100['Model_Num'], ncf_k100['Time_Step_sec'], 's--', 
                               color=ncf_colors[1], label=r'NCF ($K$=100)', linewidth=4, markersize=16)

# Setting titles and labels
ax1.set_title('(a)', fontsize=26, pad=15)
ax1.set_xlabel(r'Taylor order $k$', fontsize=28)
# ax1.set_ylabel('FlashCAVIA Memory (MB)', fontsize=14, color='#1f77b4')
ax1_twin.set_ylabel('Memory usage (GB)', fontsize=28, color='k')

ax2.set_title('(b)', fontsize=26, pad=15)
ax2.set_xlabel(r'Taylor order $k$', fontsize=28)
# ax2.set_ylabel('FlashCAVIA Time/Step (sec)', fontsize=14, color='#1f77b4')
ax2_twin.set_ylabel('Time/Step (sec)', fontsize=28, color='k')

# Set x-ticks to be integers 0, 1, 2, 3 only
for ax in [ax1, ax2]:
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels([0, 1, 2, 3], fontsize=28)
    ax.set_xlim(-0.2, 3.2)
    ax.grid(True, linestyle='--', alpha=0.7)

## Increase the size of the y-ticks
for ax in [ax1, ax2]:
    ax.tick_params(axis='y', labelsize=20)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}'))

# Adjust y-axis formatting to show K, M (thousands, millions) for memory
def format_mem(x, pos):
    if x >= 1000:
        return f'{x/1000:.1f}K'
    return f'{x:.0f}'

# ax1.yaxis.set_major_formatter(ticker.FuncFormatter(format_mem))
# ax1_twin.yaxis.set_major_formatter(ticker.FuncFormatter(format_mem))

# Format time axis to show scientific notation for small numbers
def format_time(x, pos):
    if x < 0.01:
        return f'{x:.0e}'
    return f'{x:.3f}'

# ax2.yaxis.set_major_formatter(ticker.FuncFormatter(format_time))
# ax2_twin.yaxis.set_major_formatter(ticker.FuncFormatter(format_time))

# # Add note about the missing data point
# ax2.annotate('Note: FlashCAVIA-3 (K=100) time data is missing',
#              xy=(0.5, -0.2), xycoords='axes fraction',
#              ha='center', fontsize=12, style='italic')

# # Combine legends
# lines1, labels1 = ax1.get_legend_handles_labels()
# lines2, labels2 = ax1_twin.get_legend_handles_labels()
# ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=12)

# lines3, labels3 = ax2.get_legend_handles_labels()
# lines4, labels4 = ax2_twin.get_legend_handles_labels()
# ax2.legend(lines3 + lines4, labels3 + labels4, loc='upper left', fontsize=12)

ax1.legend(loc='upper left', fontsize=24)
# ax2.legend(loc='upper left', fontsize=12)

ax2.set_yscale('log')

# Adjust layout
plt.tight_layout()

# Save figure with high quality
plt.savefig('Inkscape/memory_time_plot.pdf', dpi=300, bbox_inches='tight')

# Show the plots
plt.show()
