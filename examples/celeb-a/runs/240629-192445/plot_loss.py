#%%

import re
import matplotlib.pyplot as plt

# Read the log file
log_file_path = 'nohup.log'
with open(log_file_path, 'r') as file:
    log_lines = file.readlines()

# Initialize lists to store parsed loss values
epoch_numbers = []
lossmodel_values = []

# Regular expression to match lines with lossmodel values
### Epoch:  19      LossModel: 0.00809668     ContextsNorm: 0.00815169

lossmodel_pattern = re.compile(r'Epoch:\s+(\d+)\s+LossModel:\s+([0-9.]+)')

# Parse the log file
for line in log_lines:
    match = lossmodel_pattern.search(line)
    if match:
        epoch = int(match.group(1))
        lossmodel = float(match.group(2))
        epoch_numbers.append(epoch)
        lossmodel_values.append(lossmodel)

print(lossmodel_values)

# Plot the lossmodel values
plt.figure(figsize=(10, 6))
plt.plot(lossmodel_values, label='lossmodel')
plt.yscale('log')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Training lossmodel Over Epochs')
plt.legend()
plt.grid(True)
plt.show()
