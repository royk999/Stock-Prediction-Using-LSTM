import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

'''
data = pd.read_csv('results/results_features_rmse_mape.txt', sep=' ')

# Splitting the data into features, RMSE, and MAPE
features, rmse, mape = [], [], []

print(f'data: {data}')

for i in range(len(data)):
    features.append(data.iloc[i, 0])
    rmse.append(data.iloc[i, 1])
    mape.append(data.iloc[i, 2])
print(features)

# Creating the chart with two y-axes
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plotting RMSE
color = 'tab:blue'
ax1.set_xlabel('Number of Features')
ax1.set_ylabel('RMSE', color=color)
ax1.plot(features, rmse, color=color, marker='o', label='RMSE')
ax1.tick_params(axis='y', labelcolor=color)

# Creating a second y-axis for MAPE
ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('MAPE', color=color)  
ax2.plot(features, mape, color=color, marker='x', label='MAPE')
ax2.tick_params(axis='y', labelcolor=color)

# Title and grid
plt.title('RMSE and MAPE vs Number of Features')
fig.tight_layout()
plt.savefig('images/features_chart.png')  
plt.show()

'''


# Data for the updated request
models = ['Kaggle Model', 'Optimized Model']
rmse_values = [7.74578395, 3.54291136]

# Create the bar plot
fig, ax = plt.subplots()
ax.bar(models, rmse_values, color=['blue', 'orange'])

# Add labels and title
ax.set_ylabel('RMSE')
ax.set_title('RMSE: Kaggle Model vs Optimized Model')
ax.set_ylim(0, max(rmse_values) + 1)  # Setting the y-axis limit a bit higher for better visualization

# Display the plot
plt.savefig('images/rmse_comparison.png')

# Data for the MAPE comparison
mape_values = [4.22778489, 1.78532436]

# Create the bar plot for MAPE
fig, ax = plt.subplots()
ax.bar(models, mape_values, color=['blue', 'orange'])

# Add labels and title
ax.set_ylabel('MAPE')
ax.set_title('MAPE Comparison between Models')
ax.set_ylim(0, max(mape_values) + 1)  # Setting the y-axis limit a bit higher for better visualization

# Display the plot
plt.savefig('images/mape_comparison.png')
