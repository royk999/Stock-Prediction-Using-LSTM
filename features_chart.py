import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

features = [10, 25, 50, 100, 150, 200, 250, 300, 350, 400, 500, 600, 700, 800, 900]
rmse = [4.33031544, 4.11832856, 3.81549082, 3.68967349, 3.62545782, 3.56707019, 3.57445159,
        3.61364948, 3.59228132, 3.56560424, 3.56809917, 3.5513335, 3.54291136, 3.56032676, 3.56196746]


# Updated data with MAPE values
mape = [2.17085935, 2.08126698, 1.93068776, 1.8692754, 1.83478075, 1.82721202, 1.80644031, 
        1.82766092, 1.815465, 1.8006983, 1.80192267, 1.7923392, 1.78532436, 1.79511613, 1.79583961]

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
