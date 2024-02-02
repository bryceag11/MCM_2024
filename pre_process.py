import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Read the provided CSV file
df = pd.read_csv('Wimbledon_featured_matches.csv')

# Read the correct/reference data

# Check if the DataFrames are equal




# feature and target extraction
features = df.drop('game_victor', axis=1)
target = df['game_victor']

# Checking label distribution
plt.hist(target)
plt.show()

# Data normalization