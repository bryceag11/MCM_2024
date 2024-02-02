import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Read the CSV file
df = pd.read_csv('Wimbledon_featured_matches.csv')

# feature and target extraction
features = df.drop('game_victor', axis=1)
target = df['game_victor']

# Checking label distribution
plt.hist(target)
plt.show()

# Data normalization