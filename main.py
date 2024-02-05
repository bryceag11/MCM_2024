import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from itertools import groupby
import datetime
import torch 
import torch.nn as nn
from lstm import LSTMMomentumPredictor
from preproc import MomentumFeatures
from sklearn.metrics import precision_score, recall_score, f1_score


# Read the provided CSV file into initial data frame
df = pd.read_csv('Wimbledon_featured_matches.csv')

label_encoders = {}
# Imputation of data
imputation_features = df[['winner_shot_type', 'serve_width', 'serve_depth', 'return_depth']].copy()

for feature in imputation_features.columns:
    if imputation_features[feature].dtype == 'object':
        
        # Initialization of the LabelEncoder for feature instance 
        le = LabelEncoder()

        # Fit and transform the current feature
        imputation_features[feature] = le.fit_transform(imputation_features[feature].fillna('Unknown'))

        # Store LabelEncoder in dict
        label_encoders[feature] = le

# Print the mapping
for feature, le in label_encoders.items():
    label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    # print(f"{feature} - {label_mapping}")

# Apply the same transformations to the corresponding columns in df
for feature, le in label_encoders.items():
    # Transform the current feature in df
    df[feature] = le.transform(df[feature].fillna('Unknown'))

# Time conversion

# Replace '25' with '1' and '24' with '0' in 'elapsed_time'
time_column = df['elapsed_time'].replace({'25:': '1:', '24:': '0:'}, regex=True)


# Convert time data to seconds
time_in_minutes = [(int(str(datetime.datetime.strptime(time_str, "%H:%M:%S").time()).split(':')[0])*60 +
                   int(str(datetime.datetime.strptime(time_str, "%H:%M:%S").time()).split(':')[1])+
                   int(str(datetime.datetime.strptime(time_str, "%H:%M:%S").time()).split(':')[2])/60)  for time_str in time_column]

# Normalize the time data

time = np.array(time_in_minutes)

# scaler = MinMaxScaler()
# normalized_time = scaler.fit_transform(time_in_seconds)

df['elapsed_time'] = time


# Score normalization

# col_norm = ['set_no', 'game_no','p1_sets', 'p2_sets']

# df[col_norm] = df[col_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

# Additional score normalization
df['p1_score'] = df['p1_score'].replace('AD', 50).astype(int)
# df['p1_score'] = df['p1_score'].replace(40, 0.03).astype(int)
# df['p1_score'] = df['p1_score'].replace(30, 0.02).astype(int)
# df['p1_score'] = df['p1_score'].replace(15, 0.01).astype(int)

df['p2_score'] = df['p2_score'].replace('AD', 50).astype(int)
# df['p2_score'] = df['p2_score'].replace(40, 0.03).astype(int)
# df['p2_score'] = df['p2_score'].replace(30, 0.02).astype(int)
# df['p2_score'] = df['p2_score'].replace(15, 0.01).astype(int)


# # Normalize the distances
# scaler = MinMaxScaler(feature_range=(-1, 1))
# df['p1_distance_run'] = scaler.fit_transform(df['p1_distance_run'].values.reshape(-1, 1))

# scaler = MinMaxScaler(feature_range=(-1, 1))
# df['p2_distance_run'] = scaler.fit_transform(df['p2_distance_run'].values.reshape(-1, 1))

# # Impute NA values in 'speed_mph' with 0
df['speed_mph'].fillna(0, inplace=True)

# # Scale 'speed_mph'
# scaler = MinMaxScaler()
# df['speed_mph'] = scaler.fit_transform(df[['speed_mph']])
dft = df.copy()
df = df.iloc[0:302]

dft = dft.iloc[6952:7286]
        

momentum = MomentumFeatures(df)


momentum_dfs = []
momentum_df = []
for i in range(len(df)):
    momentum_df = momentum.process_row(i)
momentum_dfs.append(momentum_df)

momentum2 = MomentumFeatures(dft)



momentum_df = pd.concat(momentum_dfs, ignore_index=True)

momentum_dfst = []
momentum_dft = []
for i in range(len(dft)):
    momentum_dft = momentum2.process_row(i)
momentum_dfst.append(momentum_dft)

momentum_dft = pd.concat(momentum_dfst, ignore_index=True)

training_set = momentum_df.iloc[:, :16]
test_set = momentum_dft.iloc[:, :16]

normalized_df = (training_set - training_set.mean()) / training_set.std()
normalized_dft = (test_set - test_set.mean()) / test_set.std()

nan_columns_train = normalized_df.columns[normalized_df.isna().any()].tolist()
nan_rows_train = normalized_df[normalized_df.isna().any(axis=1)]

nan_columns_test = normalized_dft.columns[normalized_dft.isna().any()].tolist()
nan_rows_test = normalized_dft[normalized_dft.isna().any(axis=1)]

print("Columns with NaN values in training data:", nan_columns_train)
print("Rows with NaN values in training data:", nan_rows_train)

print("Columns with NaN values in test data:", nan_columns_test)
print("Rows with NaN values in test data:", nan_rows_test)

feature_weights = np.array([0.7, 0.7, 0.8, 0.8, 0.9, 0.9, 1.0, 1.0, 0.2, 0.2, 0.5, 0.5, 0.4, 0.4, 0.2, 0.2])
weighted_df = normalized_df * feature_weights 
weighted_dft = normalized_dft * feature_weights

momentum_p1 = weighted_df.iloc[:, [0, 2, 4, 6, 8, 10, 12, 14]].sum(axis=1)
momentum_p2 = weighted_df.iloc[:, [1, 3, 5, 7, 9, 11, 13, 15]].sum(axis=1)

momentum_p1t = weighted_dft.iloc[:, [0, 2, 4, 6, 8, 10, 12, 14]].sum(axis=1)
momentum_p2t = weighted_dft.iloc[:, [1, 3, 5, 7, 9, 11, 13, 15]].sum(axis=1)

# Combine into a matrix
raw_momentums = np.vstack([momentum_p1, momentum_p2])
raw_momentumst = np.vstack([momentum_p1t, momentum_p2t])

# Apply softmax activation
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=0, keepdims=True)

normalized_momentums = softmax(raw_momentums)
normalized_momentumst = softmax(raw_momentumst)


# # Separate back into individual player momentums
normalized_momentum_p1 = normalized_momentums[0, :]
normalized_momentum_p1t = normalized_momentumst[0, :]
normalized_momentum_p2 = normalized_momentums[1, :]
normalized_momentum_p2t = normalized_momentumst[1, :]

# ###################################################################################


# #######################
# # Assuming feature_df is your feature DataFrame and y_train_p1, y_train_p2 are target data
feature_df_np = normalized_df.to_numpy(dtype=np.float32)
feature_dft_np = normalized_dft.to_numpy(dtype=np.float32)

x_train = torch.from_numpy(feature_df_np).float().unsqueeze(0)
x_test = torch.from_numpy(feature_dft_np).float().unsqueeze(0)

# Reshape x_train to (1, 10, 142) assuming you have 10 features and 142 data points
x_train = x_train.transpose(1, 2)
x_test = x_test.transpose(1, 2)


# Assuming y_train_p1 and y_train_p2 are your target data for momentum_p1 and momentum_p2
y_train_p1 = torch.tensor(normalized_momentum_p1, dtype=torch.float32)
y_train_p2 = torch.tensor(normalized_momentum_p2, dtype=torch.float32)
y_test_p1 = torch.tensor(normalized_momentum_p1t, dtype=torch.float32)
y_test_p2 = torch.tensor(normalized_momentum_p2t, dtype=torch.float32)






# Instantiate the model
model = LSTMMomentumPredictor()

# Define the loss function
criterion = nn.MSELoss()

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 100
batch_size = 1
training_losses = []  # To store training losses for visualization

print(len(x_train))
for epoch in range(epochs):
    epoch_loss = 0.0  # Accumulator for epoch loss

    # Training loop
    for i in range(0, len(x_train), batch_size):
        x_batch = x_train[i:i+batch_size]
        y_batch_p1 = y_train_p1[i:i+batch_size]
        y_batch_p2 = y_train_p2[i:i+batch_size]

        # Forward pass
        predictions_p1, predictions_p2 = model(x_batch)
        if torch.isnan(predictions_p1).any() or torch.isnan(predictions_p2).any() or torch.isinf(predictions_p1).any() or torch.isinf(predictions_p2).any():
            raise ValueError("Model output contains NaN or Infinite values.")

        # Compute individual losses
        loss_p1 = criterion(predictions_p1.squeeze(), y_batch_p1)
        loss_p2 = criterion(predictions_p2.squeeze(), y_batch_p2)

        # Total loss
        total_loss = loss_p1 + loss_p2
        epoch_loss += total_loss.item()  # Accumulate loss for the epoch

        # Backward pass and optimization
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    # Store average epoch loss
    training_losses.append(epoch_loss / len(x_train))

    # Print training statistics
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {training_losses[-1]}')


# model.eval()  # Set the model to evaluation mode

# with torch.no_grad():
#     # Assuming x_test, y_test_p1, and y_test_p2 are your test set features and targets
#     # x_test.transpose(1, 2)

#     # Forward pass
#     predictions_p1, predictions_p2 = model(x_test)

#     # Assuming y_test_p1 and y_test_p2 are the true labels for the test set
#     y_test_p1 = y_test_p1.flatten().numpy()

#     y_test_p2 = y_test_p2.flatten().numpy()


#     # Set the threshold for classification
#     threshold = 0.5

#     # Convert continuous predictions to binary labels
#     binary_predictions_p1 = (predictions_p1 > threshold).float()
#     binary_predictions_p2 = (predictions_p2 > threshold).float()

#     # Print the shapes of binary predictions
#     print("Binary Predictions Player 1:", binary_predictions_p1.shape)
#     print("Binary Predictions Player 2:", binary_predictions_p2.shape)
#     # Compute metrics for player 1
#     precision_p1 = precision_score(y_test_p1, binary_predictions_p1)
#     recall_p1 = recall_score(y_test_p1, binary_predictions_p1)
#     f1_p1 = f1_score(y_test_p1, binary_predictions_p1)

#     # Compute metrics for player 2
#     precision_p2 = precision_score(y_test_p2, binary_predictions_p2)
#     recall_p2 = recall_score(y_test_p2, binary_predictions_p2)
#     f1_p2 = f1_score(y_test_p2, binary_predictions_p2)

#     # Print evaluation metrics
#     print('\nEvaluation Metrics:')
#     print(f'Player 1 - Precision: {precision_p1}, Recall: {recall_p1}, F1: {f1_p1}')
#     print(f'Player 2 - Precision: {precision_p2}, Recall: {recall_p2}, F1: {f1_p2}')