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

# Additional score normalization
df['p1_score'] = df['p1_score'].replace('AD', 50).astype(int)
df['p2_score'] = df['p2_score'].replace('AD', 50).astype(int)
df['speed_mph'].fillna(0, inplace=True)


dft = df.copy()
df = df.iloc[0:302]

dft = dft.iloc[6952:7254]
        
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

mean_training_set = training_set.mean()
std_training_set = training_set.std()

# Check if standard deviation is zero and replace with a small value
std_training_set[std_training_set == 0] = 1e-6

# Normalize the training set
normalized_df = (training_set - mean_training_set) / std_training_set

# Use the mean and standard deviation from the training set to normalize the test set
normalized_dft = (test_set - mean_training_set) / std_training_set

# Assuming you have normalized_df and normalized_dft defined

feature_weights = np.array([0.7, 0.7, 0.8, 0.8, 0.9, 0.9, 1.0, 1.0, 0.2, 0.2, 0.5, 0.5, 0.4, 0.4, 0.2, 0.2])
weighted_df = normalized_df * feature_weights 
weighted_dft = normalized_dft * feature_weights

weighted_df.fillna(0, inplace=True)
weighted_dft.fillna(0, inplace=True)

mom_p1 = weighted_df.iloc[:, ::2]
mom_p2 = weighted_df.iloc[:, 1::2]

mom_p1t = weighted_dft.iloc[:, ::2]
mom_p2t = weighted_dft.iloc[:, 1::2]


momentum_p1 = pd.DataFrame(index=mom_p1.index, columns=['Momentum_P1'])
momentum_p2 = pd.DataFrame(index=mom_p2.index, columns=['Momentum_P2'])

momentum_p1t = pd.DataFrame(index=mom_p1t.index, columns=['Momentum_P1T'])
momentum_p2t = pd.DataFrame(index=mom_p2t.index, columns=['Momentum_P2T'])


def normalize_values(mom_sum):
    mom_sum = np.maximum(0, mom_sum)  # Set negative values to 0
    mom_sum = np.minimum(1, mom_sum)  # Set values greater than 1 to 1
    return mom_sum

for i in range(len(mom_p1)):
    mom_p1_sum = mom_p1.iloc[i, :].sum()
    mom_p2_sum = mom_p2.iloc[i, :].sum()

    # Normalize the sum to ensure it equals 1
    total_momentum = mom_p1_sum + mom_p2_sum
    if total_momentum != 1:
        mom_p1_sum /= total_momentum
        mom_p2_sum /= total_momentum

    mom_p1_sum = normalize_values(mom_p1_sum)
    mom_p2_sum = normalize_values(mom_p2_sum)
    # Update DataFrame
    momentum_p1.loc[i, 'Momentum_P1'] = mom_p1_sum
    momentum_p2.loc[i, 'Momentum_P2'] = mom_p2_sum

for i in range(len(mom_p1t)):
    mom_p1t_sum = mom_p1t.iloc[i, :].sum()
    mom_p2t_sum = mom_p2t.iloc[i, :].sum()

    # Normalize the sum to ensure it equals 1
    total_momentum_t = mom_p1t_sum + mom_p2t_sum
    if total_momentum_t != 1:
        mom_p1t_sum /= total_momentum_t
        mom_p2t_sum /= total_momentum_t

    mom_p1t_sum = normalize_values(mom_p1t_sum)
    mom_p2t_sum = normalize_values(mom_p2t_sum)

    # Update DataFrame
    momentum_p1t.loc[i, 'Momentum_P1T'] = mom_p1t_sum
    momentum_p2t.loc[i, 'Momentum_P2T'] = mom_p2t_sum
print(momentum_p1, momentum_p2)

# Assuming feature_df is your feature DataFrame and y_train_p1, y_train_p2 are target data
feature_df_np = normalized_df.to_numpy(dtype=np.float32)
feature_dft_np = normalized_dft.to_numpy(dtype=np.float32)

x_train = torch.from_numpy(feature_df_np).float().unsqueeze(0)
x_test = torch.from_numpy(feature_dft_np).float().unsqueeze(0)

# Reshape x_train to (1, 16, 302) assuming you have 16 features and 302 data points
x_train = x_train.transpose(1, 2)
x_test = x_test.transpose(1, 2)

numpy_momentum_p1 = momentum_p1.values.astype(np.float32)
numpy_momentum_p2 = momentum_p2.values.astype(np.float32)
numpy_momentum_p1t = momentum_p1t.values.astype(np.float32)
numpy_momentum_p2t = momentum_p2t.values.astype(np.float32)

# Convert to PyTorch tensors
y_train_p1 = torch.tensor(numpy_momentum_p1, dtype=torch.float32)
y_train_p2 = torch.tensor(numpy_momentum_p2, dtype=torch.float32)
y_test_p1 = torch.tensor(numpy_momentum_p1t, dtype=torch.float32)
y_test_p2 = torch.tensor(numpy_momentum_p2t, dtype=torch.float32)

plt.figure(figsize=(15, 8))

# Calculate the delta or change in scores for both P1 and P2
delta_p1 = df['p1_score'].diff()
delta_p2 = df['p2_score'].diff()

# Normalize the scores between 0 and 1
normalized_p1 = (df['p1_score'] - df['p1_score'].min()) / (df['p1_score'].max() - df['p1_score'].min())
normalized_p2 = (df['p2_score'] - df['p2_score'].min()) / (df['p2_score'].max() - df['p2_score'].min())

# Plot the delta or change in points against the momentum for P1
fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(24, 16))

# Plot the scaled delta scores against the momentum for P1
axes[0].plot(range(302), momentum_p1, label='Momentum P1 (Train)')
axes[0].plot(range(302), normalized_p1, label='Scaled Delta P1 Score')
axes[0].set_ylabel('Momentum P1 / Scaled Delta P1 Score')
axes[0].set_title('Momentum P1 vs Scaled Delta P1 Score (Train)')
axes[0].legend()

# Plot the scaled delta scores against the momentum for P2
axes[1].plot(range(302), momentum_p2, label='Momentum P2 (Train)')
axes[1].plot(range(302), normalized_p2, label='Scaled Delta P2 Score')
axes[1].set_xlabel('Data Index')
axes[1].set_ylabel('Momentum P2 / Scaled Delta P2 Score')
axes[1].set_title('Momentum P2 vs Scaled Delta P2 Score (Train)')
axes[1].legend()

# Adjust layout for better spacing
plt.tight_layout()

plt.show()

# Instantiate the model
model = LSTMMomentumPredictor()

# Define the loss function
criterion = nn.MSELoss()

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 1000
batch_size = 1
training_losses = []  # To store training losses for visualization

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

epochs_range = range(1, epochs + 1)

from sklearn.metrics import mean_squared_error, r2_score, precision_recall_fscore_support, precision_recall_curve

def evaluate_model(model, x_test, y_test_p1, y_test_p2, threshold_p1, threshold_p2):
    model.eval()

    with torch.no_grad():
        # Forward pass on the test set
        y_pred_p1, y_pred_p2 = model(x_test)

        # Regression Metrics
        mse_p1 = mean_squared_error(y_test_p1, y_pred_p1.squeeze().numpy())
        mse_p2 = mean_squared_error(y_test_p2, y_pred_p2.squeeze().numpy())

        r2_p1 = r2_score(y_test_p1, y_pred_p1.squeeze().numpy())
        r2_p2 = r2_score(y_test_p2, y_pred_p2.squeeze().numpy())

        print(f'Mean Squared Error (MSE) - Momentum P1: {mse_p1:.4f}')
        print(f'Mean Squared Error (MSE) - Momentum P2: {mse_p2:.4f}')
        print(f'R-squared (R2) - Momentum P1: {r2_p1:.4f}')
        print(f'R-squared (R2) - Momentum P2: {r2_p2:.4f}')

        # Classification Metrics
        y_pred_p1_binary = (y_pred_p1.squeeze().numpy() > threshold_p1).astype(int)
        y_pred_p2_binary = (y_pred_p2.squeeze().numpy() > threshold_p2).astype(int)

        y_test_p1_binary = (y_test_p1 > threshold_p1).numpy().astype(int)
        y_test_p2_binary = (y_test_p2 > threshold_p2).numpy().astype(int)

        precision_p1, recall_p1, f1_p1, _ = precision_recall_fscore_support(
            y_test_p1_binary, y_pred_p1_binary, average='binary')
        
        precision_p2, recall_p2, f1_p2, _ = precision_recall_fscore_support(
            y_test_p2_binary, y_pred_p2_binary, average='binary')
        f1_p1 = f1_score(y_test_p1_binary, y_pred_p1_binary)

        # Classification Metrics - Momentum P2
        y_pred_p2_binary = (y_pred_p2.squeeze().numpy() > threshold_p2).astype(int)
        y_test_p2_binary = (y_test_p2 > threshold_p2).numpy().astype(int)

        precision_p2, recall_p2, _, _ = precision_recall_fscore_support(
            y_test_p2_binary, y_pred_p2_binary, average='binary')

        f1_p2 = f1_score(y_test_p2_binary, y_pred_p2_binary)

        # Plot Precision-Recall Curve
        plt.figure(figsize=(12, 6))

        plt.subplot(2, 2, 1)
        precision_curve, recall_curve, _ = precision_recall_curve(y_test_p1_binary, y_pred_p1_binary)
        plt.plot(recall_curve, precision_curve, label='Momentum P1')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve - Momentum P1')
        plt.legend()

        plt.subplot(2, 2, 2)
        precision_curve, recall_curve, _ = precision_recall_curve(y_test_p2_binary, y_pred_p2_binary)
        plt.plot(recall_curve, precision_curve, label='Momentum P2')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve - Momentum P2')
        plt.legend()

        plt.show()

        # Plot Predicted vs True Values
        plt.figure(figsize=(12, 6))

        plt.subplot(3, 2, 1)
        plt.scatter(y_test_p1.numpy(), y_pred_p1.squeeze().numpy(), c=y_test_p1_binary, cmap='viridis', label='Momentum P1')
        plt.xlabel('True Values - Momentum P1')
        plt.ylabel('Predicted Values - Momentum P1')
        plt.title('Predicted vs True Values - Momentum P1')
        plt.legend()

        plt.subplot(3, 2, 2)
        plt.scatter(y_test_p2.numpy(), y_pred_p2.squeeze().numpy(), c=y_test_p1_binary, cmap='viridis', label='Momentum P2')
        plt.xlabel('True Values - Momentum P2')
        plt.ylabel('Predicted Values - Momentum P2')
        plt.title('Predicted vs True Values - Momentum P2')
        plt.legend()

        plt.show()

        print(f'Threshold - Momentum P1: {threshold_p1}')
        print(f'Precision - Momentum P1: {precision_p1:.4f}')
        print(f'Recall - Momentum P1: {recall_p1:.4f}')
        print(f'F1 Score - Momentum P1: {f1_p1:.4f}')

        print(f'Threshold - Momentum P2: {threshold_p2}')
        print(f'Precision - Momentum P2: {precision_p2:.4f}')
        print(f'Recall - Momentum P2: {recall_p2:.4f}')
        print(f'F1 Score - Momentum P2: {f1_p2:.4f}')


        plt.tight_layout()
        plt.show()
# Assuming x_test is your test data
evaluate_model(model, x_test, y_test_p1, y_test_p2, threshold_p1=0.5, threshold_p2=0.75)
