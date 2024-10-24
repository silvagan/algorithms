import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load data from TSV file
data = pd.read_csv('data_for_task3.tsv', sep='\t')

# Extract input and output variables
X = data[['LotArea', 'OverallQual', 'YearBuilt']].values
y = data['SalePrice'].values

# Normalize the input data
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# Normalize the target variable with StandardScaler
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# Initialize weights
W = np.array([1, 0.2, 0.1, 1, 0.2, 1, 0.2, 0.1, 1, 0.2, 1, 0.2, 0.1, 1, 0.2])

# Neural Network Forward Pass
def forward_pass(X, W):
    n1, n2, n3 = X[:, 0], X[:, 1], X[:, 2]
    n4 = n1 * W[0] + n2 * W[5] + n3 * W[10]
    n5 = n1 * W[1] + n2 * W[6] + n3 * W[11]
    n6 = n1 * W[2] + n2 * W[7] + n3 * W[12]
    n7 = n1 * W[3] + n2 * W[8] + n3 * W[13]
    n8 = n1 * W[4] + n2 * W[9] + n3 * W[14]
    n9 = n4 + n5 + n6 + n7 + n8
    return n9

# Mean Squared Error
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Compute Gradients
def compute_gradients(X, y, W):
    y_pred = forward_pass(X, W)
    error = y_pred - y
    gradients = np.zeros_like(W)

    for i in range(len(W)):
        if i < 5:
            gradients[i] = 2 * np.mean(error * X[:, 0])
        elif i < 10:
            gradients[i] = 2 * np.mean(error * X[:, 1])
        else:
            gradients[i] = 2 * np.mean(error * X[:, 2])
    
    return gradients

# Steepest Descent Optimization
def steepest_descent(X, y, W, learning_rate=0.0001, epochs=2000, lambda_reg=0.01):
    mse_history = []
    for epoch in range(epochs):
        y_pred = forward_pass(X, W)
        mse = mean_squared_error(y, y_pred)
        mse_history.append(mse)
        
        gradients = compute_gradients(X, y, W)

        # Calculate step size for steepest descent
        XtX = X.T @ X
        gradient_XtX = np.dot(gradients[:3], XtX @ gradients[:3])
        step_size = np.dot(gradients, gradients) / (gradient_XtX + lambda_reg * np.dot(gradients, gradients))
        
        W -= step_size * gradients
        
        # Clipping weights to prevent explosion
        W = np.clip(W, -1, 1)
        
    return W, mse_history


# Initial Mean Squared Error
y_pred_initial = forward_pass(X_scaled, W)
initial_mse = mean_squared_error(y_scaled, y_pred_initial)
initial_weights = W.copy()

# Optimize Weights
W_optimized, mse_history = steepest_descent(X_scaled, y_scaled, W, learning_rate=0.0001, epochs=500)

# Final Mean Squared Error
y_pred_final = forward_pass(X_scaled, W_optimized)
final_mse = mean_squared_error(y_scaled, y_pred_final)

# Denormalize predictions to compare actual values
y_pred_initial = scaler_y.inverse_transform(y_pred_initial.reshape(-1, 1)).flatten()
y_pred_final = scaler_y.inverse_transform(y_pred_final.reshape(-1, 1)).flatten()

# Plot MSE History
plt.plot(mse_history)
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.title('Mean Squared Error During Optimization')
plt.show()

# Print Errors
print(f'Initial MSE: {initial_mse}')
print(f'Final MSE: {final_mse}')

# Table of Weights Before and After Optimization
weights_df = pd.DataFrame({
    'Weight': [f'w{i}' for i in range(1, 16)],
    'Initial': initial_weights,
    'Optimized': W_optimized
})
print(weights_df)

# Table of First 10 Predictions
predictions_df = pd.DataFrame({
    'LotArea': X[:10, 0],
    'OverallQual': X[:10, 1],
    'YearBuilt': X[:10, 2],
    'Known SalePrice': y[:10],
    'Predicted SalePrice (Initial)': y_pred_initial[:10],
    'Predicted SalePrice (Optimized)': y_pred_final[:10]
})
print(predictions_df)

# Table of Error Values Before and After Optimization
errors_df = pd.DataFrame({
    'Error Type': ['Mean Squared Error', 'Mean Absolute Error'],
    'Initial': [initial_mse, np.mean(np.abs(y_scaled - forward_pass(X_scaled, initial_weights)))],
    'Final': [final_mse, np.mean(np.abs(y_scaled - forward_pass(X_scaled, W_optimized)))]
})
print(errors_df)
