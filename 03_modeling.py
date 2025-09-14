import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 1. Load and Prepare Data ---

# Create a directory for plots if it doesn't exist
plots_dir = r"plots"
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

file_path = r"Train\Assessment 2 - MMM Weekly.csv"
df = pd.read_csv(file_path)

# Convert 'week' to datetime and set as index
df['week'] = pd.to_datetime(df['week'])
df.set_index('week', inplace=True)

# --- 2. Feature Engineering ---

# Target transformation
df['revenue_log'] = np.log1p(df['revenue'])

# Time-based features
df['time_trend'] = np.arange(len(df))
df['month'] = df.index.month
df['week_of_year'] = df.index.isocalendar().week.astype(int)


# Lagged features for spend
for col in ['facebook_spend', 'google_spend', 'tiktok_spend', 'instagram_spend', 'snapchat_spend', 'emails_send', 'sms_send']:
    for i in range(1, 5): # 4 weeks lag
        df[f'{col}_lag_{i}'] = df[col].shift(i)

# Drop rows with NaNs created by lagging
df.dropna(inplace=True)


# --- 3. Two-Stage Modeling (Causal Framing) ---

# Define features for Stage 1
features_stage1 = [
    'facebook_spend', 'tiktok_spend', 'instagram_spend', 'snapchat_spend',
    'promotions', 'emails_send', 'sms_send',
    'time_trend', 'month', 'week_of_year'
] + [col for col in df.columns if '_lag_' in col and 'google_spend' not in col]

X1 = df[features_stage1]
y1 = df['google_spend']

# Scale features for Stage 1
scaler1 = StandardScaler()
X1_scaled = scaler1.fit_transform(X1)

# Stage 1 Model: Predict Google Spend
# Using RidgeCV to find the best alpha
stage1_model = RidgeCV(alphas=np.logspace(-3, 3, 100), store_cv_results=True)
stage1_model.fit(X1_scaled, y1)

# Get residuals from Stage 1
google_spend_residuals = y1 - stage1_model.predict(X1_scaled)
df['google_spend_residuals'] = google_spend_residuals


# --- 4. Prepare Data for Stage 2 ---

# Define features for Stage 2
features_stage2 = [
    'facebook_spend', 'tiktok_spend', 'instagram_spend', 'snapchat_spend',
    'promotions', 'emails_send', 'sms_send', 'social_followers', 'average_price',
    'time_trend', 'month', 'week_of_year',
    'google_spend_residuals'  # Key causal feature
] + [col for col in df.columns if '_lag_' in col]


X2 = df[features_stage2]
y2 = df['revenue_log']

# Time-series split
tscv = TimeSeriesSplit(n_splits=5, test_size=10)
for train_index, test_index in tscv.split(X2):
    X_train, X_test = X2.iloc[train_index], X2.iloc[test_index]
    y_train, y_test = y2.iloc[train_index], y2.iloc[test_index]


# Scale features for Stage 2
scaler2 = StandardScaler()
X_train_scaled = scaler2.fit_transform(X_train)
X_test_scaled = scaler2.transform(X_test)


# --- 5. Stage 2 Model: Predict Revenue ---

# Using RidgeCV to find the best alpha for the revenue model
stage2_model = RidgeCV(alphas=np.logspace(-3, 3, 100))
stage2_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_log = stage2_model.predict(X_test_scaled)
y_pred = np.expm1(y_pred_log) # Inverse transform
y_test_actual = np.expm1(y_test) # Inverse transform for comparison

# --- 6. Evaluation and Diagnostics ---

rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
print(f"Root Mean Squared Error (RMSE) on the test set: {rmse:.2f}")


# Plot actual vs. predicted revenue
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test_actual, label='Actual Revenue')
plt.plot(y_test.index, y_pred, label='Predicted Revenue', linestyle='--')
plt.title('Actual vs. Predicted Revenue (Test Set)')
plt.xlabel('Week')
plt.ylabel('Revenue')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plots_dir, 'actual_vs_predicted.png'))
plt.close()


# Residual analysis
residuals = y_test_actual - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.title('Distribution of Residuals (Test Set)')
plt.xlabel('Residual (Actual - Predicted)')
plt.grid(True)
plt.savefig(os.path.join(plots_dir, 'residuals_distribution.png'))
plt.close()


# --- 7. Insights and Interpretation ---

# Get coefficients from the final model
coefficients = pd.DataFrame({
    'feature': X2.columns,
    'coefficient': stage2_model.coef_
})

# Sort by absolute value of coefficient
coefficients['abs_coefficient'] = coefficients['coefficient'].abs()
coefficients = coefficients.sort_values(by='abs_coefficient', ascending=False).drop(columns=['abs_coefficient'])

print("\n--- Model Coefficients (Drivers of Revenue) ---")
print(coefficients.to_string())


# Save coefficients to a file
coefficients.to_csv(r"revenue_drivers.csv", index=False)

print("\nModeling script finished. Plots and coefficients saved.")
