
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create a directory for plots if it doesn't exist
if not os.path.exists('c:\\Users\\KAILAS\\Desktop\\assignment 1\\plots'):
    os.makedirs('c:\\Users\\KAILAS\\Desktop\\assignment 1\\plots')

# Load the data
file_path = r"c:\\Users\\KAILAS\\Desktop\\assignment 1\\Train\\Assessment 2 - MMM Weekly.csv"
df = pd.read_csv(file_path)

# Convert 'week' to datetime and set as index
df['week'] = pd.to_datetime(df['week'])
df.set_index('week', inplace=True)

# Plot 1: Revenue over time
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['revenue'])
plt.title('Revenue Over Time')
plt.xlabel('Week')
plt.ylabel('Revenue')
plt.grid(True)
plt.savefig('c:\\Users\\KAILAS\\Desktop\\assignment 1\\plots/revenue_over_time.png')
plt.close()

# Plot 2: Spend variables over time
plt.figure(figsize=(12, 6))
for col in ['facebook_spend', 'google_spend', 'tiktok_spend', 'instagram_spend', 'snapchat_spend']:
    plt.plot(df.index, df[col], label=col)
plt.title('Spend Over Time')
plt.xlabel('Week')
plt.ylabel('Spend')
plt.legend()
plt.grid(True)
plt.savefig('c:\\Users\\KAILAS\\Desktop\\assignment 1\\plots/spend_over_time.png')
plt.close()

# Plot 3: Histogram of Revenue
plt.figure(figsize=(10, 6))
sns.histplot(df['revenue'], kde=True)
plt.title('Distribution of Revenue')
plt.xlabel('Revenue')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig('c:\\Users\\KAILAS\\Desktop\\assignment 1\\plots/revenue_histogram.png')
plt.close()

# Plot 4: Correlation Matrix Heatmap
plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Variables')
plt.savefig('c:\\Users\\KAILAS\\Desktop\\assignment 1\\plots/correlation_heatmap.png')
plt.close()

print("Plots saved to the 'plots' directory.")
