import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

csv_path = 'test.csv'
df = pd.read_csv(csv_path)

X = df['x'].values
Y = df['y'].values

mean_X = np.mean(X)
mean_Y = np.mean(Y)

numerator = np.sum((X - mean_X) * (Y - mean_Y))
denominator = np.sum((X - mean_X) ** 2)
m = numerator / denominator
b = mean_Y - m * mean_X

Y_pred = m * X + b

plt.figure(figsize=(10, 6))
plt.scatter(X, Y, color='blue', edgecolor='black', alpha=0.7, label='Actual Data')
plt.plot(X, Y_pred, color='green', linewidth=2, label='Linear Regression Line')
plt.xlabel('X', fontsize=14)
plt.ylabel('Y', fontsize=14)
plt.title('Enhanced Linear Regression Plot', fontsize=16)
plt.legend(frameon=True, shadow=True, borderpad=1)
plt.grid(True)
plt.show()
