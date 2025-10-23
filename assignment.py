import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('d:\sales.csv')

num_vars = ['Daily_Customer_Count', 'Store_Sales']  # Adjust with your column names

Q1 = df[num_vars].quantile(0.25)
Q3 = df[num_vars].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = {}
for var in num_vars:
    outliers[var] = df[(df[var] < lower_bound[var]) | (df[var] > upper_bound[var])][var].tolist()

print("Outliers:")
for var, outlier_values in outliers.items():
    print(f"{var}: {outlier_values}")

df_no_outliers = df.copy()
for var in num_vars:
    df_no_outliers = df_no_outliers[~df_no_outliers[var].isin(outliers[var])]

print(f"\nOriginal dataset shape: {df.shape}")
print(f"Dataset shape after removing outliers: {df_no_outliers.shape}")

for var in num_vars:
    # Calculate the IQR and bin width for the variable
    IQR_var = IQR[var]
    bin_width = (2 * IQR_var) / np.cbrt(len(df[var]))
    
    var_range = df[var].max() - df[var].min()
    
    num_bins = int(np.ceil(var_range / bin_width))
    
    plt.hist(df_no_outliers[var], bins=num_bins, edgecolor='black')

    plt.xlabel(var)
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {var} after removing outliers')

    plt.show()
