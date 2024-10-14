import pandas as pd

# Load the CSV file, assuming the first row contains the header
df = pd.read_csv('bitcoin.csv', names=['ds', 'y'])

# Convert 'ds' column to datetime and format it to 'HH-MM-SS'
df['ds'] = pd.to_datetime(df['ds'], errors='coerce').dt.strftime('%H-%M-%S')
df['y'] = df['y'].astype(str)

# Select only the necessary columns
result_df = df[['ds', 'y']]

# Save the result to a new CSV file
result_df.to_csv('qwe.csv', header=False, index=False)

print("Data has been transformed and saved to 'qwe.csv'.")
