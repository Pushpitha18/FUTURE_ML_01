import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("Sample - Superstore.csv", encoding="latin1")

# Convert Order Date to datetime
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Group sales by month
monthly_sales = df.groupby(
    df['Order Date'].dt.to_period('M')
)['Sales'].sum().reset_index()

monthly_sales['Order Date'] = monthly_sales['Order Date'].astype(str)

# Create month numbers
monthly_sales['Month_Number'] = range(1, len(monthly_sales) + 1)

# Features and target
X = monthly_sales[['Month_Number']]
y = monthly_sales['Sales']

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict next 6 months
future_months = pd.DataFrame({
    'Month_Number': range(len(monthly_sales)+1,
                          len(monthly_sales)+7)
})

predictions = model.predict(future_months)

print("Future Sales Forecast:")
for i, value in enumerate(predictions, start=1):
    print(f"Future Month {i}: {value:.2f}")

# Plot actual sales
plt.figure(figsize=(10,5))
plt.plot(monthly_sales['Month_Number'],
         monthly_sales['Sales'],
         label='Actual Sales')

# Plot forecast
plt.plot(future_months['Month_Number'],
         predictions,
         label='Forecasted Sales')

plt.xlabel("Months")
plt.ylabel("Sales")
plt.title("Sales Forecasting")
plt.legend()
plt.grid()

plt.savefig("forecast_chart.png")
plt.show()