# Add import statements
import pandas as pd
from sklearn.linear_model import LinearRegression

# Assign the dataframe to this variable.
# Load the data
bmi_life_data = pd.read_csv("bmi_and_life_expectancy.csv")
y_values = bmi_life_data[['Life expectancy']]
x_values = bmi_life_data[['BMI']]

# Make and fit the linear regression model
# Fit the model and Assign it to bmi_life_model
bmi_life_model = LinearRegression()
bmi_life_model.fit(x_values, y_values)

# Mak a prediction using the model
# Predict life expectancy for a BMI value of 21.07931
laos_life_exp = bmi_life_model.predict(21.07931)
print(laos_life_exp)