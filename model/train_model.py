import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load Dataset
df = pd.read_csv("train.csv")

# Select Features (simplified)
X = df[['GrLivArea', 'OverallQual', 'YearBuilt']]
y = df['SalePrice']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'house_model.pkl')
