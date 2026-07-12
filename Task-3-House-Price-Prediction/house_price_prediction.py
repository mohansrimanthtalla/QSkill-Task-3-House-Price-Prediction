import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load dataset
data = pd.read_csv("data/Housing.csv")

# Convert Yes/No columns
binary_columns = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea"
]

for column in binary_columns:
    data[column] = data[column].map({"yes": 1, "no": 0})

# Convert furnishing status
data["furnishingstatus"] = data["furnishingstatus"].map({
    "furnished": 2,
    "semi-furnished": 1,
    "unfurnished": 0
})

# Features and Target
X = data.drop("price", axis=1)
y = data["price"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Accuracy
accuracy = r2_score(y_test, predictions)

print("===== House Price Prediction =====")
print(f"Model Accuracy (R² Score): {accuracy:.2f}")

print("\nSample Predictions")

for i in range(5):
    print(f"Actual Price: {y_test.iloc[i]}")
    print(f"Predicted Price: {int(predictions[i])}")
    print("-" * 40)

    print("\n===== Predict Your Own House Price =====")

area = int(input("Enter Area: "))
bedrooms = int(input("Enter Bedrooms: "))
bathrooms = int(input("Enter Bathrooms: "))
stories = int(input("Enter Stories: "))
mainroad = int(input("Main Road (1=Yes, 0=No): "))
guestroom = int(input("Guest Room (1=Yes, 0=No): "))
basement = int(input("Basement (1=Yes, 0=No): "))
hotwaterheating = int(input("Hot Water Heating (1=Yes, 0=No): "))
airconditioning = int(input("Air Conditioning (1=Yes, 0=No): "))
parking = int(input("Parking Spaces: "))
prefarea = int(input("Preferred Area (1=Yes, 0=No): "))
furnishingstatus = int(input("Furnishing Status (2=Furnished, 1=Semi-Furnished, 0=Unfurnished): "))

new_house = [[
    area,
    bedrooms,
    bathrooms,
    stories,
    mainroad,
    guestroom,
    basement,
    hotwaterheating,
    airconditioning,
    parking,
    prefarea,
    furnishingstatus
]]

predicted_price = model.predict(new_house)

print("\n=======================================")
print(f"Predicted House Price: ₹{predicted_price[0]:,.2f}")
print("=======================================")