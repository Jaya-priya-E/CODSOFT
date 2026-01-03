import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv(r"E:\Priya\MyWorks\Codsoft\MovieRatingPrediction\movies.csv")

print("\nDataset Preview:")
print(data.head())

genre_encoder = LabelEncoder()
data["Genre"] = genre_encoder.fit_transform(data["Genre"])

X = data[["Genre", "Year", "Duration", "Votes"]]
y = data["Rating"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nModel Performance:")
print("Mean Absolute Error:", round(mean_absolute_error(y_test, y_pred), 2))
print("R2 Score:", round(r2_score(y_test, y_pred), 2))


print("\nAvailable Genres:", list(genre_encoder.classes_))

genre_input = input("Enter Genre: ")
year = int(input("Enter Release Year: "))
duration = int(input("Enter Duration (minutes): "))
votes = int(input("Enter Number of Votes: "))

genre_encoded = genre_encoder.transform([genre_input])[0]

user_movie = np.array([[genre_encoded, year, duration, votes]])

predicted_rating = model.predict(user_movie)

print("\nPredicted Movie Rating:", round(predicted_rating[0], 2))
