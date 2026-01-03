import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier  
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt                     

iris_data = load_iris()
X = iris_data.data         
y = iris_data.target       
feature_names = iris_data.feature_names


df = pd.DataFrame(X, columns=feature_names)
df['species'] = y
print("First 5 rows of the dataset:")
print(df.head())


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


log_model = LogisticRegression(max_iter=200)
log_model.fit(X_train, y_train)


knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)


log_predictions = log_model.predict(X_test)
knn_predictions = knn_model.predict(X_test)

print("\n--- Logistic Regression ---")
print("Accuracy:", accuracy_score(y_test, log_predictions))
print(classification_report(y_test, log_predictions))

print("\n--- KNN Classifier ---")
print("Accuracy:", accuracy_score(y_test, knn_predictions))


try:
    sample_input = input(
        "\nEnter sepal length, sepal width, petal length, petal width (comma-separated): "
    )
    sample = [float(x) for x in sample_input.split(",")]
    predicted_class = log_model.predict([sample])
    print("Predicted species:", iris_data.target_names[predicted_class[0]])
except:
    print("Skipping manual input (no valid data entered).")


plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.title("Iris Dataset Scatter Plot")
plt.savefig("iris_scatter.png")  # saves plot in your folder
print("Plot saved as iris_scatter.png")

