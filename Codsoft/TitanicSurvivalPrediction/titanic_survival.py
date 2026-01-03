import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('train.csv')


data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

data.drop(['Cabin', 'Ticket', 'Name'], axis=1, inplace=True)

le = LabelEncoder()
data['Sex'] = le.fit_transform(data['Sex'])
data['Embarked'] = le.fit_transform(data['Embarked'])

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = data[features]
y = data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

def predict_survival():
    print("Enter new passenger details:")
    Pclass = int(input("Passenger Class (1/2/3): "))
    Sex = input("Sex (male/female): ").lower()
    Sex = 1 if Sex == 'male' else 0
    Age = float(input("Age: "))
    SibSp = int(input("Number of siblings/spouses aboard: "))
    Parch = int(input("Number of parents/children aboard: "))
    Fare = float(input("Fare: "))
    Embarked = input("Port of Embarkation (C/Q/S): ").upper()
    embarked_mapping = {'C':0, 'Q':1, 'S':2}
    Embarked = embarked_mapping.get(Embarked, 2)  # default to 2 if invalid
       
    passenger = np.array([[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]])
    passenger_scaled = scaler.transform(passenger)
        
    prediction = model.predict(passenger_scaled)[0]
    result = "Survived" if prediction == 1 else "Did not survive"
    print(f"\nPrediction: {result}")

predict_survival()
