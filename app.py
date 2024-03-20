from flask import Flask, request, jsonify
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load the Iris dataset
iris = pd.read_csv("Dataset_Iris/iris.csv")

# Split the dataset into features (X) and labels (y)
X = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = iris['species']

# Train an SVM model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = svm.SVC()
model.fit(X_train, y_train)

# Define a route for predicting the species of Iris flowers
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    sepal_length = data['sepal_length']
    sepal_width = data['sepal_width']
    petal_length = data['petal_length']
    petal_width = data['petal_width']
    
    # Make prediction
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    
    return jsonify({'predicted_species': prediction[0]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
