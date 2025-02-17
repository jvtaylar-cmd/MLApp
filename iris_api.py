from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

@app.route('/predict/DecisionTree', methods=['POST'])
def predict_decision_tree():
    try:
        data = request.json
        model = joblib.load("models/DecisionTree.pkl")
        prediction = model.predict([data])[0]
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict/KNN', methods=['POST'])
def predict_knn():
    try:
        data = request.json
        model = joblib.load("models/KNN.pkl")
        prediction = model.predict([data])[0]
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict/LogisticRegression', methods=['POST'])
def predict_logistic_regression():
    try:
        data = request.json
        model = joblib.load("models/logistic_regression_model.pkl")
        prediction = model.predict([data])[0]
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)