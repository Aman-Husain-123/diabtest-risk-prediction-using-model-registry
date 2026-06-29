from flask import Flask, render_template, request, jsonify
import mlflow.pyfunc
import pandas as pd

app = Flask(__name__)

# Load the registered model once at startup
model_name = "diabetes-rf-hp"
model_version = 3
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([[
            int(data["pregnancies"]),
            int(data["glucose"]),
            int(data["blood_pressure"]),
            int(data["skin_thickness"]),
            int(data["insulin"]),
            float(data["bmi"]),
            float(data["dpf"]),
            int(data["age"]),
        ]], columns=[
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ])

        prediction = model.predict(input_df)
        result = int(prediction[0])

        return jsonify({
            "prediction": result,
            "label": "Diabetic" if result == 1 else "Not Diabetic",
            "confidence_note": "Based on Pima Indians Diabetes Dataset"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5001)
