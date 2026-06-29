# inference

import mlflow.pyfunc
import pandas as pd

data = pd.DataFrame([[1, 85, 66, 29, 0, 26.6, 0.351, 31]],
                     columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                              'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

model_name = "diabetes-rf-hp"
model_version = 3

model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")

print(model.predict(data))
