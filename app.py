# Dependencies
from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import numpy as np

# Your API definition
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    if joblib.load("model.pkl"):
        try:
            json_ = request.json
            print(json_)
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=joblib.load("model_columns.pkl"), fill_value=0)

            prediction = list(joblib.load("model.pkl").predict(query))

            return jsonify({'prediction': str(prediction)})

        except BaseException:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print('Train the model first')
        return ('No model here to use')


if __name__ == '__main__':


    regr = joblib.load("model.pkl")  # Load "model.pkl"
    print('Model loaded')
    # Load "model_columns.pkl"
    model_columns = joblib.load("model_columns.pkl")
    print('Model columns loaded')

    app.run(debug=True)
