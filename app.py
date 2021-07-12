from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("forest_fire.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[float(x) for x in request.form.values()]
    final=[np.array(int_features)]
    print(int_features)
    print(final)
    prediction=model.predict_proba(final)
    output='{0:.{1}f}'.format(prediction[0][1], 4)

    if output>str(0.5):
        return render_template('forest_fire.html',pred='Patient is suffering from Parkinsons .\nProbability of Disease is {}'.format(output),bhai="Immidiate Treatment should be done")
    else:
        return render_template('forest_fire.html',pred='Patient is not suffering from Parkinsos .\n Probability of Disease is {}'.format(output),bhai="Your Cattle is Safe for now")


if __name__ == '__main__':
    app.run(debug=False)