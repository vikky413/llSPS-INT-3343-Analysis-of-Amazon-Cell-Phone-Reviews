import numpy as np
from flask import Flask , request , jsonify , render_template
import pickle
from keras.models import load_model
from sklearn.feature_extraction.text import CountVectorizer



app = Flask(__name__)

model = load_model('mymodel.h5')
Labels = [1, 2, 3, 4, 5]

def Predict(data):
    pred = model.predict(data)
    pred = np.argmax(pred , axis = 1)
    result = [Labels[i] for i in pred]
    return result


@app.route('/' )
def home():
   return render_template('index.html')


@app.route('/review' , methods = ['POST'])
def review():
    
    text = request.form.values()
    print(text)
    loaded_vec = CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("feature.pkl", "rb")))
    text = loaded_vec.fit_transform(text).toarray()
    result = Predict(text)
    result = result[0]
    return render_template('answer.html', review = 'user has given {} star rating.'.format(result))


print('hello')
if __name__ == '__main__':
    app.run(debug=True)