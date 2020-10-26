from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('iri.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    year = request.form['year']
    team1 = request.form['tm1']
    team2 = request.form['tm2']
    city = request.form['city']
    toss = request.form['t1']
    toss_win = request.form['tm2']
    venue = request.form['Venue']
    arr = np.array([[year, team1, team2, city, toss, toss_win, venue]])
    pred = model.predict(arr)
    teamname = {1:'Mumbai Indians', 2:'Chennai Super Kings', 3:'Kolkata Knight Riders', 4:'Royal Challengers Bangalore',
                  5:'Kings XI Punjab', 6:'Rajasthan Royals', 7:'Sunrisers Hyderaba', 8:'Delhi Capitals'}
    pred = teamname[pred[0]]
    return render_template('resultsform.html', data=pred)


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    app.run(debug=True)
