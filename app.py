from flask import Flask, render_template, request
from dataset_analysis import fig

from emotion_classifier import perform_emotion_classification

# Initialize the Flask app
app = Flask(__name__, template_folder='templates')

@app.route("/", methods=["GET"])
def home():
    return render_template('index.html')

@app.route("/dataset", methods=["GET"])
def dataset():
    return render_template('dataset.html', plot=fig.to_html())

@app.route("/emotion_classification", methods=["GET", "POST"])
def emotion_classification():
    if request.method == 'GET':
        return render_template('emotion_classification.html', predicted_emotion=None, confidence_score=None)
    
    if request.method == 'POST':
        input_text = request.form.get('inputText')
        
        if not input_text:
            return render_template('emotion_classification.html', predicted_emotion=None, confidence_score=None)
        try:
            predicted_emotion, confidence_score, fig = perform_emotion_classification(input_text)
            return render_template('emotion_classification.html', predicted_emotion=predicted_emotion, confidence_score=confidence_score, input_text=input_text, plot=fig.to_html(), loading=False, error=False)
        except:
            return render_template('emotion_classification.html', predicted_emotion=None, confidence_score=None, input_text=input_text, loading=False, error=True)

if __name__ == "__main__":
    app.run(debug=True)
