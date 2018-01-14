from flask import Flask
from initial_training import load_features, text_to_wordlist, document_vector, word_features, trainAndTest
from sklearn.externals import joblib
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from flask import request, render_template
from sklearn import tree
import numpy as np
import os

dir = os.path.dirname(__file__)
app = Flask(__name__, template_folder=os.path.join(dir,"templates"))
classifier = joblib.load(os.path.join(dir, "model.pkl"))
load_features()

@app.route("/train", methods=["POST"])
def train():
    """training script post request, takes the text from the request and stores it in new_resumes directory
    if the parameter isResume is false, the name is appended with .neg while storing it to denote that
    its not a resume. Later when there is atleast one example of a positive and a negetive data point
    in the directory, the training is commenced. Training takes hardly a second since there usually are
    very few examples. Response is sent as "success"."""
    global classifier
    form = request.form
    resume_directory = "./new_resumes/"
    files = os.listdir(resume_directory)
    text = form["Text"]
    new_file_number = len(files)
    isResume = True if form["isTextResume"] == "yes" else False
    filename = "res" + str(new_file_number) + ".txt" if isResume else "res" + str(new_file_number) + ".neg.txt"
    with open(resume_directory + filename, 'w') as f:
        f.write(text.encode('utf-8'))

    files = os.listdir(resume_directory)
    directory_contains_both_types = False
    containsneg = False
    containspos = False
    for file in files:
        print file
        if file.split(".")[-2] == "neg":
            containsneg = True
        if file.split(".")[-2] != "neg":
            containspos = True
        if containspos and containsneg:
            directory_contains_both_types = True
            break

    if directory_contains_both_types:
        n = len(word_features)
        vector = np.zeros((1, n), dtype="int32")
        yvector = np.zeros((1,), dtype="int32")
        # classifier = RandomForestClassifier()

        for file in files:
            if file.split(".")[-1] == "txt":
                isResume = True
                if file.split(".")[-2] == "neg":
                    isResume = False
                with open(resume_directory + file) as f:
                    text = f.read()
                wordlist = text_to_wordlist(text)
                X, Y = document_vector(wordlist, isResume)
                
                vector = np.concatenate((vector, X), axis=0)
                yvector = np.concatenate((yvector, Y), axis=0)

		os.remove(resume_directory + file)

        classifier.fit(vector, yvector)
	print vector, yvector
        joblib.dump(classifier, "model.pkl")
        return "success"

    return "success"


@app.route("/test", methods=["POST"])
def test():
    """test resume post request, takes the text from the request and tests if its a resume against the 
    existing classifier model. Sends "yes" or "no" as output response."""
    form = request.form
    text = form["Text"]
    wordlist = text_to_wordlist(text)
    X, Y = document_vector(wordlist, True)
    output = classifier.predict(X)
    nonzeros = np.count_nonzero(X)
    if nonzeros == 0:
        output[0] = 0
    return "Yes" if output[0] == 1 else "No"

@app.route("/reset", methods=["GET"])
def reset():
    """reinitializes the classifier model and saves it in model.pkl. Use it with caution and only when the
    model seems to go bad"""
    global classifier
    classifier = RandomForestClassifier()
    joblib.dump(classifier, "model.pkl")
    load_features()
    return "successfully reset"

@app.route("/RunInitialTraining", methods=["GET"])
def runInitialTraining():
    """runs initial training script, resets the classifier and runs the training on select few data
    same as reset except you have a basic usable model."""
    trainAndTest()
    return "done"

# User Interface
@app.route("/", methods=["GET"])
def render_ui():
    return render_template("user_interface.html")


if __name__=="__main__":
    app.run(port=5000)
