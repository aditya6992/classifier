import re, os
import numpy as np
from string import punctuation
from sklearn import linear_model
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

dir = os.path.dirname(__file__)
word_features = {}

def load_features():
    # load features from "features" file, features are simply some keywords in our model
    all_words = []
    with open(os.path.join(dir,"features")) as f:
        all_words = f.read().split("\n")
    for i in xrange(len(all_words)):
        word_features[all_words[i]] = i

def text_to_wordlist(text, remove_stop_words=False, stem_words=False):
    # Clean the text, with the option to remove stop_words and to stem words.

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"what's", "", text)
    text = re.sub(r"What's", "", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"I'm", "I am", text)
    text = re.sub(r" m ", " am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"60k", " 60000 ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e-mail", "email", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"quikly", "quickly", text)
    text = re.sub(r" usa ", " America ", text)
    text = re.sub(r" USA ", " America ", text)
    text = re.sub(r" u s ", " America ", text)
    text = re.sub(r" uk ", " England ", text)
    text = re.sub(r" UK ", " England ", text)
    text = re.sub(r"india", "India", text)
    text = re.sub(r"switzerland", "Switzerland", text)
    text = re.sub(r"china", "China", text)
    text = re.sub(r"chinese", "Chinese", text) 
    text = re.sub(r"imrovement", "improvement", text)
    text = re.sub(r"intially", "initially", text)
    text = re.sub(r"quora", "Quora", text)
    text = re.sub(r" dms ", "direct messages ", text)  
    text = re.sub(r"demonitization", "demonetization", text) 
    text = re.sub(r"actived", "active", text)
    text = re.sub(r"kms", " kilometers ", text)
    text = re.sub(r"KMs", " kilometers ", text)
    text = re.sub(r" cs ", " computer science ", text) 
    text = re.sub(r" upvotes ", " up votes ", text)
    text = re.sub(r" iPhone ", " phone ", text)
    text = re.sub(r"\0rs ", " rs ", text) 
    text = re.sub(r"calender", "calendar", text)
    text = re.sub(r"ios", "operating system", text)
    text = re.sub(r"gps", "GPS", text)
    text = re.sub(r"gst", "GST", text)
    text = re.sub(r"programing", "programming", text)
    text = re.sub(r"bestfriend", "best friend", text)
    text = re.sub(r"dna", "DNA", text)
    text = re.sub(r"III", "3", text) 
    text = re.sub(r"the US", "America", text)
    text = re.sub(r"Astrology", "astrology", text)
    text = re.sub(r"Method", "method", text)
    text = re.sub(r"Find", "find", text) 
    text = re.sub(r"banglore", "Banglore", text)
    text = re.sub(r" J K ", " JK ", text)
    
    # Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation])
    
    # Optionally, remove stop words
    if remove_stop_words:
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)
    
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    # Return a list of words
    return(text)

def document_vector(wordlist, isResume):
    # create a vector from a document, here we are counting the frequency of the features in the document and
    # creating a vector from that. here yvector is simply the output 0 or 1, 1 = resume, 0 = not resume
    n = len(word_features)
    vector = np.zeros((1, n), dtype="int32")
    yvector = np.zeros((1,), dtype="int32")

    wordslist = wordlist.split(" ")
    if isResume:
        yvector[0] = 1

    for word in wordslist:
        if word in word_features:
            index = word_features[word]
            vector[0, index] += 1
    return vector, yvector

def trainAndTest():
    # take all files from the resumes directory, create vectors from each of those documents using document_
    # vector function and stack them vertically ie concatenate vertically, now create a classifier object,
    # then fit the data on the classifier, now test your ckassifier on a.test doc
    resume_directory = os.path.join(dir,"resumes/")
    files = os.listdir(resume_directory)
    # classifier = linear_model.LogisticRegression(solver="liblinear", multi_class="ovr")
    classifier = RandomForestClassifier()
    n = len(word_features)
    vector = np.zeros((1, n), dtype="int32")
    yvector = np.zeros((1,), dtype="int32")

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

    classifier.fit(vector, yvector)

    with open(os.path.join(dir,"a.test")) as f:
        text = f.read()
    wordlist = text_to_wordlist(text)
    X, Y = document_vector(wordlist, True)
    out = classifier.predict(X)
    joblib.dump(classifier, "model.pkl")
    print out[0]


if __name__=="__main__":
    load_features()
    trainAndTest()
