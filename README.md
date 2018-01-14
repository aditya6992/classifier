# classifier
train carefully, please do not train using biased data. 

train with almost equal number of positive and negetive examples.

the classifier in use is a random forest classifier, this particular one requires less amount of data, but can be biased easily.


API Documentation:

Request: train - train the data using the given text or resume
```
POST /train
{
	Text: "Objective: To work in a company ...
			Work Experience: ....",
	isTextResume: yes
}
```
Response:
```
success
```

Request: test - test the current text to see if its a resume according to our trained classifier
```
POST /test
{
	Text: "Objective: To work in a company ...
			Work Experience: ...."
}
```
Response: yes or no
```
yes
```

Request: reset - reinitialize the model, discarding all the training so far
```
GET /reset
```
Response: success

Request: RunInitialTraining - reinitialize the model and run the training on the initial dataset
```
GET /RunInitialTraining
```
Response: done