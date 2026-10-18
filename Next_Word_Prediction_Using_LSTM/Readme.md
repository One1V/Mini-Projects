# Project Description: Next Word Prediction Using LSTM
The project aims to develop a deep learning model  for predicting the next word in a given sequence of words.The model is built using Long-Short Memory(LSTM) 
networks, which are well-suited for sequence prediction.

### The project includes the following steps:
### 1. Data Collection:
We use the test of Shakespeare's *"Hamlet"* as our dataset(because it is really complex to predict and train on).This rich,complex text provides a good challenge for our model.
### 2. Data Processing:
The text is tokenized, converted into sequences, and padded to ensure uniform input lengths. The sequences are then split iinto training and testing sets.
### 3. Model Building:
An LSYM model is constructed with an embedding layer, two LSTM layers, and a dense output layer with a softmax activation function to predict the probability of the next word.
### 4.Word Training:
The model is trained iusing the prepared sequences, with early stoppingd implementes to prevent overfitting. Early stopping monitors the validation loss and stops training when the loss stops improving.
### 5.Model Evaluation: 
The model is evaluated  using a set of example sentences to test its ability to predict the next word accurately.
### 6.Deployment:
A Streamlit web application is deployed to allow users to input a sequence of words and get the predicted next word in real-life.





