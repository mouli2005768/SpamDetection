import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st
data = pd.read_csv("mail_data.csv")

data.drop_duplicates(inplace=True)
data['Category'] = data['Category'].replace(['ham','spam'],['Not Spam','Spam'])

mess=data['Message']
cat =data['Category']

(mess_train,mess_test,cat_train,cat_test) = train_test_split(mess,cat,test_size=0.2)
#Converting text to decimal which understand for the model 
cv = CountVectorizer(stop_words='english')
features=cv.fit_transform(mess_train)

#creating a model
 
model = MultinomialNB()
model.fit(features, cat_train)

# Testing the model

features_test = cv.transform(mess_test)

 # [" to get the score of accuracy"]print(model.score(features_test,cat_test))

 #predict Data in realTime
def predict(message):
   input_message=cv.transform([message]).toarray() 
   result = model.predict(input_message)
   return result


st.header('Spam Detection ')


input_mes = st.text_input('Enter your message here')
if st.button('Validate'):
   output = predict(input_mes)
   st.text(output)








