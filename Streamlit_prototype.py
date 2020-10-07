"""
implementation of preds in streamlit
using the cleaned data
"""



# main imports

import pandas as pd
import numpy as np
import streamlit as st
import random as rd
import pickle
#from sklearn.model_selection import cross_val_score #only when testing
np.random.seed(13) #random seed to keep predictions consistent

forest_clf = pickle.load(open("model.pickle",'rb'))


## STREAMLIT
st.write("""
         # Would you have survived the Titanic Disaster?""")
st.image("https://media1.faz.net/ppmedia/aktuell/83311481/1.1703919/format_top1_breit/der-untergang-der-titanic-1912.jpg",
         caption = "This disaster.")
st.image("http://3.bp.blogspot.com/-C6WJdUOdAaA/UJ7WLxU6lUI/AAAAAAAAAXs/GkjXrSqV2go/s320/titanic2.jpg", caption = "Not this one....")

st.write("""
         ## How it works:

         Given some inputs the algorithm will give you a prediction for your survival
         
         ### Some information to help in your selection
         
         #### Women and children first
         
         *Only about 32% of passengers survived.*
         
         If you were a man, you're chances of survival were much lower. The overall survival rate for men was about 20%.
         For women, it was 74%, and for children, 52%.
         

         
         #### Titanic ticket prices were:
         
         - First Class (parlor suite) = £870/$4,350 ($113,075.78 2018)
         - First Class (berth) = £30/$150 ($3,899.16 2018)
         - Second Class = £12/$60 ($1,559.67 2018)
         - Third Class = £3 - £8/$15 - $40 ($389.92 to $1,039.78 2018)


         To put that into cotext, the average US wage in 1912 was about 22 cents per hour. The average person in the US made between $200 and $400 per year.
         
         The oldest passenger onboard was 74 year old Johan Svensson.
         """)

st.markdown("[You can find more facts about the Titanic here](https://www.telegraph.co.uk/travel/lists/titanic-fascinating-facts/#:~:text=1.,2.)")
st.markdown("[and here](https://titanicfacts.net/titanic-survivors/)")
st.markdown("[Could Jack have lived? More about the famous door scene from the Titanic Movie](http://colgatephys111.blogspot.com/2012/11/could-jack-have-lived.html)")
        



st.sidebar.header("User Input Parameters")

### input needs to be scaled! geht raw X_train and then scale

def user_input_features():
    age = st.sidebar.slider("Your Age", 1,75,30)
    fare = st.sidebar.slider("Fare in 1912 $ (See Info)",15,500,40)
    SibSp = st.sidebar.selectbox("How many siblings or spouses are travelling with you?",[0,1,2,3,4,5,6,7,8])
    Parch = st.sidebar.selectbox("How many parents or children are travelling with you?",[0,1,2,3,4,5,6,7,8])
    cabin_multiple = st.sidebar.selectbox("How many additional cabins have you booked?",[0,1,2,3,4])
    numeric_ticket = rd.randint(0, 1)
    Sex = st.sidebar.selectbox("Select your Gender (due to the historical nature of the data only male/female are available)",["male","female"])
    Sex_female = 0 if Sex == "male" else 1
    Sex_male = 0 if Sex == "female" else 1
    Pclass = st.sidebar.selectbox("Which Class is your ticket from?  (For accurate predictions match with Fare)", [1,2,3])
    Pclass_1 = 1 if Pclass == 1 else 0; Pclass_2 = 1 if Pclass == 2 else 0; Pclass_3 = 1 if Pclass == 3 else 0
    boarding = st.sidebar.selectbox("Where did you board the Titanic?", ["Cherbourg","Queenstown","Southampton"])
    Embarked_C = 1 if boarding == "Cherbourg" else 0; Embarked_Q = 1 if boarding == "Queenstown" else 0; Embarked_S = 1 if boarding == "Southampton" else 0
    data = {"Age": age,"norm_fare":np.log(fare+1),"SibSp":SibSp,"Parch":Parch,"cabin_multiple": cabin_multiple,
            "numeric_ticket":numeric_ticket,"Sex_female":Sex_female,"Sex_male":Sex_male,"Pclass_1": Pclass_1, "Pclass_2": Pclass_2, "Pclass_3": Pclass_3,
            "Embarked_C":Embarked_C,"Embarked_Q":Embarked_Q, "Embarked_S":Embarked_S}
    data = pd.DataFrame(data, index = [0])
    #scaler.transform(data)
    #age,np.log(fare+1), SibSp, Parch, cabin_multiple, numeric_ticket, Sex_female,Sex_male, Pclass_1,Pclass_2,Pclass_3, Embarked_C, Embarked_Q, Embarked_S
    return data

user_data = user_input_features()
#st.subheader("Your input Parameters:")
#st.dataframe(user_data.to_dict())

prediction = forest_clf.predict(user_data)
st.title("Survival Prediction")
#st.write(prediction[0])
if prediction[0] == 1:
    st.write("**You probably would have made it!**")
    st.image("https://thumbs-prod.si-cdn.com/pn1W-PCw0pwa_EpefSOduW74gcM=/fit-in/1072x0/https://public-media.si-cdn.com/filer/Titanic-survivors-drifting-2.jpg")
else: 
    st.write("Well...how do I put this...**you're probably safer just watching the movie!**")
    st.image("https://i2-prod.irishmirror.ie/incoming/article9830920.ece/ALTERNATES/s615b/0_Kate-Winslet-as-Rose-DeWitt-Bukater-and-Leonardo-DiCaprio-as-Jack-Dawson-in-Titanic.jpg")


######TESTING#####

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())




#display_scores(cross_val_score(forest_clf,X_train,y_train, scoring = "accuracy",cv = 5))