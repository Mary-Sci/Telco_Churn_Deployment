import streamlit as st

from PIL import Image
im = Image.open("DA_TelcoChurn_SolutionsFlyer_ResourceLibrary_Thumbnail_1200x628 (1).jpg")
st.image(im, width=700, caption="Customer Churn")
#st.help(st.image)

st.sidebar.title("Churn Probablity of Single Customer")

html_temp = """
<div style="background-color:green;padding:1.5px">
<h2 style="color:white;text-align:center;">Churn Prediction App </h1>
</div><br>"""
st.markdown(html_temp,unsafe_allow_html=True)

import pandas as pd
import numpy as np
import pickle

df=pd.read_csv("model_out.csv")

#st.table(df.head())


if st.checkbox("Churn Probablity of Randomly Selected Customers"):
	st.markdown("**How many customers to be selected randomly?**")
	num = st.selectbox("Please select the number of customers",(1,5,10,20,30,40,50,100)) 
	st.button("Analyze")
	st.success(f"The Churn Probablity of randomly selected {num} Customers")
	st.dataframe(df.iloc[np.random.randint(7031, size=num, dtype='l')])




if  st.checkbox("Top Customers to Churn"):
	st.markdown("**How many customers to be selected?**")
	num2 = st.selectbox("Please select the number of top customers to churn",(1,5,10,20,30,40,50,100)) 
	st.button("Show")
	st.success(f"The Churn Probablity of selected {num2} Customers")
	st.dataframe(df.sort_values("Churn Probability", axis=0, ascending=False).head(num2))
	#st.table(df.sort_values("Churn Probability", axis=0, ascending=False).head(num2))


if  st.checkbox("Top Loyal Customers"):
	st.markdown("**How many customers to be selected?**")
	num3 = st.selectbox("Please select the number of loyal customers to churn",(1,5,10,20,30,40,50,100)) 
	st.button("Run")
	st.success(f"The Churn Probablity of selected {num3} Customers")
	st.dataframe(df.sort_values("Churn Probability", axis=0, ascending=True).head(num3))


tenure=st.sidebar.slider("Number of months the customer has stayed with the company", 1, 72, step=1)
MonthlyCharges=st.sidebar.slider("The amount charged to the customer monthly", 18.25,118.75, step=0.05)
TotalCharges = st.sidebar.slider("The total amount charged to the customer", 18.25,118.75, step=0.05)
Contract=st.sidebar.radio("The contract term of the customer", ('Month to month', 'One year', 'Two years'))
OnlineSecurity=st.sidebar.radio("Whether the customer has online security or not", ('No', 'Yes', 'No internet service'))
InternetService=st.sidebar.radio("Customer’s internet service provider", ('DSL', 'Fiber optic', 'No'))
TechSupport=st.sidebar.selectbox("Whether the customer has tech support or not", ('No', 'Yes', 'No internet service'))


def single_customer():
    my_dict = {"tenure" :tenure,
        "OnlineSecurity":OnlineSecurity,
        "Contract": Contract,
        "TotalCharges": TotalCharges ,
        "InternetService": InternetService,
        "TechSupport": TechSupport,
        "MonthlyCharges":MonthlyCharges}
    df_sample = pd.DataFrame.from_dict([my_dict])
    return df_sample
columns = ['tenure', 'MonthlyCharges', 'TotalCharges', 'OnlineSecurity_No',
       'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
       'InternetService_DSL', 'InternetService_Fiber optic',
       'InternetService_No', 'TechSupport_No',
       'TechSupport_No internet service', 'TechSupport_Yes',
       'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year']

dfx = single_customer()
dfx = pd.get_dummies(dfx).reindex(columns = columns, fill_value = 0)
#st.sidebar.table(dfx.head())

model = pickle.load(open("model_out.pkl", "rb"))

if st.sidebar.button("Submit"):
	result = model.predict(dfx)
	if result == 1:
		st.sidebar.warning("Churn Yes")
	else:
		st.sidebar.success("Churn No")



