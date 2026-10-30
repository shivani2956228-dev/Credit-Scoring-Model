import streamlit as st
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier

st.title("credit scoring model")
st.write("machine learning based credit risk prediction")

df = pd.read_csv("credit_data.csv", on_bad_lines='skip')
df.columns =df.columns.str.strip() #yeh column names ke extra spaces hata dega

x = df[["income", "loan_amount", "credit_history"]]
y = df["approved"]
model= DecisionTreeClassifier(max_depth=3, random_state=42) 
model.fit(x,y)

st.subheader("application information")

income =st.number_input("monthly income (Rs.)", min_value=0, value=30000) 
loan_amount = st.number_input("loan amount (Rs.)", min_value=0, value= 100000)

history = st.selectbox("good credit history?", ["yes", "no"])

if st.button("predict credit risk"):
    application= pd.dataframe([{
        "income": income,
        "loan_amount": loan_amount,
        "credit_history": 1 if history== "yes " else 0
        }])

    result = model.predict(application)[0]

    if result == 1:
        st.success("lower  estimated credit risk")
else:
    st.warning("higher estimated credit risk")

    st.caption("education demo using sample data only.")
    