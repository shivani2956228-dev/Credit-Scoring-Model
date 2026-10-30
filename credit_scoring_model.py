import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from  sklearn.metrics import accuracy_score

#1. load the dataset 
data = pd.read_csv(r'c:\users\pc\credit_data.csv')
data = pd.read_csv('credit_data.csv')
data.columns = data.columns.str.strip()
print("dataset loaded successfully:")
print(data.head())

print("datasset loaded successfully:")
print(data.head())

#2. features aur target define karo 
# (yahan hum maan rhe hain ki ' credit_score' hamara target column hai)
x = data[['age', 'income', 'loan_amaunt']]
y = data['credit_score']

#3. train aur test data mein split karo
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#4. model create aur train karo
model = RandomForestClassifier()
model.fit(x_train, y_train)

#5. predictions aur accuracy check karo 
predictions = model.predict(x_test)
print("\nmodel training coplete!")
print("predicttions:", predictions)