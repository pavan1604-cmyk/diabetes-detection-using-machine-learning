#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv(r"C:\Users\91994\Downloads\diabetes.csv")
data.head()


# In[2]:


data.tail()


# In[3]:


data.shape


# In[4]:


print("Number of Rows",data.shape[0])
print("Number of Columns",data.shape[1])


# In[5]:


data.info()


# In[6]:


data.isnull().sum()


# In[7]:


data.describe()


# In[8]:


sns.countplot(x='Outcome', data=data)


# In[2]:


corr_mat=data.corr()
sns.heatmap(corr_mat, annot=True)
plt.show()


# In[3]:


data.hist(bins=10,figsize=(10,10))
plt.show()


# In[11]:


sns.set(style="ticks")
sns.pairplot(data, hue="Outcome")


# In[12]:


data_copy = data.copy(deep=True)
data.columns


# In[13]:


#box plot for outlier visualization
sns.set(style="whitegrid")
data.boxplot(figsize=(15,6))


# In[14]:


data_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI']] = data_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI']].replace(0,np.nan)
data_copy.isnull().sum()


# In[15]:


#box plot
sns.set(style="whitegrid")

sns.set(rc={'figure.figsize':(4,2)})
sns.boxplot(x=data['Insulin'])
plt.show()
sns.boxplot(x=data['BloodPressure'])
plt.show()
sns.boxplot(x=data['DiabetesPedigreeFunction'])
plt.show()


# In[4]:


data['Glucose'] = data['Glucose'].replace(0,data['Glucose'].mean())
data['BloodPressure'] = data['BloodPressure'].replace(0,data['BloodPressure'].mean())
data['SkinThickness'] = data['SkinThickness'].replace(0,data['SkinThickness'].mean())
data['Insulin'] = data['Insulin'].replace(0,data['Insulin'].mean())
data['BMI'] = data['BMI'].replace(0,data['BMI'].mean())


# In[5]:


X = data.drop('Outcome',axis=1)
y = data['Outcome']


# In[6]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,
                                               random_state=42)


# In[7]:


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.pipeline import Pipeline
pipeline_lr  = Pipeline([('scalar1',StandardScaler()),
                         ('lr_classifier',LogisticRegression())])

pipeline_knn = Pipeline([('scalar2',StandardScaler()),
                          ('knn_classifier',KNeighborsClassifier())])

pipeline_svc = Pipeline([('scalar3',StandardScaler()),
                         ('svc_classifier',SVC())])

pipeline_dt = Pipeline([('dt_classifier',DecisionTreeClassifier())])
pipeline_rf = Pipeline([('rf_classifier',RandomForestClassifier(max_depth=3))])
pipeline_gbc = Pipeline([('gbc_classifier',GradientBoostingClassifier())])


# In[8]:


pipelines = [pipeline_lr,
            pipeline_knn,
            pipeline_svc,
            pipeline_dt,
            pipeline_rf,
            pipeline_gbc]
pipelines


# In[9]:


for pipe in pipelines:
    pipe.fit(X_train,y_train)
pipe_dict = {0:'LR',
             1:'KNN',
             2:'SVC',
             3:'DT',
             4: 'RF',
             5: 'GBC'}
pipe_dict


# In[22]:


for i,model in enumerate(pipelines):
    print("{} Test Accuracy:{}".format(pipe_dict[i],model.score(X_test,y_test)*100))


# In[23]:


from sklearn.ensemble import RandomForestClassifier
X = data.drop('Outcome',axis=1)
y = data['Outcome']
rf =RandomForestClassifier(max_depth=3)
rf.fit(X,y)


# In[24]:


new_data = pd.DataFrame({
    'Pregnancies':6,
    'Glucose':148.0,
    'BloodPressure':72.0,
    'SkinThickness':35.0,
    'Insulin':79.799479,
    'BMI':33.6,
    'DiabetesPedigreeFunction':0.627,
    'Age':50,    
},index=[0])
p = rf.predict(new_data)
if p[0] == 0:
    print('non-diabetic')
else:
    print('you are diabetic so please take care of your health')


# In[26]:


import joblib
joblib.dump(rf,'model_joblib_diabetes')


# In[27]:


model = joblib.load('model_joblib_diabetes')
model.predict(new_data)


# In[15]:


import pandas as pd

# Load the diabetes dataset
diabetes_dataset = pd.read_csv(r"C:\Users\91994\Downloads\diabetes.csv")

# Print the column names
print(diabetes_dataset.columns)


# In[16]:


# Separate features (X) and target variable (y)
X = diabetes_dataset.drop('Outcome', axis=1)
y = diabetes_dataset['Outcome']


# In[17]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import pandas as pd

# Load the diabetes dataset
diabetes_dataset = pd.read_csv(r"C:\Users\91994\Downloads\diabetes.csv")

# Separate features (X) and target variable (y)
X = diabetes_dataset.drop('Outcome', axis=1)
y = diabetes_dataset['Outcome']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a RandomForestClassifier (you can use any other classifier)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.2%}')

# Save the trained model to a file
joblib.dump(model, 'model_joblib_diabetes')


# In[18]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import pandas as pd

# Load the diabetes dataset
diabetes_dataset = pd.read_csv(r"C:\Users\91994\Downloads\diabetes.csv")

# Separate features (X) and target variable (y)
X = diabetes_dataset.drop('Outcome', axis=1)
y = diabetes_dataset['Outcome']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a RandomForestClassifier (you can use any other classifier)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.2%}')

# Save the trained model to a file
joblib.dump(model, 'model_joblib_diabetes')


# In[22]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import pandas as pd

# Load the diabetes dataset
diabetes_dataset = pd.read_csv(r"C:\Users\91994\Downloads\diabetes.csv")

# Separate features (X) and target variable (y)
X = diabetes_dataset.drop('Outcome', axis=1)
y = diabetes_dataset['Outcome']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a RandomForestClassifier (you can use any other classifier)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.2%}')

# Save the trained model to a file
joblib.dump(model, 'model_joblib_diabetes')

print("Model saved successfully.")


# In[23]:


import os


current_directory = os.getcwd()


model_filename = os.path.join(current_directory, 'model_joblib_diabetes')
joblib.dump(model, model_filename)

print(f"Model saved successfully to: {model_filename}")


# In[23]:


model = joblib.load('model_joblib_diabetes')


# In[3]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import validation_curve
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

# Generate a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_classes=2, random_state=42)

# Create a RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Vary the complexity parameter (e.g., number of trees in the forest)
param_range = np.arange(1, 200, 10)

# Calculate training and test scores across the specified parameter range
train_scores, test_scores = validation_curve(model, X, y, param_name="n_estimators", param_range=param_range, cv=5, scoring="accuracy", n_jobs=-1)

# Calculate mean and standard deviation for training and test scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot the validation curve
plt.figure(figsize=(10, 6))
plt.plot(param_range, train_mean, label="Training score", color="blue", marker="o")
plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.15, color="blue")

plt.plot(param_range, test_mean, label="Cross-validation score", color="green", marker="o")
plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha=0.15, color="green")

plt.title("Validation Curve for RandomForestClassifier")
plt.xlabel("Number of Trees")
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.grid(True)
plt.show()


# In[7]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Assuming you have X and y as your feature matrix and target variable
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the logistic regression model with increased max_iter and solver='lbfgs'
model = LogisticRegression(max_iter=1000, solver='lbfgs')

# Train the model
model.fit(X_train_scaled, y_train)

# Predictions on the training set
train_predictions = model.predict(X_train_scaled)
train_accuracy = accuracy_score(y_train, train_predictions)

# Predictions on the testing set
test_predictions = model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, test_predictions)

print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")


# In[11]:


import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Assuming you have already trained your RandomForestClassifier (model)
# If not, you can train it using: model.fit(X_train, y_train)

# Create a RandomForestClassifier
rf_model = RandomForestClassifier(max_depth=3)

# Train the model
rf_model.fit(X, y)

# Get feature importances from the model
feature_importances = rf_model.feature_importances_

# Get feature names
feature_names = X.columns

# Sort feature importances in descending order
indices = feature_importances.argsort()[::-1]

# Rearrange feature names based on feature importances
sorted_feature_names = [feature_names[i] for i in indices]

# Plot the bar chart
plt.figure(figsize=(10, 6))
plt.bar(range(X.shape[1]), feature_importances[indices])
plt.xticks(range(X.shape[1]), sorted_feature_names, rotation=90)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Feature Importances in Random Forest Model")
plt.show()


# In[12]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Create a DecisionTreeClassifier
dt_model = DecisionTreeClassifier()

# Train the model
dt_model.fit(X_train, y_train)

# Predictions on the training set
train_predictions_dt = dt_model.predict(X_train)
train_accuracy_dt = accuracy_score(y_train, train_predictions_dt)

# Predictions on the testing set
test_predictions_dt = dt_model.predict(X_test)
test_accuracy_dt = accuracy_score(y_test, test_predictions_dt)

print(f"Decision Tree Training Accuracy: {train_accuracy_dt * 100:.2f}%")
print(f"Decision Tree Testing Accuracy: {test_accuracy_dt * 100:.2f}%")


# In[13]:


from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# Create a DecisionTreeClassifier
dt_model = DecisionTreeClassifier()

# Train the model
dt_model.fit(X_train, y_train)

# Get feature importances from the model
feature_importances = dt_model.feature_importances_

# Get feature names
feature_names = X.columns

# Create a bar plot of feature importances
plt.figure(figsize=(10, 6))
plt.barh(range(len(feature_importances)), feature_importances, align='center')
plt.yticks(range(len(feature_importances)), feature_names)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Decision Tree Feature Importances')
plt.show()


# In[14]:


from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Assuming you have X and y as your feature matrix and target variable
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVM with Linear Kernel
svm_linear = SVC(kernel='linear')
svm_linear.fit(X_train, y_train)
linear_predictions = svm_linear.predict(X_test)
linear_accuracy = accuracy_score(y_test, linear_predictions)
print(f'Linear Kernel Accuracy: {linear_accuracy:.2%}')

# SVM with Polynomial Kernel
svm_poly = SVC(kernel='poly', degree=3)
svm_poly.fit(X_train, y_train)
poly_predictions = svm_poly.predict(X_test)
poly_accuracy = accuracy_score(y_test, poly_predictions)
print(f'Polynomial Kernel Accuracy: {poly_accuracy:.2%}')

# SVM with Radial Basis Function (RBF) Kernel
svm_rbf = SVC(kernel='rbf')
svm_rbf.fit(X_train, y_train)
rbf_predictions = svm_rbf.predict(X_test)
rbf_accuracy = accuracy_score(y_test, rbf_predictions)
print(f'RBF Kernel Accuracy: {rbf_accuracy:.2%}')


# In[15]:


import matplotlib.pyplot as plt
import numpy as np

# Accuracy values
accuracies = [linear_accuracy, poly_accuracy, rbf_accuracy]

# Kernel names
kernels = ['Linear', 'Polynomial', 'RBF']

# Create a bar graph
plt.bar(kernels, accuracies, color=['blue', 'green', 'red'])
plt.ylim([0, 1])  # Set y-axis limits to represent accuracy percentage

# Add labels and title
plt.xlabel('Kernel')
plt.ylabel('Accuracy')
plt.title('SVM with Different Kernels')

# Display the accuracy values on each bar
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.02, f'{acc:.2%}', ha='center')

# Show the plot
plt.show()


# In[20]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd

# Load your dataset (replace 'your_dataset.csv' with your actual dataset file)
diabetes_dataset = pd.read_csv(r"C:\Users\91994\Downloads\diabetes.csv")

# Assuming 'Outcome' is the target variable
X = diabetes_dataset.drop('Outcome', axis=1)
y = diabetes_dataset['Outcome']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(),
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier()
}

# Train and evaluate each classifier
results = {'Algorithm': [], 'Training Accuracy': [], 'Testing Accuracy': []}

for name, classifier in classifiers.items():
    # Train the model
    classifier.fit(X_train, y_train)

    # Training accuracy
    train_predictions = classifier.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_predictions)

    # Testing accuracy
    test_predictions = classifier.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)

    # Update results
    results['Algorithm'].append(name)
    results['Training Accuracy'].append(train_accuracy)
    results['Testing Accuracy'].append(test_accuracy)

# Display results in a table
df = pd.DataFrame(results)
print(df)


# In[ ]:





# In[1]:


from tkinter import *
from PIL import Image, ImageTk
import joblib
import pandas as pd
from tkinter import messagebox
from sklearn.exceptions import DataConversionWarning
import pyttsx3
from datetime import datetime
import warnings

# Suppress warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Print the model file path before loading
model_path = r"C:\Users\91994\model_joblib_diabetes"
print(f"Loading model from: {model_path}")
model = joblib.load(model_path)

# Function to add a funny sticker to the result label
def add_funny_sticker(label, result):
    try:
        sticker_size = (200, 200)  # Set the desired sticker size here
        if result == 0:  # Non-Diabetic
            sticker_path = "C:\\Users\\91994\\Downloads\\na.png"  # Replace with the path to the non-diabetic sticker image
        else:
            sticker_path = "C:\\Users\\91994\\Downloads\\hs.png"  # Replace with the path to the diabetic sticker image

        sticker_img = Image.open(sticker_path)
        sticker_img = sticker_img.resize(sticker_size, Image.ANTIALIAS)
        sticker_img = ImageTk.PhotoImage(sticker_img)
        sticker_label = Label(label, image=sticker_img, bg="#E8E8E8")  # Background color for the sticker label
        sticker_label.image = sticker_img  # Keep a reference to the image
        sticker_label.pack(side="right")
    except Exception as e:
        print("Failed to load the sticker:", e)

# Function to clear the funny sticker
def clear_funny_sticker(label):
    for widget in label.winfo_children():
        widget.destroy()

# Function to update the result labels with animation
def update_result_labels_animation(label, text, color, delay=50, count=0):
    if count < len(text):
        label.config(text=text[:count], fg=color, bg="#E8E8E8")  # Background color for the result label
        label.after(delay, update_result_labels_animation, label, text, color, delay, count + 1)
    else:
        label.config(text=text, fg=color, bg="#E8E8E8")  # Background color for the result label
        # Speak the prediction
        engine.say(text)
        engine.runAndWait()

# Load the diabetes dataset (assuming you have this CSV file)
diabetes_dataset = pd.read_csv(r"C:\Users\91994\Downloads\diabetes.csv")

# History to store past predictions
history = []

# GUI setup
root = Tk()
root.title("Diabetes Detection Using Machine Learning")

# Set the background color for the entire window
root.configure(bg="#E8E8E8")  # Super background color

# Create a frame to center-align the elements
frame = Frame(root, bg="#E8E8E8")  # Background color for the frame
frame.grid(row=0, column=0, columnspan=2)

label = Label(frame, text="Diabetes Detection Using Machine Learning", bg="#E8E8E8", fg="#333333", font=("Helvetica", 20))
label.grid(row=0, column=0, columnspan=2, pady=(20, 10))

# Labels and entry fields for user input (assuming you have a dataset to get the fields)
fields = diabetes_dataset.columns[:-1]
entry_fields = []
for i, field in enumerate(fields):
    Label(frame, text=f"{field}:", font=("Helvetica", 14), bg="#E8E8E8", fg="#333333").grid(row=i + 1, column=0, sticky="e")
    entry = Entry(frame, font=("Helvetica", 14), width=10)
    entry.grid(row=i + 1, column=1, padx=(0, 10))
    entry_fields.append(entry)

    # Validate input to allow only numeric and decimal values
    validate_numeric = root.register(lambda s: s.replace('.', '', 1).isdigit() or s == "")
    entry.config(validate="key", validatecommand=(validate_numeric, "%P"))

result_label = Label(frame, text="", font=("Helvetica", 16), bg="#E8E8E8")
result_label.grid(row=len(fields) + 1, column=0, columnspan=2, pady=(10, 0))

result_text_label = Label(frame, text="", font=("Helvetica", 16), bg="#E8E8E8", bd=3, relief="solid")
result_text_label.grid(row=len(fields) + 2, column=0, columnspan=2, pady=(10, 0))

# Button to predict diabetes (assuming you have a function to predict diabetes)
def predict_diabetes():
    try:
        # Get values from entry fields
        values = [entry.get() for entry in entry_fields]

        # Check if all fields are non-empty
        if any(not val for val in values):
            messagebox.showwarning("Warning", "Please enter numeric values in all fields.")
            return

        # Convert values to floats
        values = [float(val) for val in values]

        print("Input Values:", values)

        # Clear existing funny sticker
        clear_funny_sticker(result_label)

        # Predict diabetes
        result = model.predict([values])

        # Calculate suffering percentage
        suffering_percentage = model.predict_proba([values])[0][1] * 100

        print("Prediction Result:", result)
        print("Suffering Percentage:", suffering_percentage)

        # Display the result
        if result[0] == 0:  # Non-Diabetic
            prediction_text = "You are Non-Diabetic. Enjoy your life"
            prediction_color = "green"  # Change color to green for non-diabetic
        else:
            prediction_text = "You are Diabetic. Please take care"
            prediction_color = "red"  # Change color to red for diabetic

        # Add a funny sticker based on the result
        add_funny_sticker(result_label, result[0])

        # Update the label with the prediction text, color, and suffering percentage
        update_result_labels_animation(result_text_label, f"{prediction_text}\nSuffering Percentage: {suffering_percentage:.2f}%", prediction_color)

        # Save prediction to history with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        history.append((timestamp, values, result[0], suffering_percentage))

    except Exception as e:
        print("Error predicting diabetes:", e)

predict_button = Button(frame, text='Predict', command=predict_diabetes, bg="#4CAF50", fg="white", font=("Helvetica", 14))
predict_button.grid(row=len(fields) + 3, column=0, columnspan=2, pady=(10, 0))

# Button to show history
def show_history():
    history_str = "\n".join([f"{entry[0]} - Input: {entry[1]}, Result: {entry[2]}, Suffering Percentage: {entry[3]:.2f}%" for entry in history])
    messagebox.showinfo("Prediction History", history_str)

show_history_button = Button(frame, text='Show History', command=show_history, bg="#008CBA", fg="white", font=("Helvetica", 14))
show_history_button.grid(row=len(fields) + 4, column=0, columnspan=2, pady=(10, 0))

# Button to clear values (assuming you have a function to clear values)
def clear_values():
    for entry in entry_fields:
        entry.delete(0, END)
    result_label.config(text="")
    result_text_label.config(text="", fg="#333333")
    # Clear the funny sticker by destroying it
    clear_funny_sticker(result_label)

clear_button = Button(frame, text='Clear', command=clear_values, bg="#808080", fg="white", font=("Helvetica", 14))
clear_button.grid(row=len(fields) + 5, column=0, columnspan=2)

# Configure row and column weights to center-align the frame
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)

# Run the Tkinter event loop
root.mainloop()


# In[16]:


pip install pyttsx3


# In[24]:


pip install SpeechRecognition


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




