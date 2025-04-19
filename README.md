# Spam-Mail-Prediction
This project demonstrates how to build a spam mail classifier using machine learning techniques in Python. The main objective is to accurately classify email messages as either **spam** or **ham (not spam)** using text-based features and logistic regression.

## Objective 
Email spam classification is a classic problem in natural language processing (NLP). In this project, we:
- Preprocess a dataset of labeled emails
- Transform text data into numerical features using `TfidfVectorizer`
- Train a **Logistic Regression** model
- Evaluate the model's performance on unseen data
  
## Dataset
This project uses one dtatsets, `mail_data.csv`. This dataset contains **5572 rows** and **2 columns**
- `Category`: Indicates the label of the message (`spam` or `ham`) 
- `Message`: Contains the actual text of the mail

## Tools and Libraries 
-**Python:** Core language used for analysis and modeling.
-**Pandas:** For data manipulation and preprocessing.
-**Numpy:** For numerical operations and creating ranges used in plotting.
-**Scikit-Learn:** For machine learning modeling and evaluation.

## Step-by-Step Process
### **step 1: Importing the Dependencies**
- Imported essential libraries such as `pandas`, `numpy`, `sklearn`, and `TfidfVectorizer` for preprocessing and model building.
- Ensured all tools were available for text vectorization, model training, and evaluation.
- 
### **step 2: Data Collection & Pre-Processing**
- Loaded the dataset using `pandas`.
- Checked and removed any missing values to ensure clean data.
- 
### **step 3: Feature Engineering**
- Encoded the labels (`ham` as 0 and `spam` as 1).
- Applied `TfidfVectorizer` to convert email text into numerical feature vectors.
- Removed common stopwords and converted text to lowercase.
- Split the data into training and testing sets using `train_test_split`.
- 
### **step 4: Model Training**
- Trained a **Logistic Regression** model on the TF-IDF feature vectors.
- Chose logistic regression for its simplicity and strong performance in binary classification tasks.
- 
### **step 5: Evaluation**
- Used the trained model to predict on the test dataset.
- Evaluated model performance using **Accuracy Score**.
- 
### **step 6: Building a Predictive System
- Created a simple pipeline to classify new/unseen email messages.
- Allowed input of raw text to predict whether the email is spam or ham.

## Results
**Accuracy**: ~97% 

## Future Improvements
- Test with other classifiers (Naive Bayes, SVM, Random Forest)
- Use advanced preprocessing (lemmatization, stemming)
- Apply deep learning models like LSTM or BERT
