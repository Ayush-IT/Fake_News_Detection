Fake News Detection 

This project aims to classify news articles as "Fake" or "Real" using machine learning models. The dataset used consists of two CSV files: one containing fake news articles and the other containing real (true) news articles. The project utilizes Natural Language Processing (NLP) techniques and various machine learning classifiers such as Logistic Regression, Decision Trees, Gradient Boosting, and Random Forest to predict whether a given news article is fake or not. 

Features 

Data Preprocessing: Text cleaning and feature extraction using TfidfVectorizer. 

Multiple Classifier Models: Evaluates multiple machine learning models (Logistic Regression, Decision Trees, Gradient Boosting, and Random Forest). 

Manual Testing: Allows users to manually input a news article for prediction using trained models. 

Classification Report: Provides a detailed classification report for each model to evaluate its performance. 

Project Structure 

Data Preprocessing: Involves text cleaning, feature extraction, and splitting the data into training and test sets. 

Model Training: Multiple machine learning classifiers are trained on the data. 

Evaluation: The performance of each model is evaluated using metrics such as accuracy and F1-score. 

Manual Testing: A function that allows users to manually test the models with their own input text. 

Installation 

Clone the repository: 

bash 

Copy code 

git clone https://github.com/yourusername/fake-news-detection.git 
cd fake-news-detection 
 

Install dependencies: 

bash 

Copy code 

pip install -r requirements.txt 
 

Dataset: 

The dataset consists of two files: Fake.csv and True.csv. You can upload the dataset to your Google Drive or specify a path where they are stored. 

Run the script: Run the script to start the training and testing process: 

bash 

Copy code 

python fake_news_detection.py 
 

The script will output the classification reports for each of the models and allow you to input a news article for manual testing. 

Usage 

After running the script, you can test the model by entering a news article text when prompted. The system will predict whether it is Fake news or Real news based on the trained models. The prediction results will be displayed for each model. 

Example 

bash 

Copy code 

Enter news article text: 
"The stock market crashes as tech companies face regulations." 
 

Output: 

bash 

Copy code 

LR Prediction: Fake news 
DT Prediction: Fake news 
GBC Prediction: Real news 
RFC Prediction: Fake news 
 

Evaluation Metrics 

The following metrics are calculated for each model: 

Precision 

Recall 

F1-score 

Accuracy 

Technologies Used 

Python: Main programming language. 

Pandas: Data manipulation and analysis. 

NumPy: Numerical computing. 

Scikit-learn: Machine learning library for model training, evaluation, and metrics. 

Seaborn: Data visualization library. 

Matplotlib: Plotting library for charts. 

Regular Expressions (regex): Text preprocessing and cleaning. 
