# Fake-news-Detection-model
This model can deetect whether a news article is fake or real. It uses raw Title and Test column of article as input and give an output declaring if its fake or real news article.
# Steps Taken:
1. **Text Preprocessing**: Cleaned the text data by removing punctuation, special characters, and converting it to lowercase.
2. **Feature Engineering**: Used the TF-IDF technique to convert the text content and titles into numerical feature vectors that machine learning models can understand. Created separate TF-IDF vectorizers for the text and the title.
3. **Model Selection and Training**: Trained three different models
 + **Naive Bayes**: A simple probabilistic classifier as a baseline.
 + **Random Forest**: A more complex ensemble learning model that we tuned using GridSearchCV to find the best hyperparameters.
 + **LSTM Neural Network**: A deep learning model particularly well-suited for sequential data like text.
4. **Model Evaluation**: Trained each model on a portion of the data (training set) and evaluated their performance on a separate, unseen portion (test set) using metrics like accuracy, precision, recall, and F1-score.
6. **Deployment**: Chose the tuned Random Forest model as the best-performing model (achieving the highest accuracy, 99%) and prepared it for deployment by saving the trained model and the TF-IDF vectorizers using joblib. Built a web interface using Streamlit so that users can easily input new news article text and titles and get a prediction from the model.
