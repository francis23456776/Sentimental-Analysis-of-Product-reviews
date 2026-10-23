📊 Product Review Sentiment Analysis & Interactive Dashboard

This project analyzes product reviews to identify customer sentiment using Python, Natural Language Processing (NLP), and machine learning. It features an interactive Streamlit web application for analyzing individual reviews, uploading CSV datasets, visualizing sentiment distributions, and downloading analysis results.

🎯 Project Overview

The goal of this project is to understand customer opinions from product reviews through sentiment analysis and interactive data visualization.
* Text Analysis (Python & NLTK): Analyze review text using the VADER sentiment analysis tool.
* Machine Learning (Scikit-learn): Train an optional Random Forest classifier using TF-IDF text features from a labeled dataset.
* Bulk Data Analysis (Pandas): Upload CSV files to analyze multiple product reviews.
* Interactive Visualization (Plotly): Visualize sentiment distributions using interactive charts.
* Web Application (Streamlit): Provide an interactive interface for sentiment prediction and customer review analysis.

🛠️ Technologies Used

Python
Pandas
NumPy
NLTK (VADER Sentiment Analysis)
Scikit-learn (TF-IDF & Random Forest)
Plotly Express
Streamlit

🔄 Project Workflow

* Input Data: Enter a product review manually or upload a CSV dataset.
* Text Processing: Prepare review text for sentiment analysis.
* Sentiment Prediction: Analyze sentiment using VADER or the optional TF-IDF and Random Forest model.
* Data Visualization: Explore sentiment distributions through interactive charts.
* Results & Export: View analyzed reviews and download results as a CSV file.
  
✨ Key Features

* Real-time sentiment prediction for individual reviews.
* Bulk sentiment analysis using CSV uploads.
* Positive, negative, and neutral sentiment classification using VADER.
* Optional custom machine learning model training with labeled review data.
* Interactive sentiment distribution visualization.
* Review analysis results displayed in a table.
* Downloadable CSV analysis results.

🚀 How to Run This Project

1.Clone the Repository
   
git clone https://github.com/francis23456776/Sentimental-Analysis-of-Product-reviews.git

2.Install Dependencies

pip install -r requirements.txt

3.Run the Streamlit Application

streamlit run app.py

4. Explore the Application
   
Open the local URL displayed in your terminal. Enter a product review to analyze its sentiment, or upload a CSV dataset to perform bulk sentiment analysis.

📈 Project Outcomes

* Applied NLP techniques to analyze customer product reviews.
* Explored text preprocessing and TF-IDF feature extraction.
* Implemented sentiment classification using VADER and an optional Random Forest model.
* Developed an interactive analytics application using Streamlit.
* Presented sentiment patterns through data visualization.
  
🔮 Future Improvements

* Use a larger labeled dataset for more reliable model training.
* Evaluate model performance using accuracy, precision, recall, and F1-score.
* Compare VADER with other machine learning approaches.
* Add more interactive charts and filtering options.
* Deploy the application online using Streamlit Community Cloud.

👨‍💻 Author

Antony Jeba Francis
B.Sc. Information Technology Graduate | Aspiring Data Analyst
Interested in Data Analytics, Business Intelligence, Natural Language Processing, and Machine Learning.

