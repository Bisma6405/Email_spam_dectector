# Email_spam_dectector

A machine learning-powered web application that classifies email content as **Spam** or **Ham (Legitimate)** with high accuracy. The project features a trained Random Forest model, a clean Streamlit interface, and real-time confidence scoring.

##  Features
-  **Instant Classification**: Paste any email text and get immediate spam/ham predictions
-  **Confidence Scores**: Visual probability meters for both Spam and Ham classes
-  **Keyword Analysis**: Highlights top matched vocabulary words from the training dataset
-  **Sample Emails**: Built-in sidebar with pre-loaded spam and ham examples for quick testing
-  **Optimized Inference**: Lightweight `.pkl` bundle with no external scaler dependency
-  **Modern UI**: Responsive Streamlit layout with metrics, progress bars, and clear status banners

##  Model & Performance
The model was trained on a dataset of **5,172 emails** with **3,000 pre-extracted word-count features**. Multiple algorithms were evaluated:

| Model                  | Accuracy | ROC-AUC | Precision (Spam) | Recall (Spam) |
|------------------------|----------|---------|------------------|---------------|
| Logistic Regression    | 95.65%   | 0.9897  | 0.88             | 0.98          |
| Naive Bayes            | 94.40%   | 0.9751  | 0.87             | 0.94          |
| **Random Forest**    | **97.39%** | **0.9966** | **0.94**         | **0.97**      |
| Voting Ensemble        | 96.91%   | 0.9960  | 0.92             | 0.97          |

 **Final Selected Model:** `Random Forest Classifier` (chosen for highest AUC & accuracy)

##  Tech Stack
- **Backend/ML**: `Python 3`, `scikit-learn`, `NumPy`, `Pandas`
- **Frontend**: `Streamlit`
- **Serialization**: `pickle`
- **Notebook Environment**: Google Colab 

