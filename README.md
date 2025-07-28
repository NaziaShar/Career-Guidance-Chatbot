
# 🤖 Career Guidance Chatbot

This project develops a **Career Guidance Chatbot** using **Machine Learning** and **Streamlit**. The chatbot assists users by understanding career-related questions, classifying them into specific career roles, and providing detailed guidance using a predefined dataset.

## 🌟 Features

- **Natural Language Understanding**: Accepts user queries in plain English.
- **Career Role Classification**: Predicts the most relevant career field using a trained ML model (Naive Bayes, SVM, or Logistic Regression).
- **Informative Guidance**: Delivers detailed answers based on the predicted career role.
- **Interactive Web Interface**: Built with Streamlit for an intuitive user experience.

## 📊 Dataset

The project uses a **Career Guidance Dataset (CSV)** with over **1,620 entries**. It includes:

- `role`: The career field (e.g., Data Scientist, Product Manager)
- `question`: A user inquiry about that role
- `answer`: A detailed response to guide career decisions

## 🧰 Technologies Used

- **Python**: Core programming language  
- **Pandas**: Data manipulation and analysis  
- **Scikit-learn**: Machine learning (TF-IDF, Naive Bayes, SVM, Logistic Regression)  
- **Joblib**: Model and vectorizer serialization  
- **Streamlit**: Web app interface

## 🗂️ Project Structure

```
career_chatbot_project/
├── app.py                        # Streamlit web app
├── train_model.py                # Model training script
├── Career QA Dataset.csv         # Dataset
├── intent_model_naive_bayes.pkl  # Trained model (example: Naive Bayes)
├── vectorizer_naive_bayes.pkl    # TF-IDF vectorizer
└── requirements.txt              # Project dependencies
```

## ⚙️ Setup and Installation

### 1. Clone the Repository
```bash
git clone <repository_url>
cd career_chatbot_project
```

### 2. Create and Activate a Virtual Environment

**Windows:**
```bash
python -m venv venv
.env\Scriptsctivate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

> ⚠️ If you encounter an `InconsistentVersionWarning`, retrain the model using the current environment.

## 🧠 Model Training

Train your classification model by running:

```bash
python train_model.py
```

This generates:
- `intent_model_naive_bayes.pkl`
- `vectorizer_naive_bayes.pkl`

## 🚀 Running the Chatbot

Launch the chatbot using:

```bash
streamlit run app.py
```

This opens the chatbot in your browser at `http://localhost:8501`.

## 💬 Usage

1. Enter a career-related question (e.g., "What does a product manager do?")
2. Click **Get Career Guidance**
3. The chatbot predicts the career role and provides guidance

## 🧪 Model Details

- **Model**: Multinomial Naive Bayes (can be replaced with Logistic Regression or SVM)
- **Vectorization**: TF-IDF
- **Training**: Maps user queries to career roles

## 🙏 Acknowledgements

- Project developed as part of the **NextGen Summer Internship 2025**
- Dataset sourced from **[Hugging Face Datasets](https://huggingface.co/datasets)**

## 📜 License

This project is intended for educational and research purposes only.
