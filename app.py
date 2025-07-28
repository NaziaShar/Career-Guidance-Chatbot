import streamlit as st
import joblib
import pandas as pd
import re
import string

# Custom CSS for better styling
st.set_page_config(
    page_title="Career Guidance Chatbot",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .guidance-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .input-container {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.2);
    }
    
    .success-message {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    .info-box {
        background: rgba(255,255,255,0.1);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# --- Step 1: Define the text preprocessing function ---
# This function must be identical to the one used during model training.
def preprocess_text(text):
    text = str(text).lower()  # Convert to string and lowercase
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)  # Remove punctuation
    return text

# --- Step 2: Load the saved Naive Bayes model and vectorizer ---
# Ensure these filenames match what you used when saving in train_model.py
MODEL_PATH = 'intent_model_naive_bayes.pkl'
VECTORIZER_PATH = 'vectorizer_naive_bayes.pkl'
DATASET_PATH = 'Career QA Dataset.csv' # Path to your original dataset

try:
    model = joblib.load(MODEL_PATH)
    tfidf_vectorizer = joblib.load(VECTORIZER_PATH)
    df_answers = pd.read_csv(DATASET_PATH)
    # Ensure 'role' and 'answer' columns are available and correct in df_answers
    # It's good practice to preprocess the 'question' column of the df_answers too,
    # if you want to use it for exact answer matching, but the primary use is 'role' -> 'answer'.
    # For now, we'll just ensure the 'role' and 'answer' columns are there.
    if 'role' not in df_answers.columns or 'answer' not in df_answers.columns:
        st.error("Error: 'role' or 'answer' columns not found in the dataset. Please check the CSV file.")
        st.stop()
    
    # Create a mapping from role to a list of answers for that role
    # This ensures that if multiple answers exist for a role, we can pick one or combine.
    # For simplicity, we'll just get the first answer for a given role in this example.
    role_to_answer = df_answers.groupby('role')['answer'].apply(list).to_dict()

    # Success message with custom styling
    st.markdown('<div class="success-message">✅ Model, vectorizer, and dataset loaded successfully!</div>', unsafe_allow_html=True)
    
except FileNotFoundError:
    st.error(f"Error: One or more files not found. Make sure '{MODEL_PATH}', '{VECTORIZER_PATH}', and '{DATASET_PATH}' are in the same directory as 'app.py'.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading files: {e}")
    st.stop()

# --- Step 3: Enhanced Streamlit Web Interface ---

# Header with gradient background
st.markdown("""
<div class="main-header">
    <h1>🎯 Career Guidance Chatbot</h1>
    <p style="font-size: 1.2rem; margin-top: 1rem;">Your AI-powered career companion</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for additional features
with st.sidebar:
    st.markdown("### 🚀 Quick Actions")
    st.markdown("---")
    
    # Sample questions
    st.markdown("#### 💡 Try these questions:")
    sample_questions = [
        "What does a data scientist do?",
        "How to become a software engineer?",
        "What skills do I need for marketing?",
        "Tell me about product management",
        "What is UX design?"
    ]
    
    for question in sample_questions:
        if st.button(question, key=f"sample_{question}"):
            st.session_state.user_question = question

# Main content area
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    # Input container with better styling
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    
    # Get user input with placeholder
    user_question = st.text_input(
        "Ask your career question:",
        value=st.session_state.get('user_question', ''),
        placeholder="e.g., What does a data scientist do?",
        key="main_input"
    )
    
    # Submit button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        submit_button = st.button("🚀 Get Career Guidance", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Process the question
if submit_button or st.session_state.get('user_question'):
    if user_question:
        with st.spinner("🤖 Analyzing your question..."):
            # Preprocess the user's question
            cleaned_question = preprocess_text(user_question)

            # Transform the cleaned question using the loaded TF-IDF vectorizer
            question_vector = tfidf_vectorizer.transform([cleaned_question])

            # Make a prediction using the loaded model
            predicted_role = model.predict(question_vector)[0]

            # Get confidence score
            confidence_scores = model.predict_proba(question_vector)[0]
            max_confidence = max(confidence_scores)
            
            # Display results with enhanced styling
            st.markdown(f"""
            <div class="prediction-box">
                <h3>🎯 Predicted Career Field</h3>
                <h2>{predicted_role}</h2>
                <p>Confidence: {max_confidence:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Retrieve and display guidance
            answers_for_role = role_to_answer.get(predicted_role, ["No specific guidance found for this role."])
            
            st.markdown(f"""
            <div class="guidance-box">
                <h3>💡 Career Guidance</h3>
                <div class="info-box">
                    {answers_for_role[0] if answers_for_role else "Sorry, I don't have specific guidance for this role yet."}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Additional insights
            if max_confidence < 0.5:
                st.warning("⚠️ Low confidence prediction. Consider rephrasing your question for better results.")
            
    else:
        st.warning("Please enter a question to get guidance.")

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>🤖 Powered by Naive Bayes Machine Learning</p>
    <p>Built with ❤️ using Streamlit</p>
</div>
""", unsafe_allow_html=True)

# Add some spacing
st.markdown("<br><br>", unsafe_allow_html=True)