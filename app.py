import re
import pandas as pd
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

#Page configuration
st.set_page_config(
    page_title="Social Media Sentiment Analysis",
    layout="wide"
)

#Page Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 10px;
    }
    .sub-text {
        font-size: 18px;
        color: #4f4f4f;
        margin-bottom: 20px;
    }
    .info-card {
        background-color: #e9d8c7;
        padding: 18px;
        border-radius: 10px;
        border-left: 6px solid #4a90e2;
        margin-bottom: 20px;
    }
    .warning-card {
        background-color: #654321;
        padding: 16px;
        border-radius: 10px;
        border-left: 6px solid #f5b942;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

#Load Trained RoBERTa Model and AutoTokenizer
@st.cache_resource
def load_model():
    model_path = "Eugene2004/Roberta-for-Sentiment-Analysis" 
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    model.eval() 
    return tokenizer, model

try:
    tokenizer, model = load_model()
except Exception as e:
    st.error("Model loading failed. Please ensure Hugging Face repository name is correct.")
    st.exception(e)
    st.stop()


#Text Preprocessing and Prediction
def preprocess_text(text):
    text = str(text)
    # Remove URLs and Mentions, but keep punctuation and stop words
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict_sentiment(comment):
    cleaned_text = preprocess_text(comment)
    
    # Tokenize the input text
    inputs = tokenizer(
        cleaned_text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=512, 
        padding=True
    )
    
    # Perform inference without calculating gradients
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=-1).item()
    
    # Map the predicted ID to labels 
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    prediction = label_map[predicted_class_id]
    
    return prediction, cleaned_text

#Sidebar design
st.sidebar.title("💻 System Information")

st.sidebar.markdown(
    """
    **Project:** Social Media & Public Opinion  
    **Deployed Model:** RoBERTa (Transformer)  
    **Output Classes:** Positive, Neutral, Negative  
    """
)

st.sidebar.divider()

st.sidebar.markdown(
    """
    ### How to Use
    Enter a social media comment for single prediction OR Upload a CSV file for batch analysis.
    For Batch CSV Sentiment Analysis:
    1. After CSV file upload, select the column that contains comments.
    2. View prediction results and public opinion summary.
    3. Download the prediction results.
    """
)

st.sidebar.divider()

#Main Header Design
st.markdown(
    """
    <div class="main-title">💬 Social Media and Public Opinion Sentiment Analysis System</div>
    <div class="sub-text">
    This web application analyses Malaysian social media comments and predicts whether the sentiment is 
    <b>Positive</b>, <b>Neutral</b>, or <b>Negative</b>.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="info-card">
    <b>Model Information:</b><br>
    The deployed model is a fine-tuned <b>RoBERTa (Transformer)</b> architecture. 
    It leverages deep learning and self-attention to understand context and nuance better than traditional models.
    </div>
    """,
    unsafe_allow_html=True
)

#Tabs (System Function)
tab1, tab2 = st.tabs([
    "Single Comment Prediction",
    "Batch CSV Sentiment Analysis"
])


#Function 1: Single Comment Sentiment Prediction
with tab1:
    st.header("Single Comment Sentiment Prediction")

    user_comment = st.text_area(
        "Enter a social media comment:",
        placeholder="Example: This Raya promotion is not worth it, the discount is fake."
    )

    if st.button("Predict Sentiment", key="single_predict_button"):
        if user_comment.strip() == "":
            st.warning("Please enter a comment first.")
        else:
            prediction, cleaned_text = predict_sentiment(user_comment)

            st.subheader("Prediction Result")

            if prediction == "Positive":
                st.success(f"Sentiment: {prediction}")
            elif prediction == "Negative":
                st.error(f"Sentiment: {prediction}")
            else:
                st.info(f"Sentiment: {prediction}")

            with st.expander("Show pre-processed text"):
                st.code(cleaned_text)

            st.caption("Sentiment predicted using RoBERTa.")


#Function 2: Batch CSV Sentiment Analysis
with tab2:
    st.header("Batch CSV Sentiment Analysis")

    uploaded_file = st.file_uploader(
        "Upload a CSV file containing social media comments",
        type=["csv"]
    )

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Dataset Preview")
        st.dataframe(batch_df.head())

        columns = list(batch_df.columns)

        if "comment" in columns:
            default_index = columns.index("comment")
        elif "message" in columns:
            default_index = columns.index("message")
        elif "translated_message" in columns:
            default_index = columns.index("translated_message")
        else:
            default_index = 0

        text_column = st.selectbox(
            "Select the column that contains comments:",
            columns,
            index=default_index
        )

        if st.button("Analyse CSV", key="batch_analyse_button"):
            st.info("Predicting sentiments... (This may take a moment with Transformer models)")
            
            # Apply the new prediction function to each row
            results = batch_df[text_column].apply(predict_sentiment)
            
            # Unpack the tuple returned by predict_sentiment into two new columns
            batch_df["predicted_sentiment"] = [res[0] for res in results]
            batch_df["cleaned_text"] = [res[1] for res in results]

            st.subheader("Prediction Results")
            st.dataframe(batch_df)

            # Public Opinion Summary
            st.subheader("Public Opinion Summary")

            sentiment_order = ["Positive", "Neutral", "Negative"]
            total_comments = len(batch_df)

            sentiment_counts = (
                batch_df["predicted_sentiment"]
                .value_counts()
                .reindex(sentiment_order, fill_value=0)
            )

            sentiment_percentages = (sentiment_counts / total_comments * 100).round(2)

            summary_df = pd.DataFrame({
                "Sentiment": sentiment_order,
                "Count": sentiment_counts.values,
                "Percentage (%)": sentiment_percentages.values
            })

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Total Comments", total_comments)
            col2.metric("Positive", f"{sentiment_counts['Positive']} ({sentiment_percentages['Positive']}%)")
            col3.metric("Neutral", f"{sentiment_counts['Neutral']} ({sentiment_percentages['Neutral']}%)")
            col4.metric("Negative", f"{sentiment_counts['Negative']} ({sentiment_percentages['Negative']}%)")

            st.write("Summary table:")
            st.dataframe(summary_df)

            max_count = sentiment_counts.max()
            dominant_sentiments = sentiment_counts[sentiment_counts == max_count].index.tolist()

            if len(dominant_sentiments) == 1:
                dominant_sentiment = dominant_sentiments[0]

                if dominant_sentiment == "Positive":
                    st.success(f"Dominant Public Opinion: {dominant_sentiment}")
                    st.write("Overall, the analysed comments show a more positive public opinion.")
                elif dominant_sentiment == "Negative":
                    st.error(f"Dominant Public Opinion: {dominant_sentiment}")
                    st.write("Overall, the analysed comments show a more negative public opinion.")
                else:
                    st.info(f"Dominant Public Opinion: {dominant_sentiment}")
                    st.write("Overall, the analysed comments are mostly neutral or informational.")
            else:
                st.warning(
                    "Dominant Public Opinion: Tie between "
                    + ", ".join(dominant_sentiments)
                )
                st.write(
                    "The analysed comments show a mixed public opinion because two or more sentiment classes have the same highest count."
                )

            # Sentiment Distribution Chart
            st.subheader("Sentiment Distribution")

            chart_df = summary_df.set_index("Sentiment")[["Count"]]
            st.bar_chart(chart_df)

            # Optional Evaluation if expected_sentiment exists (For testing purposes)
            if "expected_sentiment" in batch_df.columns:
                st.subheader("Optional Evaluation on Uploaded Sample")

                correct_predictions = (
                    batch_df["expected_sentiment"] == batch_df["predicted_sentiment"]
                ).sum()

                sample_accuracy = correct_predictions / total_comments * 100

                st.write(
                    f"Sample Accuracy based on `expected_sentiment`: "
                    f"**{sample_accuracy:.2f}%** "
                    f"({correct_predictions}/{total_comments} correct)"
                )

                st.caption(
                    "This accuracy is only for the uploaded sample file when an expected_sentiment column is provided. "
                    "The official model evaluation is reported separately using the test set."
                )

            # Download Results
            csv_output = batch_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Prediction Results",
                data=csv_output,
                file_name="sentiment_prediction_results.csv",
                mime="text/csv"
            )

# Footer design
st.divider()
st.caption("TNL6323 Natural Language Processing Group Project")
