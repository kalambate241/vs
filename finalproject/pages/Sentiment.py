import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import altair as alt
import io

# Function to analyze sentiment using VADER
def analyze_sentiment_vader(text):
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    sentiment = "Positive" if scores['compound'] > 0 else "Negative" if scores['compound'] < 0 else "Neutral"
    return sentiment, scores

# Function to analyze sentiment using TextBlob
def analyze_sentiment_textblob(text):
    blob = TextBlob(text)
    sentiment = "Positive" if blob.sentiment.polarity > 0 else "Negative" if blob.sentiment.polarity < 0 else "Neutral"
    return sentiment, blob.sentiment.polarity

# Updated function to plot sentiment scores
def plot_sentiment(scores):
    labels = ['Positive', 'Neutral', 'Negative']
    values = [scores['pos'], scores['neu'], scores['neg']]
    data = pd.DataFrame({'Sentiment': labels, 'Score': values})

    fig, ax = plt.subplots()
    sns.barplot(x='Sentiment', y='Score', hue='Sentiment', data=data, palette=["green", "gray", "red"], legend=False, ax=ax)
    
    ax.set_title("Sentiment Distribution")
    ax.set_ylabel("Score")
    st.pyplot(fig)

# Function to save analysis results to CSV
def save_analysis_to_csv(text, vader_sentiment, vader_scores, blob_sentiment, blob_polarity):
    data = {
        "Text": [text],
        "VADER Sentiment": [vader_sentiment],
        "VADER Compound": [vader_scores['compound']],
        "TextBlob Sentiment": [blob_sentiment],
        "TextBlob Polarity": [blob_polarity]
    }
    df = pd.DataFrame(data)
    return df

# Function to save analysis as text
def save_analysis_to_text(text, vader_sentiment, vader_scores, blob_sentiment, blob_polarity):
    text_output = f"Text: {text}\n"
    text_output += f"VADER Sentiment: {vader_sentiment}\n"
    text_output += f"VADER Compound Score: {vader_scores['compound']}\n"
    text_output += f"TextBlob Sentiment: {blob_sentiment}\n"
    text_output += f"TextBlob Polarity Score: {blob_polarity}\n\n"
    return text_output

# Function to show sentiment trend using Altair
def show_sentiment_trend(text_list):
    data = []
    for text in text_list:
        sentiment, scores = analyze_sentiment_vader(text)
        data.append({"Text": text, "Compound Score": scores['compound']})
    df = pd.DataFrame(data)
    
    chart = alt.Chart(df).mark_line().encode(
        x=alt.X("Text:N", title="Text Samples"),
        y=alt.Y("Compound Score:Q", title="Sentiment Score"),
        tooltip=["Text", "Compound Score"]
    ).properties(title="Sentiment Trend Over Multiple Texts")
    
    st.altair_chart(chart, use_container_width=True)

# Streamlit UI
st.set_page_config(page_title="Sentiment Analysis Tool", layout="wide")
st.title("📊 Sentiment Analysis Tool")
st.write("Analyze the sentiment of a given text using VADER and TextBlob with interactive visualizations and data export.")

# User Input
text_list = st.text_area("Enter multiple texts (one per line) for trend analysis:").split("\n")

if st.button("Analyze"):
    if text_list and any(text.strip() for text in text_list):
        st.subheader("📈 Sentiment Trend Analysis")
        show_sentiment_trend(text_list)
        
        text_file_content = ""  # Store all text results for a single text file download
        
        for text in text_list:
            if text.strip():
                vader_sentiment, vader_scores = analyze_sentiment_vader(text)
                blob_sentiment, blob_polarity = analyze_sentiment_textblob(text)
                
                st.subheader(f"🔍 Sentiment Analysis for: {text[:50]}...")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**VADER Sentiment:** {vader_sentiment}")
                    st.json(vader_scores)
                    plot_sentiment(vader_scores)
                
                with col2:
                    st.write(f"**TextBlob Sentiment:** {blob_sentiment}")
                    st.write(f"**Polarity Score:** {blob_polarity:.2f}")
                
                # Save and Download CSV
                df = save_analysis_to_csv(text, vader_sentiment, vader_scores, blob_sentiment, blob_polarity)
                st.download_button(label="Download CSV", data=df.to_csv(index=False), file_name=f"sentiment_analysis_{text_list.index(text)}.csv", mime="text/csv", key=f"download_csv_{text_list.index(text)}")
                
                # Save text data
                text_file_content += save_analysis_to_text(text, vader_sentiment, vader_scores, blob_sentiment, blob_polarity)
        
        # Provide a single text file download with all results
        text_file = io.BytesIO(text_file_content.encode('utf-8'))
        st.download_button(label="Download Text Report", data=text_file, file_name="sentiment_analysis.txt", mime="text/plain")
    
    else:
        st.warning("Please enter some text to analyze.")
