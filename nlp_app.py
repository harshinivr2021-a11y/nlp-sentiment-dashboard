import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load preprocessed data & model
df = pd.read_csv("preprocessed_reviews.csv")
model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

st.sidebar.title("Filters")

# Platform filter
platform_filter = st.sidebar.selectbox("Platform", ["All"] + df['platform'].dropna().unique().tolist())

# Location filter
location_filter = st.sidebar.selectbox("Location", ["All"] + df['location'].dropna().unique().tolist())

# Apply filters
df_filtered = df.copy()
if platform_filter != "All":
    df_filtered = df_filtered[df_filtered['platform'] == platform_filter]
if location_filter != "All":
    df_filtered = df_filtered[df_filtered['location'] == location_filter]

st.title("📊 Sentiment Analysis Dashboard")

st.subheader("Sentiment Distribution")
sentiment_counts = df_filtered['sentiment'].value_counts()

fig, ax = plt.subplots()
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="viridis", ax=ax)
ax.set_xlabel("Sentiment")
ax.set_ylabel("Count")
st.pyplot(fig)


st.subheader("Average Rating Over Time")
if "date" in df_filtered.columns:
    df_filtered['date'] = pd.to_datetime(df_filtered['date'], errors='coerce')
    trend = df_filtered.groupby(df_filtered['date'].dt.to_period("M"))['rating'].mean()

    fig, ax = plt.subplots()
    trend.plot(kind="line", marker="o", ax=ax)
    ax.set_ylabel("Average Rating")
    st.pyplot(fig)


st.subheader("🔮 Try Your Own Review")
user_input = st.text_area("Enter a review:")

if st.button("Predict Sentiment"):
    # simple clean function
    import re
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\\s]', '', text)
        tokens = text.split()
        tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
        return " ".join(tokens)

    clean = clean_text(user_input)
    vec = vectorizer.transform([clean])
    prediction = model.predict(vec)[0]
    st.success(f"Predicted Sentiment: **{prediction}**")



