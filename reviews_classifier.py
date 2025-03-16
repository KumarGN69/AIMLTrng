import pandas as pd
import time, re,csv
from sentiment_analyzer import SentimentAnalyzer


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter



# ---------------read the csv file-------------------------------------------------
df = pd.read_csv("./all_posts.csv")

# ---------------combine tile and review text into cpmbinedreview-------------------------------------------------
df["combined_reviews"] = df['post_title'] + "." + df['self_text']
print(df.columns)

# ---------------analyze sentiments -------------------------------------------------
start = time.time()
print(f"Starting Sentiment analysis")
posts = pd.read_csv('./all_posts.csv')

sentiments = SentimentAnalyzer()
df['sentiment'] = df['combined_reviews'].apply(sentiments.assessSentiment)
print(df['sentiment'])
# sentiments.assessSentiments(reviews=posts)
# # print the sentiment analysis summary
# print(
#     f"Positive: {sentiments.positive_sentiments}, Negative:{sentiments.negative_sentiments}, "
#     f" Neutral: {sentiments.neutral_sentiments}, Unclassified: {sentiments.unclassified_sentiments}")
# end = time.time()
# print(f"time taken for sentiment analysis", end - start)

# ---------------cluster using -------------------------------------------------

# df = pd.read_csv("./all_posts.csv")
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\W+', ' ', text)  # Remove punctuation
    return text

df['Cleaned_Review'] = df['combined_reviews'].apply(clean_text)

# Step 2: Convert to TF-IDF Features
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)  # Adjust feature size for speed
X = vectorizer.fit_transform(df['Cleaned_Review'])

# Step 3: Apply K-Means Clustering
num_clusters = 6  # Since you need 6 labels
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X)
df.to_json("review_clusters.json",index=False)
df.to_csv("reviews_clusters.csv",index=False,quoting=csv.QUOTE_ALL,quotechar='"')
