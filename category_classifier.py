from sentence_transformers import SentenceTransformer, util
import pandas as pd

import re, os, csv


#----------------reading the extracted posts into a dataframe---------------------------------------
df = pd.read_csv("./all_posts.csv")

#----------------combining the title and review text into a single text-----------------------------
df["combined_reviews"] = df['post_title'].astype(str).str.cat(df['self_text'].astype(str), na_rep='')


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\W+', ' ', text)  # Remove punctuation
    return text


df['cleaned_reviews'] = df['combined_reviews'].apply(clean_text)

#----------------defining themes-------------------------------------------------------------------

themes = ["Audio issues", "call quality","Video Issues" "Wifi issues", "Bluetooth issues", "Ecosystem","Other"]

#----------------creating embeddings for review and the themes-------------------------------------

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
theme_embedding = model.encode(themes)


def encode_review(review):
    return model.encode(review)


def find_similarity(review, theme_embedding):
    review_embedding = encode_review(review)
    similarities = util.pytorch_cos_sim(review_embedding, theme_embedding)
    return themes[similarities.argmax().item()]


#---------------find similarity and theme--------------------------------------------------------
# Convert 'cleaned_reviews' column to a list and classify themes
df['category'] = [find_similarity(review, theme_embedding) for review in df['cleaned_reviews']]

df.to_csv('./classified_posts.csv',index=False, quoting=csv.QUOTE_ALL, quotechar='"' )