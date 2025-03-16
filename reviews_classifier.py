import pandas as pd


df = pd.read_csv("./all_posts.csv")
print(df.columns)
df["combined_reviews"] = df['post_title'] + "." + df['self_text']
print(df.columns)
print(df['combined_reviews'][0])
print(df['post_title'][0])
print(df['self_text'][0])