import pandas as pd
import time
from cleantext import clean
from custom_llm import CustomLLMModel
from custom_rag import CustomRAG

# read the file of extract user reviews and clean them up
df = pd.read_csv('./all_posts.csv')
df['user_reviews'] = df['post_title'].astype(str)+ df['self_text'].astype(str)
df['user_reviews'] = df['user_reviews'].apply(lambda x: clean(x, no_emoji=True))
df['user_reviews'] = df['user_reviews'].str.replace(r'\n+', ' ', regex=True)
df['user_reviews'] = df['user_reviews'].str.strip()
df['user_reviews'] = df['user_reviews'].str.replace(r'\s+', ' ', regex=True)

# create a list of documents
user_reviews = [df['user_reviews'].iloc[record] for record in range(0,df['user_reviews'].size)]

# create a model instance
model = CustomLLMModel()
vector_store = model.create_vectorstore(user_reviews)

#create a RAG instance
data_RAG = CustomRAG(model=model)

print(data_RAG.do_similarity_search(vector_store=vector_store, query="Watch connection issues"))
print('Summary of the issues')
start = time.time()
print(data_RAG.get_summary(vector_store))
end = time.time()
print(f'Time taken to summarize the content: {end-start} seconds')