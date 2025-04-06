import chromadb, pprint

vector = [1,2,3,4,5,6]
print(f'vector :{vector}')

client = chromadb.Client()

collection = client.create_collection("all-my-documents")
collection.add(
    documents=[
        "This is about food",
        "This is about animal's food",
        "This is about cats and dogs"
        ],
    metadatas = [{"topic1":"food"},{"topic2":"animal"},{"topic3":"animal"}],
    ids=["doc1","doc2","doc3"],
)
results = collection.query(
    query_texts = ["This is about food"],
    n_results =2
)
pprint.pprint(results)