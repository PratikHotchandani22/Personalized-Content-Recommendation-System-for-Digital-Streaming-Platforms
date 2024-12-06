import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def prepare_data_and_model(netflix_data):
    # Load the dataset
    #netflix_data = pd.read_csv(file_path)

    # Select relevant columns and handle missing values
    netflix_data = netflix_data[['title', 'description', 'listed_in']]
    netflix_data['description'] = netflix_data['description'].fillna('')
    netflix_data['listed_in'] = netflix_data['listed_in'].fillna('')

    # Combine description and listed_in into a single text field for embedding
    netflix_data['text'] = netflix_data['description'] + " " + netflix_data['listed_in']

    # Initialize the model for SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Embed the text data for each movie
    netflix_data['embedding'] = netflix_data['text'].apply(lambda x: model.encode(x, convert_to_tensor=True))

    # Return the prepared data and model
    return netflix_data, model


def recommend_movies(user_query, model, netflix_data, top_n=5):
    # Embed the user's query
    query_embedding = model.encode(user_query, convert_to_tensor=True)

    # Compute cosine similarities between the query and all dataset embeddings
    similarities = [
        cosine_similarity(query_embedding.cpu().numpy().reshape(1, -1), embedding.cpu().numpy().reshape(1, -1))[0][0]
        for embedding in netflix_data['embedding']
    ]

    # Add similarities to the dataframe for ranking
    netflix_data['similarity'] = similarities

    # Sort movies by similarity in descending order
    recommendations = netflix_data.sort_values(by='similarity', ascending=False).head(top_n)

    return recommendations[['title', 'description', 'listed_in', 'similarity']]
