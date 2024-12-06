#!/usr/bin/env python
# coding: utf-8

# ### Exploratory Data Analysis

# In[166]:


#get_ipython().system('pip install fuzzywuzzy')


# In[167]:


import pandas as pd
import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt

from fuzzywuzzy import process
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from community import community_louvain  # Louvain clustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from networkx.algorithms.community import greedy_modularity_communities
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import streamlit as st


# ### Loading the data

# In[168]:


df = pd.read_csv('netflix_titles.csv')


# In[169]:


df.head()


# ### Check for missing values

# In[170]:


missing_values = df.isnull().sum()
print(missing_values)


# In[171]:


df['rating'].unique()


# In[172]:


df['listed_in']


# ### Splitting the genres

# In[173]:


df['genres'] = df['listed_in'].str.split(', ')


# In[174]:


df['genres']


# In[175]:


df_exploded = df.explode('genres')


# In[176]:


df_exploded


# ### Country wise mode

# In[177]:


genre_country_mode = df_exploded.groupby('genres')['country'].apply(
    lambda x: x.mode()[0] if not x.mode().empty else 'Unknown'
)


# In[178]:


genre_country_mode


# ### Filling in the missing values for country column

# In[179]:


# Map the mode back to the original DataFrame for missing 'country' values
df['country'] = df.apply(
    lambda row: genre_country_mode[row['genres'][0]] if pd.isnull(row['country']) and row['genres'] else row['country'],
    axis=1
)


# In[180]:


missing_values = df.isnull().sum()
print(missing_values)


# ### Exploring the rating column

# In[181]:


df.rating.value_counts()
df.rating.replace(['74 min', '84 min', '66 min'], 'TV-MA',inplace=True)


# In[182]:


df['rating'].unique()


# In[183]:


# Get the row with index label '3'
specific_row = df.loc[211]
print(specific_row)


# In[184]:


df['genres']


# In[185]:


rating_replacements = {
    211: 'TV-14',
    2411: 'TV-14',
    3288: 'PG-13',
    4056: 'TV-G',
    4402: 'TV-G',
    4403: 'TV-G',
    4706: 'TV-14',
    5015: 'TV-14',
    5234: 'TV-14',
    6231: 'TV-Y'
}

for id, rate in rating_replacements.items():
    df.iloc[id, 8] = rate

df['rating'].isna().sum()


# In[186]:


df.loc[[5989], ['rating']] = 'TV-PG'
df.loc[[6827], ['rating']] = 'TV-14'
df.loc[[7312], ['rating']] = 'TV-PG'
df.loc[[7537], ['rating']] = 'PG-13'


# In[187]:


df['rating'].isna().sum()


# In[188]:


df.loc[df.rating.isin(['TV-Y7-FV']), ['rating']] = 'TV-Y7'
df.loc[df.rating.isin(['TV-G']), ['rating']] = 'G'
df.loc[df.rating.isin(['TV-PG']), ['rating']] = 'PG'
df.loc[df.rating.isin(['TV-MA']), ['rating']] = 'R'
df.loc[df.rating.isin(['NR', 'UR']), ['rating']] = 'nrur'


# In[189]:


df


# ### Number of ratings per rating category

# In[190]:


fig, ax = plt.subplots(figsize =(12,8))
ax = sns.countplot(y=df.rating,order = df.rating.value_counts().index,palette='rocket')
ax.bar_label(ax.containers[0])
ax.set_title('Number of ratings')


# ### Amount of Content per Country

# In[191]:


country_counts = df['country'].value_counts()[:10]
plt.figure(figsize=(10, 6))
country_counts.plot(kind='bar', color='green')
plt.title('Top 10 Countries with Most Content')
plt.xlabel('Country')
plt.ylabel('Number of Titles')
plt.tight_layout()
plt.show()


# ### Date time column

# In[192]:


# Data Cleaning
df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')  # Convert to datetime
df['year_added'] = df['date_added'].dt.year  # Extract year
df['month_added'] = df['date_added'].dt.month  # Extract month
# df['duration'] = df['duration'].fillna('Unknown')  # Handle missing durations


# In[193]:


df


# In[194]:


df.isnull().sum()


# In[195]:


# Convert 'year_added' to integers, handling missing values
df['year_added'] = df['year_added'].fillna(0).astype(int)
df['month_added'] = df['month_added'].fillna(0).astype(int)

# Replace 0 with NaN for clarity (optional)
df['year_added'] = df['year_added'].replace(0, pd.NA)
df['month_added'] = df['month_added'].replace(0, pd.NA)


# In[196]:


df


# In[197]:


df['duration'].isna().sum()


# ### Number of Movies and TV Shows on Netflix

# In[198]:


plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='type', palette='pastel')
plt.title('Content Type Distribution (Movies vs TV Shows)')
plt.xlabel('Content Type')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# 

# In[199]:


df['genres']


# ### Plot 1: Popular Genres by Region

# In[200]:


df_exploded = df.explode('genres')

# Filter countries for visualization
top_countries = ['United States', 'India', 'United Kingdom', 'Japan', 'South Korea']
df_filtered = df_exploded[df_exploded['country'].isin(top_countries)]

# Count genres by country
genre_country_counts = df_filtered.groupby(['genres', 'country']).size().unstack().fillna(0)

# Plot
plt.figure(figsize=(16, 8))
genre_country_counts.plot(kind='bar', stacked=True, colormap='viridis', figsize=(16, 8))
plt.title('Popular Genres by Region', fontsize=16)
plt.xlabel('Genres', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend(title='Country', fontsize=10)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# ### # Plot 2: Temporal Trends in Content Production
# 

# In[201]:


df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce')
top_genres = ['International Movies', 'Dramas', 'Comedies', 'International TV Shows', 'Documentaries']
df_filtered = df_exploded[df_exploded['genres'].isin(top_genres)]

# Count by year and genre
temporal_trends = df_filtered.groupby(['release_year', 'genres']).size().unstack().fillna(0)

# Plot
plt.figure(figsize=(12, 6))
temporal_trends.plot(marker='o', figsize=(12, 6))
plt.title('Temporal Trends in Content Production', fontsize=16)
plt.xlabel('Release Year', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(title='Genre', fontsize=10)
plt.tight_layout()
plt.show()


# ### Movie Recommendation

# In[202]:

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from fuzzywuzzy import process
from community import community_louvain

def prepare_models_and_data_old(df):
    """
    Prepares models, similarity matrices, and clustering data for movie recommendations.
    Returns necessary components for recommendations.
    """
    # Step 1: Preprocess Data
    def preprocess_data(df):
        kid = df[df.rating.isin(['TV-Y', 'TV-Y7', 'G', 'PG'])].index
        teen = df[df.rating.isin(['PG-13', 'TV-14'])].index
        adult = df[df.rating.isin(['R', 'NC-17', 'nrur'])].index
        df.loc[kid, 'maturity_level'] = 'kid'
        df.loc[teen, 'maturity_level'] = 'teen'
        df.loc[adult, 'maturity_level'] = 'adult'

        df['content'] = (df['title'].astype(str) + ' ' + df['director'].astype(str) +
                         ' ' + df['cast'].astype(str) + ' ' + df['country'].astype(str) +
                         ' ' + df['rating'].astype(str) + ' ' + df['duration'].astype(str) +
                         ' ' + df['listed_in'].astype(str) + ' ' + df['description'].astype(str) +
                         ' ' + df['maturity_level'].astype(str))
        df['content'] = df['content'].fillna('')

    preprocess_data(df)

    # Step 2: Create TF-IDF and BoW Matrices
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['content'])

    count_vectorizer = CountVectorizer()
    bow_matrix = count_vectorizer.fit_transform(df['content'])

    # Step 3: Train Word2Vec Model
    df['tokenized_content'] = df['content'].apply(simple_preprocess)
    w2v_model = Word2Vec(vector_size=100, window=5, min_count=1, workers=4)
    w2v_model.build_vocab(df['tokenized_content'])
    w2v_model.train(df['tokenized_content'], total_examples=w2v_model.corpus_count, epochs=10)

    def averaged_word_vectorizer(corpus, model, num_features):
        vocabulary = set(model.wv.index_to_key)
        features = [
            np.mean([model.wv[word] for word in words if word in vocabulary] or
                    [np.zeros(num_features)], axis=0)
            for words in corpus
        ]
        return np.array(features)

    w2v_features = averaged_word_vectorizer(df['tokenized_content'], w2v_model, 100)

    # Step 4: Build Similarity Graph
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    threshold = 0.1
    G = nx.Graph()

    for i, title in enumerate(df['title']):
        G.add_node(i, title=title)

    for i in range(cosine_sim.shape[0]):
        for j in range(i + 1, cosine_sim.shape[1]):
            if cosine_sim[i, j] > threshold:
                G.add_edge(i, j, weight=cosine_sim[i, j])

    partition = community_louvain.best_partition(G)
    nx.set_node_attributes(G, partition, 'cluster')

    return {
        "df": df,
        "tfidf_matrix": tfidf_matrix,
        "bow_matrix": bow_matrix,
        "w2v_features": w2v_features,
        "w2v_model": w2v_model,
        "G": G,
        "partition": partition
    }

def prepare_models_and_data(df):
    """
    Prepares models, similarity matrices, and clustering data for movie recommendations.
    Returns necessary components for recommendations.
    """
    
    # Step 1: Preprocess Data
    def preprocess_data(df):
        # Categorize movies into maturity levels
        kid = df[df.rating.isin(['TV-Y', 'TV-Y7', 'G', 'PG'])].index
        teen = df[df.rating.isin(['PG-13', 'TV-14'])].index
        adult = df[df.rating.isin(['R', 'NC-17', 'nrur'])].index
        df.loc[kid, 'maturity_level'] = 'kid'
        df.loc[teen, 'maturity_level'] = 'teen'
        df.loc[adult, 'maturity_level'] = 'adult'

        # Combine multiple columns into a single 'content' column for similarity calculation
        df['content'] = (df['title'].astype(str) + ' ' + df['director'].astype(str) +
                         ' ' + df['cast'].astype(str) + ' ' + df['country'].astype(str) +
                         ' ' + df['rating'].astype(str) + ' ' + df['duration'].astype(str) +
                         ' ' + df['listed_in'].astype(str) + ' ' + df['description'].astype(str) +
                         ' ' + df['maturity_level'].astype(str))
        df['content'] = df['content'].fillna('')

    preprocess_data(df)

    # Step 2: Create TF-IDF and BoW Matrices
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['content'])

    count_vectorizer = CountVectorizer()
    bow_matrix = count_vectorizer.fit_transform(df['content'])

    # Step 3: Train Word2Vec Model
    df['tokenized_content'] = df['content'].apply(simple_preprocess)  # Tokenize content
    w2v_model = Word2Vec(vector_size=100, window=5, min_count=1, workers=4)
    w2v_model.build_vocab(df['tokenized_content'])
    w2v_model.train(df['tokenized_content'], total_examples=w2v_model.corpus_count, epochs=10)

    # Averaged Word Vectors
    def averaged_word_vectorizer(corpus, model, num_features):
        vocabulary = set(model.wv.index_to_key)
        features = [
            np.mean([model.wv[word] for word in words if word in vocabulary] or
                    [np.zeros(num_features)], axis=0)
            for words in corpus
        ]
        return np.array(features)

    w2v_features = averaged_word_vectorizer(df['tokenized_content'], w2v_model, 100)

    # Step 4: Build Similarity Graph based on TF-IDF
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    threshold = 0.1  # Set threshold for edge creation based on similarity score
    G = nx.Graph()

    # Add nodes to the graph with movie titles as attributes
    for i, title in enumerate(df['title']):
        G.add_node(i, title=title)

    # Add edges between movies that are sufficiently similar
    for i in range(cosine_sim.shape[0]):
        for j in range(i + 1, cosine_sim.shape[1]):
            if cosine_sim[i, j] > threshold:
                G.add_edge(i, j, weight=cosine_sim[i, j])

    # Step 5: Community Detection (Louvain Method)
    partition = community_louvain.best_partition(G)
    nx.set_node_attributes(G, partition, 'cluster')  # Assign clusters to nodes

    # Return all necessary components for recommendation and analysis
    return {
        "df": df,
        "tfidf_matrix": tfidf_matrix,
        "bow_matrix": bow_matrix,
        "w2v_features": w2v_features,
        "w2v_model": w2v_model,
        "G": G,
        "partition": partition
    }

def generate_recommendations(prepared_data, user_movie):
    """
    Generates recommendations based on various models and approaches, including similarity scores.
    """
    df = prepared_data["df"]
    tfidf_matrix = prepared_data["tfidf_matrix"]
    bow_matrix = prepared_data["bow_matrix"]
    w2v_features = prepared_data["w2v_features"]
    G = prepared_data["G"]
    partition = prepared_data["partition"]

    def find_similar_movies_fuzzy(df, movie_name):
        top_movies = process.extract(movie_name, df['title'], limit=5)
        return top_movies[0][0]  # Returns the closest movie name

    def get_tfidf_recommendations():
        movie_name = find_similar_movies_fuzzy(df, user_movie)
        movie_index = df[df['title'] == movie_name].index[0]
        similarity_scores = cosine_similarity(tfidf_matrix[movie_index], tfidf_matrix).flatten()
        top_indices = similarity_scores.argsort()[-20:][::-1]
        return [(df.loc[i, 'title'], similarity_scores[i]) for i in top_indices]

    def get_bow_recommendations():
        movie_name = find_similar_movies_fuzzy(df, user_movie)
        movie_index = df[df['title'] == movie_name].index[0]
        similarity_scores = cosine_similarity(bow_matrix[movie_index], bow_matrix).flatten()
        top_indices = similarity_scores.argsort()[-20:][::-1]
        return [(df.loc[i, 'title'], similarity_scores[i]) for i in top_indices]

    def get_w2v_recommendations():
        movie_name = find_similar_movies_fuzzy(df, user_movie)
        movie_index = df[df['title'] == movie_name].index[0]
        similarity_scores = cosine_similarity(w2v_features[movie_index].reshape(1, -1), w2v_features).flatten()
        top_indices = similarity_scores.argsort()[-20:][::-1]
        return [(df.loc[i, 'title'], similarity_scores[i]) for i in top_indices]

    def get_louvain_recommendations():
        # Step 1: Use fuzzy matching to find the closest movie title
        movie_name = find_similar_movies_fuzzy(df, user_movie)
        if not movie_name:
            return []  # No recommendations if movie is not found

        # Step 2: Find the node corresponding to the matched movie title
        movie_node = None
        for node, data in G.nodes(data=True):
            if data['title'] == movie_name:
                movie_node = node
                break            

        # Step 3: Handle case where no node is found
        if movie_node is None:
            print(f"Movie '{movie_name}' not found in the graph")
            return []

        # Step 4: Get the movie cluster from the Louvain partition
        movie_cluster = partition.get(movie_node)
        cluster_movies = [node for node, cluster in partition.items() if cluster == movie_cluster]
        
        # Step 5: Get recommendations from movies in the same cluster
        recommendations = [
            (df.iloc[node]['title'], G[movie_node][node]['weight'])
            for node in cluster_movies if G.has_edge(movie_node, node)
        ]
        
        # Step 6: Sort recommendations based on similarity weight and return top 10
        sorted_recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)
        return sorted_recommendations[:10]

    def get_louvain_recommendation_old():
        movie_node = None
        for node, data in G.nodes(data=True):
            print("node is: ", node)
            print("data is: ", data)
            
            if data['title'] == user_movie:
                print("movie found in for loop!!!!")
                movie_node = node
                break            

        if movie_node is None:
            print("movie searched is: ", user_movie)
            print("no movies node found")
            return []

        movie_cluster = partition.get(movie_node)
        cluster_movies = [node for node, cluster in partition.items() if cluster == movie_cluster]
        recommendations = [
            (df.iloc[node]['title'], G[movie_node][node]['weight'])
            for node in cluster_movies if G.has_edge(movie_node, node)
        ]
        sorted_recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)
        return sorted_recommendations[:10]

    return {
        "TF-IDF": get_tfidf_recommendations(),
        "BoW": get_bow_recommendations(),
        "Word2Vec": get_w2v_recommendations(),
        "Louvain": get_louvain_recommendations()
    }

def combine_recommendations_to_dataframe(recommendations):
    """
    Combines the list output from generate_recommendations into a single DataFrame.

    Parameters:
        recommendations (dict): A dictionary where keys are method names and 
                                 values are lists of tuples (recommendation_name, similarity_score).

    Returns:
        pd.DataFrame: A DataFrame with columns - method_name, recommendation_name, similarity_score.
    """
    combined_data = []
    for method_name, recommendation_list in recommendations.items():
        for recommendation_name, similarity_score in recommendation_list:
            combined_data.append({
                'method_name': method_name,
                'recommendation_name': recommendation_name,
                'similarity_score': similarity_score
            })
    
    return pd.DataFrame(combined_data)

def split_recommendations_into_dataframes(recommendations):
    """
    Splits the recommendations dictionary into separate DataFrames for each method.

    Parameters:
        recommendations (dict): A dictionary where keys are method names and 
                                 values are lists of tuples (recommendation_name, similarity_score).

    Returns:
        dict: A dictionary where keys are method names and values are DataFrames with 
              columns - recommendation_name, similarity_score.
    """
    dataframes = {}
    for method_name, recommendation_list in recommendations.items():
        # Create a DataFrame for each method
        dataframes[method_name] = pd.DataFrame(
            recommendation_list, 
            columns=["recommendation_name", "similarity_score"]
        )
    return dataframes

def get_all_data_for_recommendations_old(recommendation_df ,final_df):

    # Step 1: Find unique movie names in the 'recommendation_name' column
    unique_movie_names = recommendation_df["recommendation_name"].unique()

    # Step 2: Filter rows in the DataFrame based on these unique movie names
    filtered_df = final_df[final_df["title"].isin(unique_movie_names)]

    return filtered_df

def get_all_data_for_recommendations(recommendation_df, final_df):
    # Step 1: Find unique movie names in the 'recommendation_name' column
    unique_movie_names = recommendation_df["recommendation_name"].unique()
    
    # Step 2: Filter rows in the final_df based on these unique movie names
    filtered_final_df = final_df[final_df["title"].isin(unique_movie_names)]
    
    # Step 3: Merge recommendation_df with filtered_final_df on the movie titles
    merged_df = recommendation_df.merge(
        filtered_final_df,
        left_on="recommendation_name",
        right_on="title",
        how="inner"
    )
    
    # Step 4: Create separate DataFrames based on the 'method_name'
    tfidf_df = merged_df[merged_df["method_name"] == "TF-IDF"]
    louvain_df = merged_df[merged_df["method_name"] == "Louvain"]
    bow_df = merged_df[merged_df["method_name"] == "BoW"]
    word2vec_df = merged_df[merged_df["method_name"] == "Word2Vec"]

    # Step 5: Drop the specified columns from each DataFrame
    columns_to_drop = ["method_name", "recommendation_name", "show_id", "genres", "tokenized_content", "content"]
    
    tfidf_df = tfidf_df.drop(columns=columns_to_drop, errors='ignore')
    louvain_df = louvain_df.drop(columns=columns_to_drop, errors='ignore')
    bow_df = bow_df.drop(columns=columns_to_drop, errors='ignore')
    word2vec_df = word2vec_df.drop(columns=columns_to_drop, errors='ignore')

    return tfidf_df, louvain_df, bow_df, word2vec_df



"""

def visualize_louvain_recommendations_old(prepared_data, user_movie, recommendations):
    df = prepared_data["df"]
    G = prepared_data["G"]
    partition = prepared_data["partition"]
    
    # Check if the recommendations DataFrame is empty
    if recommendations.empty:
        print("No Louvain recommendations found.")
        return

    # Extract recommended movie titles from the "recommendation_name" column
    recommended_movies = recommendations["recommendation_name"].tolist()
    
    # Step 1: Subgraph containing recommended movies
    subgraph_nodes = set()
    for movie in recommended_movies:
        # Find the corresponding node in the graph
        for node, data in G.nodes(data=True):
            if data['title'] == movie:
                subgraph_nodes.add(node)

    # Step 2: Create the subgraph for visualization
    subgraph = G.subgraph(subgraph_nodes)

    # Step 3: Define the layout and plot the subgraph
    pos = nx.spring_layout(subgraph)  # You can experiment with different layouts

    # Step 4: Draw the graph
    plt.figure(figsize=(10, 8))
    nx.draw(subgraph, pos, with_labels=True, node_size=500, node_color='lightblue', font_size=10, font_weight='bold', edge_color='gray')

    # Display the plot
    plt.title(f"Louvain Recommendations for '{user_movie}'", fontsize=14)
    # Replace plt.show() with st.pyplot
    st.pyplot(plt)

def visualize_louvain_recommendations(prepared_data, user_movie, recommendations):
    df = prepared_data["df"]
    G = prepared_data["G"]
    partition = prepared_data["partition"]

    # Check if the recommendations DataFrame is empty
    if recommendations.empty:
        print("No Louvain recommendations found.")
        return

    # Extract recommended movie titles from the "recommendation_name" column
    recommended_movies = recommendations["recommendation_name"].tolist()

    # Create a set of recommended movie nodes
    recommended_nodes = set()
    for movie in recommended_movies:
        for node, data in G.nodes(data=True):
            if data.get("title") == movie:
                recommended_nodes.add(node)

    # Define positions for all nodes
    pos = nx.spring_layout(G, seed=42)  # Use a fixed seed for consistent layouts

    # Plot the entire graph
    plt.figure(figsize=(12, 10))
    nx.draw(
        G, pos, with_labels=False, node_size=50, node_color="gray", edge_color="lightgray"
    )

    # Highlight recommended movies
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=recommended_nodes,
        node_size=500,
        node_color="lightblue",
        label="Recommended Movies",
    )

    # Label nodes with movie titles
    node_labels = {node: data["title"] for node, data in G.nodes(data=True) if node in recommended_nodes}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, font_weight="bold")

    # Title and legend
    plt.title(f"Louvain Recommendations for '{user_movie}'", fontsize=14)
    plt.legend(scatterpoints=1)
    st.pyplot(plt)



def preprocess_data(df):
    #Setting Maturity Levels for kids, Teens, Adults
    kid = df[df.rating.isin(['TV-Y','TV-Y7','G','PG'])].index
    teen = df[df.rating.isin(['PG-13','TV-14'])].index
    adult = df[df.rating.isin(['R','NC-17','nrur'])].index
    df.loc[kid, 'maturity_level'] = 'kid'
    df.loc[teen, 'maturity_level'] = 'teen'
    df.loc[adult, 'maturity_level'] = 'adult'

    # combining all the contents making a big string of knowledge
    df['content'] = df['title'].astype(str) + ' ' + df['director'].astype(str) + ' ' + df['cast'].astype(str) + ' ' + df['country'].astype(str) + ' ' + df['rating'].astype(str) + df['duration'].astype(str) + ' ' + df['listed_in'].astype(str) + ' ' + df['description'].astype(str)  + ' ' + df['maturity_level'].astype(str)
    df['content'] = df['content'].fillna('')

def create_tfidf_matrix(df):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['content'])
    return tfidf_matrix

# Function to create Bag of Words (BoW) matrix
def create_bow_matrix(df):
    count_vectorizer = CountVectorizer()
    bow_matrix = count_vectorizer.fit_transform(df['content'])
    return bow_matrix

# Function to compute TF-IDF cosine similarity
def tfidf_cosine_similarity(tfidf_matrix):
    cosine_sim = cosine_similarity(tfidf_matrix)
    return cosine_sim

# Function to compute BoW cosine similarity
def bow_cosine_similarity(bow_matrix):
    cosine_sim = cosine_similarity(bow_matrix)
    return cosine_sim

# Function to train Word2Vec model
def train_word2vec(df):
    df['tokenized_content'] = df['content'].apply(simple_preprocess)
    model = Word2Vec(vector_size=100, window=5, min_count=1, workers=4)
    model.build_vocab(df['tokenized_content'])
    model.train(df['tokenized_content'], total_examples=model.corpus_count, epochs=10)
    return model

# Function to average word vectors for a text
def average_word_vectors(words, model, vocabulary, num_features):
    feature_vector = np.zeros((num_features,), dtype="float64")
    nwords = 0.

    for word in words:
        if word in vocabulary:
            nwords = nwords + 1.
            feature_vector = np.add(feature_vector, model.wv[word])

    if nwords:
        feature_vector = np.divide(feature_vector, nwords)

    return feature_vector

def averaged_word_vectorizer(corpus, model, num_features):
    vocabulary = set(model.wv.index_to_key)
    features = [average_word_vectors(tokenized_sentence, model, vocabulary, num_features) for tokenized_sentence in corpus]
    return np.array(features)

# Function to compute Word2Vec-based similarity
def word2vec_similarity(user_movie, df):
    user_movie = find_similar_movies_fuzzy(df, user_movie)
    df['tokenized_content'] = df['content'].apply(simple_preprocess)
    model = Word2Vec(vector_size=100, window=5, min_count=1, workers=4)
    model.build_vocab(df['tokenized_content'])
    model.train(df['tokenized_content'], total_examples=model.corpus_count, epochs=10)
    movie_index = df[df['title'] == user_movie].index[0]
    w2v_feature_array = averaged_word_vectorizer(corpus=df['tokenized_content'], model=model, num_features=100)

    # Compute the cosine similarities between the user movie and all other movies
    user_movie_vector = w2v_feature_array[movie_index].reshape(1, -1)
    similarity_scores = cosine_similarity(user_movie_vector, w2v_feature_array)

    # Get the top 10 most similar movies
    similar_movies = list(enumerate(similarity_scores[0]))
    sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1:20]

    # Print the top 10 similar movies
    for i, score in sorted_similar_movies:
        print("{}: {}".format(i, df.loc[i, 'title']))


# Function to compute BoW-based similarity
def bow_similarity(user_movie, df, bow_matrix):
    user_movie = find_similar_movies_fuzzy(df, user_movie)
    movie_index = df[df['title'] == user_movie].index[0]
    similarity_scores = bow_cosine_similarity(bow_matrix)
    similar_movies = list(enumerate(similarity_scores[movie_index]))
    sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1:20]
    for i, score in sorted_similar_movies:
        print("{}: {}".format(i, df.loc[i, 'title']))

# Function to compute TF-IDF-based similarity
def tfidf_similarity(user_movie, df, tfidf_matrix):
    user_movie = find_similar_movies_fuzzy(df, user_movie)
    movie_index = df[df['title'] == user_movie].index[0]
    similarity_scores = tfidf_cosine_similarity(tfidf_matrix)
    similar_movies = list(enumerate(similarity_scores[movie_index]))
    sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1:20]
    for i, score in sorted_similar_movies:
        print("{}: {}".format(i, df.loc[i, 'title']))

# Function to find similar movies using fuzzy string matching
def find_similar_movies_fuzzy(df, movie_name):
    top_movies = process.extract(movie_name, df['title'], limit=5)
    return top_movies[0][0]

def similar_movies_fuzzy(df, movie_name):
    top_movies = process.extract(movie_name, df['title'], limit=5)
    print("Advanced Search and similar Alternatives")
    for movie, score, index in top_movies:
        print(f"Movie: {movie}, Similarity Score: {score}")


# In[203]:


copied_df = df.copy()


# In[204]:


# Preprocess data and create the 'content' column
preprocess_data(copied_df)

# Create the TF-IDF matrix and BoW matrix
tfidf_matrix = create_tfidf_matrix(copied_df)
bow_matrix = create_bow_matrix(copied_df)


# In[205]:


# Assuming 'df' contains the cleaned dataset
df['content'] = df['title'] + " " + df['description'].fillna("")
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['content'])

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Build the graph
threshold = 0.1  # Similarity threshold
G = nx.Graph()

# Add nodes
for i, title in enumerate(df['title']):
    G.add_node(i, title=title)

# Add edges
for i in range(cosine_sim.shape[0]):
    for j in range(i + 1, cosine_sim.shape[1]):
        if cosine_sim[i, j] > threshold:
            G.add_edge(i, j, weight=cosine_sim[i, j])


# In[206]:


def compute_Louvain_partition(G):
  # Apply Louvain clustering
  partition = community_louvain.best_partition(G)
  # Add cluster information to each node
  nx.set_node_attributes(G, partition, 'cluster')
  # Display number of clusters
  num_clusters = len(set(partition.values()))
  print(f"Number of clusters: {num_clusters}")
  return partition


# In[207]:


def recommend_from_cluster(movie_title, df, G, partition, top_n=10):
    # Find the node corresponding to the movie title
    movie_node = None
    for node, data in G.nodes(data=True):
        if data['title'] == movie_title:
            movie_node = node
            break

    if movie_node is None:
        print(f"Movie '{movie_title}' not found in the graph.")
        return

    print(f"Node for '{movie_title}': {movie_node}")
    print(f"Edges for '{movie_title}': {list(G.edges(movie_node, data=True))}")

    # Find the cluster of the movie
    movie_cluster = partition.get(movie_node)
    if movie_cluster is None:
        print(f"'{movie_title}' does not belong to any cluster.")
        return

    # Get all movies in the same cluster
    cluster_movies = [node for node, cluster in partition.items() if cluster == movie_cluster]
    print(f"Movies in the same cluster: {len(cluster_movies)}")

    # Rank cluster movies by similarity
    recommendations = []
    for node in cluster_movies:
        if node != movie_node and G.has_edge(movie_node, node):
            similarity = G[movie_node][node]['weight']
            recommendations.append((df.iloc[node]['title'], similarity))

    # Sort by similarity and return top N
    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:top_n]

    if not recommendations:
        print(f"No recommendations found for '{movie_title}' in the same cluster.")
        return

    print(f"Recommendations for '{movie_title}':")
    for title, score in recommendations:
        print(f"  - {title} (Similarity Score: {score:.2f})")


# In[208]:


def Louvain_Movie_Recommendation(user_movie):
  partition = compute_Louvain_partition(G)
  recommend_from_cluster("The Matrix", df, G, partition, top_n=12)


# In[209]:


def movie_recommendations(user_movie):
  similar_movies_fuzzy(copied_df, user_movie)

  print("\nSimilar Movies (TF-IDF Cosine Similarity):")
  tfidf_similarity(user_movie, copied_df, tfidf_matrix)

  print("\nSimilar Movies (BoW Cosine Similarity):")
  bow_similarity(user_movie, copied_df, bow_matrix)

  print("\nSimilar Movies (Word2Vec Similarity):")
  similarity_scores = word2vec_similarity(user_movie, copied_df)

  print("\nSimilar Movies (Graph Louvain Algorithm)")
  Louvain_Movie_Recommendation(user_movie)


# In[210]:


# Get user input
user_movie = input("Enter a movie title: ")
movie_recommendations(user_movie)


# In[210]:
"""



