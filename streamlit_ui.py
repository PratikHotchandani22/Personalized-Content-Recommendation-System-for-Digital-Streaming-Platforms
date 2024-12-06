import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from models import prepare_models_and_data, generate_recommendations, combine_recommendations_to_dataframe, get_all_data_for_recommendations
from semantic_search import prepare_data_and_model, recommend_movies

# Set Streamlit page configuration (removing the sidebar)
st.set_page_config(page_title="Movie Recommendations", layout="wide", initial_sidebar_state="collapsed")

# Load the dataset (replace the path with your actual CSV file path)
df_final = pd.read_csv('final_dataset.csv')

df_semantic_search = pd.read_csv('netflix_titles.csv')

# Load and process the data using the first function
df_semantic, model_semantic = prepare_data_and_model(df_semantic_search)

prepared_data = prepare_models_and_data(df_final)

# Set the title for the Streamlit app
st.title("🎬 Movie Recommendations by Method")

# Add a nice background and customize the input field with black text color
st.markdown("""
    <style>
    .stTextInput>div>div>input {
        background-color: #f0f0f5;
        border-radius: 5px;
        padding: 10px;
        font-size: 16px;
        color: black;  /* Set font color to black */
    }
    .stButton>button {
        background-color: #ff7f50;
        border-radius: 5px;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Text input for movie recommendation preferences
user_query = st.text_input("Enter your movie preferences (e.g., genre, duration, type):")

# If the user enters a query, pass it to the movie_recommendations function
if user_query:
    # Step 1: Generate recommendations and create the recommendation DataFrame
    recommendations = generate_recommendations(prepared_data, user_query)
    df_recommendations = combine_recommendations_to_dataframe(recommendations)

    # Step 2: Get detailed data for recommendations split by method
    tfidf_df, louvain_df, bow_df, word2vec_df = get_all_data_for_recommendations(df_recommendations, df_final)

    # Step 3: Combine the dataframes into a dictionary for easy handling in UI
    dataframes = {
        "TF-IDF Recommendations": tfidf_df,
        "Louvain Recommendations": louvain_df,
        "BoW Recommendations": bow_df,
        "Word2Vec Recommendations": word2vec_df,
    }

    # Streamlit UI
    if len(df_recommendations) > 0:
        # Step 4: Display each recommendation method as a row
        for method_name, df in dataframes.items():
            # Displaying method name in a more aesthetic manner
            st.markdown(f"### 🏆 {method_name}")
            if not df.empty:
                # Styling the dataframe for a better aesthetic view
                st.dataframe(df.style.set_table_styles([
                    {'selector': 'thead th', 'props': [('background-color', '#f1f1f1'), ('color', '#333')]},
                    {'selector': 'tbody td', 'props': [('font-size', '14px'), ('color', '#666')]},
                    {'selector': 'tr:nth-child(even)', 'props': [('background-color', '#f9f9f9')]},
                ]))
            else:
                st.write(f"No recommendations found for {method_name}.")

    else:
        st.write("No recommendations found based on your preferences.")

    # Get recommendations based on the user's query
    recommendations_semantic = recommend_movies(user_query, model_semantic, df_semantic)

    # Display the recommended movies
    st.markdown(f"### 🏆 Semantic Search Recommendations")
    st.write(recommendations_semantic)
    
else:
    st.write("Please enter your preferences to get recommendations.")

# Add footer or additional notes
st.markdown("---")
st.markdown("#### Made with ❤️ for movie enthusiasts!")

