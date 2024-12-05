import streamlit as st
import pandas as pd

# Load the dataset (replace the path with your actual CSV file path)
df = pd.read_csv('final_dataset.csv')

# Extract unique genres for each row
df['unique_genre'] = df['listed_in'].apply(
    lambda x: ', '.join(sorted(set([g.strip() for g in x.split(',')])))
)

# Explode genres for filtering
df_exploded = df.assign(unique_genre=df['unique_genre'].str.split(',')).explode('unique_genre')
df_exploded['unique_genre'] = df_exploded['unique_genre'].str.strip()  # Remove extra spaces

# Extract Duration (for Movies and TV Shows)
# For Movies, extract duration in minutes
def extract_movie_duration(x):
    if isinstance(x, str) and 'min' in x:
        try:
            return int(x.split(' ')[0])  # Extract the numeric value
        except ValueError:
            return None  # In case the value is not a valid integer
    return None

df['movie_duration'] = df['duration'].apply(extract_movie_duration)

# For TV Shows, extract the number of seasons
def extract_tv_seasons(x):
    if isinstance(x, str) and 'Season' in x:
        try:
            return int(x.split(' ')[0])  # Extract the number of seasons
        except ValueError:
            return None  # In case the value is not a valid integer
    return None

df['tv_seasons'] = df['duration'].apply(extract_tv_seasons)

# Streamlit UI
st.title('Netflix-like Movie Recommendation System')

# Filter by Type (Movie or TV Show)
type_options = df['type'].dropna().unique()
selected_type = st.selectbox('Select Type:', ['All'] + list(type_options))

# Filter genres dynamically based on Type
if selected_type == 'Movie':
    genres = sorted(df_exploded[df_exploded['type'] == 'Movie']['unique_genre'].unique())
elif selected_type == 'TV Show':
    genres = sorted(df_exploded[df_exploded['type'] == 'TV Show']['unique_genre'].unique())
else:
    genres = sorted(df_exploded['unique_genre'].unique())  # All unique genres if "All" is selected

# Filter by Genre (Multiple Selection)
selected_genres = st.multiselect('Select Genres:', options=['All'] + genres)

# Checkbox to enable "Contains all selected genres"
contains_all_genres = st.checkbox('Contains all selected genres')

# Filter by Maturity Rating
maturity_ratings = df['rating'].dropna().unique()
selected_maturity_rating = st.selectbox('Select Maturity Rating:', ['All'] + list(maturity_ratings))

# Filter by Year Range
min_year = int(df['release_year'].min())
max_year = int(df['release_year'].max())
selected_year_range = st.slider(
    'Select Year Range:',
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year)
)

# Filter by Duration (for Movies or TV Shows)
duration_options = ['All', 'Short (90 min or less)', 'Medium (91-120 min)', 'Long (120 min+)']

if selected_type == 'Movie':
    selected_duration = st.selectbox('Select Movie Duration:', options=duration_options)
elif selected_type == 'TV Show':
    selected_duration = st.selectbox('Select Number of Seasons for TV Shows:', options=duration_options)

# Apply filters
filtered_df = df.copy()

# Type filter
if selected_type != 'All':
    filtered_df = filtered_df[filtered_df['type'] == selected_type]

# Multi-Genre filter
if 'All' not in selected_genres and selected_genres:
    if contains_all_genres:
        # If "Contains all selected genres" is checked, apply the all-genre condition
        genre_filter = filtered_df['unique_genre'].apply(
            lambda genres: all(genre.strip() in genres for genre in selected_genres)
        )
    else:
        # If "Contains all selected genres" is not checked, apply the original filter
        genre_filter = filtered_df['unique_genre'].apply(
            lambda genres: any(genre.strip() in genres for genre in selected_genres)
        )
    filtered_df = filtered_df[genre_filter]

# Maturity Rating filter
if selected_maturity_rating != 'All':
    filtered_df = filtered_df[filtered_df['rating'] == selected_maturity_rating]

# Year Range filter
filtered_df = filtered_df[(filtered_df['release_year'] >= selected_year_range[0]) & 
                          (filtered_df['release_year'] <= selected_year_range[1])]

# Duration filter (for Movies or TV Shows)
if selected_type == 'Movie' and selected_duration != 'All':
    if selected_duration == 'Short (90 min or less)':
        filtered_df = filtered_df[filtered_df['movie_duration'] <= 90]
    elif selected_duration == 'Medium (91-120 min)':
        filtered_df = filtered_df[(filtered_df['movie_duration'] > 90) & (filtered_df['movie_duration'] <= 120)]
    elif selected_duration == 'Long (120 min+)':
        filtered_df = filtered_df[filtered_df['movie_duration'] > 120]
elif selected_type == 'TV Show' and selected_duration != 'All':
    if selected_duration == 'Short (90 min or less)':
        filtered_df = filtered_df[filtered_df['tv_seasons'] <= 3]
    elif selected_duration == 'Medium (91-120 min)':
        filtered_df = filtered_df[(filtered_df['tv_seasons'] > 3) & (filtered_df['tv_seasons'] <= 5)]
    elif selected_duration == 'Long (120 min+)':
        filtered_df = filtered_df[filtered_df['tv_seasons'] > 5]

# Show filtered movies
st.subheader('Available Titles Based on Your Filters:')
if filtered_df.empty:
    st.write("No titles match your filters.")
else:
    st.dataframe(filtered_df[['title', 'type', 'unique_genre', 'rating', 'release_year', 'duration']])

# Optional: Display the original data if user checks the box
if st.checkbox('Show all titles'):
    st.subheader('All Titles:')
    st.dataframe(df[['title', 'type', 'unique_genre', 'rating', 'release_year', 'duration']])
