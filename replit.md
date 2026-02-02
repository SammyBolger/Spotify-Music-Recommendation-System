# Spotify Music Recommendation System

## Overview
A content-based music recommendation system with a Spotify-inspired design. The app uses machine learning (cosine similarity) to recommend songs based on audio features like danceability, energy, valence, tempo, and more.

**Dataset Source:** [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) - The original dataset contains 114,000+ tracks with audio features. This project uses a curated subset of 100 representative songs across 17 genres for demonstration purposes.

## Tech Stack
- **Frontend:** Streamlit with custom CSS (Spotify-themed animations)
- **ML/Data:** scikit-learn (cosine similarity, MinMaxScaler), pandas, numpy
- **Visualization:** Plotly (radar charts, bar charts)
- **Python:** 3.11

## Project Structure
```
├── app.py                    # Main Streamlit application
├── recommendation_engine.py  # ML recommendation engine
├── data/
│   ├── __init__.py
│   └── sample_data.py        # 100 curated songs with audio features
├── .streamlit/
│   └── config.toml           # Streamlit configuration
└── pyproject.toml            # Python dependencies
```

## Features
1. **Song-based recommendations:** Select a song and get similar tracks
2. **Mood-based recommendations:** Choose a mood (Happy, Sad, Energetic, Chill, Party, Focus)
3. **Feature-based recommendations:** Customize audio features with sliders
4. **Search & Browse:** Search songs or browse by genre
5. **Visualizations:** Radar charts for audio fingerprints, feature comparisons

## ML Approach: Content-Based Filtering

### How It Works
1. **Feature Extraction:** Each song has 9 audio features from Spotify's audio analysis
2. **Normalization:** Features are scaled using MinMaxScaler (0-1 range) to ensure equal weighting
3. **Similarity Computation:** Cosine similarity measures the angle between feature vectors
4. **Ranking:** Songs are ranked by similarity score, with higher scores indicating more similar audio profiles

### Why Cosine Similarity?
- Works well with high-dimensional feature spaces
- Captures directional similarity (pattern of features) rather than magnitude
- Computationally efficient for real-time recommendations
- Industry-standard approach used by major music platforms

### Feature Selection Rationale
The 9 audio features capture different aspects of music:
- **Rhythm:** danceability, tempo
- **Intensity:** energy, loudness
- **Mood:** valence (positiveness)
- **Sound profile:** acousticness, instrumentalness
- **Vocal content:** speechiness, liveness

## Audio Features Used
| Feature | Description | Range |
|---------|-------------|-------|
| Danceability | How suitable for dancing | 0.0 - 1.0 |
| Energy | Intensity and activity measure | 0.0 - 1.0 |
| Valence | Musical positiveness/mood | 0.0 - 1.0 |
| Tempo | Beats per minute | 60 - 200 |
| Acousticness | Acoustic vs electronic sound | 0.0 - 1.0 |
| Instrumentalness | Vocal presence prediction | 0.0 - 1.0 |
| Speechiness | Presence of spoken words | 0.0 - 1.0 |
| Liveness | Audience presence detection | 0.0 - 1.0 |
| Loudness | Overall volume in dB | -60 - 0 |

## Libraries & Skills Demonstrated
- **pandas:** Data manipulation and DataFrame operations
- **scikit-learn:** MinMaxScaler for normalization, cosine_similarity for recommendations
- **numpy:** Numerical operations and array handling
- **plotly:** Interactive radar charts and bar charts
- **Streamlit:** Web application framework with custom CSS styling

## Running the App
```bash
streamlit run app.py --server.port 5000
```

## Extending the Project
To use the full Kaggle dataset:
1. Download from the Kaggle link above
2. Replace the sample data in `data/sample_data.py`
3. The recommendation engine automatically adapts to larger datasets

## Recent Changes
- Initial build: Complete Spotify-themed music recommendation system
- Added 100 curated songs across 17 genres
- Implemented content-based filtering with cosine similarity
- Created animated Spotify-style UI with green/black theme
