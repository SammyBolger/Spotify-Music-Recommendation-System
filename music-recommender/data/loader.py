"""
Data loader for the full Spotify tracks dataset.
Handles loading and caching of 114,000+ tracks efficiently.
"""

import pandas as pd
import os

FULL_DATASET_PATH = os.path.join(os.path.dirname(__file__), 'spotify_full.csv')

AUDIO_FEATURES = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
]

_cached_df = None

def load_full_dataset():
    """Load the full Spotify dataset with caching."""
    global _cached_df
    
    if _cached_df is not None:
        return _cached_df
    
    df = pd.read_csv(FULL_DATASET_PATH)
    
    df = df.rename(columns={'track_genre': 'genre'})
    
    df['track_id'] = df['track_id'].astype(str)
    df['artists'] = df['artists'].fillna('Unknown Artist')
    df['album_name'] = df['album_name'].fillna('Unknown Album')
    df['track_name'] = df['track_name'].fillna('Unknown Track')
    df['genre'] = df['genre'].fillna('unknown')
    
    for col in AUDIO_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    df = df.drop_duplicates(subset=['track_id'], keep='first')
    
    df = df.dropna(subset=AUDIO_FEATURES)
    
    df = df.reset_index(drop=True)
    
    _cached_df = df
    return df


def get_audio_features_columns():
    """Return the list of audio feature column names."""
    return AUDIO_FEATURES


def get_dataset_stats():
    """Get statistics about the dataset."""
    df = load_full_dataset()
    return {
        'total_tracks': len(df),
        'total_genres': df['genre'].nunique(),
        'total_artists': df['artists'].nunique(),
        'audio_features': len(AUDIO_FEATURES)
    }
