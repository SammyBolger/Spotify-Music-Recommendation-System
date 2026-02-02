"""
Music Recommendation Engine using Content-Based Filtering.
Uses cosine similarity on audio features to find similar songs.

Libraries used:
- scikit-learn: For cosine similarity and feature scaling
- pandas: For data manipulation
- numpy: For numerical operations
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from data.loader import load_full_dataset, get_audio_features_columns


class MusicRecommender:
    """
    Content-based music recommendation system using audio features.
    
    This recommender uses the following audio features:
    - Danceability: How suitable a track is for dancing (0.0 to 1.0)
    - Energy: Intensity and activity measure (0.0 to 1.0)
    - Loudness: Overall loudness in dB
    - Speechiness: Presence of spoken words (0.0 to 1.0)
    - Acousticness: Confidence of acoustic sound (0.0 to 1.0)
    - Instrumentalness: Predicts vocal presence (0.0 to 1.0)
    - Liveness: Presence of audience (0.0 to 1.0)
    - Valence: Musical positiveness (0.0 to 1.0)
    - Tempo: Estimated tempo in BPM
    """
    
    def __init__(self):
        self.df = load_full_dataset()
        self.feature_columns = get_audio_features_columns()
        self.scaler = MinMaxScaler()
        self._prepare_features()
    
    def _prepare_features(self):
        """Prepare and scale audio features for similarity calculation."""
        features = self.df[self.feature_columns].values
        self.scaled_features = self.scaler.fit_transform(features)
    
    def get_all_tracks(self):
        """Return all tracks in the dataset."""
        return self.df[['track_id', 'track_name', 'artists', 'album_name', 
                        'popularity', 'genre']].to_dict('records')
    
    def get_popular_tracks(self, n_tracks=200):
        """Return the most popular tracks for initial display."""
        popular = self.df.nlargest(n_tracks, 'popularity')
        return popular[['track_id', 'track_name', 'artists', 'album_name', 
                        'popularity', 'genre']].to_dict('records')
    
    def get_track_by_id(self, track_id):
        """Get a single track by its ID."""
        track = self.df[self.df['track_id'] == str(track_id)]
        if len(track) > 0:
            return track.iloc[0].to_dict()
        return None
    
    def get_track_features(self, track_id):
        """Get audio features for a specific track."""
        track = self.df[self.df['track_id'] == str(track_id)]
        if len(track) > 0:
            features = track[self.feature_columns].iloc[0].to_dict()
            return features
        return None
    
    def get_recommendations(self, track_id, n_recommendations=10, exclude_same_artist=False):
        """
        Get song recommendations based on a seed track.
        Computes similarity on-demand for memory efficiency with large datasets.
        
        Args:
            track_id: The ID of the seed track
            n_recommendations: Number of recommendations to return
            exclude_same_artist: Whether to exclude songs by the same artist
            
        Returns:
            List of recommended tracks with similarity scores
        """
        try:
            idx = self.df[self.df['track_id'] == str(track_id)].index[0]
        except IndexError:
            return []
        
        seed_features = self.scaled_features[idx].reshape(1, -1)
        similarities = cosine_similarity(seed_features, self.scaled_features)[0]
        
        if exclude_same_artist:
            seed_artist = self.df.iloc[idx]['artists']
            mask = self.df['artists'] != seed_artist
            mask[idx] = False
            valid_indices = np.where(mask)[0]
            valid_similarities = similarities[valid_indices]
            top_indices_local = np.argsort(valid_similarities)[::-1][:n_recommendations]
            top_indices = valid_indices[top_indices_local]
            top_scores = valid_similarities[top_indices_local]
        else:
            similarities[idx] = -1
            top_indices = np.argsort(similarities)[::-1][:n_recommendations]
            top_scores = similarities[top_indices]
        
        recommendations = []
        for i, track_idx in enumerate(top_indices):
            track = self.df.iloc[track_idx].to_dict()
            track['similarity_score'] = round(top_scores[i] * 100, 1)
            recommendations.append(track)
        
        return recommendations
    
    def get_recommendations_by_features(self, features_dict, n_recommendations=10):
        """
        Get recommendations based on custom audio feature preferences.
        
        Args:
            features_dict: Dictionary of audio features (danceability, energy, etc.)
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of recommended tracks
        """
        feature_vector = np.array([[
            features_dict.get('danceability', 0.5),
            features_dict.get('energy', 0.5),
            features_dict.get('loudness', -10),
            features_dict.get('speechiness', 0.1),
            features_dict.get('acousticness', 0.5),
            features_dict.get('instrumentalness', 0.0),
            features_dict.get('liveness', 0.2),
            features_dict.get('valence', 0.5),
            features_dict.get('tempo', 120)
        ]])
        
        scaled_vector = self.scaler.transform(feature_vector)
        similarities = cosine_similarity(scaled_vector, self.scaled_features)[0]
        
        top_indices = np.argsort(similarities)[::-1][:n_recommendations]
        
        recommendations = []
        for idx in top_indices:
            track = self.df.iloc[idx].to_dict()
            track['similarity_score'] = round(similarities[idx] * 100, 1)
            recommendations.append(track)
        
        return recommendations
    
    def get_tracks_by_genre(self, genre, n_tracks=20):
        """Get tracks filtered by genre."""
        genre_tracks = self.df[self.df['genre'] == genre.lower()]
        return genre_tracks.head(n_tracks).to_dict('records')
    
    def get_genre_stats(self):
        """Get statistics for each genre."""
        stats = self.df.groupby('genre')[self.feature_columns].mean()
        return stats.to_dict('index')
    
    def get_all_genres(self):
        """Return list of all unique genres."""
        return sorted(self.df['genre'].unique().tolist())
    
    def search_tracks(self, query):
        """Search for tracks by name or artist."""
        query = query.lower()
        mask = (
            self.df['track_name'].str.lower().str.contains(query, na=False) |
            self.df['artists'].str.lower().str.contains(query, na=False)
        )
        return self.df[mask].to_dict('records')
    
    def get_mood_based_recommendations(self, mood, n_recommendations=10):
        """
        Get recommendations based on mood presets.
        
        Moods:
        - happy: High valence, high energy
        - sad: Low valence, low energy
        - energetic: High energy, high tempo
        - chill: Low energy, high acousticness
        - party: High danceability, high energy
        - focus: High instrumentalness, low speechiness
        """
        mood_presets = {
            'happy': {'valence': 0.8, 'energy': 0.7, 'danceability': 0.7},
            'sad': {'valence': 0.2, 'energy': 0.3, 'acousticness': 0.6},
            'energetic': {'energy': 0.9, 'tempo': 140, 'danceability': 0.7},
            'chill': {'energy': 0.3, 'acousticness': 0.7, 'valence': 0.5},
            'party': {'danceability': 0.9, 'energy': 0.8, 'valence': 0.7},
            'focus': {'instrumentalness': 0.7, 'speechiness': 0.05, 'energy': 0.4}
        }
        
        if mood.lower() not in mood_presets:
            return []
        
        preset = mood_presets[mood.lower()]
        return self.get_recommendations_by_features(preset, n_recommendations)


def create_recommender():
    """Factory function to create a MusicRecommender instance."""
    return MusicRecommender()
