"""
🎵 Spotify Music Recommendation System
A content-based music recommendation app with Spotify-inspired design.

Built with:
- Streamlit for the web interface
- scikit-learn for ML (cosine similarity, MinMaxScaler)
- pandas & numpy for data manipulation
- plotly for interactive visualizations

Dataset: Spotify Tracks Dataset
https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from recommendation_engine import create_recommender


st.set_page_config(
    page_title="Spotify Music Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700;800&display=swap');
    
    :root {
        --spotify-green: #1DB954;
        --spotify-green-light: #1ed760;
        --spotify-green-dark: #1aa34a;
        --spotify-black: #191414;
        --spotify-dark: #121212;
        --spotify-gray: #282828;
        --spotify-light-gray: #b3b3b3;
        --spotify-white: #ffffff;
    }
    
    .stApp {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f0f23 100%);
        font-family: 'Montserrat', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main title with gradient animation */
    .main-title {
        font-size: 4rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #1DB954 0%, #1ed760 25%, #ffffff 50%, #1ed760 75%, #1DB954 100%);
        background-size: 400% 400%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradientFlow 3s ease infinite;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 30px rgba(29, 185, 84, 0.3);
    }
    
    @keyframes gradientFlow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .subtitle {
        text-align: center;
        color: #b3b3b3;
        font-size: 1.2rem;
        font-weight: 400;
        margin-bottom: 2rem;
        animation: fadeInUp 0.8s ease-out;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Glowing music wave animation */
    .music-waves {
        display: flex;
        justify-content: center;
        align-items: flex-end;
        height: 50px;
        gap: 4px;
        margin-bottom: 2rem;
    }
    
    .wave-bar {
        width: 6px;
        background: linear-gradient(to top, #1DB954, #1ed760);
        border-radius: 3px;
        animation: wave 1s ease-in-out infinite;
        box-shadow: 0 0 10px rgba(29, 185, 84, 0.5);
    }
    
    .wave-bar:nth-child(1) { animation-delay: 0s; height: 20px; }
    .wave-bar:nth-child(2) { animation-delay: 0.1s; height: 35px; }
    .wave-bar:nth-child(3) { animation-delay: 0.2s; height: 25px; }
    .wave-bar:nth-child(4) { animation-delay: 0.3s; height: 40px; }
    .wave-bar:nth-child(5) { animation-delay: 0.4s; height: 30px; }
    .wave-bar:nth-child(6) { animation-delay: 0.5s; height: 45px; }
    .wave-bar:nth-child(7) { animation-delay: 0.6s; height: 20px; }
    .wave-bar:nth-child(8) { animation-delay: 0.7s; height: 35px; }
    .wave-bar:nth-child(9) { animation-delay: 0.8s; height: 28px; }
    
    @keyframes wave {
        0%, 100% { transform: scaleY(1); }
        50% { transform: scaleY(0.5); }
    }
    
    /* Song cards with hover effects */
    .song-card {
        background: linear-gradient(145deg, #282828 0%, #1a1a1a 100%);
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.8rem 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid rgba(255,255,255,0.05);
        position: relative;
        overflow: hidden;
    }
    
    .song-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(29, 185, 84, 0.1), transparent);
        transition: left 0.5s ease;
    }
    
    .song-card:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 20px 40px rgba(0,0,0,0.4), 0 0 30px rgba(29, 185, 84, 0.2);
        border-color: #1DB954;
    }
    
    .song-card:hover::before {
        left: 100%;
    }
    
    .song-title {
        color: #ffffff;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.3rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .song-artist {
        color: #e0e0e0;
        font-size: 0.9rem;
        font-weight: 400;
    }
    
    .song-album {
        color: #b3b3b3;
        font-size: 0.8rem;
        font-style: italic;
        margin-top: 0.2rem;
    }
    
    .song-genre {
        display: inline-block;
        background: linear-gradient(135deg, #1DB954, #1aa34a);
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 0.5rem;
    }
    
    .similarity-badge {
        background: linear-gradient(135deg, #1DB954, #1ed760);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        display: inline-block;
        margin-top: 0.5rem;
        box-shadow: 0 4px 15px rgba(29, 185, 84, 0.4);
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { box-shadow: 0 4px 15px rgba(29, 185, 84, 0.4); }
        50% { box-shadow: 0 4px 25px rgba(29, 185, 84, 0.7); }
    }
    
    /* Section headers */
    .section-header {
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 2rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .section-header::after {
        content: '';
        flex: 1;
        height: 2px;
        background: linear-gradient(90deg, #1DB954, transparent);
        margin-left: 15px;
    }
    
    /* Feature radar chart styling */
    .feature-container {
        background: linear-gradient(145deg, #282828 0%, #1a1a1a 100%);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(255,255,255,0.05);
        margin: 1rem 0;
    }
    
    /* Mood buttons with glow */
    .mood-btn {
        background: linear-gradient(145deg, #282828, #1a1a1a);
        border: 2px solid #1DB954;
        color: #1DB954;
        padding: 0.8rem 1.5rem;
        border-radius: 25px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        display: inline-block;
        margin: 0.3rem;
    }
    
    .mood-btn:hover {
        background: #1DB954;
        color: #000000;
        box-shadow: 0 0 30px rgba(29, 185, 84, 0.5);
        transform: scale(1.05);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0d0d 0%, #1a1a1a 100%);
        border-right: 1px solid #282828;
    }
    
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label {
        color: #ffffff !important;
        font-weight: 500;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #121212;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #1DB954;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #1ed760;
    }
    
    /* Stats cards */
    .stat-card {
        background: linear-gradient(145deg, #1DB954 0%, #1aa34a 100%);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 10px 30px rgba(29, 185, 84, 0.3);
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(29, 185, 84, 0.5);
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 800;
        color: #ffffff;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }
    
    .stat-label {
        color: rgba(255,255,255,0.9);
        font-size: 0.9rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Animated background circles */
    .bg-circles {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        overflow: hidden;
        z-index: -1;
    }
    
    .circle {
        position: absolute;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(29, 185, 84, 0.1) 0%, transparent 70%);
        animation: float 20s ease-in-out infinite;
    }
    
    .circle:nth-child(1) {
        width: 300px;
        height: 300px;
        top: 10%;
        left: 10%;
        animation-delay: 0s;
    }
    
    .circle:nth-child(2) {
        width: 500px;
        height: 500px;
        top: 50%;
        right: -100px;
        animation-delay: -5s;
    }
    
    .circle:nth-child(3) {
        width: 200px;
        height: 200px;
        bottom: 10%;
        left: 30%;
        animation-delay: -10s;
    }
    
    @keyframes float {
        0%, 100% { transform: translate(0, 0) scale(1); }
        25% { transform: translate(30px, -30px) scale(1.1); }
        50% { transform: translate(-20px, 20px) scale(0.9); }
        75% { transform: translate(20px, 30px) scale(1.05); }
    }
    
    /* Search box styling */
    .stTextInput > div > div > input {
        background: #282828 !important;
        border: 1px solid #404040 !important;
        border-radius: 8px !important;
        color: #ffffff !important;
        padding: 0.6rem 1rem !important;
        font-size: 0.85rem !important;
        transition: all 0.3s ease !important;
        outline: none !important;
    }
    
    .stTextInput > div > div {
        background: transparent !important;
        border-radius: 8px !important;
        border: none !important;
    }
    
    .stTextInput > div {
        background: transparent !important;
        border: none !important;
    }
    
    .stTextInput [data-baseweb="input"] {
        background: #282828 !important;
        border: 1px solid #404040 !important;
        border-radius: 8px !important;
    }
    
    .stTextInput [data-baseweb="base-input"] {
        background: transparent !important;
        border-color: #404040 !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #1DB954 !important;
        box-shadow: 0 0 10px rgba(29, 185, 84, 0.2) !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #b3b3b3 !important;
        font-size: 0.85rem !important;
    }
    
    /* Text input label styling */
    .stTextInput > label {
        color: #b3b3b3 !important;
        font-size: 0.85rem !important;
    }
    
    /* Select box styling */
    .stSelectbox > div > div {
        background: #1e1e1e !important;
        border: 2px solid #404040 !important;
        border-radius: 12px !important;
        color: #ffffff !important;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #1DB954 !important;
    }
    
    .stSelectbox [data-baseweb="select"] span {
        color: #ffffff !important;
    }
    
    /* Slider styling */
    .stSlider > div > div > div > div {
        background: #1DB954 !important;
    }
    
    .stSlider [data-baseweb="slider"] div {
        color: #ffffff !important;
    }
    
    /* Slider thumb (the circle) - green instead of red */
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background-color: #1DB954 !important;
        border-color: #1DB954 !important;
    }
    
    .stSlider [data-baseweb="slider"] [data-testid="stThumbValue"] {
        color: #1DB954 !important;
    }
    
    /* Slider track fill */
    .stSlider [data-baseweb="slider"] > div > div:first-child {
        background-color: #1DB954 !important;
    }
    
    /* Checkbox styling - green instead of red */
    .stCheckbox [data-baseweb="checkbox"] {
        background-color: transparent !important;
        border-color: #1DB954 !important;
    }
    
    .stCheckbox [data-baseweb="checkbox"][aria-checked="true"] {
        background-color: #1DB954 !important;
        border-color: #1DB954 !important;
    }
    
    .stCheckbox [data-baseweb="checkbox"] svg {
        fill: #000000 !important;
    }
    
    /* Checkbox label - same gray as Settings/Recommendation Mode headers */
    .stCheckbox label span,
    .stCheckbox label p {
        color: #b3b3b3 !important;
    }
    
    section[data-testid="stSidebar"] .stCheckbox label,
    section[data-testid="stSidebar"] .stCheckbox label span,
    section[data-testid="stSidebar"] .stCheckbox label p,
    section[data-testid="stSidebar"] .stCheckbox > label > div:last-child {
        color: #b3b3b3 !important;
    }
    
    /* Tooltip/help icon styling - match label color */
    section[data-testid="stSidebar"] [data-testid="stTooltipIcon"],
    section[data-testid="stSidebar"] .stTooltipIcon,
    section[data-testid="stSidebar"] svg[data-testid="stTooltipIcon"] {
        color: #ffffff !important;
        fill: #ffffff !important;
    }
    
    /* Question mark icons next to labels */
    section[data-testid="stSidebar"] [data-testid="tooltipHoverTarget"] svg,
    section[data-testid="stSidebar"] .stElementContainer svg[width="14"] {
        color: #ffffff !important;
        fill: #ffffff !important;
    }
    
    /* Make slider labels white */
    .stSlider label {
        color: #ffffff !important;
    }
    
    /* Fix all gray text to be more readable */
    .stMarkdown p, .stMarkdown span {
        color: #e0e0e0 !important;
    }
    
    /* Slider value display - remove green highlight completely */
    [data-testid="stTickBarMin"], [data-testid="stTickBarMax"] {
        color: #b3b3b3 !important;
        background: transparent !important;
        background-color: transparent !important;
    }
    
    .stSlider [data-testid="stTickBarMin"],
    .stSlider [data-testid="stTickBarMax"] {
        background-color: transparent !important;
        background: none !important;
        color: #b3b3b3 !important;
    }
    
    .stSlider [data-testid="stTickBarMin"] div,
    .stSlider [data-testid="stTickBarMax"] div {
        background-color: transparent !important;
        background: none !important;
        color: #b3b3b3 !important;
    }
    
    /* Override any Streamlit slider highlighting */
    .stSlider div[data-testid="stTickBarMin"],
    .stSlider div[data-testid="stTickBarMax"],
    .stSlider [data-testid="stTickBarMin"] *,
    .stSlider [data-testid="stTickBarMax"] * {
        background: transparent !important;
        background-color: transparent !important;
    }
    
    /* Hide slider min/max values (5 and 20) completely */
    section[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMin"],
    section[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMax"] {
        display: none !important;
        visibility: hidden !important;
    }
    
    /* Remove any green background from slider value display */
    section[data-testid="stSidebar"] .stSlider [data-testid="stThumbValue"] {
        background: transparent !important;
        color: #b3b3b3 !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #1DB954 0%, #1aa34a 100%) !important;
        color: #000000 !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    
    .stButton > button:hover {
        transform: scale(1.05) !important;
        box-shadow: 0 10px 30px rgba(29, 185, 84, 0.4) !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #1a1a1a;
        padding: 0.5rem;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #b3b3b3;
        padding: 0.8rem 1.5rem;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: #1DB954 !important;
        color: #000000 !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: #282828 !important;
        border-radius: 12px !important;
        color: #ffffff !important;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        color: #1DB954 !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #b3b3b3 !important;
    }
    
    /* Loading animation */
    .loading-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
    }
    
    .loading-spinner {
        width: 50px;
        height: 50px;
        border: 4px solid #282828;
        border-top: 4px solid #1DB954;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Playing indicator */
    .now-playing {
        display: inline-flex;
        align-items: flex-end;
        gap: 2px;
        height: 16px;
    }
    
    .now-playing span {
        width: 3px;
        background: #1DB954;
        border-radius: 2px;
        animation: playing 0.8s ease-in-out infinite;
    }
    
    .now-playing span:nth-child(1) { animation-delay: 0s; }
    .now-playing span:nth-child(2) { animation-delay: 0.2s; }
    .now-playing span:nth-child(3) { animation-delay: 0.4s; }
    
    @keyframes playing {
        0%, 100% { height: 4px; }
        50% { height: 16px; }
    }
    
    /* Feature bar styling */
    .feature-bar {
        background: #404040;
        border-radius: 10px;
        height: 8px;
        overflow: hidden;
        margin: 0.3rem 0;
    }
    
    .feature-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #1DB954, #1ed760);
        border-radius: 10px;
        transition: width 0.5s ease;
    }
    
    .feature-label {
        color: #ffffff;
        font-size: 0.8rem;
        display: flex;
        justify-content: space-between;
    }
    
    /* Playlist card */
    .playlist-header {
        background: linear-gradient(135deg, #1DB954 0%, #1aa34a 50%, #0d7a32 100%);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 1.5rem;
    }
    
    .playlist-icon {
        width: 120px;
        height: 120px;
        background: rgba(0,0,0,0.3);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 3rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    .playlist-info h2 {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
    }
    
    .playlist-info p {
        color: rgba(255,255,255,0.8);
        margin: 0.5rem 0 0 0;
    }
    
    /* Tooltip */
    .tooltip {
        position: relative;
        display: inline-block;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        background: #282828;
        color: #fff;
        text-align: center;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        transform: translateX(-50%);
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 0.8rem;
        white-space: nowrap;
        border: 1px solid #1DB954;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2.5rem;
        }
        
        .song-card {
            padding: 1rem;
        }
        
        .stat-number {
            font-size: 1.8rem;
        }
    }
    </style>
    
    <div class="bg-circles">
        <div class="circle"></div>
        <div class="circle"></div>
        <div class="circle"></div>
    </div>
    """, unsafe_allow_html=True)


def render_header():
    st.markdown("""
    <div class="main-title">🎵 Music Recommender</div>
    <div class="subtitle">Discover your next favorite song using AI-powered recommendations</div>
    <div class="music-waves">
        <div class="wave-bar"></div>
        <div class="wave-bar"></div>
        <div class="wave-bar"></div>
        <div class="wave-bar"></div>
        <div class="wave-bar"></div>
        <div class="wave-bar"></div>
        <div class="wave-bar"></div>
        <div class="wave-bar"></div>
        <div class="wave-bar"></div>
    </div>
    """, unsafe_allow_html=True)


def render_song_card(track, show_similarity=False):
    similarity_html = ""
    if show_similarity and 'similarity_score' in track:
        similarity_html = f'<div class="similarity-badge">🎯 {track["similarity_score"]}% Match</div>'
    
    st.markdown(f"""
    <div class="song-card">
        <div class="song-title">
            <div class="now-playing">
                <span></span>
                <span></span>
                <span></span>
            </div>
            {track['track_name']}
        </div>
        <div class="song-artist">🎤 {track['artists']}</div>
        <div class="song-album">💿 {track['album_name']}</div>
        <div class="song-genre">{track['genre']}</div>
        {similarity_html}
    </div>
    """, unsafe_allow_html=True)


def render_feature_bars(features):
    feature_labels = {
        'danceability': ('💃 Danceability', 'How suitable for dancing'),
        'energy': ('⚡ Energy', 'Intensity and activity'),
        'valence': ('😊 Mood (Valence)', 'Musical positiveness'),
        'acousticness': ('🎸 Acousticness', 'Acoustic vs electronic'),
        'instrumentalness': ('🎹 Instrumentalness', 'Vocal presence'),
        'speechiness': ('🎤 Speechiness', 'Spoken words'),
        'liveness': ('🎪 Liveness', 'Audience presence'),
    }
    
    for key, (label, tooltip) in feature_labels.items():
        if key in features:
            value = features[key]
            percentage = value * 100
            st.markdown(f"""
            <div class="feature-label">
                <span>{label}</span>
                <span>{percentage:.0f}%</span>
            </div>
            <div class="feature-bar">
                <div class="feature-bar-fill" style="width: {percentage}%"></div>
            </div>
            """, unsafe_allow_html=True)


def create_radar_chart(features, title="Audio Features"):
    categories = ['Danceability', 'Energy', 'Valence', 'Acousticness', 
                  'Instrumentalness', 'Speechiness', 'Liveness']
    values = [
        features.get('danceability', 0),
        features.get('energy', 0),
        features.get('valence', 0),
        features.get('acousticness', 0),
        features.get('instrumentalness', 0),
        features.get('speechiness', 0),
        features.get('liveness', 0)
    ]
    values.append(values[0])
    categories.append(categories[0])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(29, 185, 84, 0.3)',
        line=dict(color='#1DB954', width=2),
        marker=dict(size=8, color='#1ed760'),
        name=title
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(color='#e0e0e0'),
                gridcolor='#404040'
            ),
            angularaxis=dict(
                tickfont=dict(color='#ffffff', size=11),
                gridcolor='#404040'
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ffffff'),
        showlegend=False,
        margin=dict(l=60, r=60, t=40, b=40),
        height=350
    )
    
    return fig


def create_feature_comparison_chart(tracks_data):
    if not tracks_data:
        return None
    
    df = pd.DataFrame(tracks_data)
    features = ['danceability', 'energy', 'valence']
    
    fig = go.Figure()
    
    colors = ['#1DB954', '#1ed760', '#1aa34a', '#0d7a32', '#085c26']
    
    for i, track in enumerate(tracks_data[:5]):
        values = [track.get(f, 0) for f in features]
        fig.add_trace(go.Bar(
            name=track['track_name'][:20] + ('...' if len(track['track_name']) > 20 else ''),
            x=features,
            y=values,
            marker_color=colors[i % len(colors)],
            text=[f'{v:.2f}' for v in values],
            textposition='auto'
        ))
    
    fig.update_layout(
        barmode='group',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ffffff'),
        xaxis=dict(
            title=dict(text='Audio Features', font=dict(color='#ffffff')),
            gridcolor='#404040',
            tickfont=dict(color='#e0e0e0')
        ),
        yaxis=dict(
            title=dict(text='Value', font=dict(color='#ffffff')),
            gridcolor='#404040',
            tickfont=dict(color='#e0e0e0'),
            range=[0, 1]
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            font=dict(size=11, color='#ffffff')
        ),
        margin=dict(l=50, r=20, t=80, b=50),
        height=380
    )
    
    return fig


@st.cache_resource
def get_recommender():
    """Cache the recommender to avoid reloading on each interaction."""
    return create_recommender()


def main():
    load_css()
    
    recommender = get_recommender()
    
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <div style="font-size: 3rem;">🎵</div>
            <div style="color: #1DB954; font-weight: 700; font-size: 1.2rem;">Music Recommender</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown('<p style="color: #ffffff; font-weight: 600; font-size: 1rem; margin-bottom: 0.5rem;">🎯 Recommendation Mode</p>', unsafe_allow_html=True)
        mode = st.selectbox(
            "Choose how to discover music",
            ["🎵 By Song", "🎭 By Mood", "🎚️ By Features", "🔍 Search"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown('<p style="color: #ffffff; font-weight: 600; font-size: 1rem; margin-bottom: 0.5rem;">⚙️ Settings</p>', unsafe_allow_html=True)
        
        n_recommendations = st.slider(
            "Number of recommendations",
            min_value=5,
            max_value=20,
            value=10,
            help="How many songs to recommend"
        )
        
        exclude_same_artist = st.checkbox(
            "Exclude same artist",
            value=False,
            help="Don't recommend songs from the same artist"
        )
    
    render_header()
    
    col1, col2, col3, col4 = st.columns(4)
    
    from data.loader import get_dataset_stats
    stats = get_dataset_stats()
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{stats['total_tracks']:,}</div>
            <div class="stat-label">Total Songs</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{stats['total_genres']}</div>
            <div class="stat-label">Genres</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{stats['total_artists']:,}</div>
            <div class="stat-label">Artists</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{stats['audio_features']}</div>
            <div class="stat-label">Audio Features</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if mode == "🎵 By Song":
        st.markdown('<div class="section-header">🎵 Search for a Song</div>', unsafe_allow_html=True)
        
        search_query = st.text_input("Search by song or artist name", placeholder="e.g., Blinding Lights, Drake, Adele...")
        
        if search_query and len(search_query) >= 2:
            search_results = recommender.search_tracks(search_query)[:100]
            if search_results:
                track_options = {f"{t['track_name']} - {t['artists']}": t['track_id'] for t in search_results}
            else:
                st.info("No songs found. Try a different search term.")
                track_options = {}
        else:
            popular_tracks = recommender.get_popular_tracks(200)
            track_options = {f"{t['track_name']} - {t['artists']}": t['track_id'] for t in popular_tracks}
            if not search_query:
                st.caption("Showing top 200 popular tracks. Search to find more!")
        
        if track_options:
            selected_track = st.selectbox(
                "Choose a seed song",
                options=list(track_options.keys()),
                label_visibility="collapsed"
            )
        else:
            selected_track = None
        
        if selected_track:
            track_id = track_options[selected_track]
            seed_track = recommender.get_track_by_id(track_id)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown('<div class="section-header">🎧 Your Selection</div>', unsafe_allow_html=True)
                render_song_card(seed_track)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**Audio DNA:**")
                features = recommender.get_track_features(track_id)
                if features:
                    render_feature_bars(features)
            
            with col2:
                st.markdown('<div class="section-header">📊 Audio Fingerprint</div>', unsafe_allow_html=True)
                if features:
                    fig = create_radar_chart(features, seed_track['track_name'])
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
            st.markdown('<div class="section-header">🎯 Recommended For You</div>', unsafe_allow_html=True)
            
            recommendations = recommender.get_recommendations(
                track_id, 
                n_recommendations=n_recommendations,
                exclude_same_artist=exclude_same_artist
            )
            
            if recommendations:
                cols = st.columns(2)
                for i, rec in enumerate(recommendations):
                    with cols[i % 2]:
                        render_song_card(rec, show_similarity=True)
                
                st.markdown('<div class="section-header">📈 Feature Comparison</div>', unsafe_allow_html=True)
                comparison_fig = create_feature_comparison_chart(recommendations[:5])
                if comparison_fig:
                    st.plotly_chart(comparison_fig, use_container_width=True, config={'displayModeBar': False})
    
    elif mode == "🎭 By Mood":
        st.markdown('<div class="section-header">🎭 How Are You Feeling?</div>', unsafe_allow_html=True)
        
        mood_descriptions = {
            "Happy": ("😊", "Upbeat, positive vibes", "High valence, high energy"),
            "Sad": ("😢", "Melancholic, emotional", "Low valence, acoustic"),
            "Energetic": ("⚡", "High-intensity, workout", "High energy, fast tempo"),
            "Chill": ("🌊", "Relaxed, laid-back", "Low energy, acoustic"),
            "Party": ("🎉", "Dance floor ready", "High danceability, upbeat"),
            "Focus": ("🎯", "Concentration mode", "Instrumental, minimal vocals")
        }
        
        cols = st.columns(3)
        selected_mood = None
        
        for i, (mood, (emoji, desc, features)) in enumerate(mood_descriptions.items()):
            with cols[i % 3]:
                if st.button(f"{emoji} {mood}", key=f"mood_{mood}", use_container_width=True):
                    selected_mood = mood.lower()
                st.markdown(f"""
                <div style="text-align: center; color: #b3b3b3; font-size: 0.75rem; margin-bottom: 1rem;">
                    {desc}<br><span style="color: #1DB954;">{features}</span>
                </div>
                """, unsafe_allow_html=True)
        
        if 'selected_mood' not in st.session_state:
            st.session_state.selected_mood = None
        
        if selected_mood:
            st.session_state.selected_mood = selected_mood
        
        if st.session_state.selected_mood:
            st.markdown(f'<div class="section-header">🎵 {st.session_state.selected_mood.title()} Vibes</div>', unsafe_allow_html=True)
            
            recommendations = recommender.get_mood_based_recommendations(
                st.session_state.selected_mood, 
                n_recommendations=n_recommendations
            )
            
            cols = st.columns(2)
            for i, rec in enumerate(recommendations):
                with cols[i % 2]:
                    render_song_card(rec, show_similarity=True)
    
    elif mode == "🎚️ By Features":
        st.markdown('<div class="section-header">🎚️ Customize Your Sound</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            danceability = st.slider("💃 Danceability", 0.0, 1.0, 0.5, 0.01)
            energy = st.slider("⚡ Energy", 0.0, 1.0, 0.5, 0.01)
            valence = st.slider("😊 Mood (Valence)", 0.0, 1.0, 0.5, 0.01)
            tempo = st.slider("🎵 Tempo (BPM)", 60, 200, 120)
        
        with col2:
            acousticness = st.slider("🎸 Acousticness", 0.0, 1.0, 0.3, 0.01)
            instrumentalness = st.slider("🎹 Instrumentalness", 0.0, 1.0, 0.0, 0.01)
            speechiness = st.slider("🎤 Speechiness", 0.0, 1.0, 0.1, 0.01)
            liveness = st.slider("🎪 Liveness", 0.0, 1.0, 0.2, 0.01)
        
        features_dict = {
            'danceability': danceability,
            'energy': energy,
            'valence': valence,
            'tempo': tempo,
            'acousticness': acousticness,
            'instrumentalness': instrumentalness,
            'speechiness': speechiness,
            'liveness': liveness
        }
        
        st.markdown('<div class="section-header">📊 Your Custom Audio Profile</div>', unsafe_allow_html=True)
        fig = create_radar_chart(features_dict, "Custom Profile")
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        if st.button("🎯 Get Recommendations", use_container_width=True):
            recommendations = recommender.get_recommendations_by_features(
                features_dict, 
                n_recommendations=n_recommendations
            )
            
            st.markdown('<div class="section-header">🎵 Songs Matching Your Profile</div>', unsafe_allow_html=True)
            
            cols = st.columns(2)
            for i, rec in enumerate(recommendations):
                with cols[i % 2]:
                    render_song_card(rec, show_similarity=True)
    
    elif mode == "🔍 Search":
        st.markdown('<div class="section-header">🔍 Search for Songs</div>', unsafe_allow_html=True)
        
        search_query = st.text_input(
            "Search",
            placeholder="Search by song name or artist...",
            label_visibility="collapsed"
        )
        
        if search_query:
            results = recommender.search_tracks(search_query)
            
            if results:
                st.markdown(f'<div class="section-header">📋 Found {len(results)} Results</div>', unsafe_allow_html=True)
                
                cols = st.columns(2)
                for i, track in enumerate(results):
                    with cols[i % 2]:
                        render_song_card(track)
            else:
                st.info("No songs found matching your search. Try different keywords!")
        else:
            st.markdown("### 🎵 Browse by Genre")
            
            genres = recommender.get_all_genres()
            genre_cols = st.columns(4)
            for i, genre in enumerate(genres):
                with genre_cols[i % 4]:
                    if st.button(genre.upper(), key=f"genre_{genre}", use_container_width=True):
                        st.session_state.selected_genre = genre
            
            if 'selected_genre' in st.session_state and st.session_state.selected_genre:
                genre_tracks = recommender.get_tracks_by_genre(st.session_state.selected_genre)
                st.markdown(f'<div class="section-header">🎵 {st.session_state.selected_genre.title()} Tracks</div>', unsafe_allow_html=True)
                
                cols = st.columns(2)
                for i, track in enumerate(genre_tracks):
                    with cols[i % 2]:
                        render_song_card(track)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; color: #ffffff; padding: 2rem; border-top: 1px solid #282828;">
        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🎵</div>
        <div style="font-weight: 600; color: #1DB954; font-size: 1.1rem;">Music Recommendation System</div>
        <div style="font-size: 0.85rem; margin-top: 0.5rem; color: #e0e0e0;">
            Built with Python • Streamlit • CSS/HTML<br>
            Dataset: <a href="https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset" 
                       style="color: #1DB954;">Spotify Tracks Dataset</a>
        </div>
        <div style="margin-top: 1rem; font-size: 0.85rem;">
            <span style="color: #1DB954; font-weight: 600;">Sam Bolger</span><br>
            <a href="https://www.linkedin.com/in/sambolger" style="color: #1DB954; text-decoration: underline;">LinkedIn</a> • 
            <a href="mailto:sbolger@cord.edu" style="color: #1DB954; text-decoration: underline;">sbolger@cord.edu</a> • 
            <a href="https://github.com/sammybolger" style="color: #1DB954; text-decoration: underline;">GitHub</a> • 
            <a href="https://www.sammybolger.com" style="color: #1DB954; text-decoration: underline;">sammybolger.com</a>
        </div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
