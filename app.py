import streamlit as st
import pandas as pd
from main import get_video_id, get_video_title, get_video_transcript, save_to_csv, display_topics

# Custom CSS for styling
st.markdown("""
    <style>
    /* Body background color */
    body {
        background-color: #f8d7da;
    }
    /* Title and main text style */
    .big-font {
        font-size: 30px;
        font-weight: bold;
        color: #000000;
        text-align: center;
        margin-top: 20px;
    }
    .icon {
        display: block;
        margin-left: auto;
        margin-right: auto;
        margin-top: 20px;
    }
    .youtube-icon {
        width: 100px;
        height: 100px;
    }
    .section-header {
        background-color: #f8d7da;
        color: #000000;
        padding: 12px;
        border-radius: 8px;
        text-align: center;
        font-size: 24px;
        margin-top: 20px;
    }
    .subheader {
        font-size: 22px;
        color: #000000;
    }
    /* Card-style background color */
    .card {
        background-color: #f8d7da;
        border: 1px solid #e6e6e6;
        border-radius: 8px;
        padding: 20px;
        margin-top: 15px;
        color: #000000;
    }
    .card p {
        font-size: 18px;
        color: #000000;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit interface
st.title("YouTube Video Chaptering App")
st.markdown('<p class="big-font">Generate logical chapters for YouTube videos based on transcript.</p>', unsafe_allow_html=True)

# Display YouTube icon
st.markdown("""
    <div class="icon">
        <img class="youtube-icon" src="https://i.pinimg.com/originals/79/7b/96/797b96e0880a4fe147f1eaa89d7c7013.gif" alt="YouTube">
    </div>
""", unsafe_allow_html=True)

# Input for YouTube URL
url = st.text_input("Enter YouTube video URL:", placeholder="https://www.youtube.com/watch?v=example")

# When the user clicks the process button
if st.button("Process Video"):
    if url:
        video_id = get_video_id(url)
        if not video_id:
            st.error("Invalid YouTube URL. Please check and try again.")
        else:
            with st.spinner("Fetching video details..."):
                title = get_video_title(video_id)
                transcript = get_video_transcript(video_id)

            if not transcript:
                st.error("No transcript available for this video.")
            else:
                st.success(f"Video Title: {title}")
                filename = f"{video_id}_transcript.csv"
                save_path = save_to_csv(title, transcript, filename)

                # Read the saved transcript
                transcript_df = pd.read_csv(save_path)
                st.markdown('<div class="section-header">Transcript Preview</div>', unsafe_allow_html=True)
                st.dataframe(transcript_df.head())

                # Analyze and display chapters
                st.markdown('<div class="section-header">Generated Chapters</div>', unsafe_allow_html=True)
                transcript_df['start'] = pd.to_numeric(transcript_df['start'], errors='coerce')
                n_features = 1000
                n_topics = 10
                n_top_words = 10

                # Create logical breaks (same as main.py logic)
                from sklearn.feature_extraction.text import CountVectorizer
                from sklearn.decomposition import NMF
                tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
                tf = tf_vectorizer.fit_transform(transcript_df['text'])
                nmf = NMF(n_components=n_topics, random_state=42).fit(tf)
                tf_feature_names = tf_vectorizer.get_feature_names_out()

                topic_distribution = nmf.transform(tf)
                topic_distribution_trimmed = topic_distribution[:len(transcript_df)]
                transcript_df['dominant_topic'] = topic_distribution_trimmed.argmax(axis=1)

                logical_breaks = []
                for i in range(1, len(transcript_df)):
                    if transcript_df['dominant_topic'].iloc[i] != transcript_df['dominant_topic'].iloc[i - 1]:
                        logical_breaks.append(transcript_df['start'].iloc[i])

                threshold = 60  # seconds
                consolidated_breaks = []
                last_break = None

                for break_point in logical_breaks:
                    if last_break is None or break_point - last_break >= threshold:
                        consolidated_breaks.append(break_point)
                        last_break = break_point

                final_chapters = []
                last_chapter = (consolidated_breaks[0], transcript_df['dominant_topic'].iloc[0])
                for break_point in consolidated_breaks[1:]:
                    current_topic = transcript_df.loc[transcript_df['start'] == break_point, 'dominant_topic'].values[0]
                    if current_topic == last_chapter[1]:
                        last_chapter = (last_chapter[0], current_topic)
                    else:
                        final_chapters.append(last_chapter)
                        last_chapter = (break_point, current_topic)

                final_chapters.append(last_chapter)

                chapter_points = []
                chapter_names = []

                for i, (break_point, topic_idx) in enumerate(final_chapters):
                    chapter_time = pd.to_datetime(break_point, unit='s').strftime('%H:%M:%S')
                    chapter_points.append(chapter_time)

                    chapter_text = transcript_df[
                        (transcript_df['start'] >= break_point) & (transcript_df['dominant_topic'] == topic_idx)
                    ]['text'].str.cat(sep=' ')
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    vectorizer = TfidfVectorizer(stop_words='english', max_features=3)
                    tfid_matrix = vectorizer.fit_transform([chapter_text])
                    feature_names = vectorizer.get_feature_names_out()
                    chapter_name = " ".join(feature_names)

                    chapter_names.append(f"Chapter {i + 1}: {chapter_name}")

                # Display chapters in card format
                for time, name in zip(chapter_points, chapter_names):
                    st.markdown(f"""
                        <div class="card">
                            <strong>{time}</strong> - {name}
                            <p>Click to jump to this chapter</p>
                        </div>
                    """, unsafe_allow_html=True)

    else:
        st.error("Please enter a YouTube URL.")
