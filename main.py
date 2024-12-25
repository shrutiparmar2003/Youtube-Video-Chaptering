import re
import csv
import os
import pandas as pd
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF

API_KEY = 'YOUR API KEY'

def get_video_id(url):
    video_id_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
    return video_id_match.group(1) if video_id_match else None

def get_video_title(video_id):
    youtube = build('youtube', 'v3', developerKey=API_KEY)
    request = youtube.videos().list(part='snippet', id=video_id)
    response = request.execute()
    title = response['items'][0]['snippet']['title'] if response['items'] else 'Unknown Title'
    return title

def get_video_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return transcript
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def save_to_csv(title, transcript, filename):
    output_dir = "D:\\VideoChaptering"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_path = os.path.join(output_dir, filename)
    transcript_data = [{'start': entry['start'], 'text': entry['text']} for entry in transcript]
    df = pd.DataFrame(transcript_data)
    df.to_csv(save_path, index=False)

    with open(save_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Title:', title])

    print(f"Transcript saved to: {save_path}")
    return save_path

def display_topics(model, feature_names, n_top_words):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        topic_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        topics.append(" ".join(topic_words))
    return topics

def main():
    url = input("Enter YouTube video link: ")
    video_id = get_video_id(url)

    if not video_id:
        print("Invalid URL")
        return

    title = get_video_title(video_id)
    transcript = get_video_transcript(video_id)

    if not transcript:
        print("No transcript available for this video.")
        return

    filename = f"{video_id}_transcript.csv"
    save_path = save_to_csv(title, transcript, filename)

    transcript_df = pd.read_csv(save_path)
    print(transcript_df.head())

    # Convert `start` column to numeric type
    transcript_df['start'] = pd.to_numeric(transcript_df['start'], errors='coerce')

    transcript_df['text_length'] = transcript_df['text'].apply(len)
    plt.figure(figsize=(10, 5))
    plt.hist(transcript_df['text_length'], bins=50, color='blue', alpha=0.7)
    plt.title('Distribution of Text Lengths')
    plt.xlabel('Text Length')
    plt.ylabel('Frequency')
    plt.show()

    n_features = 1000
    n_topics = 10
    n_top_words = 10
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
        vectorizer = TfidfVectorizer(stop_words='english', max_features=3)
        tfid_matrix = vectorizer.fit_transform([chapter_text])
        feature_names = vectorizer.get_feature_names_out()
        chapter_name = " ".join(feature_names)

        chapter_names.append(f"Chapter {i + 1}: {chapter_name}")

    # Display the final chapter points with names
    print("\nFinal Chapter Points with Names:")
    for time, name in zip(chapter_points, chapter_names):
        print(f"{time} - {name}")





if __name__ == '__main__':
    main()
