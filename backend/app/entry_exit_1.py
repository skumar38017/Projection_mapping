import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from facenet_pytorch import InceptionResnetV1, extract_face
from ultralytics import YOLO
import streamlit as st
from app.config import settings

# Paths and folders
EMBEDDING_FOLDER = 'embeddings'
VIDEO_FOLDER = 'uploaded_videos'
CSV_FILE = 'matches.csv'
OUTPUT_VIDEO_FILE = 'output_video.mp4'

# Function to create directories if they don't exist
def create_directories_if_not_exist():
    if not os.path.exists(EMBEDDING_FOLDER):
        os.makedirs(EMBEDDING_FOLDER)
    if not os.path.exists(VIDEO_FOLDER):
        os.makedirs(VIDEO_FOLDER)

# Create necessary directories
create_directories_if_not_exist()

# Initialize model and face embedder
model = YOLO('yolov8n.pt')
face_embedder = InceptionResnetV1(pretrained='vggface2').eval()

# Function to calculate the cosine similarity between two embeddings
def cosine_similarity(emb1, emb2):
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

# Load image embeddings from the first phase
image_embeddings = {}
for file in os.listdir(EMBEDDING_FOLDER):
    if file.endswith(".npy"):
        name = os.path.splitext(file)[0].split('_')[0]
        image_embeddings[name] = np.load(os.path.join(EMBEDDING_FOLDER, file))

# Function to delete existing CSV file
def delete_csv_file():
    if os.path.exists(CSV_FILE):
        os.remove(CSV_FILE)

def process_video(video_path):
    delete_csv_file()

    video_capture = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'vp80')
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_video_file = os.path.join(VIDEO_FOLDER, 'output_video.webm')
    out = cv2.VideoWriter(output_video_file, fourcc, 20.0, (frame_width, frame_height))

    matches = {}
    assigned_names = set()  # Set to keep track of assigned names
    face_trackers = {}  # Dictionary to store face trackers

    tracker_results = model.track(video_path, tracker='bytetrack.yaml', show=False)

    for frame_idx, result in enumerate(tracker_results):
        frame_rgb = result.orig_img
        frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)
        boxes = result.boxes.xyxy.cpu().numpy()
        track_ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else None

        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            box = [int(x1), int(y1), int(x2), int(y2)]
            track_id = int(track_ids[i]) if track_ids is not None else None

            face = extract_face(frame_rgb, box)

            if face is not None and face.size(1) > 0 and face.size(2) > 0:
                face = face.unsqueeze(0)
                face_embedding = face_embedder(face).detach().numpy().flatten()
                video_embedding = face_embedding

                if track_id in face_trackers:
                    best_match = face_trackers[track_id]['name']
                else:
                    best_match = None
                    best_similarity = 0

                    for name, image_embedding in image_embeddings.items():
                        if name not in assigned_names:  # Only consider unassigned names
                            similarity = cosine_similarity(image_embedding, video_embedding)
                            if similarity > best_similarity:
                                best_similarity = similarity
                                best_match = name

                    if best_similarity > 0.8:  # Adjust this threshold as needed
                        assigned_names.add(best_match)  # Add to assigned names
                        face_trackers[track_id] = {'name': best_match, 'embedding': video_embedding}
                    else:
                        best_match = f"Unknown_{track_id}"
                        face_trackers[track_id] = {'name': best_match, 'embedding': video_embedding}

                if best_match.startswith("Unknown_"):
                    label_color = (255, 0, 0)  # Red for unknown faces
                else:
                    label_color = (0, 255, 0)  # Green for known faces
                    if best_match not in matches:
                        matches[best_match] = {'entry_time': current_time, 'exit_time': current_time}
                    else:
                        matches[best_match]['exit_time'] = current_time

                # Draw rectangle around the face
                cv2.rectangle(frame_rgb, (box[0], box[1]), (box[2], box[3]), label_color, 2)
                
                # Add name above the bounding box for matched faces
                cv2.putText(frame_rgb, best_match, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, label_color, 2)

        # Write frame to output video
        out.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

    video_capture.release()
    out.release()

    # Save matches to CSV (only known faces)
    df = pd.DataFrame([(name, times['entry_time'], times['exit_time']) for name, times in matches.items()],
                      columns=['Name', 'Entry Time', 'Exit Time'])
    df.index = df.index + 1
    df.to_csv(CSV_FILE, index=False)

    return output_video_file

# Streamlit App
st.title('Video Upload for Face Recognition')

uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    video_path = os.path.join(VIDEO_FOLDER, uploaded_video.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_video.getbuffer())
    st.success(f"Video file saved at {video_path}")
    st.subheader('Uploaded Video')
    st.video(video_path)

    output_video_file = process_video(video_path)
    st.success(f"Video processed. Output video saved to {output_video_file}")

    st.video(output_video_file)

    if os.path.exists(CSV_FILE):
        st.write(pd.read_csv(CSV_FILE))