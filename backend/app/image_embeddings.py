from facenet_pytorch import MTCNN, InceptionResnetV1, extract_face
import os
import cv2
import numpy as np
import pandas as pd
from app.config import settings

# Directory containing images
images_folder = 'images'  

# Initialize face detector and embedder
face_detector = MTCNN(keep_all=True)
face_embedder = InceptionResnetV1(pretrained='vggface2').eval()

# Function to process and extract embeddings
def process_images(folder_path):
    # Create a list to collect data
    embeddings_list = []
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            
            # Read and preprocess image
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            faces = face_detector.detect(image_rgb)
            
            if faces[0] is not None:
                for i, face in enumerate(faces[0]):
                    # Extract face and calculate embedding
                    face_image = extract_face(image_rgb, face)
                    face_embedding = face_embedder(face_image.unsqueeze(0)).squeeze().detach().numpy()
                    
                    # Add the embedding to the list
                    embeddings_list.append({
                        'filename': filename,
                        'embedding': face_embedding
                    })
                    
                    print(f"Processed {filename} with face {i+1}")
    
    # Convert the list to a DataFrame
    embeddings_df = pd.DataFrame(embeddings_list)
    
    return embeddings_df

# Process images and get embeddings
embeddings_df = process_images(images_folder)

# Print the embeddings
for index, row in embeddings_df.iterrows():
    print(f"Filename: {row['filename']}")
    print(f"Embedding: {row['embedding']}")
    print()

# Save embeddings to a CSV file
embeddings_df.to_csv('face_embeddings.csv', index=False)
print("Embeddings saved to face_embeddings.csv")

#-------------------------------------------------------------------------------------------

# import os
# import numpy as np
# import random
# import cv2
# import torch
# from facenet_pytorch import MTCNN, InceptionResnetV1, extract_face
# from PIL import Image

# # Path to the images folder
# images_folder = 'images'

# # Function to process image, detect face, and calculate embeddings
# def process_image(image_path, face_detector, face_embedder, label):
#     # Read the image
#     image = cv2.imread(image_path)
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     # Detect faces using MTCNN
#     boxes, probs = face_detector.detect(image_rgb)

#     if boxes is not None:
#         # Apply transformations and normalization
#         transformed_faces = [face_embedder(extract_face(image_rgb, box).unsqueeze(0)) for box in boxes]

#         # Convert PyTorch tensor to NumPy array
#         transformed_faces_np = [transformed_face.squeeze().detach().numpy() for transformed_face in transformed_faces]

#         # Save embeddings to .npy files
#         for i, embeddings in enumerate(transformed_faces_np):
#             # Create a filename for the embeddings
#             npy_filename = f"{label}_{os.path.basename(image_path).split('.')[0]}_face{i+1}.npy"
#             np.save(os.path.join(images_folder, npy_filename), embeddings)
#             print(f"Saved embeddings to {npy_filename}")

# # Initialize MTCNN for face detection with adjusted parameters
# face_detector = MTCNN(margin=20, post_process=False, select_largest=False)

# # Initialize InceptionResnetV1 for face embedding
# face_embedder = InceptionResnetV1(pretrained='vggface2').eval()

# # Get a list of all images in the folder
# all_images = [img for img in os.listdir(images_folder) if img.endswith(('.jpg', '.png'))]

# # Process each image
# for image_file in all_images:
#     image_path = os.path.join(images_folder, image_file)
#     label = os.path.basename(image_path).split('.')[0]  
#     process_image(image_path, face_detector, face_embedder, label)
