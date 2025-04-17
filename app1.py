import os
import io
import glob
import base64
from datetime import datetime
from zipfile import ZipFile
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageDraw
import cv2

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    from mtcnn import MTCNN
except ImportError:
    raise ImportError("MTCNN package is not installed. Please install it using: pip install mtcnn")

from transformers import AutoProcessor, TFSwinModel

import streamlit as st

from dotenv import load_dotenv
load_dotenv()

if 'error_count' not in st.session_state:
    st.session_state.error_count = 0
if 'upload_state' not in st.session_state:
    st.session_state.upload_state = {
        'progress': 0,
        'current_file': '',
        'processed_files': 0,
        'total_files': 0
    }

try:
    mtcnn_detector = MTCNN(min_face_size=20)
    st.session_state['mtcnn_loaded'] = True
except Exception as e:
    st.error(f"MTCNN initialization failed: {str(e)}")
    st.session_state['mtcnn_loaded'] = False
    st.stop()

def cleanup_temp_files():
    """Clean up temporary files."""
    try:
        if os.path.exists("extracted_images"):
            for root, dirs, files in os.walk("extracted_images", topdown=False):
                for name in files:
                    try:
                        os.remove(os.path.join(root, name))
                    except Exception as e:
                        st.warning(f"Failed to remove file {name}: {str(e)}")
                for name in dirs:
                    try:
                        os.rmdir(os.path.join(root, name))
                    except Exception as e:
                        st.warning(f"Failed to remove directory {name}: {str(e)}")
            try:
                os.rmdir("extracted_images")
            except Exception as e:
                st.warning(f"Failed to remove extracted_images directory: {str(e)}")
    except Exception as e:
        st.warning(f"Error during cleanup: {str(e)}")

def correct_image_orientation(image):
    """Correct image orientation using EXIF data."""
    try:
        return ImageOps.exif_transpose(image)
    except Exception:
        return image

def safe_image_open(file_or_path):
    """Safely open and validate an image file."""
    try:
        # Normalize the path
        if isinstance(file_or_path, (str, bytes)):
            file_or_path = os.path.normpath(file_or_path)
            if not os.path.exists(file_or_path):
                st.error(f"Image file not found: {file_or_path}")
                return None, "Image file not found."
        
        image = Image.open(file_or_path)
        image.verify()
        image = Image.open(file_or_path)
        if image.mode not in ('RGB', 'RGBA'):
            image = image.convert('RGB')
        
        width, height = image.size
        if min(width, height) < 224:
            ratio = 224 / min(width, height)
            new_size = (int(width * ratio), int(height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        return image, None
    except Exception as e:
        st.error(f"Error opening image: {str(e)}")
        return None, str(e)

def enhance_image(image):
    """Enhance image quality for better face detection."""
    try:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.5)
        return image
    except Exception as e:
        st.warning(f"Image enhancement failed: {str(e)}")
        return image

def detect_multiple_faces(image):
    """Detect faces using MTCNN."""
    try:
        if image is None:
            return None, "Invalid image input."
        
        enhanced_image = enhance_image(image)
        image_np = np.array(enhanced_image)
        original_h, original_w = image_np.shape[:2]
        
        mtcnn_faces = mtcnn_detector.detect_faces(image_np)
        detected_faces = []
        face_images = []
        
        for face in mtcnn_faces:
            if face['confidence'] > 0.9:
                x, y, w, h = face['box']
                margin_x, margin_y = int(w * 0.2), int(h * 0.2)
                x1 = max(0, x - margin_x)
                y1 = max(0, y - margin_y)
                x2 = min(original_w, x + w + margin_x)
                y2 = min(original_h, y + h + margin_y)
                
                face_img = image.crop((x1, y1, x2, y2))
                face_img = face_img.resize((224, 224), Image.Resampling.LANCZOS)
                face_images.append(face_img)
                detected_faces.append((x, y, w, h))
        
        if not face_images:
            return None, "No valid faces detected."
        
        image_with_faces = image.copy()
        draw = ImageDraw.Draw(image_with_faces)
        for i, (x, y, w, h) in enumerate(detected_faces):
            draw.rectangle(((x, y), (x+w, y+h)), outline="green", width=2)
            draw.text((x, y-20), f"Face {i+1}", fill="green")
        
        return image_with_faces, face_images
    except Exception as e:
        st.error(f"Error in face detection: {str(e)}")
        return None, str(e)

@st.cache_resource
def load_model():
    """Load Swin-Base model and processor."""
    try:
        HF_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
        if not HF_TOKEN:
            raise ValueError("HUGGING_FACE_TOKEN not set")
        
        processor = AutoProcessor.from_pretrained(
            "microsoft/swin-base-patch4-window7-224",
            token=HF_TOKEN
        )
        model = TFSwinModel.from_pretrained(
            "microsoft/swin-base-patch4-window7-224",
            token=HF_TOKEN
        )
        return processor, model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        raise

def get_embedding_batch(images, processor, model):
    """Compute embeddings for a batch of face images using Swin-Base."""
    try:
        if not images:
            return None
        
        processed_images = [img.convert('RGB').resize((224, 224)) for img in images if img]
        if not processed_images:
            return None
        
        inputs = processor(images=processed_images, return_tensors="tf", padding=True)
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        return embeddings
    except Exception as e:
        st.error(f"Error computing embeddings: {str(e)}")
        return None

def cosine_similarity(emb1, emb2):
    """Compute cosine similarity between two embeddings."""
    try:
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(emb1, emb2) / (norm1 * norm2)
    except Exception as e:
        st.error(f"Error computing similarity: {str(e)}")
        return 0.0

def create_download_link(img, filename, text):
    """Create a download link for an image."""
    try:
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f'<a href="data:image/png;base64,{b64}" download="{filename}">{text}</a>'
    except Exception as e:
        st.error(f"Error creating download link: {str(e)}")
        return None

def create_zip_download(similar_faces):
    """Create a ZIP file of matched images."""
    try:
        zip_buffer = io.BytesIO()
        with ZipFile(zip_buffer, 'w', ZipFile.ZIP_DEFLATED) as zip_file:
            for idx, match in enumerate(similar_faces):
                img_buffer = io.BytesIO()
                match['image_with_faces'].save(img_buffer, format='PNG')
                filename = f"match_{idx+1}_sim_{match['similarity']:.2f}_{os.path.basename(match['image_path'])}"
                zip_file.writestr(filename, img_buffer.getvalue())
        
        zip_buffer.seek(0)
        b64 = base64.b64encode(zip_buffer.getvalue()).decode()
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f'<a href="data:application/zip;base64,{b64}" download="matched_faces_{date_str}.zip">Download All Matches (ZIP)</a>'
    except Exception as e:
        st.error(f"Error creating ZIP download: {str(e)}")
        return None

def display_results_with_download(similar_faces):
    """Display results with download options."""
    try:
        if not similar_faces:
            st.warning("No similar faces found.")
            return
        
        st.write(f"Found {len(similar_faces)} similar faces.")
        zip_href = create_zip_download(similar_faces)
        if zip_href:
            st.markdown("### Download All Matches")
            st.markdown(zip_href, unsafe_allow_html=True)
        
        st.write("### Matches")
        for i in range(0, len(similar_faces), 3):
            cols = st.columns(3)
            for j in range(3):
                if i + j < len(similar_faces):
                    match = similar_faces[i + j]
                    with cols[j]:
                        st.image(match['image_with_faces'], caption=f"Similarity: {match['similarity']:.2f}", width=200)
                        filename = f"match_{i+j+1}_{os.path.basename(match['image_path'])}"
                        href = create_download_link(match['image_with_faces'], filename, "Download")
                        st.markdown(href, unsafe_allow_html=True)
                        st.write(f"Source: {os.path.basename(match['image_path'])}")
                        st.write(f"Similarity: {match['similarity']:.2f}")
        
        avg_similarity = sum(m['similarity'] for m in similar_faces) / len(similar_faces)
        st.write(f"Average similarity: {avg_similarity:.2f}")
    except Exception as e:
        st.error(f"Error displaying results: {str(e)}")

def process_images(image_files, user_image, processor, model):
    """Process images to find similar faces using Swin-Base."""
    try:
        reference_image, reference_faces = detect_multiple_faces(user_image)
        if not reference_faces:
            st.error("No faces detected in reference image.")
            return
        
        st.write("### Reference Faces")
        cols = st.columns(min(len(reference_faces), 3))
        for idx, face_img in enumerate(reference_faces):
            cols[idx % 3].image(face_img, caption=f"Reference Face {idx + 1}", width=150)
        
        reference_embeddings = get_embedding_batch(reference_faces, processor, model)
        if reference_embeddings is None:
            st.error("Failed to generate embeddings for reference faces.")
            return
        
        similar_faces = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        total_images = len(image_files)
        processed = 0
        errors = 0
        
        for idx, img_path in enumerate(image_files):
            try:
                # Construct and normalize the full path
                full_path = os.path.normpath(os.path.join("extracted_images", img_path))
                
                # Update status
                status_text.text(f"Processing: {img_path}")
                
                img, error = safe_image_open(full_path)
                if error:
                    errors += 1
                    continue
                
                img_with_faces, detected_faces = detect_multiple_faces(img)
                if not detected_faces:
                    continue
                
                face_embeddings = get_embedding_batch(detected_faces, processor, model)
                if face_embeddings is None:
                    continue
                
                for face_idx, face_emb in enumerate(face_embeddings):
                    for ref_idx, ref_emb in enumerate(reference_embeddings):
                        similarity = cosine_similarity(face_emb, ref_emb)
                        if similarity > 0.85:
                            similar_faces.append({
                                'image_path': img_path,
                                'face_index': face_idx + 1,
                                'ref_face_index': ref_idx + 1,
                                'similarity': similarity,
                                'image_with_faces': img_with_faces
                            })
                
                processed += 1
                progress_bar.progress((idx + 1) / total_images)
                status_text.text(f"Processed: {processed}/{total_images} (Errors: {errors})")
            
            except Exception as e:
                errors += 1
                st.warning(f"Error processing {img_path}: {str(e)}")
        
        status_text.empty()
        
        if similar_faces:
            display_results_with_download(similar_faces)
        else:
            st.warning(f"No similar faces found. Processed {processed} images with {errors} errors.")
        
        return similar_faces
    except Exception as e:
        st.error(f"Error in process_images: {str(e)}")
        return None
    finally:
        cleanup_temp_files()

def process_large_zip(zip_file):
    """Process ZIP file with images."""
    try:
        temp_dir = "extracted_images"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        progress_bar = st.progress(0)
        image_files = []
        
        with ZipFile(zip_file, 'r') as zf:
            all_files = zf.namelist()
            total_files = len([f for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            processed = 0
            
            for f in all_files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    try:
                        # Normalize the path and create subdirectories if needed
                        normalized_path = os.path.normpath(f)
                        target_path = os.path.join(temp_dir, normalized_path)
                        
                        # Create subdirectories if they don't exist
                        os.makedirs(os.path.dirname(target_path), exist_ok=True)
                        
                        # Extract the file
                        zf.extract(f, temp_dir)
                        
                        # Store the normalized path
                        image_files.append(normalized_path)
                        
                        processed += 1
                        progress_bar.progress(processed / total_files)
                        
                    except Exception as e:
                        st.warning(f"Failed to extract {f}: {str(e)}")
            
            progress_bar.progress(1.0)
            
            if image_files:
                st.success(f"Successfully extracted {len(image_files)} images")
            else:
                st.warning("No valid images found in ZIP file")
        
        return image_files
    except Exception as e:
        st.error(f"Error processing ZIP file: {str(e)}")
        return []
    finally:
        cleanup_temp_files()

def process_zip_file(user_image, processor, model):
    """Handle ZIP file upload."""
    try:
        st.write("### Upload Images to Search")
        zip_file = st.file_uploader("Upload ZIP file", type=["zip"])
        
        if zip_file:
            st.info("Processing zip file...")
            image_files = process_large_zip(zip_file)
            if image_files:
                process_images(image_files, user_image, processor, model)
            else:
                st.warning("No valid images found in ZIP.")
    except Exception as e:
        st.error(f"Error in process_zip_file: {str(e)}")
    finally:
        cleanup_temp_files()

st.title("Face Similarity Detection with Swin-Base")
st.markdown("""
### Features:
- Detects multiple faces using MTCNN
- Compares faces using fine-tuned Swin-Base Transformer
- Supports ZIP file uploads
- Minimum image size: 224x224 pixels
- Expected accuracy: ~97-99% on LFW after fine-tuning
""")

try:
    processor, model = load_model()
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    st.stop()

st.write("### Upload Reference Image")
user_image_file = st.file_uploader("Upload reference image", type=["jpg", "jpeg", "png"])

if user_image_file:
    user_image, error = safe_image_open(user_image_file)
    if error:
        st.error(error)
        st.stop()
    
    reference_image, reference_faces = detect_multiple_faces(user_image)
    if reference_faces:
        st.image(reference_image, caption="Reference Image (Detected Faces)", width=400)
        process_zip_file(user_image, processor, model)
    else:
        st.error("No valid faces detected in reference image.")

cleanup_temp_files()