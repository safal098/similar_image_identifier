# Standard imports
import os
import io
import glob
import base64
from datetime import datetime
from zipfile import ZipFile
import warnings
warnings.filterwarnings('ignore')

# Data processing imports
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageEnhance, UnidentifiedImageError
import cv2

# Deep learning imports
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Face detection imports
try:
    from mtcnn import MTCNN
except ImportError:
    st.error("MTCNN package is not installed. Please install it using: pip install mtcnn")
    
try:
    from deepface import DeepFace
except ImportError:
    st.error("DeepFace package is not installed. Please install it using: pip install deepface")

# Transformers import
try:
    from transformers import AutoProcessor, TFSwinModel
except ImportError:
    st.error("Failed to import transformers. Installing required packages...")
    try:
        import subprocess
        subprocess.check_call(["pip", "install", "transformers[torch]", "torch", "--quiet"])
        from transformers import AutoProcessor, TFSwinModel
    except Exception as e:
        st.error(f"Failed to install/import transformers: {str(e)}")

# Streamlit import
import streamlit as st

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize session state for error tracking
if 'error_count' not in st.session_state:
    st.session_state.error_count = 0

# Initialize session state for file upload tracking
if 'upload_state' not in st.session_state:
    st.session_state.upload_state = {
        'progress': 0,
        'current_file': '',
        'processed_files': 0,
        'total_files': 0
    }

# Initialize face detector with error handling
try:
    mtcnn_detector = MTCNN(min_face_size=20)
    st.session_state['mtcnn_loaded'] = True
except Exception as e:
    st.warning(f"MTCNN initialization failed: {str(e)}")
    st.session_state['mtcnn_loaded'] = False

# Initialize OpenCV face cascade
try:
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if not os.path.exists(cascade_path):
        st.error("Face detection cascade file not found. Please check OpenCV installation.")
        face_cascade = None
    else:
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            st.error("Failed to load face detection cascade.")
            face_cascade = None
except Exception as e:
    st.error(f"Error initializing face cascade: {str(e)}")
    face_cascade = None

def initialize_deepface():
    """Initialize DeepFace models with error handling."""
    try:
        # Force DeepFace to download and initialize models
        DeepFace.build_model('VGG-Face')
        return True
    except Exception as e:
        st.error(f"Error initializing DeepFace: {str(e)}")
        return False

def cleanup_temp_files():
    """Clean up temporary files and directories created during processing."""
    try:
        # Clean up extracted images directory
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
                
        # Clean up any other temporary files that might be created
        temp_patterns = ['*.temp', '*.tmp']
        for pattern in temp_patterns:
            for temp_file in glob.glob(pattern):
                try:
                    os.remove(temp_file)
                except Exception as e:
                    st.warning(f"Failed to remove temporary file {temp_file}: {str(e)}")
                    
    except Exception as e:
        st.warning(f"Error during cleanup: {str(e)}")

def correct_image_orientation(image):
    """Correct image orientation using EXIF data."""
    try:
        # First try using EXIF transpose
        try:
            return ImageOps.exif_transpose(image)
        except Exception:
            pass
            
        # Manual EXIF orientation handling
        try:
            exif = image._getexif()
            if exif and 274 in exif:  # 274 is the orientation tag
                orientation = exif[274]
                rotations = {
                    3: Image.Transpose.ROTATE_180,
                    6: Image.Transpose.ROTATE_270,
                    8: Image.Transpose.ROTATE_90
                }
                if orientation in rotations:
                    return image.transpose(rotations[orientation])
        except Exception:
            pass
        
        return image
    except Exception:
        return image

def safe_image_open(file_or_path):
    """Safely open and validate an image file with enhanced error handling."""
    try:
        # Log the input path for debugging
        st.debug(f"Attempting to open image: {file_or_path}")
        
        # Handle string paths
        if isinstance(file_or_path, (str, bytes)):
            if not os.path.exists(file_or_path):
                st.error(f"Image file not found: {file_or_path}")
                return None, "Image file not found."
            
            # Log file size
            file_size = os.path.getsize(file_or_path)
            st.debug(f"File size: {file_size} bytes")
            
            if file_size == 0:
                st.error(f"Empty file: {file_or_path}")
                return None, "Empty file."
        
        # Attempt to open the image
        try:
            image = Image.open(file_or_path)
        except UnidentifiedImageError:
            st.error(f"Unidentified image format: {file_or_path}")
            return None, "Unidentified image format."
        except IOError as e:
            st.error(f"IO Error opening image: {str(e)}")
            return None, f"IO Error: {str(e)}"
        except Exception as e:
            st.error(f"Error opening image: {str(e)}")
            return None, f"Error opening image: {str(e)}"
        
        # Verify image integrity
        try:
            image.verify()
            image = Image.open(file_or_path)  # Reopen after verify
        except Exception as e:
            st.error(f"Image verification failed: {str(e)}")
            return None, f"Corrupted image: {str(e)}"
            
        # Check image mode and convert if necessary
        if image.mode not in ('RGB', 'RGBA'):
            st.debug(f"Converting image from {image.mode} to RGB")
            try:
                image = image.convert('RGB')
            except Exception as e:
                st.error(f"Color mode conversion failed: {str(e)}")
                return None, f"Color mode conversion failed: {str(e)}"
        
        # Check image dimensions
        width, height = image.size
        st.debug(f"Image dimensions: {width}x{height}")
        
        min_dimension = 224  # Minimum dimension required by most models
        if width < min_dimension or height < min_dimension:
            st.warning(f"Image dimensions ({width}x{height}) are below minimum {min_dimension}x{min_dimension}. Resizing...")
            # Calculate new dimensions maintaining aspect ratio
            ratio = min_dimension / min(width, height)
            new_size = (int(width * ratio), int(height * ratio))
            try:
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                st.debug(f"Resized image to: {new_size}")
            except Exception as e:
                st.error(f"Image resize failed: {str(e)}")
                return None, f"Image resize failed: {str(e)}"
        
        return image, None
        
    except Exception as e:
        st.error(f"Unexpected error in safe_image_open: {str(e)}")
        return None, f"Unexpected error: {str(e)}"

def enhance_image(image):
    """Enhance image quality for better face detection."""
    try:
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.5)
        
        # Enhance brightness
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.1)
        
        return image
    except Exception as e:
        st.warning(f"Image enhancement failed: {str(e)}")
        return image

def get_face_embeddings(face_image):
    """Get face embeddings using DeepFace."""
    try:
        # Convert PIL image to numpy array if needed
        if isinstance(face_image, Image.Image):
            face_image = np.array(face_image)
            
        # Get embeddings using DeepFace
        embedding = DeepFace.represent(
            face_image, 
            model_name='VGG-Face',
            enforce_detection=False
        )
        
        if embedding:
            return embedding[0]['embedding']
        return None
    except Exception as e:
        st.warning(f"Face embedding failed: {str(e)}")
        return None

def detect_multiple_faces(image):
    """
    Enhanced face detection using multiple methods with fallback options.
    Returns tuple: (image_with_faces, list_of_face_images)
    """
    try:
        if image is None:
            return None, "Invalid image input."
            
        # Initialize results
        detected_faces = []
        face_images = []
        detection_methods_used = []
        
        # Enhance and preprocess image
        enhanced_image = enhance_image(image)
        image_np = np.array(enhanced_image)
        original_h, original_w = image_np.shape[:2]
        
        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # 1. Try MTCNN detection with different min_face_sizes
        if st.session_state.get('mtcnn_loaded', False):
            try:
                min_face_sizes = [20, 40, 60]  # Try different minimum face sizes
                for min_face_size in min_face_sizes:
                    mtcnn_detector = MTCNN(min_face_size=min_face_size)
                    mtcnn_faces = mtcnn_detector.detect_faces(image_np)
                    
                    for face in mtcnn_faces:
                        if face['confidence'] > 0.90:  # Slightly lower threshold
                            x, y, w, h = face['box']
                            if not any(is_duplicate_face(x, y, w, h, existing) for existing in detected_faces):
                                detected_faces.append((x, y, w, h, 'mtcnn', face['confidence']))
                
                if detected_faces:
                    detection_methods_used.append('MTCNN')
            except Exception as e:
                st.warning(f"MTCNN detection failed: {str(e)}")

        # 2. Try DeepFace detection if needed
        if len(detected_faces) < 1:
            try:
                backends = ['retinaface', 'mtcnn', 'opencv']
                for backend in backends:
                    try:
                        face_objs = DeepFace.extract_faces(
                            image_np,
                            detector_backend=backend,
                            enforce_detection=False,
                            align=True
                        )
                        
                        for face_obj in face_objs:
                            facial_area = face_obj.get('facial_area', {})
                            if all(k in facial_area for k in ['x', 'y', 'w', 'h']):
                                x = facial_area['x']
                                y = facial_area['y']
                                w = facial_area['w']
                                h = facial_area['h']
                                if not any(is_duplicate_face(x, y, w, h, existing) for existing in detected_faces):
                                    detected_faces.append((x, y, w, h, 'deepface', 0.95))
                        
                        if face_objs:
                            detection_methods_used.append(f'DeepFace-{backend}')
                            break
                    except Exception:
                        continue
            except Exception as e:
                st.warning(f"DeepFace detection failed: {str(e)}")

        # 3. Try OpenCV Cascade as final backup
        if len(detected_faces) < 1:
            try:
                gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                gray = clahe.apply(gray)
                
                scale_factors = [1.05, 1.1, 1.15]
                min_neighbors_values = [3, 4, 5]
                
                for scale_factor in scale_factors:
                    for min_neighbors in min_neighbors_values:
                        faces = face_cascade.detectMultiScale(
                            gray,
                            scaleFactor=scale_factor,
                            minNeighbors=min_neighbors,
                            minSize=(30, 30),
                            flags=cv2.CASCADE_SCALE_IMAGE
                        )
                        
                        for (x, y, w, h) in faces:
                            if not any(is_duplicate_face(x, y, w, h, existing) for existing in detected_faces):
                                detected_faces.append((x, y, w, h, 'opencv', 0.85))
                
                if faces.any():
                    detection_methods_used.append('OpenCV')
            except Exception as e:
                st.warning(f"OpenCV detection failed: {str(e)}")

        # Process detected faces
        image_with_faces = image.copy()
        draw = ImageDraw.Draw(image_with_faces)
        
        for i, (x, y, w, h, method, conf) in enumerate(detected_faces):
            # Calculate quality metrics
            face_ratio = (w * h) / (original_w * original_h)
            aspect_ratio = w / h
            
            # More lenient quality thresholds
            if face_ratio < 0.001 or face_ratio > 0.98 or aspect_ratio < 0.5 or aspect_ratio > 2.0:
                continue
            
            # Add margin to face region
            margin_x = int(w * 0.2)
            margin_y = int(h * 0.2)
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(original_w, x + w + margin_x)
            y2 = min(original_h, y + h + margin_y)
            
            # Extract and enhance face region
            face_img = image.crop((x1, y1, x2, y2))
            face_img = enhance_image(face_img)
            
            # Draw rectangle and label with confidence
            color = {
                'mtcnn': "green",
                'deepface': "blue",
                'opencv': "yellow"
            }.get(method, "white")
            
            draw.rectangle(((x, y), (x+w, y+h)), outline=color, width=2)
            draw.text((x, y-20), f"Face {i+1} ({method}: {conf:.2f})", fill=color)
            
            face_images.append(face_img)
        
        # Log detection results
        if detection_methods_used:
            st.info(f"Face detection successful using: {', '.join(detection_methods_used)}")
        
        if not face_images:
            return None, "No valid faces detected in the image."
            
        return image_with_faces, face_images
        
    except Exception as e:
        st.error(f"Error in face detection pipeline: {str(e)}")
        return None, str(e)

def is_duplicate_face(x, y, w, h, existing_face):
    """Check if a detected face overlaps significantly with an existing detection."""
    ex, ey, ew, eh = existing_face[:4]
    
    # Calculate intersection
    x_left = max(x, ex)
    y_top = max(y, ey)
    x_right = min(x + w, ex + ew)
    y_bottom = min(y + h, ey + eh)
    
    if x_right < x_left or y_bottom < y_top:
        return False
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = w * h
    area2 = ew * eh
    overlap = intersection / min(area1, area2)
    
    return overlap > 0.5

def validate_and_preprocess_image(image, check_face=True):
    """Validate and preprocess image for the model."""
    try:
        if image is None:
            return None
            
        # Check image mode and convert if necessary
        if image.mode not in ('RGB', 'L'):
            try:
                image = image.convert('RGB')
            except Exception as e:
                st.error(f"Failed to convert image to RGB: {str(e)}")
                return None
        
        # Get original dimensions
        try:
            width, height = image.size
        except Exception as e:
            st.error(f"Failed to get image dimensions: {str(e)}")
            return None
        
        # Check minimum dimensions
        if width < 224 or height < 224:
            st.warning(f"Image dimensions ({width}x{height}) are too small. Minimum required is 224x224 pixels.")
            return None
            
        # Check maximum dimensions and resize if necessary
        max_dimension = 1920
        if width > max_dimension or height > max_dimension:
            try:
                # Calculate new dimensions maintaining aspect ratio
                ratio = min(max_dimension/width, max_dimension/height)
                new_size = tuple(int(dim * ratio) for dim in (width, height))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                st.info(f"Image has been resized from {width}x{height} to {new_size[0]}x{new_size[1]} to improve processing speed.")
            except Exception as e:
                st.error(f"Failed to resize image: {str(e)}")
                return None
        
        # Correct orientation using EXIF data
        try:
            image = correct_image_orientation(image)
        except Exception as e:
            st.warning(f"Failed to correct image orientation: {str(e)}")
            # Continue with original image
        
        # Validate face if required
        if check_face:
            image, error_message = detect_multiple_faces(image)
            if error_message:
                st.warning(error_message)
                return None
        
        return image
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

@st.cache_resource
def load_model():
    """Load the model and processor with proper error handling."""
    try:
        # Verify HF_TOKEN is available
        HF_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
        if not HF_TOKEN:
            raise ValueError("HUGGING_FACE_TOKEN environment variable is not set")

        # Load processor with error handling
        try:
            processor = AutoProcessor.from_pretrained(
                "microsoft/swin-tiny-patch4-window7-224",
                token=HF_TOKEN
            )
            st.debug("Successfully loaded image processor")
        except Exception as e:
            st.error(f"Failed to load image processor: {str(e)}")
            st.info("Attempting to install/update required packages...")
            try:
                import subprocess
                subprocess.check_call(["pip", "install", "--upgrade", "transformers[torch]", "torch", "--quiet"])
                processor = AutoProcessor.from_pretrained(
                    "microsoft/swin-tiny-patch4-window7-224",
                    token=HF_TOKEN
                )
            except Exception as e:
                raise Exception(f"Failed to load image processor even after update: {str(e)}")

        # Load model with error handling
        try:
            model = TFSwinModel.from_pretrained(
                "microsoft/swin-tiny-patch4-window7-224",
                token=HF_TOKEN
            )
            st.debug("Successfully loaded Swin model")
        except Exception as e:
            st.error(f"Failed to load model: {str(e)}")
            raise

        return processor, model
        
    except Exception as e:
        st.error(f"Error in load_model: {str(e)}")
        raise

def get_embedding_batch(images, processor, model):
    """Compute embeddings for a batch of images."""
    try:
        if not images:
            return None

        # Ensure images are in correct format
        processed_images = []
        for img in images:
            if img is None:
                continue
                
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize if needed (maintain aspect ratio)
            if img.size[0] > 1920 or img.size[1] > 1920:
                img.thumbnail((1920, 1920), Image.Resampling.LANCZOS)
                
            processed_images.append(img)

        if not processed_images:
            return None

        # Process images with the processor
        try:
            inputs = processor(images=processed_images, return_tensors="tf", padding=True)
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].numpy()
            return embeddings
        except Exception as e:
            st.error(f"Error computing embeddings: {str(e)}")
            return None

    except Exception as e:
        st.error(f"Error processing images for embedding: {str(e)}")
        return None

def cosine_similarity(emb1, emb2):
    """Compute cosine similarity between two embeddings."""
    try:
        if emb1 is None or emb2 is None:
            return 0.0
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
        # Convert PIL image to bytes
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        
        # Create base64 string
        b64 = base64.b64encode(buf.getvalue()).decode()
        href = f'<a href="data:image/png;base64,{b64}" download="{filename}">{text}</a>'
        return href
    except Exception as e:
        st.error(f"Error creating download link: {str(e)}")
        return None

def create_zip_download(similar_faces):
    """Create a ZIP file containing all matched images."""
    try:
        # Create ZIP file in memory
        zip_buffer = io.BytesIO()
        with ZipFile(zip_buffer, 'w', ZipFile.ZIP_DEFLATED) as zip_file:
            for idx, match in enumerate(similar_faces):
                # Convert PIL image to bytes
                img_buffer = io.BytesIO()
                match['image_with_faces'].save(img_buffer, format='PNG')
                img_buffer.seek(0)
                
                # Create filename with match details
                filename = f"match_{idx+1}_similarity_{match['similarity']:.2f}_{os.path.basename(match['image_path'])}"
                zip_file.writestr(filename, img_buffer.getvalue())

        # Create download link for ZIP
        zip_buffer.seek(0)
        b64 = base64.b64encode(zip_buffer.getvalue()).decode()
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        download_filename = f"matched_faces_{date_str}.zip"
        href = f'<a href="data:application/zip;base64,{b64}" download="{download_filename}">Download All Matched Images (ZIP)</a>'
        return href
    except Exception as e:
        st.error(f"Error creating ZIP download: {str(e)}")
        return None

def display_results_with_download(similar_faces):
    """Display results with download options and preview gallery."""
    try:
        if not similar_faces:
            st.warning("No similar faces found in the processed images.")
            return

        st.write(f"Found {len(similar_faces)} similar faces.")

        # Add bulk download button
        st.write("### Download Options")
        zip_href = create_zip_download(similar_faces)
        if zip_href:
            st.markdown(zip_href, unsafe_allow_html=True)

        # Create preview gallery with individual downloads
        st.write("### Preview Gallery")
        st.write("Click on any image to download it individually.")

        # Sort by similarity
        similar_faces.sort(key=lambda x: x['similarity'], reverse=True)

        # Display results in a grid with download buttons
        for i in range(0, len(similar_faces), 3):
            cols = st.columns(3)
            for j in range(3):
                if i + j < len(similar_faces):
                    match = similar_faces[i + j]
                    with cols[j]:
                        # Display image
                        st.image(match['image_with_faces'],
                                caption=f"Match {i+j+1}\nSimilarity: {match['similarity']:.2f}",
                                width=200)
                        
                        # Create download button for individual image
                        filename = f"match_{i+j+1}_{os.path.basename(match['image_path'])}"
                        href = create_download_link(
                            match['image_with_faces'],
                            filename,
                            "⬇️ Download this image"
                        )
                        if href:
                            st.markdown(href, unsafe_allow_html=True)
                        
                        # Add match details
                        st.write(f"📁 Source: {os.path.basename(match['image_path'])}")
                        st.write(f"👤 Face: {match['face_index']}")
                        st.write(f"🎯 Matches Reference Face: {match['ref_face_index']}")
                        st.write(f"✨ Similarity: {match['similarity']:.2f}")

        # Add summary information
        st.write("### Summary")
        st.write(f"Total matches found: {len(similar_faces)}")
        avg_similarity = sum(m['similarity'] for m in similar_faces) / len(similar_faces)
        st.write(f"Average similarity score: {avg_similarity:.2f}")

    except Exception as e:
        st.error(f"Error displaying results: {str(e)}")

def compare_faces(face1, face2):
    """Compare two faces using DeepFace."""
    try:
        # Convert PIL images to numpy arrays if needed
        if isinstance(face1, Image.Image):
            face1 = np.array(face1)
        if isinstance(face2, Image.Image):
            face2 = np.array(face2)
            
        # Compare faces using DeepFace
        result = DeepFace.verify(
            face1, 
            face2,
            model_name='VGG-Face',
            enforce_detection=False,
            distance_metric='cosine'
        )
        
        # Convert distance to similarity score (1 - distance)
        similarity = 1 - result['distance']
        return similarity
        
    except Exception as e:
        st.warning(f"Face comparison failed: {str(e)}")
        return 0.0

def process_images(image_files, user_image, processor, model):
    """Process the comparison images and find similar faces with enhanced debugging."""
    try:
        # Create debug expander
        with st.expander("Processing Debug Information", expanded=True):
            debug_container = st.container()

        def debug_log(message):
            with debug_container:
                st.write(message)

        # First, get reference faces
        debug_log("### Processing Reference Image")
        reference_image, reference_faces = detect_multiple_faces(user_image)
        
        if reference_faces is None or len(reference_faces) == 0:
            st.error("No faces detected in reference image.")
            return
            
        debug_log(f"Found {len(reference_faces)} faces in reference image")
        
        # Display reference faces
        st.write("### Reference Faces")
        ref_cols = st.columns(min(len(reference_faces), 3))
        reference_embeddings = []
        
        for idx, face_img in enumerate(reference_faces):
            with ref_cols[idx % 3]:
                st.image(face_img, caption=f"Reference Face {idx + 1}", width=150)
                # Get embedding for reference face
                try:
                    embedding = get_face_embeddings(face_img)
                    if embedding is not None:
                        reference_embeddings.append(embedding)
                        debug_log(f"✅ Generated embedding for reference face {idx + 1}")
                    else:
                        debug_log(f"❌ Failed to generate embedding for reference face {idx + 1}")
                except Exception as e:
                    debug_log(f"❌ Error getting embedding for reference face {idx + 1}: {str(e)}")

        if not reference_embeddings:
            st.error("Could not generate embeddings for any reference faces.")
            return

        # Process comparison images
        debug_log("\n### Processing Comparison Images")
        similar_images = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_images = len(image_files)
        processed_count = 0
        match_count = 0
        error_count = 0

        for img_path in image_files:
            try:
                full_path = os.path.join("extracted_images", img_path)
                debug_log(f"\nProcessing: {img_path}")
                
                # Load and validate image
                img, error = safe_image_open(full_path)
                if error:
                    debug_log(f"❌ Error loading image: {error}")
                    error_count += 1
                    continue

                # Detect faces in comparison image
                img_with_faces, detected_faces = detect_multiple_faces(img)
                if detected_faces is None or len(detected_faces) == 0:
                    debug_log(f"❌ No faces detected in: {img_path}")
                    error_count += 1
                    continue

                debug_log(f"Found {len(detected_faces)} faces in comparison image")
                
                # Compare each detected face with reference faces
                for face_idx, face_img in enumerate(detected_faces):
                    try:
                        # Get embedding for detected face
                        face_embedding = get_face_embeddings(face_img)
                        if face_embedding is None:
                            debug_log(f"❌ Failed to get embedding for face {face_idx + 1}")
                            continue

                        # Compare with all reference faces
                        for ref_idx, ref_embedding in enumerate(reference_embeddings):
                            similarity = cosine_similarity(face_embedding, ref_embedding)
                            debug_log(f"Face {face_idx + 1} similarity with reference {ref_idx + 1}: {similarity:.3f}")
                            
                            if similarity > 0.85:  # Adjust threshold as needed
                                match_count += 1
                                similar_images.append({
                                    'image_path': img_path,
                                    'face_index': face_idx + 1,
                                    'ref_face_index': ref_idx + 1,
                                    'similarity': similarity,
                                    'image_with_faces': img_with_faces
                                })
                                debug_log(f"✅ Match found! Similarity: {similarity:.3f}")
                                
                                # Display matched face for verification
                                with st.expander(f"Match Details - {os.path.basename(img_path)}", expanded=False):
                                    cols = st.columns(2)
                                    with cols[0]:
                                        st.image(reference_faces[ref_idx], caption=f"Reference Face {ref_idx + 1}", width=150)
                                    with cols[1]:
                                        st.image(face_img, caption=f"Matched Face (Similarity: {similarity:.3f})", width=150)

                    except Exception as e:
                        debug_log(f"❌ Error comparing face {face_idx + 1}: {str(e)}")
                        error_count += 1

            except Exception as e:
                debug_log(f"❌ Error processing {img_path}: {str(e)}")
                error_count += 1
            finally:
                processed_count += 1
                progress = processed_count / total_images
                progress_bar.progress(progress)
                status_text.text(
                    f"Processed: {processed_count}/{total_images}\n"
                    f"Matches found: {match_count}\n"
                    f"Errors: {error_count}"
                )

        # Display final results
        st.write("\n### Processing Summary")
        st.write(f"- Total images processed: {total_images}")
        st.write(f"- Total matches found: {match_count}")
        st.write(f"- Total errors: {error_count}")

        if similar_images:
            display_results_with_download(similar_images)
        else:
            st.warning("No similar faces found.")
            if error_count > 0:
                st.error(f"Encountered {error_count} errors while processing images.")
                
        return similar_images

    except Exception as e:
        st.error(f"Error in process_images: {str(e)}")
        return None
    finally:
        cleanup_temp_files()

def process_large_zip(zip_file):
    """Process large zip files with enhanced error handling and progress tracking."""
    try:
        # Initialize or reset session state for detailed progress tracking
        if 'zip_progress' not in st.session_state:
            st.session_state.zip_progress = {
                'total_files': 0,
                'processed_files': 0,
                'successful_files': 0,
                'failed_files': 0,
                'current_file': '',
                'errors': [],
                'skipped_files': []
            }
        
        # Create temporary directory if it doesn't exist
        temp_dir = "extracted_images"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        # Initialize progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        error_log = st.empty()
        
        with ZipFile(zip_file, 'r') as zf:
            # Get list of all files
            all_files = zf.namelist()
            
            # Filter valid image files and handle encoding
            image_files = []
            for f in all_files:
                try:
                    # Skip macOS system files
                    if '__MACOSX' in f or f.startswith('.'):
                        continue
                        
                    # Check file extension
                    if not f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                        st.session_state.zip_progress['skipped_files'].append(
                            f"Skipped {f}: Not a supported image format"
                        )
                        continue
                    
                    # Try to decode filename
                    try:
                        # Handle potential encoding issues
                        decoded_name = f.encode('cp437').decode('utf-8')
                    except UnicodeError:
                        try:
                            decoded_name = f.encode('utf-8').decode('utf-8')
                        except UnicodeError:
                            st.session_state.zip_progress['errors'].append(
                                f"Failed to decode filename: {f}"
                            )
                            continue
                    
                    image_files.append(decoded_name)
                    
                except Exception as e:
                    st.session_state.zip_progress['errors'].append(
                        f"Error processing file {f}: {str(e)}"
                    )
                    continue
            
            if not image_files:
                st.warning("No valid image files found in the ZIP archive.")
                return []
            
            # Update session state
            st.session_state.zip_progress['total_files'] = len(image_files)
            st.session_state.zip_progress['processed_files'] = 0
            st.session_state.zip_progress['successful_files'] = 0
            st.session_state.zip_progress['failed_files'] = 0
            st.session_state.zip_progress['errors'] = []
            
            valid_images = []
            
            # Process each file
            for file_path in image_files:
                try:
                    # Update progress tracking
                    st.session_state.zip_progress['current_file'] = file_path
                    st.session_state.zip_progress['processed_files'] += 1
                    
                    # Extract file with error handling
                    try:
                        zf.extract(file_path, temp_dir)
                    except Exception as e:
                        error_msg = f"Failed to extract {file_path}: {str(e)}"
                        st.session_state.zip_progress['errors'].append(error_msg)
                        st.session_state.zip_progress['failed_files'] += 1
                        continue
                    
                    # Verify extracted file
                    extracted_path = os.path.join(temp_dir, file_path)
                    if not os.path.exists(extracted_path):
                        st.session_state.zip_progress['errors'].append(
                            f"Extracted file not found: {file_path}"
                        )
                        st.session_state.zip_progress['failed_files'] += 1
                        continue
                    
                    # Validate image file
                    try:
                        with Image.open(extracted_path) as img:
                            # Verify image can be loaded
                            img.verify()
                            # Try to load it again to ensure it's valid
                            img = Image.open(extracted_path)
                            # Try to convert to RGB to ensure color mode compatibility
                            if img.mode not in ('RGB', 'RGBA'):
                                img = img.convert('RGB')
                            # Check minimum dimensions
                            if min(img.size) < 224:
                                st.session_state.zip_progress['skipped_files'].append(
                                    f"Skipped {file_path}: Image too small ({img.size[0]}x{img.size[1]})"
                                )
                                continue
                                
                        st.session_state.zip_progress['successful_files'] += 1
                        valid_images.append(file_path)
                        
                    except UnidentifiedImageError:
                        error_msg = f"Unidentified image format: {file_path}"
                        st.session_state.zip_progress['errors'].append(error_msg)
                        st.session_state.zip_progress['failed_files'] += 1
                        try:
                            os.remove(extracted_path)
                        except:
                            pass
                        continue
                    except Exception as e:
                        error_msg = f"Invalid or corrupted image {file_path}: {str(e)}"
                        st.session_state.zip_progress['errors'].append(error_msg)
                        st.session_state.zip_progress['failed_files'] += 1
                        try:
                            os.remove(extracted_path)
                        except:
                            pass
                        continue
                    
                    # Update progress display
                    progress = st.session_state.zip_progress['processed_files'] / st.session_state.zip_progress['total_files']
                    progress_bar.progress(progress)
                    
                    status_text.text(
                        f"Processing: {file_path}\n"
                        f"Progress: {st.session_state.zip_progress['processed_files']}/{st.session_state.zip_progress['total_files']}\n"
                        f"Successful: {st.session_state.zip_progress['successful_files']}"
                    )
                    
                    # Update error log if needed
                    if st.session_state.zip_progress['errors']:
                        error_log.markdown(
                            "**Recent Errors:**\n" + 
                            "\n".join(st.session_state.zip_progress['errors'][-5:])
                        )
                    
                except Exception as e:
                    error_msg = f"Error processing {file_path}: {str(e)}"
                    st.session_state.zip_progress['errors'].append(error_msg)
                    st.session_state.zip_progress['failed_files'] += 1
                    continue
            
            # Final status update
            status_text.text(
                f"Completed processing {len(image_files)} files:\n"
                f"✅ Successful: {st.session_state.zip_progress['successful_files']}\n"
                f"❌ Failed: {st.session_state.zip_progress['failed_files']}"
            )
            
            # Show detailed error log
            if st.session_state.zip_progress['errors'] or st.session_state.zip_progress['skipped_files']:
                with st.expander("Processing Details", expanded=True):
                    if st.session_state.zip_progress['errors']:
                        st.markdown("**Errors:**")
                        for error in st.session_state.zip_progress['errors']:
                            st.error(error)
                    
                    if st.session_state.zip_progress['skipped_files']:
                        st.markdown("**Skipped Files:**")
                        for skipped in st.session_state.zip_progress['skipped_files']:
                            st.warning(skipped)
            
            return valid_images

    except Exception as e:
        st.error(f"Error processing ZIP file: {str(e)}")
        return []
    finally:
        # Ensure progress bar is complete
        progress_bar.progress(1.0)

def split_zip_if_needed(zip_file):
    """Split large zip files into smaller chunks if needed."""
    try:
        file_size_mb = len(zip_file.getvalue()) / (1024 * 1024)  # Size in MB
        
        if file_size_mb > 200:
            st.error(f"""
            File size ({file_size_mb:.1f}MB) exceeds Streamlit's limit of 200MB.
            
            Please split your ZIP file into smaller parts (less than 200MB each) and upload them separately.
            You can use tools like 7-Zip or WinRAR to split your ZIP file.
            
            Steps to split a ZIP file:
            1. Using 7-Zip:
               - Right-click the ZIP file
               - Select '7-Zip' → 'Split file...'
               - Enter 200M as the split size
               
            2. Using WinRAR:
               - Right-click the ZIP file
               - Select 'Split archive...'
               - Set split size to 200M
            """)
            return None
            
        return zip_file
    except Exception as e:
        st.error(f"Error checking file size: {str(e)}")
        return None

def process_zip_file(user_image, processor, model):
    """Process the zip file containing comparison images with enhanced debugging."""
    try:
        # Verify model initialization
        if not hasattr(st.session_state, 'models_initialized'):
            st.info("Initializing face detection models...")
            if initialize_models():
                st.session_state.models_initialized = True
            else:
                st.error("Failed to initialize face detection models.")
                return
        
        st.write("### Upload Images to Search")
        st.write("Maximum file size: 200MB per upload (Streamlit limitation)")
        st.info("""
        💡 If your ZIP file is larger than 200MB:
        1. Split it into smaller ZIP files (< 200MB each)
        2. Upload each part separately
        3. The app will process all images from each upload
        """)
        
        # Configure file uploader
        zip_file = st.file_uploader(
            "Upload ZIP file",
            type=["zip"],
            help="Upload a ZIP file containing images (max 200MB per upload)",
            accept_multiple_files=False
        )
        
        if not zip_file:
            return
            
        try:
            # Check and handle file size
            processed_zip = split_zip_if_needed(zip_file)
            if processed_zip is None:
                return
                
            st.info("Processing zip file...")
            
            # Process the zip file with progress tracking
            image_files = process_large_zip(processed_zip)
            
            if not image_files:
                st.warning("No valid images found in the zip folder.")
                return
            
            # Process images with progress tracking
            process_images(image_files, user_image, processor, model)
            
        except Exception as e:
            st.error(f"Error processing zip file: {str(e)}")
        finally:
            cleanup_temp_files()
            
    except Exception as e:
        st.error(f"Error in process_zip_file: {str(e)}")
    finally:
        cleanup_temp_files()

def initialize_models():
    """Initialize all face detection models with error handling."""
    try:
        # Initialize MTCNN
        try:
            global mtcnn_detector
            mtcnn_detector = MTCNN(min_face_size=20)
            st.session_state['mtcnn_loaded'] = True
        except Exception as e:
            st.error(f"MTCNN initialization failed: {str(e)}")
            st.session_state['mtcnn_loaded'] = False
            return False
        
        # Initialize DeepFace
        try:
            DeepFace.build_model('VGG-Face')
        except Exception as e:
            st.error(f"DeepFace initialization failed: {str(e)}")
            return False
        
        # Initialize OpenCV cascade
        try:
            global face_cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if not os.path.exists(cascade_path):
                st.error("Face detection cascade file not found.")
                return False
            face_cascade = cv2.CascadeClassifier(cascade_path)
            if face_cascade.empty():
                st.error("Failed to load face detection cascade.")
                return False
        except Exception as e:
            st.error(f"OpenCV cascade initialization failed: {str(e)}")
            return False
        
        return True
        
    except Exception as e:
        st.error(f"Error initializing models: {str(e)}")
        return False

def process_local_folder(folder_path, user_image, processor, model):
    """Process images from a local folder."""
    try:
        if not os.path.exists(folder_path):
            st.error("Selected folder does not exist.")
            return

        # Create a placeholder for logging
        log_placeholder = st.empty()
        def log_message(message):
            log_placeholder.markdown(message)

        # Get all image files
        image_files = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_files.append(os.path.join(root, file))

        if not image_files:
            st.warning("No valid images found in the selected folder.")
            return

        # Process images with progress tracking
        st.info(f"Found {len(image_files)} images. Processing...")
        progress_bar = st.progress(0)
        status_text = st.empty()

        similar_faces = []
        processed_count = 0
        total_files = len(image_files)

        # Get reference faces
        log_message("Processing reference image...")
        reference_image, reference_faces = detect_multiple_faces(user_image)
        if reference_faces is None or len(reference_faces) == 0:
            st.error("No valid faces detected in reference image.")
            return

        # Show reference faces
        st.write("### Reference Faces")
        ref_cols = st.columns(min(len(reference_faces), 3))
        
        for idx, face_img in enumerate(reference_faces):
            with ref_cols[idx % 3]:
                st.image(face_img, caption=f"Reference Face {idx + 1}", width=150)

        # Process each image
        skipped_files = 0
        no_faces_files = 0
        
        # Create preview columns
        preview_cols = st.columns(3)
        preview_idx = 0
        
        for img_path in image_files:
            try:
                # Update progress
                processed_count += 1
                progress = processed_count / total_files
                progress_bar.progress(progress)
                status_text.text(f"Processing: {os.path.basename(img_path)} ({processed_count}/{total_files})")
                log_message(f"Processing: {img_path}")

                # Load and process image
                img, error = safe_image_open(img_path)
                if error:
                    log_message(f"❌ Error loading image: {error}")
                    skipped_files += 1
                    continue

                # Detect faces
                img_with_faces, detected_faces = detect_multiple_faces(img)
                if detected_faces is None or len(detected_faces) == 0:
                    log_message(f"⚠️ No faces detected in: {os.path.basename(img_path)}")
                    no_faces_files += 1
                    continue

                log_message(f"Found {len(detected_faces)} faces in: {os.path.basename(img_path)}")

                # Show live preview
                with preview_cols[preview_idx % 3]:
                    st.image(img_with_faces, caption=f"Processing: {os.path.basename(img_path)}", width=150)
                preview_idx += 1

                # Compare faces
                matches_found = False
                for face_idx, face_img in enumerate(detected_faces):
                    # Compare with all reference faces
                    for ref_idx, ref_face in enumerate(reference_faces):
                        # Compare using multiple methods
                        similarity = compare_faces(face_img, ref_face)
                        log_message(f"Face {face_idx + 1} similarity with reference {ref_idx + 1}: {similarity:.2f}")
                        
                        if similarity > 0.6:  # Adjusted threshold
                            matches_found = True
                            similar_faces.append({
                                'image_path': img_path,
                                'face_index': face_idx + 1,
                                'ref_face_index': ref_idx + 1,
                                'similarity': similarity,
                                'image_with_faces': img_with_faces
                            })

                if matches_found:
                    log_message(f"✅ Found matches in: {os.path.basename(img_path)}")
                else:
                    log_message(f"❌ No matches found in: {os.path.basename(img_path)}")

            except Exception as e:
                log_message(f"❌ Error processing {img_path}: {str(e)}")
                skipped_files += 1
                continue

        # Clear preview
        for col in preview_cols:
            col.empty()

        # Display summary
        status_text.text("Processing complete!")
        st.write("### Processing Summary")
        st.write(f"- Total images processed: {total_files}")
        st.write(f"- Images with no faces: {no_faces_files}")
        st.write(f"- Skipped/error files: {skipped_files}")
        st.write(f"- Successfully processed: {total_files - no_faces_files - skipped_files}")
        
        if not similar_faces:
            st.warning("No faces similar to the reference faces were found in the provided images.")
            return

        # Display results
        display_results_with_download(similar_faces)

    except Exception as e:
        st.error(f"Error processing folder: {str(e)}")

# Initialize face cascade classifier
try:
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if not os.path.exists(cascade_path):
        st.error("Face detection model not found. Please check OpenCV installation.")
        st.stop()
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        st.error("Failed to load face detection model.")
        st.stop()
except Exception as e:
    st.error(f"Error initializing face detection: {str(e)}")
    st.stop()

# Main Streamlit app
try:
    st.title("Multi-Face Detection and Comparison App")
    
    # Add updated guidelines
    st.markdown("""
    ### Features:
    - Detects multiple faces in each image
    - Supports both ZIP files and local folder browsing
    - Processes images recursively in subfolders
    - Shows face detection confidence for each match
    
    ### Upload Options:
    1. **ZIP File**: Upload a ZIP file (max 200MB)
    2. **Local Folder**: Select a folder from your computer
    
    ### Image Requirements:
    - Supports JPG, JPEG, and PNG formats
    - Can process multiple faces per image
    - Works with group photos
    - Handles various face angles and positions
    """)

    # Load model
    try:
        processor, model = load_model()
    except Exception as e:
        st.error(f"Failed to load the model: {str(e)}")
        st.stop()

    # Reference image upload
    st.write("### Step 1: Upload Reference Image")
    user_image_file = st.file_uploader(
        "Upload your reference image",
        type=["jpg", "jpeg", "png"],
        help="Upload a photo containing the face(s) you want to find"
    )

    if user_image_file:
        try:
            user_image, error = safe_image_open(user_image_file)
            if error:
                st.error(error)
                st.stop()

            # Detect faces in reference image
            reference_image, reference_faces = detect_multiple_faces(user_image)
            if reference_faces is None:
                st.error("No valid faces detected in reference image.")
                st.stop()

            st.image(reference_image, caption="Reference Image (Detected Faces)", width=400)
            st.write(f"Detected {len(reference_faces)} faces in reference image")

            # Step 2: Choose input method
            st.write("### Step 2: Choose Input Method")
            input_method = st.radio(
                "Select how you want to input images to search:",
                ["Upload ZIP File", "Browse Local Folder"]
            )

            if input_method == "Upload ZIP File":
                process_zip_file(user_image, processor, model)
            else:
                folder_path = st.text_input("Enter the full path to your images folder:")
                if folder_path:
                    process_local_folder(folder_path, user_image, processor, model)

        except Exception as e:
            st.error(f"Error: {str(e)}")
        finally:
            cleanup_temp_files()
            
except Exception as e:
    st.error(f"Application error: {str(e)}")
finally:
    cleanup_temp_files()
