# Standard imports
import os
import io
import glob
import base64
from datetime import datetime
from zipfile import ZipFile, ZIP_STORED
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

import gc
import psutil
import traceback

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
                
        # Clean up any temporary files
        temp_patterns = ['*.tmp', '*.temp']
        temp_dirs = [
            os.path.expanduser('~/Downloads'),  # User's Downloads folder
            os.getcwd(),                        # Current working directory
            os.path.dirname(os.path.abspath(__file__))  # Script directory
        ]
        
        for temp_dir in temp_dirs:
            if os.path.exists(temp_dir):
                for pattern in temp_patterns:
                    try:
                        for temp_file in glob.glob(os.path.join(temp_dir, pattern)):
                            try:
                                if os.path.exists(temp_file):
                                    os.remove(temp_file)
                            except Exception as e:
                                st.warning(f"Failed to remove temporary file {temp_file}: {str(e)}")
                    except Exception as e:
                        st.warning(f"Error searching for pattern {pattern} in {temp_dir}: {str(e)}")
                    
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
        st.write(f"🔍 Attempting to open image: {file_or_path}")
        
        # Handle UploadedFile objects
        if hasattr(file_or_path, 'read'):
            try:
                # Read the file into memory
                image_bytes = file_or_path.read()
                image = Image.open(io.BytesIO(image_bytes))
                st.write(f"Successfully opened uploaded file: {file_or_path.name}")
            except Exception as e:
                st.error(f"Error reading uploaded file: {str(e)}")
                return None, f"Error reading uploaded file: {str(e)}"
        # Handle string paths
        elif isinstance(file_or_path, (str, bytes)):
            if not os.path.exists(file_or_path):
                st.error(f"Image file not found: {file_or_path}")
                return None, "Image file not found."
            
            try:
                image = Image.open(file_or_path)
            except Exception as e:
                st.error(f"Error opening image from path: {str(e)}")
                return None, f"Error opening image: {str(e)}"
        else:
            return None, f"Unsupported input type: {type(file_or_path)}"
        
        # Log original image characteristics
        st.write(f"Original format: {image.format}")
        st.write(f"Original mode: {image.mode}")
        st.write(f"Original size: {image.size}")
        
        # Verify image integrity
        try:
            image.verify()
            # Reopen after verify
            if hasattr(file_or_path, 'read'):
                image = Image.open(io.BytesIO(image_bytes))
            else:
                image = Image.open(file_or_path)
        except Exception as e:
            st.error(f"Image verification failed: {str(e)}")
            return None, f"Corrupted image: {str(e)}"
            
        # Convert to RGB if needed
        if image.mode not in ('RGB', 'RGBA'):
            st.write(f"Converting image from {image.mode} to RGB")
            try:
                image = image.convert('RGB')
            except Exception as e:
                st.error(f"Color mode conversion failed: {str(e)}")
                return None, f"Color mode conversion failed: {str(e)}"
        
        # Check and adjust dimensions
        width, height = image.size
        st.write(f"Image dimensions: {width}x{height}")
        
        min_dimension = 224  # Minimum dimension required by most models
        max_dimension = 1920  # Maximum dimension for processing efficiency
        
        # Resize if needed
        if width < min_dimension or height < min_dimension:
            st.write(f"Image dimensions ({width}x{height}) are below minimum {min_dimension}x{min_dimension}. Resizing...")
            # Calculate new dimensions maintaining aspect ratio
            ratio = min_dimension / min(width, height)
            new_size = (int(width * ratio), int(height * ratio))
            try:
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                st.write(f"Resized image to: {new_size}")
            except Exception as e:
                st.error(f"Image resize failed: {str(e)}")
                return None, f"Image resize failed: {str(e)}"
        elif width > max_dimension or height > max_dimension:
            st.write(f"Image dimensions ({width}x{height}) exceed maximum {max_dimension}x{max_dimension}. Resizing...")
            # Calculate new dimensions maintaining aspect ratio
            ratio = max_dimension / max(width, height)
            new_size = (int(width * ratio), int(height * ratio))
            try:
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                st.write(f"Resized image to: {new_size}")
            except Exception as e:
                st.error(f"Image resize failed: {str(e)}")
                return None, f"Image resize failed: {str(e)}"
        
        # Ensure the image is in memory
        try:
            image.load()
        except Exception as e:
            st.error(f"Failed to load image into memory: {str(e)}")
            return None, f"Failed to load image: {str(e)}"
        
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
    Returns tuple: (marked_image, face_images, original_image, face_regions)
    """
    try:
        if image is None:
            return None, None, None, "Invalid image input."
            
        # Keep a clean copy of the original image
        original_image = image.copy()
        
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
        
        # Try MTCNN detection first
        try:
            detector = MTCNN()
            mtcnn_faces = detector.detect_faces(image_np)
            if mtcnn_faces:
                for face in mtcnn_faces:
                    x, y, w, h = face['box']
                    confidence = face['confidence']
                    detected_faces.append((x, y, w, h, 'mtcnn', confidence))
                detection_methods_used.append('mtcnn')
        except Exception as e:
            st.warning(f"MTCNN detection failed: {str(e)}")
        
        # If no faces found with MTCNN, try DeepFace
        if not detected_faces:
            try:
                backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface']
                for backend in backends:
                    try:
                        faces = DeepFace.extract_faces(
                            img_path=image_np,
                            target_size=(224, 224),
                            detector_backend=backend
                        )
                        if faces:
                            for face in faces:
                                facial_area = face.get('facial_area', {})
                                x = facial_area.get('x', 0)
                                y = facial_area.get('y', 0)
                                w = facial_area.get('w', 0)
                                h = facial_area.get('h', 0)
                                confidence = face.get('confidence', 0.5)
                                detected_faces.append((x, y, w, h, 'deepface', confidence))
                            detection_methods_used.append(f'deepface-{backend}')
                            break
                    except Exception as e:
                        continue
            except Exception as e:
                st.warning(f"DeepFace detection failed: {str(e)}")
        
        # If still no faces found, try OpenCV Cascade as last resort
        if not detected_faces:
            try:
                gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
                opencv_faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                
                if len(opencv_faces) > 0:
                    for (x, y, w, h) in opencv_faces:
                        detected_faces.append((x, y, w, h, 'opencv', 0.5))
                    detection_methods_used.append('opencv')
            except Exception as e:
                st.warning(f"OpenCV detection failed: {str(e)}")
        
        if not detected_faces:
            return None, None, None, "No faces detected using any method."
        
        # Process detected faces
        image_with_faces = image.copy()  # For display purposes
        draw = ImageDraw.Draw(image_with_faces)
        
        face_regions = []  # Store face regions for later use
        
        for i, (x, y, w, h, method, conf) in enumerate(detected_faces):
            # Calculate quality metrics
            face_ratio = (w * h) / (original_w * original_h)
            aspect_ratio = w / h
            
            # Skip if face region is too small/large or has unusual proportions
            if face_ratio < 0.001 or face_ratio > 0.98 or aspect_ratio < 0.5 or aspect_ratio > 2.0:
                continue
            
            # Add margin to face region
            margin_x = int(w * 0.2)
            margin_y = int(h * 0.2)
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(original_w, x + w + margin_x)
            y2 = min(original_h, y + h + margin_y)
            
            # Extract face region from original image
            face_img = original_image.crop((x1, y1, x2, y2))
            face_img = enhance_image(face_img)
            
            # Store face region coordinates
            face_regions.append({
                'coordinates': (x1, y1, x2, y2),
                'method': method,
                'confidence': conf,
                'index': i + 1
            })
            
            # Draw rectangle and label only on the display version
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
            return None, None, None, "No valid faces detected in the image."
            
        return image_with_faces, face_images, original_image, face_regions
        
    except Exception as e:
        st.error(f"Error in face detection pipeline: {str(e)}")
        return None, None, None, str(e)

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

def validate_image_dimensions(image):
    """Validate and adjust image dimensions if needed."""
    try:
        if not isinstance(image, Image.Image):
            return None, "Invalid image type"
            
        width, height = image.size
        min_dimension = 224
        max_dimension = 1920
        
        # Check if resize is needed
        if width < min_dimension or height < min_dimension:
            # Scale up to minimum dimension
            ratio = min_dimension / min(width, height)
            new_size = (int(width * ratio), int(height * ratio))
            st.write(f"Scaling up image from {width}x{height} to {new_size}")
            return image.resize(new_size, Image.Resampling.LANCZOS), None
        elif width > max_dimension or height > max_dimension:
            # Scale down to maximum dimension
            ratio = max_dimension / max(width, height)
            new_size = (int(width * ratio), int(height * ratio))
            st.write(f"Scaling down image from {width}x{height} to {new_size}")
            return image.resize(new_size, Image.Resampling.LANCZOS), None
            
        return image, None
    except Exception as e:
        return None, f"Error validating dimensions: {str(e)}"

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
            st.info("✅ Successfully loaded image processor")
        except Exception as e:
            st.error(f"Failed to load image processor: {str(e)}")
            st.info("🔄 Attempting to install/update required packages...")
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
            st.info("✅ Successfully loaded Swin model")
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

def create_download_link(image, filename, link_text):
    """Create a download link for a PIL Image."""
    try:
        # Convert PIL image to bytes
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        # Create base64 string
        b64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        # Create HTML link
        href = f'<a href="data:image/png;base64,{b64}" download="{filename}">{link_text}</a>'
        return href
    except Exception as e:
        st.error(f"Error creating download link: {str(e)}")
        return None

def display_results_with_download(similar_faces):
    """Display results with download options and preview gallery."""
    try:
        if not similar_faces:
            st.warning("No similar faces found in the processed images.")
            return

        st.write(f"Found {len(similar_faces)} similar faces.")

        # Add bulk download button for original images
        st.write("### Download Options")
        zip_href = create_zip_download(similar_faces)
        if zip_href:
            st.markdown(zip_href, unsafe_allow_html=True)

        # Create preview gallery with individual downloads
        st.write("### Preview Gallery")
        st.write("Preview shows detected faces. Downloads will be original images without detection markers.")

        # Sort by similarity
        similar_faces.sort(key=lambda x: x['similarity'], reverse=True)

        # Display results in a grid with download buttons
        for i in range(0, len(similar_faces), 3):
            cols = st.columns(3)
            for j in range(3):
                if i + j < len(similar_faces):
                    match = similar_faces[i + j]
                    with cols[j]:
                        # Display marked version for preview
                        st.image(match['image_with_faces'],
                                caption=f"Match {i+j+1}\nSimilarity: {match['similarity']:.2f}",
                                width=200)
                        
                        # Create download button for original image
                        filename = f"original_match_{i+j+1}_{os.path.basename(match['image_path'])}"
                        href = create_download_link(
                            match['original_image'],
                            filename,
                            "⬇️ Download Original Image"
                        )
                        if href:
                            st.markdown(href, unsafe_allow_html=True)
                        
                        # Add match details
                        st.write(f"📁 Source: {os.path.basename(match['image_path'])}")
                        st.write(f"👤 Face: {match['face_index']}")
                        st.write(f"🎯 Matches Reference Face: {match['ref_face_index']}")
                        st.write(f"✨ Similarity: {match['similarity']:.2f}")

    except Exception as e:
        st.error(f"Error displaying results: {str(e)}")

def create_zip_download(similar_faces):
    """Create a ZIP file containing all matched original images."""
    try:
        # Create ZIP file in memory
        zip_buffer = io.BytesIO()
        with ZipFile(zip_buffer, 'w', ZipFile.ZIP_DEFLATED) as zip_file:
            for idx, match in enumerate(similar_faces):
                # Convert PIL image to bytes
                img_buffer = io.BytesIO()
                match['original_image'].save(img_buffer, format='PNG')  # Use original image
                img_buffer.seek(0)
                
                # Create filename with match details
                filename = f"original_match_{idx+1}_similarity_{match['similarity']:.2f}_{os.path.basename(match['image_path'])}"
                zip_file.writestr(filename, img_buffer.getvalue())

        # Create download link for ZIP
        zip_buffer.seek(0)
        b64 = base64.b64encode(zip_buffer.getvalue()).decode()
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        download_filename = f"original_matched_faces_{date_str}.zip"
        href = f'<a href="data:application/zip;base64,{b64}" download="{download_filename}">⬇️ Download All Original Images (ZIP)</a>'
        return href
    except Exception as e:
        st.error(f"Error creating ZIP download: {str(e)}")
        return None

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
        # Initialize results list
        results = []
        
        # Create debug expander
        with st.expander("Processing Debug Information", expanded=True):
            debug_container = st.container()

        def debug_log(message):
            with debug_container:
                st.write(message)

        # First, get reference faces
        debug_log("### Processing Reference Image")
        reference_image_marked, reference_faces, reference_original, face_regions = detect_multiple_faces(user_image)
        
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
                img = Image.open(os.path.join("extracted_images", img_path))
                # Detect faces in comparison image
                img_marked, faces, img_original, face_regions = detect_multiple_faces(img)
                if faces is None:
                    st.warning(f"No faces detected in {img_path}")
                    continue
                
                # Display image with detection markers
                st.image(img_marked, caption=f"Processed: {img_path}", width=400)
                st.write(f"Found {len(faces)} faces in {img_path}")
                
                # Store results for download
                results.append({
                    'path': img_path,
                    'marked_image': img_marked,
                    'original_image': img_original,
                    'faces': faces,
                    'regions': face_regions
                })
                
                # Add download button for original image
                st.write(f"Download original image (without markers):")
                href = create_download_link(
                    img_original,
                    f"original_{os.path.basename(img_path)}",
                    "⬇️ Download Original Image"
                )
                st.markdown(href, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error processing {img_path}: {str(e)}")
                continue

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
                st.error("Face detection cascade file not found. Please check OpenCV installation.")
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
        # Clean and normalize the folder path
        folder_path = folder_path.strip().strip('"').strip("'")  # Remove quotes
        folder_path = os.path.normpath(folder_path)  # Normalize path separators
        folder_path = os.path.abspath(folder_path)   # Convert to absolute path
        
        st.write(f"Processing folder: {folder_path}")
        
        # Check if path exists and is a directory
        if not os.path.exists(folder_path):
            st.error(f"❌ Folder not found: {folder_path}")
            st.info("💡 Tips for folder path:\n"
                   "1. Make sure the path exists\n"
                   "2. Use forward slashes (/) or double backslashes (\\\\)\n"
                   "3. Remove any quotes around the path")
            return
        
        if not os.path.isdir(folder_path):
            if os.path.isfile(folder_path) and folder_path.lower().endswith('.zip'):
                st.error("❌ It looks like you provided a ZIP file path. Please use the 'Upload ZIP File' option instead.")
                return
            else:
                st.error(f"❌ The path is not a directory: {folder_path}")
                return

        # Create placeholders for logging and progress
        log_placeholder = st.empty()
        progress_bar = st.progress(0)
        status_text = st.empty()

        def log_message(message):
            log_placeholder.markdown(message)

        # Get all image files recursively
        image_files = []
        valid_extensions = {'.jpg', '.jpeg', '.png'}
        
        log_message("🔍 Scanning for images...")
        
        try:
            for root, _, files in os.walk(folder_path):
                for file in files:
                    file_ext = os.path.splitext(file.lower())[1]
                    if file_ext in valid_extensions:
                        full_path = os.path.join(root, file)
                        try:
                            # Quick validation of image file
                            with Image.open(full_path) as img:
                                img.verify()
                            image_files.append(full_path)
                        except Exception as e:
                            log_message(f"⚠️ Skipping invalid image {file}: {str(e)}")
                            continue
        except Exception as e:
            st.error(f"Error scanning directory: {str(e)}")
            return

        if not image_files:
            st.warning("No valid images found in the selected folder.")
            st.info("💡 Make sure the folder contains supported image formats: .jpg, .jpeg, or .png")
            return

        total_files = len(image_files)
        log_message(f"📁 Found {total_files} images to process")

        # Initialize counters
        processed_count = 0
        successful_count = 0
        error_count = 0
        similar_faces = []

        # Process each image
        for img_path in image_files:
            try:
                # Update progress
                processed_count += 1
                progress = processed_count / total_files
                progress_bar.progress(progress)
                status_text.text(
                    f"Processing: {processed_count}/{total_files}\n"
                    f"Successful: {successful_count}\n"
                    f"Errors: {error_count}"
                )

                log_message(f"📷 Processing: {os.path.basename(img_path)}")

                # Load and validate image
                img, error = safe_image_open(img_path)
                if error:
                    log_message(f"❌ Error loading {os.path.basename(img_path)}: {error}")
                    error_count += 1
                    continue

                # Detect faces
                img_with_faces, faces, img_original, face_regions = detect_multiple_faces(img)
                if faces is None or len(faces) == 0:
                    log_message(f"⚠️ No faces detected in: {os.path.basename(img_path)}")
                    continue

                log_message(f"✅ Found {len(faces)} faces in {os.path.basename(img_path)}")

                # Compare each detected face with reference faces
                for face_idx, face_img in enumerate(faces):
                    try:
                        # Get embedding for detected face
                        face_embedding = get_face_embeddings(face_img)
                        if face_embedding is None:
                            continue

                        # Compare with reference faces
                        similarity = compare_faces(face_img, user_image)
                        
                        if similarity > 0.6:  # Adjusted threshold
                            successful_count += 1
                            similar_faces.append({
                                'image_path': img_path,
                                'face_index': face_idx + 1,
                                'ref_face_index': 1,
                                'similarity': similarity,
                                'image_with_faces': img_with_faces,
                                'original_image': img_original,
                                'face_regions': face_regions
                            })
                            log_message(f"🎯 Match found in {os.path.basename(img_path)} (Similarity: {similarity:.2f})")

                    except Exception as e:
                        log_message(f"⚠️ Error processing face {face_idx + 1} in {os.path.basename(img_path)}: {str(e)}")
                        continue

            except Exception as e:
                error_count += 1
                log_message(f"❌ Error processing {os.path.basename(img_path)}: {str(e)}")
                continue

        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        log_placeholder.empty()

        # Display results
        st.write("### Processing Summary")
        st.write(f"- Total images processed: {total_files}")
        st.write(f"- Successfully processed: {successful_count}")
        st.write(f"- Errors encountered: {error_count}")

        if similar_faces:
            display_results_with_download(similar_faces)
        else:
            st.warning("No similar faces found in the provided folder.")

    except Exception as e:
        st.error(f"Error processing folder: {str(e)}")
    finally:
        cleanup_temp_files()

def standardize_and_log_image(image, source_desc=""):
    """Standardize image format and log characteristics."""
    try:
        log_file = "image_processing_log.txt"
        with open(log_file, "a") as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"Processing image from: {source_desc}\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            
            # Log original image characteristics
            f.write(f"Original image type: {type(image)}\n")
            st.write(f"Original image type: {type(image)}")
            
            if isinstance(image, np.ndarray):
                f.write(f"Original array shape: {image.shape}\n")
                f.write(f"Original array dtype: {image.dtype}\n")
                st.write(f"Original array shape: {image.shape}")
                # Convert numpy array to PIL Image
                image = Image.fromarray(image)
            
            if isinstance(image, Image.Image):
                f.write(f"Original PIL mode: {image.mode}\n")
                f.write(f"Original size: {image.size}\n")
                st.write(f"Original PIL mode: {image.mode}")
                st.write(f"Original size: {image.size}")
                
                # Ensure RGB mode
                if image.mode != 'RGB':
                    f.write(f"Converting from {image.mode} to RGB\n")
                    image = image.convert('RGB')
                
                # Check and adjust size
                width, height = image.size
                max_dimension = 1920
                min_dimension = 224
                
                if width > max_dimension or height > max_dimension:
                    ratio = min(max_dimension/width, max_dimension/height)
                    new_size = (int(width * ratio), int(height * ratio))
                    f.write(f"Resizing from {width}x{height} to {new_size[0]}x{new_size[1]}\n")
                    image = image.resize(new_size, Image.Resampling.LANCZOS)
                elif width < min_dimension or height < min_dimension:
                    ratio = min_dimension / min(width, height)
                    new_size = (int(width * ratio), int(height * ratio))
                    f.write(f"Upscaling from {width}x{height} to {new_size[0]}x{new_size[1]}\n")
                    image = image.resize(new_size, Image.Resampling.LANCZOS)
                
                # Log final characteristics
                f.write(f"Final mode: {image.mode}\n")
                f.write(f"Final size: {image.size}\n")
                st.write(f"Final mode: {image.mode}")
                st.write(f"Final size: {image.size}")
                
                return image, None
            else:
                error_msg = f"Unsupported image type: {type(image)}"
                f.write(f"Error: {error_msg}\n")
                return None, error_msg
                
    except Exception as e:
        error_msg = f"Error standardizing image: {str(e)}"
        with open(log_file, "a") as f:
            f.write(f"Error: {error_msg}\n")
            f.write(f"Stack trace:\n{traceback.format_exc()}\n")
        return None, error_msg

def main():
    """Main application function."""
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
            return

        # Reference image upload
        st.write("### Step 1: Upload Reference Image")
        user_image_file = st.file_uploader(
            "Upload your reference image",
            type=["jpg", "jpeg", "png"],
            help="Upload a photo containing the face(s) you want to find"
        )

        if user_image_file:
            try:
                # First, open and validate the image
                user_image, error = safe_image_open(user_image_file)
                if error:
                    st.error(error)
                    return
                
                # Validate dimensions
                user_image, error = validate_image_dimensions(user_image)
                if error:
                    st.error(error)
                    return
                
                # Log image characteristics after processing
                st.write("Final image characteristics:")
                st.write(f"Mode: {user_image.mode}")
                st.write(f"Size: {user_image.size}")
                
                # Process reference image
                reference_image_marked, reference_faces, reference_original, face_regions = detect_multiple_faces(user_image)
                if reference_faces is None:
                    st.error("No valid faces detected in reference image.")
                    st.stop()
                
                # Display processed reference image with markers
                st.image(reference_image_marked, caption="Reference Image (Detected Faces)", width=400)
                st.write(f"Detected {len(reference_faces)} faces in reference image")
                
                # Store original reference image for later use
                st.session_state['reference_original'] = reference_original
                st.session_state['reference_faces'] = reference_faces
                st.session_state['face_regions'] = face_regions
                
                # Create download button for original reference image
                if reference_original:
                    st.write("Download original reference image (without markers):")
                    href = create_download_link(
                        reference_original,
                        "reference_image_original.png",
                        "⬇️ Download Original Reference Image"
                    )
                    st.markdown(href, unsafe_allow_html=True)
                
                # Step 2: Choose input method
                st.write("### Step 2: Choose Input Method")
                input_method = st.radio(
                    "Select how you want to input images to search:",
                    ["Upload ZIP File", "Browse Local Folder"]
                )

                if input_method == "Upload ZIP File":
                    process_zip_file(user_image, processor, model)
                else:
                    st.write("### Select Local Folder")
                    st.write("""
                    Enter the full path to the folder containing your images.
                    
                    Examples:
                    - Windows: C:\\Users\\YourName\\Pictures
                    - Mac/Linux: /home/username/pictures
                    
                    Make sure you have read permissions for the folder.
                    """)
                    
                    folder_path = st.text_input(
                        "Folder path",
                        help="Enter the complete path to your images folder"
                    )
                    
                    if folder_path:
                        # Validate folder path
                        folder_path = os.path.expanduser(folder_path)  # Expand ~ if present
                        folder_path = os.path.abspath(folder_path)     # Convert to absolute path
                        
                        if not os.path.exists(folder_path):
                            st.error(f"Folder not found: {folder_path}")
                        elif not os.path.isdir(folder_path):
                            st.error(f"The path is not a directory: {folder_path}")
                        else:
                            # Add a button to start processing
                            if st.button("Process Folder"):
                                with st.spinner("Processing images..."):
                                    process_local_folder(folder_path, user_image, processor, model)

            except Exception as e:
                st.error(f"Error: {str(e)}")
            finally:
                cleanup_temp_files()
                
    except Exception as e:
        st.error(f"Application error: {str(e)}")
    finally:
        cleanup_temp_files()

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

if __name__ == "__main__":
    main()
