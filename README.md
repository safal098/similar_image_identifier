Here's a ready-to-use `README.md` file tailored for your `app.py`, which is a Streamlit-based application for multi-face detection and comparison using DeepFace, MTCNN, OpenCV, and Hugging Face models:

---

```markdown
# 🔍 Multi-Face Detection and Comparison App

This is a **Streamlit** web app that detects and compares multiple faces in images using state-of-the-art models like **MTCNN**, **DeepFace**, **OpenCV**, and **Swin Transformer (Hugging Face)**.

---

## 🚀 Features

- Detects multiple faces in group or individual photos
- Compares reference face(s) against faces in bulk image uploads
- Supports both:
  - Uploading ZIP files (max 200MB)
  - Browsing local folders for images
- Displays detection confidence and face match similarity
- Download detected matches (individually or in bulk)

---

## 📦 Requirements

Before running the app, make sure the following packages are installed:

```bash
pip install streamlit opencv-python-headless deepface mtcnn pillow transformers[torch] torch python-dotenv
```

---

## 🌐 Environment Setup

Create a `.env` file in the root directory and add your Hugging Face token:

```env
HUGGING_FACE_TOKEN=your_huggingface_token_here
```

---

## ▶️ How to Run

Simply run the Streamlit app:

```bash
streamlit run app.py
```

Then open the provided local URL in your browser.

---

## 📁 Usage Instructions

1. **Upload Reference Image:**  
   - Upload a face image to compare against others

2. **Choose Input Method:**  
   - **ZIP File**: Upload a .zip containing `.jpg`, `.jpeg`, `.png` images  
   - **Local Folder**: Input path to a local folder on your system

3. **Get Results:**  
   - View similar face matches
   - Download individual or all matched original images

---

## 💡 Notes

- Faces are detected using a hybrid of **MTCNN**, **DeepFace backends**, and **OpenCV Haar Cascades**
- Face embeddings are computed via **DeepFace (VGG-Face)** and **Swin-Tiny Transformer**
- ZIP files larger than **200MB** are not supported (Streamlit limitation)

---

## 📸 Example Use Cases

- Deduplicating family/group photos
- Identifying faces in large image sets
- Matching photos of the same person from different sources

---

## 🛠️ Troubleshooting

- If MTCNN or DeepFace are not installed, the app will guide you to install them
- Ensure `.env` is properly configured or the Swin Transformer model won't load

---

## 📝 License

This project is for educational and research use only.

---

## 🤝 Contributions

Feel free to open issues or pull requests. Improvements are always welcome!
```

---

Let me know if you'd like a version with images, badges, or examples added too!
