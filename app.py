import fitz  # PyMuPDF
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import base64


# Extract text from PDF
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text


# Convert PDF to base64 for embedding
def display_pdf(file):
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    return pdf_display


# Streamlit App
st.title("📄 CopyCatch - Assignment Similarity Checker")

original_file = st.file_uploader("Upload Original Assignment (PDF)", type=["pdf"])
student_file = st.file_uploader("Upload Student Assignment (PDF)", type=["pdf"])

if original_file and student_file:
    # Extract text (need fresh file handles)
    original_file.seek(0)
    original_text = extract_text_from_pdf(original_file)

    student_file.seek(0)
    student_text = extract_text_from_pdf(student_file)

    # Compute similarity
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([original_text, student_text])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100

    # Show score
    st.subheader("📊 Similarity Score")
    st.write(f"Similarity: **{similarity:.2f}%**")

    if similarity > 85:
        st.error("⚠️ Highly similar! Possible copy.")
    elif similarity > 60:
        st.warning("⚠️ Partial similarity detected.")
    else:
        st.success("✅ Low similarity. Looks original.")

    # Reset file pointer again to show PDFs
    original_file.seek(0)
    student_file.seek(0)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📑 Original PDF")
        st.markdown(display_pdf(original_file), unsafe_allow_html=True)

    with col2:
        st.subheader("📑 Student PDF")
        st.markdown(display_pdf(student_file), unsafe_allow_html=True)
