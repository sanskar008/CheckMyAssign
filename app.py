# save as app.py
import fitz  # PyMuPDF
import streamlit as st
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text


# Function to generate highlighted diff
def get_diff_html(original, student):
    diff = difflib.HtmlDiff()
    return diff.make_table(
        original.splitlines(),
        student.splitlines(),
        fromdesc="Original",
        todesc="Student",
        context=True,
        numlines=3,
    )


# Streamlit UI
st.title("📄 CopyCatch - Assignment Similarity Checker")
st.write(
    "Upload the **original assignment** and a **student assignment** to check similarity."
)

# File uploads
original_file = st.file_uploader("Upload Original Assignment (PDF)", type=["pdf"])
student_file = st.file_uploader("Upload Student Assignment (PDF)", type=["pdf"])

if original_file and student_file:
    # Extract text
    original_text = extract_text_from_pdf(original_file)
    student_text = extract_text_from_pdf(student_file)

    # Vectorize text for similarity
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([original_text, student_text])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100

    # Show result
    st.subheader("📊 Similarity Score")
    st.write(f"Similarity: **{similarity:.2f}%**")

    if similarity > 85:
        st.error("⚠️ Highly similar! Possible copy.")
    elif similarity > 60:
        st.warning("⚠️ Partial similarity detected.")
    else:
        st.success("✅ Low similarity. Looks original.")

    # Show comparison
    st.subheader("📑 Text Comparison")
    diff_html = get_diff_html(original_text, student_text)
    st.components.v1.html(diff_html, height=500, scrolling=True)
