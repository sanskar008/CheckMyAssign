import fitz  # PyMuPDF
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io
import zipfile

# --------- Helper Functions ---------


# Extract text from a single PDF
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text


# Create downloadable CSV/Excel
def convert_df(df, filetype="csv"):
    if filetype == "csv":
        return df.to_csv(index=True).encode("utf-8")
    elif filetype == "excel":
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=True, sheet_name="Plagiarism Report")
        return output.getvalue()


# --------- UI Styling ---------
st.set_page_config(
    page_title="CheckMyAssign - Plagiarism Checker", page_icon="📄", layout="wide"
)

# Custom CSS for style
st.markdown(
    """
    <style>
    .main {background-color: #181c24; color: #fff;}
    .stButton>button {background-color: #e63946; color: white; font-weight: bold;}
    .stDownloadButton>button {background-color: #457b9d; color: white; font-weight: bold;}
    .stSlider > div[data-baseweb=\"slider\"] {color: #e63946;}
    .stDataFrame {background-color: #222;}
    </style>
""",
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.image(
        "https://cdn-icons-png.flaticon.com/512/3135/3135715.png",
        width=120,
    )
    st.markdown("# 📄 CheckMyAssign")
    st.markdown("**Batch Assignment Plagiarism Checker**")
    st.markdown("---")
    st.markdown(
        "**Instructions:**\n- Upload individual PDF files or a ZIP file containing PDFs.\n- Filenames should be student IDs.\n- Adjust the plagiarism threshold as needed.\n- Download the report as CSV or Excel."
    )
    st.markdown("---")
    st.info("Made with ❤️ for educators.")

# Main Title
st.markdown(
    """
<h1 style='text-align: center; color: #e63946; font-size: 2.8rem;'>📄 CheckMyAssign</h1>
<h3 style='text-align: center; color: #fff;'>Batch Assignment Plagiarism Checker</h3>
""",
    unsafe_allow_html=True,
)

# Upload section in columns
col1, col2 = st.columns(2)
with col1:
    uploaded_files = st.file_uploader(
        "Upload student assignments (PDFs)",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_upload",
    )
with col2:
    uploaded_zip = st.file_uploader(
        "Or upload a folder as a ZIP file (containing PDFs)",
        type=["zip"],
        accept_multiple_files=False,
        key="zip_upload",
    )

if uploaded_files or uploaded_zip:
    all_files = list(uploaded_files) if uploaded_files else []
    zip_pdf_files = []
    if uploaded_zip is not None:
        with zipfile.ZipFile(uploaded_zip) as z:
            for name in z.namelist():
                if name.lower().endswith(".pdf"):
                    zip_pdf_files.append((name, z.read(name)))
        if not all_files and not zip_pdf_files:
            st.error("No PDF files found in the uploaded ZIP.")
            st.stop()
        st.success(
            f"✅ {len(all_files) + len(zip_pdf_files)} files (including {len(zip_pdf_files)} from ZIP) uploaded successfully"
        )
    elif all_files:
        st.success(f"✅ {len(all_files)} files uploaded successfully")
    else:
        st.warning("Please upload PDF files or a ZIP containing PDFs.")
        st.stop()

    texts = []
    student_ids = []

    # Extract text from each PDF (uploaded individually)
    for file in all_files:
        file.seek(0)
        text = extract_text_from_pdf(file)
        texts.append(text)
        student_ids.append(file.name.replace(".pdf", ""))  # filename = student ID

    # Extract text from each PDF in ZIP
    for name, data in zip_pdf_files:
        text = extract_text_from_pdf(io.BytesIO(data))
        texts.append(text)
        student_ids.append(name.replace(".pdf", ""))

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Cosine similarity matrix
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Convert to DataFrame with student IDs
    df = pd.DataFrame(similarity_matrix * 100, index=student_ids, columns=student_ids)
    # Remove self-comparison by setting diagonal to None
    for idx in range(len(student_ids)):
        df.iloc[idx, idx] = None

    # --------- Show Results ---------
    st.subheader("Plagiarism Percentage Matrix")
    st.write("Values represent similarity % between assignments (student IDs).")

    st.dataframe(df.style.background_gradient(cmap="Reds").format("{:.2f}"))

    # High similarity pairs
    st.subheader("Possible Plagiarized Pairs")
    threshold = st.slider("Select plagiarism threshold (%)", 50, 100, 70)
    reported = set()

    for i in range(len(student_ids)):
        for j in range(len(student_ids)):
            if i == j:
                continue  # Skip self-comparison
            if j < i:
                continue  # Avoid duplicate pairs (since matrix is symmetric)
            sim = df.iloc[i, j]
            if sim > threshold:
                pair = (student_ids[i], student_ids[j])
                if pair not in reported:
                    st.write(
                        f"📌 **{student_ids[i]}** ↔ **{student_ids[j]}** → {sim:.2f}% similar"
                    )
                    reported.add(pair)

    # --------- Download Report ---------
    st.subheader("📥 Download Report")

    csv_data = convert_df(df, "csv")
    st.download_button(
        label="⬇️ Download CSV Report",
        data=csv_data,
        file_name="plagiarism_report.csv",
        mime="text/csv",
    )

    excel_data = convert_df(df, "excel")
    st.download_button(
        label="⬇️ Download Excel Report",
        data=excel_data,
        file_name="plagiarism_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
