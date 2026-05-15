import fitz
import spacy
import pandas as pd
import streamlit as st

# Load NLP model
nlp = spacy.load("en_core_web_sm")

st.title("PDF NER Extractor")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    # Open PDF from uploaded file
    doc_pdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")

    # Extract text
    text = ""
    for page in doc_pdf:
        text += page.get_text()

    st.subheader("Extracted Text")
    st.text_area("PDF Content", text, height=300)

    # Apply NLP
    doc = nlp(text)

    # Store entities
    data = []
    st.subheader("Named Entities")
    for ent in doc.ents:
        st.write(f"{ent.text} -> {ent.label_}")
        data.append([ent.text, ent.label_])

    # Save to CSV
    df = pd.DataFrame(data, columns=["Entity", "Label"])
    df.to_csv("ner_output.csv", index=False)
    st.success("NER results saved to ner_output.csv")

    # Display CSV
    st.subheader("NER DataFrame")
    st.dataframe(df)

    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="ner_output.csv",
        mime="text/csv"
    )