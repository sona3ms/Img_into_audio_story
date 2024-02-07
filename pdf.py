import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
import os

# Set OpenAI API key
# os.environ["OPENAI_API_KEY"] = "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"
openai_api_key = os.environ.get("OPENAI_API_KEY")
# Function to extract text from PDF
def extract_text_from_pdf(file):
    raw_text = ''
    reader = PdfReader(file)
    for page in reader.pages:
        raw_text += page.extract_text()
    return raw_text

# Streamlit app
def main():
    st.title("PDF Question Answering")

    # Upload PDF file
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file is not None:
        # Extract text from PDF
        raw_text = extract_text_from_pdf(uploaded_file)
        
        # Split text
        text_splitter = CharacterTextSplitter(        
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        texts = text_splitter.split_text(raw_text)
        
        # Obtain embeddings
        embeddings = OpenAIEmbeddings()
        docsearch = FAISS.from_texts(texts, embeddings)
        
        # Load question answering chain
        chain = load_qa_chain(OpenAI(), chain_type="stuff")

        # User input: question
        question = st.text_input("Enter your question")

        # Query processing
        if st.button("Get Answer"):
            if question:
                docs = docsearch.similarity_search(question)
                answers = chain.run(input_documents=docs, question=question)
                st.write("Answer:", answers)

if __name__ == "__main__":
    main()
