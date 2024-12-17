

!pip install pdfminer.six
!pip install streamlit
!pip install pickle5
!pip install langchain
!pip install langchain-groq
!pip install faiss-cpu
!pip install huggingface_hub

!pip install -U langchain-community

!pip install pdf2image

!pip install fitz

!pip install pytesseract

!pip install pymupdf

!pip install sentence-transformers



from pdfminer.high_level import extract_text
import fitz  # PyMuPDF for image extraction
import os
import pickle
import time
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from google.colab import files


llm = ChatGroq(
    temperature=0,
    groq_api_key="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",
    model_name="llama-3.1-70b-versatile"
)


file_path = "faiss_store.pkl"


uploaded_files = files.upload()

def process_pdfs():
    all_text = ""
    image_dir = "extracted_images"

   
    os.makedirs(image_dir, exist_ok=True)

   
    for uploaded_file in uploaded_files.keys():
      
        print(f"Processing text from {uploaded_file}...")
        extracted_text = extract_text(uploaded_file)
        all_text += extracted_text + "\n"

       
        print(f"Extracting images from {uploaded_file}...")
        doc = fitz.open(uploaded_file)
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images(full=True)

            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_path = os.path.join(image_dir, f"{uploaded_file}_page{page_num+1}_img{img_index+1}.{image_ext}")

               
                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)
        doc.close()

    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text_chunks = text_splitter.split_text(all_text)

    
    print("Building embeddings and FAISS vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(text_chunks, embeddings)

   
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore, f)

    print("Processing complete! Text and images extracted.")
    print(f"Images are saved in: {image_dir}")
    print("FAISS index saved to disk.")


process_pdfs()


query = input("Ask a Question: ")
if query:
    if os.path.exists(file_path):
        print("Loading FAISS index...")
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)

       
        chain = RetrievalQA.from_llm(llm=llm, retriever=vectorstore.as_retriever())

       
        print("Processing your query...")
        result = chain.run(query)

       
        print("Answer:")
        print(result)
