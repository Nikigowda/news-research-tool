import os
import streamlit as st
import pickle
import time
from dotenv import load_dotenv
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Load environment variables
load_dotenv()  # Load environment variables from .env file

# Retrieve OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize the OpenAI model
llm = OpenAI(temperature=0.9, max_tokens=500, openai_api_key=openai_api_key)

# Initialize Streamlit app
st.title("Financial News Research Tool 📈")
st.sidebar.title("News Article URLs")

# Input URLs
urls = [
    "https://www.moneycontrol.com/news/business/tata-motors-mahindra-gain-certificates-for-production-linked-payouts-11281691.html",
    "https://www.moneycontrol.com/news/business/tata-motors-launches-punch-icng-price-starts-at-rs-7-1-lakh-11098751.html",
    "https://www.moneycontrol.com/news/business/stocks/buy-tata-motors-target-of-rs-743-kr-choksey-11080811.html"
]

# Button to process URLs
process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

# Placeholder for status updates
main_placeholder = st.empty()

if process_url_clicked:
    try:
        # Load data from URLs
        loader = UnstructuredURLLoader(urls=urls)
        main_placeholder.text("Data Loading...Started...✅✅✅")
        data = loader.load()
        
        # Debugging: Display loaded data
        st.write(f"Loaded data: {data}")
        
        if not data:
            main_placeholder.text("No data loaded. Check the URLs or loader configuration.")
        else:
            # Split data into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=1000
            )
            main_placeholder.text("Text Splitting...Started...✅✅✅")
            docs = text_splitter.split_documents(data)
            
            # Debugging: Display split documents
            st.write(f"Split documents: {docs}")

            if not docs:
                main_placeholder.text("No documents created. Check the data and splitting configuration.")
            else:
                # Create embeddings and save to FAISS index
                embeddings = OpenAIEmbeddings()
                vectorstore_openai = FAISS.from_documents(docs, embeddings)
                main_placeholder.text("Embedding Vector Building...✅✅✅")
                time.sleep(2)

                # Save the FAISS index to a pickle file
                with open(file_path, "wb") as f:
                    pickle.dump(vectorstore_openai, f)
                main_placeholder.text("Data Processing Complete! You can now query the model.")

    except Exception as e:
        main_placeholder.text(f"Error occurred: {str(e)}")

# Query input and response display
query = st.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        try:
            with open(file_path, "rb") as f:
                vectorstore = pickle.load(f)
                chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
                result = chain({"question": query}, return_only_outputs=True)

                # Display the answer
                st.header("Answer")
                st.write(result["answer"])

                # Display sources, if available
                sources = result.get("sources", "")
                if sources:
                    st.subheader("Sources:")
                    sources_list = sources.split("\n")  # Split the sources by newline
                    for source in sources_list:
                        st.write(source)
        except Exception as e:
            st.text(f"Error occurred while querying: {str(e)}")
    else:
        st.text("FAISS index not found. Please process URLs first.")
