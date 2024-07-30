
Financial News Analysis Tool
Welcome to the Financial News Analysis Tool, a robust utility for extracting valuable insights from financial news articles. Whether you're interested in stock market trends or specific companies, this tool allows you to utilize advanced technologies for efficient information retrieval.



Key Features
URL Input: Load URLs or upload text files containing URLs to retrieve article content.
LangChain Integration: Process article content through LangChain's UnstructuredURL Loader for enhanced data understanding.
Embedding Vector Construction: Utilize OpenAI's embeddings to construct embedding vectors.
FAISS Similarity Search: Leverage FAISS, a robust similarity search library, for swift and effective retrieval of relevant information.
ChatGPT Interaction: Interact with ChatGPT (LLM) by inputting queries and receiving answers along with source URLs.
How to Use
Run the Application:

Execute the main.py script to open the web app in your browser.
Input URLs:

Enter URLs directly in the sidebar or upload a text file containing URLs.
Process URLs:

Click "Process URLs" to initiate data loading and processing.
The system will perform text splitting, generate embedding vectors, and index them efficiently using FAISS.
Index Storage:

The FAISS index is stored locally in a pickle file (faiss_store_openai.pkl) for future use.
Query the Tool:

Ask questions and receive answers based on the analyzed news articles.
Example URLs for Reference
Tata Motors & Mahindra Gain Certificates for Production-Linked Payouts
Tata Motors Launches Punch ICNG
Buy Tata Motors, Target of Rs 743: KR Choksey
Project Structure
main.py: The main Streamlit application script.
requirements.txt: A list of required Python packages for the project.
faiss_store_openai.pkl: A pickle file to store the FAISS index.
.env: Configuration file for storing the OpenAI API key.
