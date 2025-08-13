📜 Legal Document Analyzer
An AI-powered legal assistant that analyzes uploaded legal documents (contracts, NDAs, policies, compliance docs) and answers clause-specific queries with references — ensuring strict context-based responses without outside assumptions.

Built using LangChain, Groq LLM, Hugging Face embeddings, and FAISS vector database for accurate document retrieval and analysis.

🚀 Features
📂 Upload PDFs – Supports multi-page legal documents.

🔍 Clause-level search – Retrieves exact clauses and sections related to your query.

📑 Precise references – Always cites section numbers and verbatim text.

🛡️ Context-restricted answers – Never uses knowledge outside the uploaded document.

💬 Conversational history – Maintains context across multiple queries.

⚡ Fast retrieval – Powered by FAISS vector database.

🛠️ Tech Stack
Frontend & App Framework: Streamlit

LLM: Groq using gemma2-9b-it model

Framework: LangChain

Embeddings: Hugging Face – all-MiniLM-L6-v2

Vector DB: FAISS

PDF Processing: PyMuPDF

📦 Installation
1.**Clone the repository**


git clone https://github.com/shravanssr11/Legal-Document-Analyzer-AI-Powered-Clause-Compliance-Insights

cd legal-document-analyzer

2.**Create virtual environment**

python -m venv venv
source venv/bin/activate   # For Mac/Linux
venv\Scripts\activate      # For Windows

3.**Install dependencies**

pip install -r requirements.txt

4.**Set up Enviornment Variables**

Create a .env file in the root directory and add:

GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_api_key

▶️ Usage
Run the Streamlit app:

streamlit run app.py