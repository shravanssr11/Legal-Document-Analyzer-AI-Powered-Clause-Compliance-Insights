import streamlit as st
import fitz  # PyMuPDF
st.markdown("""
# 📜 Legal Document Analyzer
#### Your AI-powered assistant for clause-level legal insights

Upload contracts, NDAs, compliance documents, or policies, and get:
- 🔍 Precise answers to your queries  
- 📑 Exact clause & section references  
- 🛡️ Strict context-based analysis (no outside assumptions)  
""")


with st.sidebar:
    file=st.file_uploader("upload your doc",type="pdf")

query = st.text_input(" ", label_visibility="collapsed",
                      placeholder="e.g., What is the late payment penalty in this contract?")
# extracting text from uploaded doc

def extract_text_from_pdf(file_bytes):
    text = ""
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text
if file is not None:
    bytes_data =file.read()

    text = extract_text_from_pdf(bytes_data)

# setting up google api key

    import os
    from dotenv import load_dotenv
    load_dotenv()
    os.environ['HF_TOKEN']=os.getenv('HF_TOKEN')
    groq_api_key=os.getenv('GROQ_API_KEY')

# splitting doc
    headers_to_split_on = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3"),
             ]


    from langchain_text_splitters import MarkdownHeaderTextSplitter
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    docs = splitter.split_text(text)


# create embeddings
    import asyncio
    try:
       asyncio.get_running_loop()
    except RuntimeError:
      asyncio.set_event_loop(asyncio.new_event_loop())

    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    
# create a vector store

    from langchain_community.vectorstores import FAISS
    db=FAISS.from_documents(docs,embeddings)
    


# creating chat prompt template

    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

    legal_analyzer_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a highly knowledgeable legal assistant specializing in analyzing legal documents 
    such as contracts, NDAs, compliance policies, and regulatory texts.

    You will be provided with:
    1. A user query related to legal documents.
    2. Retrieved document excerpts from a trusted database (`retrieved_context`).
    3. Metadata about the document source and section.

    Your tasks:
    - Provide a precise, legally sound answer to the query.
    - Use **only** the `retrieved_context` to formulate the answer. Do **not** use outside knowledge unless explicitly stated.
    - Always reference the exact clause/section number when possible.
    - If the answer is not in the `retrieved_context`, say: "The provided documents do not contain relevant information."
    - Maintain a formal legal tone.

    ---
    *Retrieved Context:*
    {context}

    ---
    *User Query:*
    {input}

    ---
    *Instructions:*
    1. Summarize the relevant legal information in plain language suitable for a professional audience.
    2. Include verbatim clause text where necessary.
    3. Provide the section or paragraph reference for every claim.
    4. If applicable, warn the user about possible ambiguities or conflicting clauses.
    5. Never give personal legal advice — only interpret the provided documents.

    ---
    *Answer:*"""
        ),
        MessagesPlaceholder(variable_name="history")
    ])



# define llm

    from langchain_groq import ChatGroq
    llm=ChatGroq(model="gemma2-9b-it",groq_api_key=groq_api_key,max_tokens=500)
    retriver=db.as_retriever()

# setting up chain

    from langchain.chains.combine_documents import create_stuff_documents_chain
    combine_docs_chain = create_stuff_documents_chain(llm, legal_analyzer_prompt)
    from langchain.chains import create_retrieval_chain
    retrieval_chain=create_retrieval_chain(retriver,combine_docs_chain)

#   creating get session history function

    from langchain_core.chat_history import BaseChatMessageHistory
    from langchain_core.chat_history import InMemoryChatMessageHistory

    store={}

    def get_session_history(session_id:str)->BaseChatMessageHistory:
        if session_id not in store:
            store[session_id]=InMemoryChatMessageHistory()
        return store[session_id]

# setting up history
    from langchain_core.runnables import RunnableWithMessageHistory

    with_message_history=RunnableWithMessageHistory(retrieval_chain,get_session_history,input_messages_key="input",history_messages_key="history")

    config={"configurable":{"session_id":"default"}}

#get response from llm
    if query:

        response=with_message_history.invoke({"input":query},config)

    #displaying result

        result=response['answer']
        st.write(result)
        with st.expander('Document similarity result'):
            for i,doc in enumerate(response['context']):
                st.write(doc.page_content)
                    

