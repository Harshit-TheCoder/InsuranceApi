import os
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
# from langchain_chroma import Chroma
from langchain_community.vectorstores import Chroma
from PolicyNames.edelweiss_policy_names import edelweiss_policy

from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

SESSION_ID = "default_session"

# Load LLM
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Add your key in .env as GROQ_API_KEY
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="Gemma2-9b-It")

# Load embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# Create vector store
PERSIST_DIR = "Edelweiss_Embeddings"

if os.path.exists(PERSIST_DIR):
    vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
else:
    DOCS_FOLDER = "documents/Edelweiss"
    documents = []
    for file in os.listdir(DOCS_FOLDER):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(DOCS_FOLDER, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=PERSIST_DIR)
    vectorstore.persist()
retriever = vectorstore.as_retriever()

# Create retriever chain
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question which reference context in the chat history, "
    "formulate a standalone question which can be understood without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

# Answer question
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question."
    "If you don't know the answer, SAY 'Sorry, but I failed to find the relevant details from my database with the details you provided.' and DISPLAY {edelweiss_policy}."
    "YOUR ARE TRAINED WITH THESE POLICIES {edelweiss_policy}"
    "ALWAYS PRIORITIZE SEARCHING YOUR VECTOR DATABASE RATHER THAN GIVING GENERAL ANSWERS"
    "\n\n{context}. ANSWER IN JSON FORMAT ONLY."
)
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Setup chat history
session_store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)

# # CLI Interface
# print("Welcome to PDF RAG Q&A. Ask questions (type 'exit' to quit).")
# while True:
#     question = input("\nYou: ")
#     if question.lower() == "exit":
#         break
#     response = conversational_rag_chain.invoke(
#         {"input": question},
#         config={"configurable": {"session_id": SESSION_ID}}
#     )
#     print("Assistant:", response['answer'])
