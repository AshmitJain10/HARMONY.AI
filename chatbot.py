import gradio as gr
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# Initialize LLM
from langchain_groq import ChatGroq  # Assuming ChatGroq is available

def initial_llm():
    llm = ChatGroq(
        temperature=0,
        groq_api_key="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",
        model_name="llama-3.3-70b-versatile"
    )
    return llm

def create_db():
    loader = DirectoryLoader("./", glob='*.pdf', loader_cls=PyPDFLoader)
    docs = loader.load()
    txt_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    txts = txt_splitter.split_documents(docs)
    embedd = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    call_db = Chroma.from_documents(txts, embedd, persist_directory='./chroma_db')
    call_db.persist()
    print("ChromaDB created and data has been saved!!")
    return call_db

def setup_qachain(call_db, llm):
    retrieve = call_db.as_retriever()
    prompt_tmplt = """Sharing your thoughts is a brave step. I'll respond with empathy and understanding:
    {context}
    User: {question}
    Chatbot: """
    PROMPT = PromptTemplate(template=prompt_tmplt, input_variables=['context', 'question'])
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retrieve,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

# Load or create database
db_path = './chroma_db'
if not os.path.exists(db_path):
    call_db = create_db()
else:
    embedd = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    call_db = Chroma(persist_directory=db_path, embedding_function=embedd)

llm = initial_llm()
qa_chain = setup_qachain(call_db, llm)

def chatbot_response(query, history):
    if query.lower() == "exit":
        return "Take care of yourself. I am here to help anytime you need. Goodbye! 😊"
    response = qa_chain.run(query)
    history.append((query, response))
    return "", history

# Gradio Interface
def launch_chatbot():
    with gr.Blocks() as demo:
        gr.Markdown("# HARMONY.AI - 🧠 A Mental Health AI Chatbot")
        gr.Markdown("Chatbot powered by LangChain, ChromaDB, and Llama-3. Ask anything!!!")
        chatbot = gr.Chatbot()
        query_input = gr.Textbox(placeholder="Ask a question...")
        submit_button = gr.Button("Send")
        
        submit_button.click(chatbot_response, inputs=[query_input, chatbot], outputs=[query_input, chatbot])
        
        gr.Markdown("---")
        gr.Markdown("### Made By: Ashmit Jain")
        gr.Markdown("[GitHub](https://github.com/AshmitJain10/Mental-Health-AI-Chatbot)")
        
    demo.launch(share=True)

launch_chatbot()
