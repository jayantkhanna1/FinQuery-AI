import os
import gradio as gr
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
load_dotenv()

class DocumentChatbot:
    def __init__(self, openai_api_key):
        """
        Initialize the chatbot with document processing capabilities.
        
        :param openai_api_key: Your OpenAI API key
        """
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # Initialize embeddings and LLM
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1)
        
        # Initialize vector store
        self.vector_store = None
        self.qa_chain = None
    
    def load_documents(self, file_paths):
        """
        Load and process multiple PDF documents.
        
        :param file_paths: List of PDF file paths
        """
        # Text splitter for creating chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Aggregate all document chunks
        all_splits = []
        
        # Load and split each document
        processing_log = []
        for file_path in file_paths:
            try:
                # Load PDF
                loader = PyPDFLoader(file_path)
                documents = loader.load()
                
                # Split into chunks
                splits = text_splitter.split_documents(documents)
                all_splits.extend(splits)
                
                processing_log.append(f"Successfully loaded and processed: {file_path}")
            
            except Exception as e:
                processing_log.append(f"Error processing {file_path}: {e}")
        
        # Create vector store
        if all_splits:
            self.vector_store = FAISS.from_documents(all_splits, self.embeddings)
            
            # Create retrieval QA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(search_kwargs={"k": 3})
            )
            
            return "\n".join(processing_log)
        else:
            return "No documents were successfully processed."
    
    def ask_question(self, question):
        """
        Ask a question about the loaded documents.
        
        :param question: User's question
        :return: AI's response
        """
        if not self.vector_store or not self.qa_chain:
            return "No documents loaded. Please upload documents first."
        
        try:
            # Get response from QA chain
            response = self.qa_chain.run(question)
            return response
        
        except Exception as e:
            return f"Error processing question: {e}"

def create_ui():
    """
    Create Gradio interface for the document chatbot.
    """
    # Check if OpenAI API key is set
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")
    
    # Create chatbot instance
    chatbot = DocumentChatbot(os.environ["OPENAI_API_KEY"])
    
    def process_documents(files):
        """
        Process uploaded PDF files.
        
        :param files: List of uploaded file objects
        :return: Processing log
        """
        # Extract file paths from uploaded files
        file_paths = [file.name for file in files]
        
        # Load and process documents
        return chatbot.load_documents(file_paths)
    
    def ask_question(message, chat_history):
        """
        Handle user questions about uploaded documents.
        
        :param message: User's question
        :param chat_history: Previous chat interactions
        :return: AI's response and updated chat history
        """
        # Get response from chatbot
        response = chatbot.ask_question(message)
        
        # Update chat history
        chat_history.append((message, response))
        
        return "", chat_history
    
    # Create Gradio interface
    with gr.Blocks() as demo:
        gr.Markdown("# PDF Document Chatbot")
        
        # Document Upload Component
        with gr.Row():
            file_input = gr.File(
                file_types=[".pdf"], 
                file_count="multiple", 
                label="Upload PDF Documents"
            )
            upload_button = gr.Button("Process Documents")
        
        # Processing Log Output
        processing_log = gr.Textbox(label="Processing Log", interactive=False)
        
        # Chat Interface
        chatbot_component = gr.Chatbot(label="Document Chat")
        msg_input = gr.Textbox(label="Your Question")
        submit_button = gr.Button("Ask")
        
        # Upload Event
        upload_button.click(
            fn=process_documents, 
            inputs=[file_input], 
            outputs=[processing_log]
        )
        
        # Chat Event
        submit_button.click(
            fn=ask_question, 
            inputs=[msg_input, chatbot_component], 
            outputs=[msg_input, chatbot_component]
        )
        msg_input.submit(
            fn=ask_question, 
            inputs=[msg_input, chatbot_component], 
            outputs=[msg_input, chatbot_component]
        )
    
    return demo

def main():
    # Launch Gradio interface
    interface = create_ui()
    interface.launch()

if __name__ == "__main__":
    main()

