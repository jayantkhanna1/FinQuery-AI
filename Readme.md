# Financial Document Chatbot

A specialized chatbot application designed for analyzing financial documents, particularly SEC filings like 10-K and 10-Q reports. The application uses OpenAI's GPT-3.5-turbo model and embeddings to provide accurate, context-aware responses about financial information contained in the documents.

## Features

- Upload and process multiple financial documents (10-K, 10-Q, etc.) simultaneously
- Interactive chat interface for financial document analysis
- Intelligent text splitting optimized for financial document structure
- Vector-based semantic search using FAISS
- User-friendly Gradio web interface
- Real-time processing status updates

## Sample Documents Included

The repository includes sample financial documents for testing:
- Example 10-K annual reports
- Example 10-Q quarterly reports

These sample files can be used to test the system and understand how to interact with financial documents effectively.

## Prerequisites

- Python 3.7 or higher
- OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/financial-document-chatbot.git
cd financial-document-chatbot
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Required Packages

```
python-dotenv
openai
langchain
gradio
faiss-cpu
pypdf
```

## Usage

1. Run the application:
```bash
python main.py
```

2. Open your web browser and navigate to the local URL provided by Gradio (typically `http://127.0.0.1:7860`).

3. Upload financial documents (10-K, 10-Q, etc.) using the file upload interface.
   - You can start with the provided sample files
   - Add your own financial documents as needed

4. Click "Process Documents" to analyze the uploaded files.

5. Start asking questions about the financial documents, such as:
   - "What were the total revenues in the last fiscal year?"
   - "Explain the company's risk factors"
   - "What are the key changes in operating expenses?"
   - "Summarize the financial performance trends"

## How It Works

1. **Document Processing**:
   - Financial documents are loaded using PyPDFLoader
   - Documents are split into smaller chunks using RecursiveCharacterTextSplitter
   - Text chunks are converted into embeddings using OpenAI's embedding model
   - Embeddings are stored in a FAISS vector store for efficient retrieval

2. **Question Answering**:
   - User questions are processed through a retrieval QA chain
   - The system finds the most relevant document chunks using semantic search
   - GPT-3.5-turbo generates responses based on the retrieved financial context

## Class Structure

### DocumentChatbot
The main class that handles financial document processing and question answering:
- `__init__(openai_api_key)`: Initializes the chatbot with necessary components
- `load_documents(file_paths)`: Processes and loads financial documents
- `ask_question(question)`: Handles user queries and generates responses about financial data

### UI Components
- Created using Gradio Blocks
- Includes file upload interface, processing log, and chat interface
- Supports both button clicks and Enter key submission

## Error Handling

- Validates OpenAI API key presence
- Provides feedback for document processing errors
- Handles chat interaction errors gracefully
- Displays processing status for each uploaded document

## Limitations

- Optimized for financial documents (10-K, 10-Q, etc.)
- Requires active internet connection for OpenAI API access
- Processing time depends on document size and complexity
- Subject to OpenAI API rate limits and costs
- May require domain knowledge to interpret financial responses accurately

## Use Cases

- Financial analysis and research
- Due diligence investigations
- Financial report summarization
- Trend analysis across multiple reports
- Risk factor analysis
- Competitive analysis using multiple company filings
