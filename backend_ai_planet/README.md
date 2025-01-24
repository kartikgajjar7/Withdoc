# Document Processing API with FastAPI, LangChain, and Gemini

This project is a backend API built with **FastAPI** that allows users to upload PDF documents, process them using **LangChain** and **Google Gemini**, and ask questions about the content of the documents. The API stores document metadata and content in a **Neon database** and provides endpoints for uploading documents and querying them.

---

## Features

- **Upload PDF Documents**: Users can upload PDF files, which are processed and stored in the database.
- **Ask Questions**: Users can ask questions about the content of a specific document using its ID.
- **Conversational Memory**: The API maintains a conversation history for each document, allowing follow-up questions.
- **Powered by LangChain and Gemini**: Uses Google's Gemini model for natural language processing and LangChain for document retrieval and conversation management.

---

## Technologies Used

- **Backend**: FastAPI
- **Database**: Neon (PostgreSQL)
- **Language Model**: Google Gemini (via LangChain)
- **Document Processing**: PyMuPDF (`fitz`)
- **Vector Store**: FAISS (for document embeddings)
- **Frontend**: React (not included in this repository)

---

## API Endpoints

### 1. **Upload a PDF Document**
- **Endpoint**: `POST /upload/`
- **Description**: Upload a PDF file for processing and storage.
- **Request Body**: 
  - `file`: The PDF file to upload.
- **Response**:
  ```json
  {
    "id": "document_id",
    "filename": "example.pdf"
  }```
### 2. Ask a Question About a Document
- **Endpoint**: `POST /ask/{document_id}`
- **Description**: Ask a question about the content of a specific document.
- **Request Body**:
  ```json
  {
    "question": "What is the main topic of this document?",
    "history": [
      {"role": "user", "content": "What is the document about?"},
      {"role": "assistant", "content": "The document discusses AI and machine learning."}
    ]
  }
- **Response**:
```json
{
  "answer": "The main topic is AI and machine learning.",
  "history": [
    {"role": "user", "content": "What is the document about?"},
    {"role": "assistant", "content": "The document discusses AI and machine learning."},
    {"role": "user", "content": "What is the main topic of this document?"},
    {"role": "assistant", "content": "The main topic is AI and machine learning."}
  ]
}
```
### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```
  
2. **Set Up a Virtual Environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
```
3. **Install Dependencies:**:
```bash
pip install -r requirements.txt
```
4. **Set Up Environment Variables:**:
```env
GOOGLE_API_KEY=your_google_api_key
DATABASE_URL=your_neon_database_url
```
5. **Run the Application:**:
```bash
uvicorn main:app --reload
```
