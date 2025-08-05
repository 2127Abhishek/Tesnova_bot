import os
import logging
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import CharacterTextSplitter
from tempfile import TemporaryDirectory
from dotenv import load_dotenv

# Load .env environment variables
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.DEBUG)

# Initialize FastAPI app
app = FastAPI()

# Enable CORS (optional but useful for frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Google API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY not found in environment variables.")

# Request model
class QuestionRequest(BaseModel):
    documents: str
    questions: list[str]

# Health check endpoint
@app.get("/")
def health_check():
    return {"status": "API is running"}

# Main API endpoint
@app.post("/hackrx/run")
async def answer_questions(req: QuestionRequest):
    try:
        logging.debug(f"Received {len(req.questions)} questions")
        logging.debug(f"Document URL: {req.documents}")

        # Validate the PDF URL
        if not req.documents.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="URL must point to a PDF file.")

        # Download the PDF
        pdf_response = requests.get(req.documents)
        pdf_response.raise_for_status()

        # Save and process the PDF
        with TemporaryDirectory() as tmpdir:
            pdf_path = os.path.join(tmpdir, "document.pdf")
            with open(pdf_path, "wb") as f:
                f.write(pdf_response.content)

            loader = PyPDFLoader(pdf_path)
            documents = loader.load()

        # Split the document into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)

        # Create vector store with embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )
        db = FAISS.from_documents(docs, embeddings)

        # Load QA chain using Gemini Pro
        qa_chain = load_qa_chain(
            llm=ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY),
            chain_type="stuff"
        )

        # Process each question
        answers = []
        for question in req.questions:
            relevant_docs = db.similarity_search(question)
            result = qa_chain.run(input_documents=relevant_docs, question=question)
            answers.append(result.strip())

        # Return answers in desired format
        return {"answers": answers}

    except Exception as e:
        logging.error("Error while processing request:", exc_info=e)
        raise HTTPException(status_code=500, detail=str(e))
