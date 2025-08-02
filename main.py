from fastapi import FastAPI, Request, Header, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from fastapi.responses import JSONResponse
import requests
import tempfile
import os
from dotenv import load_dotenv
import logging
import traceback  # For detailed error tracing

from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFMinerLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

# Enable detailed logging
logging.basicConfig(level=logging.DEBUG)

# Initialize FastAPI app
app = FastAPI()

# Request body model
class QARequest(BaseModel):
    documents: str
    questions: List[str]

# Response model
class QAResponse(BaseModel):
    answers: List[str]

@app.post("/hackrx/run", response_model=QAResponse)
async def answer_questions(
    request: Request,
    payload: QARequest,
    authorization: Optional[str] = Header(None)
):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        # Log start of request
        logging.debug(f"Received questions: {payload.questions}")
        logging.debug(f"Document URL: {payload.documents}")

        # Download PDF
        response = requests.get(payload.documents)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(response.content)
            pdf_path = tmp.name

        # Load and split the document
        loader = PDFMinerLoader(pdf_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)

        # Set up embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        # Create FAISS vector DB
        vectordb = FAISS.from_documents(docs, embeddings)

        # Set up RAG
        retriever = vectordb.as_retriever()
        qa = RetrievalQA.from_chain_type(
            llm=ChatGoogleGenerativeAI(model="gemini-pro", temperature=0),  # Safer default
            retriever=retriever,
            return_source_documents=False
        )

        # Process each question
        answers = []
        for q in payload.questions:
            result = qa.run(q)
            answers.append(result)

        return QAResponse(answers=answers)

    except Exception as e:
        logging.error("An error occurred during question answering:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
