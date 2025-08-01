from fastapi import FastAPI, Request, Header, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from fastapi.responses import JSONResponse
import requests
import tempfile
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFMinerLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Request body format
class QARequest(BaseModel):
    documents: str
    questions: List[str]

# Response format
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
        # Download PDF
        response = requests.get(payload.documents)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(response.content)
            pdf_path = tmp.name

        # Load and split the document using PDFMinerLoader
        loader = PDFMinerLoader(pdf_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)

        # Create vector store with Gemini embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectordb = FAISS.from_documents(docs, embeddings)

        # Set up RAG with Gemini Pro
        retriever = vectordb.as_retriever()
        qa = RetrievalQA.from_chain_type(
            llm=ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0),
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
        raise HTTPException(status_code=500, detail=str(e))
