import os
import logging
import requests
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import GooglePalmEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.DEBUG)

app = FastAPI()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY not found in environment variables.")

class QuestionRequest(BaseModel):
    documents: str
    questions: list[str]

@app.post("/hackrx/run")
async def answer_questions(req: QuestionRequest):
    try:
        logging.debug(f"Received questions: {req.questions}")
        logging.debug(f"Document URL: {req.documents}")

        pdf_response = requests.get(req.documents)
        pdf_response.raise_for_status()

        with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_response.content)
            tmp_pdf_path = tmp_file.name

        loader = PyPDFLoader(tmp_pdf_path)
        documents = loader.load()

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
        db = FAISS.from_documents(docs, embeddings)

        qa_chain = load_qa_chain(
            llm=ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY),
            chain_type="stuff"
        )

        answers = {}
        for question in req.questions:
            relevant_docs = db.similarity_search(question)
            result = qa_chain.run(input_documents=relevant_docs, question=question)
            answers[question] = result.strip()

        return {"status": "success", "answers": answers}

    except Exception as e:
        logging.error("An error occurred during question answering:", exc_info=e)
        raise HTTPException(status_code=500, detail=str(e))
