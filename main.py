import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import requests
from typing import List

# Set up logging
logging.basicConfig(level=logging.DEBUG)

app = FastAPI()

class QARequest(BaseModel):
    documents: str
    questions: List[str]

@app.post("/hackrx/run")
async def answer_questions(request: QARequest):
    document_url = request.documents
    questions = request.questions
    logging.debug(f"Received questions: {questions}")
    logging.debug(f"Document URL: {document_url}")

    try:
        response = requests.get(document_url)
        response.raise_for_status()
        logging.debug("Successfully fetched document")
    except Exception as e:
        logging.error(f"Failed to fetch document: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to fetch document: {e}")

    # In real case: extract answers using LLMs or NLP
    # For now, return mock/sample answers
    answers = {}
    for q in questions:
        answers[q] = f"(Sample answer) The answer to '{q}' will be extracted from the document."

    return {
        "status": "success",
        "answers": answers
    }
