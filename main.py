from fastapi import FastAPI, Request
from pydantic import BaseModel
import requests
import logging

app = FastAPI()

logging.basicConfig(level=logging.DEBUG)

class QuestionRequest(BaseModel):
    documents: str
    questions: list[str]

@app.post("/hackrx/run")
async def answer_questions(request: QuestionRequest):
    try:
        document_url = request.documents
        questions = request.questions

        logging.debug(f"Received questions: {questions}")
        logging.debug(f"Document URL: {document_url}")

        # Fix: Add headers to avoid GitHub raw URL being rejected
        headers = {'User-Agent': 'Mozilla/5.0'}

        response = requests.get(document_url, headers=headers)
        response.raise_for_status()
        pdf_content = response.content

        # For now, simulate processing (you can replace with actual LLM or PDF parser)
        results = {}
        for question in questions:
            results[question] = f"(Sample answer) The answer to '{question}' will be extracted from the document."

        return {"status": "success", "answers": results}

    except Exception as e:
        logging.error("An error occurred during question answering:", exc_info=True)
        return {"status": "error", "message": str(e)}
