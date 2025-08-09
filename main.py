import os
import logging
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.Youtubeing import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter # Switched to a more robust splitter
from tempfile import TemporaryDirectory
from dotenv import load_dotenv

# Load .env environment variables
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO) # Use INFO for production, DEBUG for development

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
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
        logging.info(f"Received {len(req.questions)} questions for document: {req.documents}")

        # Validate the PDF URL
        if not req.documents.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="URL must point to a PDF file.")

        # Download the PDF
        pdf_response = requests.get(req.documents)
        pdf_response.raise_for_status()

        # Save and process the PDF in a temporary directory
        with TemporaryDirectory() as tmpdir:
            pdf_path = os.path.join(tmpdir, "document.pdf")
            with open(pdf_path, "wb") as f:
                f.write(pdf_response.content)

            loader = PyPDFLoader(pdf_path)
            documents = loader.load()

        # Split the document into chunks using a better splitter for general documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = text_splitter.split_documents(documents)

        # Create vector store with embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )
        db = FAISS.from_documents(docs, embeddings)

        # **IMPROVEMENT**: Create a stricter prompt to improve accuracy
        prompt_template = """
        Use the following pieces of context to answer the question at the end.
        If you don't know the answer from the context provided, just say that you don't know. Do not try to make up an answer.
        Be concise and respond based only on the provided text.

        Context:
        {context}

        Question: {question}

        Answer:
        """
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        # **CRITICAL FIX**: Use a valid and powerful model name like "gemini-1.5-pro-latest"
        # And apply the custom prompt to the chain
        qa_chain = load_qa_chain(
            llm=ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=GOOGLE_API_KEY, temperature=0.2),
            chain_type="stuff",
            prompt=PROMPT
        )

        # Process each question
        answers = []
        for question in req.questions:
            # **IMPROVEMENT**: Retrieve more documents (k=6) to get better context
            relevant_docs = db.similarity_search(question, k=6)

            # To debug, you can uncomment the following lines to see what the LLM sees
            # logging.info(f"--- Retrieved docs for question: '{question}' ---")
            # for i, doc in enumerate(relevant_docs):
            #     logging.info(f"--- DOC {i+1} ---\n{doc.page_content}\n")
            
            # **IMPROVEMENT**: Use the modern .invoke() method
            result = qa_chain.invoke(
                {"input_documents": relevant_docs, "question": question},
                return_only_outputs=True
            )
            answers.append(result['output_text'].strip())

        # Return answers in desired format
        return {"answers": answers}

    except Exception as e:
        logging.error("Error while processing request:", exc_info=e)
        raise HTTPException(status_code=500, detail=str(e))