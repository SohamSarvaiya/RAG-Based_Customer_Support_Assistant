import logging
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from app.auth import require_api_key
from app.config import Settings, get_settings
from app.errors import AppError, DocumentError
from app.rag.embeddings import build_embedder
from app.rag.generator import Generator
from app.rag.pipeline import RAGPipeline
from app.rag.vectorstore import VectorStore
from app.schemas import AskRequest, AskResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")
STATIC = Path(__file__).parent / "static"


def create_app(settings: Settings | None = None, pipeline: RAGPipeline | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(title="RAG-Based Customer Support Assistant", version="1.0.0")
    app.state.settings = settings
    app.state.pipeline = pipeline  # built lazily so importing the app stays fast

    def get_pipeline(request: Request) -> RAGPipeline:
        if request.app.state.pipeline is None:
            s = request.app.state.settings
            request.app.state.pipeline = RAGPipeline(
                s, build_embedder(s.embedding_backend, s.embedding_model),
                VectorStore(s.chroma_path, s.collection_name), Generator.from_settings(s))
        return request.app.state.pipeline

    @app.exception_handler(AppError)
    async def app_error_handler(_: Request, exc: AppError):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})

    @app.exception_handler(Exception)
    async def unhandled(_: Request, exc: Exception):
        log.exception("Unhandled error")
        return JSONResponse(status_code=500, content={"detail": "Internal server error."})

    @app.get("/", include_in_schema=False)
    def demo_ui():
        return FileResponse(STATIC / "index.html")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/documents", status_code=201, dependencies=[Depends(require_api_key)])
    def upload_document(file: UploadFile = File(...), p: RAGPipeline = Depends(get_pipeline)):
        if not (file.filename or "").lower().endswith(".pdf"):
            raise DocumentError("Only .pdf files are supported.", 415)
        data = file.file.read(settings.max_upload_mb * 1024 * 1024 + 1)
        if len(data) > settings.max_upload_mb * 1024 * 1024:
            raise DocumentError(f"File too large (max {settings.max_upload_mb} MB).", 413)
        if not data:
            raise DocumentError("Uploaded file is empty.", 400)
        result = p.ingest_pdf(file.filename, data)
        log.info("ingest %s -> %s", file.filename, result["status"])
        return result

    @app.get("/documents", dependencies=[Depends(require_api_key)])
    def list_documents(p: RAGPipeline = Depends(get_pipeline)):
        return {"documents": p.store.list_documents()}

    @app.delete("/documents/{doc_id}", dependencies=[Depends(require_api_key)])
    def delete_document(doc_id: str, p: RAGPipeline = Depends(get_pipeline)):
        removed = p.store.delete_document(doc_id)
        if not removed:
            raise HTTPException(404, "Document not found.")
        return {"doc_id": doc_id, "chunks_removed": removed}

    @app.post("/ask", response_model=AskResponse, dependencies=[Depends(require_api_key)])
    def ask(body: AskRequest, p: RAGPipeline = Depends(get_pipeline)):
        return p.ask(body.question.strip(), body.top_k)

    return app


app = create_app()
