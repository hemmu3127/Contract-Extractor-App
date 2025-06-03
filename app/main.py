# app/main.py
import logging
import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Body, Depends, status
from pydantic import BaseModel, Field # Field for Pydantic model field customization
from typing import Optional, Dict, Any
# import pandas as pd # Not directly used in main.py anymore, but in vector_db_manager

# Import from app package
from . import config, utils # Relative imports
from .vector_db_manager import VectorDBManager
from .data_processor import extract_contract_details_from_text

# Setup logging as early as possible
utils.setup_logging() # This should be called once
logger = logging.getLogger(__name__) # Get logger for this module

# --- Pydantic Models for Request/Response ---
class PopulateDBRequest(BaseModel):
    xlsx_file_path: Optional[str] = Field(default=None, description="Optional path to the XLSX file. Uses config default if not provided.")
    force_repopulate: bool = Field(default=False, description="If true, deletes existing collection data before populating.")

class PopulateDBResponse(BaseModel):
    message: str
    collection_name: str
    item_count: int

class ExtractRequest(BaseModel):
    contract_text: str = Field(..., min_length=10, description="The full text of the contract to analyze.")
    use_rag: bool = Field(default=True, description="Whether to use Retrieval Augmented Generation.")

class ExtractedDetails(BaseModel):
    agreement_value: Optional[float] = None
    agreement_start_date: Optional[str] = None
    agreement_end_date: Optional[str] = None
    renewal_notice_days: Optional[int] = None
    party_one: Optional[str] = None # ADDED
    party_two: Optional[str] = None # ADDED

class ExtractResponse(BaseModel):
    message: str
    source_text_snippet: str
    extracted_data: Optional[ExtractedDetails] = None
    rag_enabled: bool
    error: Optional[str] = None


# --- Global Variables / Application State ---
db_manager_instance: Optional[VectorDBManager] = None

# --- FastAPI Lifespan Events ---
@asynccontextmanager
async def lifespan(app_instance: FastAPI): # Renamed 'app' to 'app_instance' to avoid conflict
    # Startup
    global db_manager_instance
    logger.info("FastAPI application startup...")
    try:
        # Config validation is now done when config.py is imported.
        # If it fails critically, the app might not even reach here.
        logger.info("Configuration should have been validated on import.")
        
        # Moved DB directory creation to VectorDBManager constructor
        # to ensure it's handled there if path is specified.
        
        db_manager_instance = VectorDBManager() # Initializes with config
        logger.info(f"VectorDBManager initialized for persistent store. Collection: '{db_manager_instance.collection_name}', Count: {db_manager_instance.get_collection_count()}")

        # Optional: Auto-populate on startup if the DB is empty and XLSX exists
        # This can be slow, consider if it's truly needed for your use case
        xlsx_exists = os.path.exists(config.XLSX_FILE_PATH)
        if db_manager_instance.get_collection_count() == 0 and xlsx_exists:
            logger.info(f"Persistent DB is empty and '{config.XLSX_FILE_PATH}' exists. Attempting initial population...")
            try:
                db_manager_instance.populate_from_xlsx(xlsx_path=config.XLSX_FILE_PATH, force_repopulate=False) # Don't force if just checking
                logger.info(f"Initial DB population complete. New count: {db_manager_instance.get_collection_count()}")
            except Exception as e:
                logger.error(f"Error during initial DB population on startup: {e}", exc_info=True)
        elif not xlsx_exists and db_manager_instance.get_collection_count() == 0:
            logger.warning(f"XLSX file for initial population not found at '{config.XLSX_FILE_PATH}'. DB remains empty.")


    except Exception as e:
        logger.error(f"Fatal error during application startup: {e}", exc_info=True)
        raise RuntimeError(f"Application startup failed: {e}") from e
    
    yield # Application runs here
    
    # Shutdown
    logger.info("FastAPI application shutdown...")


# --- FastAPI App Instance ---
app = FastAPI(
    title="Contract Data Extractor API",
    description="API to extract structured data from contract texts using RAG with Gemini and ChromaDB.",
    version="1.0.1", # Updated version
    lifespan=lifespan
)

# --- Dependency for DB Manager ---
def get_db_manager() -> VectorDBManager:
    if db_manager_instance is None:
        logger.critical("db_manager_instance is not initialized! This indicates a serious issue with application startup.")
        # This error should ideally lead to the app not starting correctly or health checks failing.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, # Service Unavailable
            detail="Database manager not available. The service is not ready."
        )
    return db_manager_instance


# --- API Endpoints ---
@app.post("/admin/populate-database", response_model=PopulateDBResponse, status_code=status.HTTP_200_OK, tags=["Admin & Maintenance"])
async def populate_database_endpoint(
    request_body: PopulateDBRequest,
    db_manager: VectorDBManager = Depends(get_db_manager)
):
    """
    Populates the persistent vector database from contract texts in an XLSX file.
    Can be a long-running operation. Use with caution in production.
    """
    effective_xlsx_path = request_body.xlsx_file_path or config.XLSX_FILE_PATH
    logger.info(f"Received admin request to populate DB from: {effective_xlsx_path}. Force: {request_body.force_repopulate}")

    if not os.path.exists(effective_xlsx_path):
        logger.error(f"XLSX file not found at: {effective_xlsx_path}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, # More appropriate for missing file
            detail=f"XLSX file not found at: {effective_xlsx_path}"
        )

    try:
        # Consider running this in a background task for non-blocking API
        # from fastapi import BackgroundTasks (and add BackgroundTasks to endpoint params)
        # background_tasks.add_task(db_manager.populate_from_xlsx, xlsx_path=effective_xlsx_path, force_repopulate=request_body.force_repopulate)
        # For now, synchronous:
        db_manager.populate_from_xlsx(
            xlsx_path=effective_xlsx_path,
            force_repopulate=request_body.force_repopulate
        )
        item_count = db_manager.get_collection_count()
        msg = f"Database population from '{os.path.basename(effective_xlsx_path)}' {'forced and ' if request_body.force_repopulate else ''}completed."
        logger.info(f"{msg} Collection '{db_manager.collection_name}' now has {item_count} items.")
        return PopulateDBResponse(
            message=msg,
            collection_name=db_manager.collection_name,
            item_count=item_count
        )
    except Exception as e:
        logger.error(f"Error during DB population request: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during DB population: {str(e)}"
        )

@app.post("/extract", response_model=ExtractResponse, tags=["Core Extraction"])
async def extract_details_endpoint(
    request_body: ExtractRequest,
    db_manager: VectorDBManager = Depends(get_db_manager)
):
    """
    Extracts structured information from contract text.
    """
    logger.info(f"Received extraction request. RAG requested: {request_body.use_rag}")
    
    use_rag_effective = request_body.use_rag
    if use_rag_effective and db_manager.get_collection_count() == 0:
        logger.warning("RAG is enabled by request, but the vector database is empty. RAG context will be unavailable.")
        # The response will reflect that RAG was enabled but might not have found context.

    try:
        extracted_data_dict = extract_contract_details_from_text(
            input_text=request_body.contract_text,
            vector_db_manager=db_manager,
            use_rag=use_rag_effective # Pass the effective RAG status
        )

        if extracted_data_dict:
            parsed_details = ExtractedDetails(**extracted_data_dict)
            logger.info(f"Extraction successful. Data: {parsed_details.model_dump_json(indent=2)}")
            return ExtractResponse(
                message="Contract details extracted successfully.",
                source_text_snippet=request_body.contract_text[:200] + "...",
                extracted_data=parsed_details,
                rag_enabled=use_rag_effective
            )
        else:
            logger.warning("Extraction process completed but returned no structured data from LLM or parsing failed.")
            return ExtractResponse(
                message="Failed to extract structured details. LLM might not have found information or parsing failed.",
                source_text_snippet=request_body.contract_text[:200] + "...",
                extracted_data=None,
                rag_enabled=use_rag_effective,
                error="No structured data could be extracted or parsed from LLM response."
            )

    except HTTPException: # Re-raise HTTPExceptions from called functions
        raise
    except Exception as e:
        logger.error(f"Unexpected error during extraction request: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during extraction: {str(e)}"
        )

@app.get("/health", status_code=status.HTTP_200_OK, tags=["System & Health"])
async def health_check():
    """Simple health check endpoint."""
    # Could add checks for DB connection or LLM API key presence here
    return {"status": "healthy", "message": "API is operational."}

@app.get("/system/db-status", response_model=Dict[str, Any], tags=["System & Health"])
async def database_status(db_manager: VectorDBManager = Depends(get_db_manager)):
    """Returns the status of the vector database."""
    return {
        "db_type": "ChromaDB (Persistent)",
        "db_path": db_manager.db_path,
        "collection_name": db_manager.collection_name,
        "item_count": db_manager.get_collection_count(),
        "is_healthy": True # Basic check, could be more sophisticated
    }

# --- To run the app (for local development) ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server for local development from main.py...")
    uvicorn.run(
        "app.main:app", # Path to the FastAPI app instance
        host="0.0.0.0",
        port=8000,
        log_level="info", # Uvicorn's own log level
        reload=True # Enable auto-reload for development
    )