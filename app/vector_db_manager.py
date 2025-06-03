# app/vector_db_manager.py
import logging
from typing import List, Optional, Tuple, Dict, Any
import pandas as pd
import chromadb
import os

# Import from app package
from . import config
from .llm_services import get_gemini_embeddings_batch # Relative import

logger = logging.getLogger(__name__)
import logging
import os
from . import config # Ensure this uses the updated config.py

logger = logging.getLogger(__name__)

class VectorDBManager:
    def __init__(self, db_path: str = config.CHROMA_DB_PATH, collection_name: str = config.COLLECTION_NAME):
        self.db_path = db_path  # This IS the directory path for ChromaDB storage
        self.collection_name = collection_name
        try:
            # Ensure the db_path directory itself exists.
            if not self.db_path: # Should be caught by config validation now
                _msg = "ChromaDB path (db_path) is critically misconfigured (empty)."
                logger.error(_msg)
                raise ValueError(_msg)

            # Create the directory if it doesn't exist
            if not os.path.exists(self.db_path):
                try:
                    os.makedirs(self.db_path, exist_ok=True)
                    logger.info(f"Created ChromaDB storage directory: {self.db_path}")
                except OSError as e:
                    logger.error(f"Could not create ChromaDB storage directory '{self.db_path}': {e}", exc_info=True)
                    raise
            elif not os.path.isdir(self.db_path): # Path exists but is not a directory
                _msg = f"Configured CHROMA_DB_PATH '{self.db_path}' exists but is not a directory."
                logger.error(_msg)
                raise NotADirectoryError(_msg)

            self.client = chromadb.PersistentClient(path=self.db_path)
            logger.info(f"ChromaDB persistent client initialized for path: {self.db_path}")

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB persistent client with path '{self.db_path}': {e}", exc_info=True)
            raise
            
        self.collection = self._get_or_create_collection()

    # ... (rest of the VectorDBManager class from previous correct version) ...
    def _get_or_create_collection(self) -> chromadb.api.models.Collection.Collection:
        try:
            collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Retrieved existing persistent collection: '{self.collection_name}' with {collection.count()} items.")
            return collection
        except Exception: 
            logger.info(f"Persistent collection '{self.collection_name}' not found at {self.db_path}. Creating new collection.")
            collection = self.client.create_collection(name=self.collection_name)
            logger.info(f"Created new persistent collection: '{self.collection_name}'.")
            return collection

    def populate_from_xlsx(self, xlsx_path: str = config.XLSX_FILE_PATH, force_repopulate: bool = False):
        logger.info(f"Attempting to populate persistent vector DB from: {xlsx_path}")
        if force_repopulate:
            logger.warning(f"Force repopulate: Deleting existing persistent collection '{self.collection_name}' before population.")
            try:
                self.client.delete_collection(name=self.collection_name)
                self.collection = self._get_or_create_collection() 
            except Exception as e:
                logger.error(f"Failed to delete or recreate persistent collection during force repopulate: {e}", exc_info=True)
                return

        if self.collection.count() > 0 and not force_repopulate:
            logger.info(f"Persistent collection '{self.collection_name}' already has {self.collection.count()} items. Skipping population. Use force_repopulate=True to override.")
            return

        try:
            df = pd.read_excel(xlsx_path)
        except FileNotFoundError:
            logger.error(f"Excel file not found at: {xlsx_path}")
            return
        except Exception as e:
            logger.error(f"Error reading Excel file {xlsx_path}: {e}", exc_info=True)
            return

        if 'Text' not in df.columns or 'File Name' not in df.columns:
            logger.error("'Text' or 'File Name' column not found in Excel. Cannot populate.")
            return

        texts_to_embed: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        ids: List[str] = []

        for index, row in df.iterrows():
            text_content = str(row['Text'])
            file_name = str(row['File Name'])

            if not text_content or pd.isna(text_content):
                logger.warning(f"Skipping row {index} (File: {file_name}) due to empty/NaN text content.")
                continue

            texts_to_embed.append(text_content)
            meta = {"file_name": file_name, "original_xlsx_index": str(index)}
            metadatas.append(meta)
            safe_file_name_part = "".join(c if c.isalnum() or c in ['-', '_'] else '_' for c in file_name[:30])
            ids.append(f"doc_{index}_{safe_file_name_part}")


        if not texts_to_embed:
            logger.info("No valid texts found in Excel to embed.")
            return

        logger.info(f"Generating embeddings for {len(texts_to_embed)} documents...")
        embeddings_list = get_gemini_embeddings_batch(texts_to_embed) 

        valid_texts = []
        valid_metadatas = []
        valid_ids = []
        valid_embeddings = []

        for i, emb_vector in enumerate(embeddings_list):
            if emb_vector: 
                valid_texts.append(texts_to_embed[i])
                valid_metadatas.append(metadatas[i])
                valid_ids.append(ids[i])
                valid_embeddings.append(emb_vector)
            else:
                logger.warning(f"Failed to generate embedding for document ID: {ids[i]} (text: '{texts_to_embed[i][:50]}...'). Skipping.")

        if not valid_embeddings:
            logger.error("No embeddings were successfully generated. DB population failed.")
            return

        try:
            logger.info(f"Adding {len(valid_embeddings)} items to ChromaDB persistent collection '{self.collection_name}'.")
            self.collection.add(
                embeddings=valid_embeddings,
                documents=valid_texts,
                metadatas=valid_metadatas,
                ids=valid_ids
            )
            logger.info(f"Successfully added {self.collection.count()} items to persistent collection '{self.collection_name}'.")
        except Exception as e:
            logger.error(f"Error adding batch to persistent ChromaDB: {e}", exc_info=True)
            logger.error(f"Details - Embeddings: {len(valid_embeddings)}, Docs: {len(valid_texts)}, Meta: {len(valid_metadatas)}, IDs: {len(valid_ids)}")


    def query_documents(self, query_text: str, k: int = config.TOP_K_RETRIEVAL) -> List[Dict[str, Any]]:
        num_items_in_collection = self.collection.count()
        if num_items_in_collection == 0:
            logger.warning("Querying an empty persistent collection. No results will be found.")
            return []
            
        logger.debug(f"Querying persistent DB for '{query_text[:50]}...' with k={k}")
        query_embedding_list = get_gemini_embeddings_batch([query_text])

        if not query_embedding_list or not query_embedding_list[0]:
            logger.error("Failed to generate embedding for query text.")
            return []

        query_embedding = query_embedding_list[0]
        
        actual_k = min(k, num_items_in_collection) 
        if actual_k < k:
             logger.info(f"Requested k={k} is greater than collection size ({num_items_in_collection}). Querying for {actual_k} results.")

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=actual_k, 
                include=['documents', 'metadatas', 'distances']
            )
            
            retrieved_docs = []
            if results and results.get('ids') and results['ids'][0]:
                for i, doc_id in enumerate(results['ids'][0]):
                    retrieved_docs.append({
                        "id": doc_id,
                        "text": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i] if results.get('metadatas') and results['metadatas'][0] else {},
                        "distance": results['distances'][0][i] if results.get('distances') and results['distances'][0] else None,
                    })
                logger.info(f"Retrieved {len(retrieved_docs)} documents for query from persistent DB.")
                return retrieved_docs
            else:
                logger.info("No documents found for the query in persistent DB. Query result was empty.")
                return []
        except Exception as e:
            logger.error(f"Error querying persistent ChromaDB: {e}", exc_info=True)
            return []

    def get_collection_count(self) -> int:
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Error getting collection count: {e}", exc_info=True)
            return 0

    def reset_collection(self):
        logger.warning(f"Resetting persistent collection: {self.collection_name} at {self.db_path}. All data will be lost.")
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception as e:
            logger.warning(f"Could not delete persistent collection '{self.collection_name}' during reset (it might not exist): {e}")
        self.collection = self._get_or_create_collection()
        logger.info(f"Persistent collection '{self.collection_name}' has been reset.")