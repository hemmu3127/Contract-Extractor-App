# app/llm_services.py
import logging
import time
from typing import List, Optional, Dict, Any

import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

# Import config from the app package
from . import config # Use relative import

logger = logging.getLogger(__name__)

# Configure the Gemini client
def _configure_gemini():
    if config.GEMINI_API_KEY:
        genai.configure(api_key=config.GEMINI_API_KEY)
        logger.info("Gemini API client configured.")
    else:
        logger.error("GEMINI_API_KEY not configured. LLM services will not function.")

_configure_gemini() # Configure on module load

def get_gemini_embeddings_batch(texts: List[str], model_name: str = config.EMBEDDING_MODEL_NAME, retries: int = 3, delay: int = 5) -> List[Optional[List[float]]]:
    if not texts:
        return []
    if not config.GEMINI_API_KEY:
        logger.error("Cannot generate embeddings: GEMINI_API_KEY not set.")
        return [None] * len(texts)

    all_embeddings: List[Optional[List[float]]] = []
    
    for attempt in range(retries):
        try:
            current_batch_embeddings_for_this_attempt = []
            for i in range(0, len(texts), config.EMBEDDING_BATCH_SIZE):
                batch_texts_segment = texts[i:i + config.EMBEDDING_BATCH_SIZE]
                logger.debug(f"Embedding batch segment {i // config.EMBEDDING_BATCH_SIZE + 1}, size: {len(batch_texts_segment)}")
                
                result = genai.embed_content(
                    model=model_name,
                    content=batch_texts_segment,
                    task_type="RETRIEVAL_DOCUMENT"
                )
                
                if isinstance(result, dict) and 'embedding' in result and isinstance(result['embedding'], list):
                    if result['embedding'] and isinstance(result['embedding'][0], list):
                        current_batch_embeddings_for_this_attempt.extend(result['embedding'])
                    else:
                        logger.warning(f"Unexpected embedding structure in segment. Filling with Nones.")
                        current_batch_embeddings_for_this_attempt.extend([None] * len(batch_texts_segment))
                else:
                    logger.warning(f"No 'embedding' key or unexpected format for batch segment. Result: {str(result)[:200]}. Filling with Nones.")
                    current_batch_embeddings_for_this_attempt.extend([None] * len(batch_texts_segment))
            
            all_embeddings = current_batch_embeddings_for_this_attempt
            if len(all_embeddings) == len(texts):
                return all_embeddings
            else:
                logger.error(f"Mismatch in embedding count. Expected {len(texts)}, got {len(all_embeddings)}. Attempt {attempt+1}")
                if attempt == retries - 1: return [None] * len(texts)
                time.sleep(delay); continue

        except (google_exceptions.ResourceExhausted, google_exceptions.ServiceUnavailable, google_exceptions.DeadlineExceeded, google_exceptions.InternalServerError) as e:
            logger.warning(f"Embeddings API error (attempt {attempt + 1}/{retries}): {e}. Retrying in {delay}s...")
            if attempt == retries - 1:
                logger.error(f"Failed to get embeddings after {retries} retries due to API errors.")
                return [None] * len(texts)
            time.sleep(delay)
        except Exception as e:
            logger.error(f"Unexpected error generating embeddings: {e}", exc_info=True)
            return [None] * len(texts)
    
    logger.error("Failed to get embeddings after all retries (embedding loop).")
    return [None] * len(texts)


def get_gemini_generation(prompt: str, model_name: str = config.GENERATIVE_MODEL_NAME, retries: int = 3, delay: int = 5) -> Optional[str]:
    if not prompt:
        logger.warning("Empty prompt received for generation.")
        return None
    if not config.GEMINI_API_KEY:
        logger.error("Cannot generate text: GEMINI_API_KEY not set.")
        return None

    logger.debug(f"Attempting generation with model: {model_name}")
    model = genai.GenerativeModel(model_name)
    
    for attempt in range(retries):
        try:
            # For SDK < 0.4.0, we rely purely on the prompt to request JSON output.
            # No special mime_type or request_options are supported for generate_content.
            generation_config_obj = genai.types.GenerationConfig(
                temperature=0.2, # Keep temperature low for factual JSON
                # max_output_tokens=..., # Consider setting if concerned about long outputs
                # top_p=...,
                # top_k=...
            )

            # The `generate_content` method in older SDKs typically takes the prompt
            # as the first argument, or via a `contents` keyword.
            # Let's ensure we're using the `contents` keyword for clarity and consistency.
            response = model.generate_content(
                contents=prompt, # Pass the prompt string to the 'contents' parameter
                generation_config=generation_config_obj
                # No other special parameters for JSON mode in this SDK version
            )
            
            if response.candidates:
                candidate = response.candidates[0] # Get the first candidate
                if candidate.content and candidate.content.parts:
                    generated_text = candidate.content.parts[0].text
                    logger.debug(f"Successfully generated text (length {len(generated_text)}). Snippet: {generated_text[:100]}")
                    return generated_text
                # Check finish reason even if parts are empty/missing
                elif candidate.finish_reason not in [None, genai.types.Candidate.FinishReason.FINISH_REASON_UNSPECIFIED, genai.types.Candidate.FinishReason.STOP]:
                    logger.warning(f"Generation finished with reason: {candidate.finish_reason}. Content might be missing or incomplete.")
                    return None # No usable text
                else: 
                    logger.warning(f"Generation response candidate has no content/parts or unexpected structure. Candidate: {candidate}")
                    return None
            elif response.prompt_feedback and response.prompt_feedback.block_reason:
                block_reason_message = getattr(response.prompt_feedback, 'block_reason_message', str(response.prompt_feedback.block_reason))
                logger.error(f"Prompt blocked for generation. Reason: {block_reason_message}")
                return None
            else:
                logger.warning(f"LLM response was empty, had no candidates, or was malformed. Response: {str(response)[:500]}")
                return None

        except (google_exceptions.ResourceExhausted, google_exceptions.ServiceUnavailable, google_exceptions.DeadlineExceeded, google_exceptions.InternalServerError) as e:
            logger.warning(f"Generation API error (attempt {attempt + 1}/{retries}) with model {model_name}: {e}. Retrying in {delay}s...")
            if attempt == retries - 1:
                logger.error(f"Failed to generate text after {retries} retries for model {model_name}.")
                return None
            time.sleep(delay)
        except Exception as e: # Catch any other unexpected errors
            logger.error(f"Unexpected error during generation with model {model_name}: {e}", exc_info=True)
            # For unexpected errors, it's often better to not retry endlessly.
            return None 
            
    logger.error(f"Failed to generate text after all retries for model {model_name} (generation loop).")
    return None