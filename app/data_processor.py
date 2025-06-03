# app/data_processor.py
import logging
import re
from typing import Optional, Dict, Any, List

# Import from app package
from . import config, utils # Relative imports
from .llm_services import get_gemini_generation
from .vector_db_manager import VectorDBManager # Relative import

logger = logging.getLogger(__name__)

def build_extraction_prompt(primary_text: str, retrieved_contexts: Optional[List[Dict[str, Any]]] = None) -> str:
    """Constructs the prompt for the LLM to extract information."""

    prompt = f"""
    Constructs a detailed and robust prompt for the LLM to extract specific contract information
    and output it in a structured JSON format.
    """

    # --- Start of the Enhanced Prompt ---
    prompt = f"""
You are an AI assistant specialized in meticulously analyzing legal contract texts. Your primary goal is to extract specific pieces of information from the provided "Primary Contract Text" and structure this information into a precise JSON output.

**Instructions for Extraction:**

1.  **Source of Information:** Extract information *exclusively* from the "Primary Contract Text" provided below. Do NOT use information from "Similar Contract Examples" to fill in values; those examples are for contextual understanding of phrasing and typical data formats ONLY.
2.  **JSON Output:** You MUST provide your response in a valid JSON format as specified. Ensure all keys are present.
3.  **Handling Missing Information:** If a specific piece of information cannot be found or confidently extracted from the "Primary Contract Text", use the JSON value `null` for that field. Do not invent or guess information.
4.  **Field-Specific Instructions:**
    *   **`agreement_value`**:
        *   Identify the primary monetary value of the agreement (e.g., monthly rent, total contract sum).
        *   Extract this as a numerical value (integer or float).
        *   Remove any currency symbols (e.g., $, £, €, Rs., PESOS, PHP) or thousands separators (e.g., commas).
        *   Example: If text says "₱6,500.00", extract `6500.0`. If "Ten Thousand Dollars", extract `10000`.
        *   If no clear single agreement value is found, use `null`.
    *   **`agreement_start_date`**:
        *   Find the official commencement or start date of the agreement.
        *   Format this date as "YYYY-MM-DD".
        *   If the day is ambiguous but month and year are clear (e.g., "May 2007"), use the first day of the month (e.g., "2007-05-01").
        *   If the date is written out (e.g., "the twentieth day of May, two thousand seven"), convert it.
        *   If no start date is found, or it's too vague to format, use `null`.
    *   **`agreement_end_date`**:
        *   Find the official termination or end date of the agreement.
        *   Format this date as "YYYY-MM-DD".
        *   Apply similar parsing logic as for `agreement_start_date`.
        *   If the agreement is open-ended or no end date is specified, use `null`.
    *   **`renewal_notice_days`**:
        *   Identify the number of days of advance notice required for contract renewal or termination.
        *   Extract this as an integer (e.g., if text says "15 days notice", extract `15`).
        *   If text says "two months notice", convert to days (e.g., `60`, assuming 30 days/month). If "one month", extract `30`.
        *   If no specific number of days for renewal/termination notice is found, use `null`.
    *   **`party_one`**:
        *   If there is any kind of person name and try to extract from the description and add here.
        *   Identify the first party to the agreement, typically the Lessor, Owner, Landlord, or Service Provider. Look for phrases like "between [Party One] and [Party Two]" or "[Party One] hereinafter called the 'LESSOR'".
        *   Extract the full name or company name as a string.
        *   Include identifying titles if they are part of the name (e.g., "Innovatech Solutions Inc.").
        *   If multiple individuals are listed for Party One (e.g., "John Doe and/or Jane Doe"), include both if concise, or the primary named entity.
        
    *   **`party_two`**:
        *   After identifying the party one the second person name is party two 
        *   Identify the second party to the agreement, typically the Lessee, Tenant, Resident, or Client. Look for phrases like "between [Party One] and [Party Two]" or "[Party Two] hereinafter called the 'LESSEE'".
        *   Extract the full name or company name as a string.
        *   Apply similar extraction logic as for `party_one`.

**Required JSON Output Structure and Example:**

```json
{{
  "agreement_value": <number_or_null>,
  "agreement_start_date": "YYYY-MM-DD_or_null",
  "agreement_end_date": "YYYY-MM-DD_or_null",
  "renewal_notice_days": <integer_or_null>,
  "party_one": "String_Name",
  "party_two": "String_Name"
}}

Primary Contract Text:
---
{primary_text}
---
"""

    if retrieved_contexts:
        prompt += "\nSimilar Contract Examples (for context only, extract information *only* from Primary Contract Text above):\n"
        for i, context in enumerate(retrieved_contexts):
            context_text_snippet = context.get('text', '')[:800] # Limit snippet length
            source_file = context.get('metadata', {}).get('file_name', 'N/A')
            distance = context.get('distance')
            distance_str = f"{distance:.4f}" if distance is not None else "N/A"
            prompt += f"\n--- Example {i+1} (Source: {source_file}, Distance: {distance_str}) ---\n"
            prompt += f"{context_text_snippet}...\n"
        prompt += "---\n"

    prompt += "\nPlease extract the information from the 'Primary Contract Text' using the specified JSON format."
    return prompt

def process_extracted_data(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """Cleans and standardizes the extracted data from LLM JSON output."""
    processed = {}
    logger.debug(f"Processing raw extracted data: {raw_data}")

    # Agreement Value
    val_str = raw_data.get("agreement_value")
    if val_str is not None:
        if isinstance(val_str, (int, float)): # Already a number
            processed["agreement_value"] = val_str
        elif isinstance(val_str, str):
            try:
                # Remove currency symbols (common ones), commas, and whitespace.
                # Be careful not to remove decimal points if they are part of the number.
                cleaned_val_str = re.sub(r"[^\d\.]", "", val_str) # Keep digits and decimal point
                if cleaned_val_str: # If something remains after stripping non-numeric
                    if '.' in cleaned_val_str:
                        processed["agreement_value"] = float(cleaned_val_str)
                    else:
                        processed["agreement_value"] = int(cleaned_val_str)
                else: # String was present but became empty after cleaning (e.g., "None" or "N/A")
                    processed["agreement_value"] = None
            except (ValueError, TypeError) as e:
                logger.warning(f"Could not parse agreement_value '{val_str}' to number: {e}. Setting to null.")
                processed["agreement_value"] = None
        else: # Not a string or number
             processed["agreement_value"] = None
    else: # Key was missing or explicitly null
        processed["agreement_value"] = None


    # Dates - uses utils.parse_date_string which returns YYYY-MM-DD or original/None
    processed["agreement_start_date"] = utils.parse_date_string(raw_data.get("agreement_start_date"))
    processed["agreement_end_date"] = utils.parse_date_string(raw_data.get("agreement_end_date"))

    # Renewal Notice Days
    renewal_days_val = raw_data.get("renewal_notice_days")
    if renewal_days_val is not None:
        if isinstance(renewal_days_val, int):
            processed["renewal_notice_days"] = renewal_days_val
        elif isinstance(renewal_days_val, str):
            try:
                # Try direct conversion first
                processed["renewal_notice_days"] = int(renewal_days_val)
            except ValueError:
                # If direct fails, try to extract digits (e.g., from "15 days")
                match = re.search(r'\d+', renewal_days_val)
                if match:
                    try:
                        processed["renewal_notice_days"] = int(match.group(0))
                    except ValueError: # Should not happen if regex matches \d+
                         processed["renewal_notice_days"] = None
                else: # No digits found
                    processed["renewal_notice_days"] = None
        else: # Not an int or string
            processed["renewal_notice_days"] = None
    else: # Key was missing or explicitly null
        processed["renewal_notice_days"] = None

    logger.debug(f"Finished processing data: {processed}")
    return processed


def extract_contract_details_from_text(
    input_text: str,
    vector_db_manager: VectorDBManager,
    use_rag: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Extracts contract details from a given text, optionally using RAG.
    Returns a dictionary suitable for ExtractedDetails Pydantic model.
    """
    if not input_text:
        logger.warning("Input text is empty. Cannot extract details.")
        return None

    retrieved_contexts = None
    if use_rag:
        logger.info("RAG enabled. Querying vector DB for similar documents...")
        retrieved_contexts = vector_db_manager.query_documents(input_text, k=config.TOP_K_RETRIEVAL)
        if retrieved_contexts:
            logger.info(f"Retrieved {len(retrieved_contexts)} documents for RAG context.")
            for i, ctx in enumerate(retrieved_contexts):
                logger.debug(f"RAG Context {i+1}: ID={ctx.get('id')}, Dist={ctx.get('distance', -1.0):.4f}, File={ctx.get('metadata', {}).get('file_name', 'N/A')}")
        else:
            logger.info("No relevant documents found for RAG context.")

    prompt = build_extraction_prompt(input_text, retrieved_contexts)
    # For debugging prompt length or content:
    # logger.debug(f"Generated LLM Prompt (length {len(prompt)}):\n{prompt[:1000]}...")

    logger.info("Sending request to Gemini for data extraction...")
    llm_response_text = get_gemini_generation(prompt)

    if not llm_response_text:
        logger.error("LLM did not return a response.")
        return None

    logger.debug(f"LLM Raw Response:\n{llm_response_text}")

    raw_extracted_data_dict = utils.robust_json_parser(llm_response_text)

    if not raw_extracted_data_dict:
        logger.error("Failed to parse JSON from LLM response.")
        # You might want to attempt to re-prompt or use a fallback parser here in a real system
        return None

    logger.info(f"Successfully parsed raw data from LLM: {raw_extracted_data_dict}")
    
    # Ensure all expected keys are present in the processed dict, even if value is None
    expected_keys = ["agreement_value", "agreement_start_date", "agreement_end_date", "renewal_notice_days"]
    processed_data = process_extracted_data(raw_extracted_data_dict)
    
    final_data = {key: processed_data.get(key) for key in expected_keys}
    logger.info(f"Processed and finalized extracted data: {final_data}")
    
    return final_data