# app/utils.py
import json
import logging
import os
from datetime import datetime
from typing import Optional, Dict, Any

# Import config from the app package
from . import config # Use relative import

def setup_logging():
    """Sets up basic logging."""
    # Ensure log directory exists
    log_dir = os.path.dirname(config.LOG_FILE_PATH)
    if log_dir and not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir, exist_ok=True)
        except OSError as e:
            print(f"Warning: Could not create log directory {log_dir}: {e}")
            # Fallback to no file handler if dir creation fails, or log to current dir.
            # For simplicity, we'll let basicConfig handle it if path is invalid.

    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL, logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(config.LOG_FILE_PATH, mode='a'), # Append mode
            logging.StreamHandler() # To console
        ]
    )
    # Suppress overly verbose logs from libraries if necessary
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING) # For google-generativeai library

def robust_json_parser(llm_output_string: str) -> Optional[Dict[str, Any]]:
    """
    Attempts to parse JSON from a string, handling common LLM output variations.
    """
    if not llm_output_string:
        return None
    try:
        # Common case: LLM wraps JSON in markdown code blocks
        cleaned_output = llm_output_string.strip()
        if cleaned_output.startswith("```json"):
            cleaned_output = cleaned_output[len("```json"):].strip()
            if cleaned_output.endswith("```"):
                cleaned_output = cleaned_output[:-len("```")].strip()
        elif cleaned_output.startswith("```") and cleaned_output.endswith("```"):
             cleaned_output = cleaned_output[3:-3].strip()


        # Find the first '{' and last '}'
        json_start = cleaned_output.find('{')
        json_end = cleaned_output.rfind('}') + 1

        if json_start != -1 and json_end != -1 and json_start < json_end:
            json_str = cleaned_output[json_start:json_end]
            return json.loads(json_str)
        else:
            logging.warning(f"Could not find a valid JSON structure in string: {llm_output_string[:200]}...")
            return None
    except json.JSONDecodeError as e:
        logging.error(f"JSONDecodeError: {e} while parsing: {llm_output_string[:200]}...")
        return None
    except Exception as e:
        logging.error(f"Unexpected error during JSON parsing: {e} - {llm_output_string[:200]}...")
        return None


def parse_date_string(date_str: Optional[str]) -> Optional[str]:
    """
    Tries to parse a date string into YYYY-MM-DD format.
    Handles common variations. Returns original string if unparseable by common formats.
    """
    if not date_str or not isinstance(date_str, str):
        return None # Or return date_str if you want to pass it through

    date_str = date_str.strip()
    if not date_str: # Empty string after strip
        return None

    common_formats = [
        "%d.%m.%Y", "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d",
        "%Y.%m.%d", "%d-%m-%Y", "%m-%d-%Y",
        "%b %d, %Y", "%d %b %Y", "%B %d, %Y", "%d %B %Y", # May 20, 2007 or May 20, 2007
        "%d.%m.%y", "%m/%d/%y", # Short year
    ]

    for fmt in common_formats:
        try:
            dt_obj = datetime.strptime(date_str, fmt)
            # Handle short year (e.g., '07' -> 2007) - strptime does this by default for %y
            return dt_obj.strftime("%Y-%m-%d")
        except ValueError:
            continue
    
    # Attempt to handle "31.11.2009" (excel text has this, which is invalid)
    # This kind of specific error correction is tricky. LLM should ideally output valid dates.
    # For now, we'll log and return original if no common format matches.
    logging.warning(f"Could not parse date string '{date_str}' into YYYY-MM-DD using common formats. Returning original.")
    return date_str # Return original string if not parseable by defined formats