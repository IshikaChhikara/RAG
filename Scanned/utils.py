"""
Utility functions for the Scanned PDF extraction pipeline (OCR-based).

Provides:
- Logging configuration
- Path and file operations
- Filename sanitization
- PDF validation
- Incremental processing checks
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Optional


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(log_file_path: Path) -> logging.Logger:
    """
    Configure logging with both file and console handlers.
    
    Creates a logger that writes to both file and console with appropriate
    verbosity levels (DEBUG for file, INFO for console).
    
    Args:
        log_file_path: Path to the log file.
        
    Returns:
        Configured logger instance.
        
    Raises:
        OSError: If log file cannot be created.
    """
    logger = logging.getLogger("rag_pipeline_scanned")
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create log file parent directory if needed
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # File handler - DEBUG level for detailed logs
    file_handler = logging.FileHandler(log_file_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler - INFO level for user feedback
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter with timestamp
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


# ============================================================================
# DIRECTORY AND PATH OPERATIONS
# ============================================================================

def create_pipeline_directories(base_path: Path) -> dict[str, Path]:
    """
    Create the complete directory structure for the Scanned OCR pipeline.

    base_path is already the scanned folder, so don't append
    "scanned" again.
    """

    directories = {
        "extracted_raw": base_path / "data" / "extracted_raw",
        "cleaned": base_path / "data" / "cleaned",
        "logs": base_path / "logs",
    }

    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    return directories


# ============================================================================
# FILENAME AND PATH OPERATIONS
# ============================================================================

def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename for use in directory names.
    
    Removes file extension and replaces invalid characters.
    
    Args:
        filename: Original filename (with or without extension).
        
    Returns:
        Sanitized filename suitable for directory names.
        
    Examples:
        "My Document.pdf" → "My_Document"
        "file-v2.pdf" → "file-v2"
    """
    # Remove .pdf extension
    name = filename.rsplit(".", 1)[0] if "." in filename else filename
    
    # Remove invalid characters for directory names
    name = re.sub(r'[<>:"/\\|?*]', "", name)
    
    # Replace spaces with underscores for cleaner paths
    name = name.replace(" ", "_")
    
    return name


def get_pdf_stem(pdf_path: Path) -> str:
    """
    Get the stem (filename without extension) of a PDF.
    
    Args:
        pdf_path: Path to the PDF file.
        
    Returns:
        Filename without .pdf extension.
    """
    return pdf_path.stem


# ============================================================================
# FILE VALIDATION
# ============================================================================

def validate_pdf_path(pdf_path: Path) -> bool:
    """
    Validate that a PDF file exists and is readable.
    
    Checks:
    - File exists
    - Is a regular file (not directory)
    - Has read permissions
    
    Args:
        pdf_path: Path to PDF file to validate.
        
    Returns:
        True if file is valid, False otherwise.
    """
    if not pdf_path.exists():
        return False
    
    if not pdf_path.is_file():
        return False
    
    if not os.access(pdf_path, os.R_OK):
        return False
    
    return True


# ============================================================================
# INCREMENTAL PROCESSING CHECKS (METADATA-BASED)
# ============================================================================

def is_extraction_complete(
    pdf_name: str,
    extracted_raw_dir: Path,
    logger: logging.Logger
) -> bool:
    """
    Check if extraction for a PDF is already complete and successful.
    
    A PDF is considered successfully extracted if:
    - Its output directory exists
    - metadata.json exists
    - metadata.json has text_extracted=true
    - processing_status="completed"
    
    This allows retry of failed extractions.
    
    Args:
        pdf_name: Name of the PDF file (with .pdf extension).
        extracted_raw_dir: Path to extracted_raw directory.
        logger: Logger instance.
        
    Returns:
        True if extraction is complete and successful, False otherwise.
    """
    sanitized_name = sanitize_filename(pdf_name)
    output_dir = extracted_raw_dir / sanitized_name
    metadata_path = output_dir / "metadata.json"
    
    if not metadata_path.exists():
        return False
    
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        # Check for successful extraction
        is_complete = (
            metadata.get("text_extracted", False) and
            metadata.get("processing_status") == "completed"
        )
        
        return is_complete
    
    except Exception as e:
        logger.debug(f"Error reading metadata for {pdf_name}: {str(e)}")
        return False


def is_cleaning_complete(
    pdf_name: str,
    cleaned_dir: Path,
    logger: logging.Logger
) -> bool:
    """
    Check if cleaning for a PDF is already complete and successful.
    
    A PDF is considered successfully cleaned if:
    - Its output directory exists
    - metadata.json exists
    - metadata.json has cleaned=true
    - processing_status="completed"
    
    This allows retry of failed cleaning operations.
    
    Args:
        pdf_name: Name of the PDF file (with .pdf extension).
        cleaned_dir: Path to cleaned directory.
        logger: Logger instance.
        
    Returns:
        True if cleaning is complete and successful, False otherwise.
    """
    sanitized_name = sanitize_filename(pdf_name)
    output_dir = cleaned_dir / sanitized_name
    metadata_path = output_dir / "metadata.json"
    
    if not metadata_path.exists():
        return False
    
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        # Check for successful cleaning
        is_complete = (
            metadata.get("cleaned", False) and
            metadata.get("processing_status") == "completed"
        )
        
        return is_complete
    
    except Exception as e:
        logger.debug(f"Error reading metadata for {pdf_name}: {str(e)}")
        return False


# ============================================================================
# PIPELINE STATUS AND REPORTING
# ============================================================================

def get_extraction_summary(
    extracted_raw_dir: Path,
    logger: logging.Logger
) -> dict[str, int]:
    """
    Get statistics about extracted PDFs.
    
    Args:
        extracted_raw_dir: Path to extracted_raw directory.
        logger: Logger instance.
        
    Returns:
        Dictionary with extraction statistics.
    """
    try:
        pdf_dirs = [d for d in extracted_raw_dir.iterdir() if d.is_dir()]
        
        return {
            "total_extracted": len(pdf_dirs),
            "has_text": sum(
                1 for d in pdf_dirs if (d / "text.txt").exists()
            ),
            "has_images": sum(
                1 for d in pdf_dirs if (d / "images").exists()
            ),
        }
    except Exception as e:
        logger.warning(f"Error getting extraction summary: {str(e)}")
        return {}


def get_cleaning_summary(
    cleaned_dir: Path,
    logger: logging.Logger
) -> dict[str, int]:
    """
    Get statistics about cleaned PDFs.
    
    Args:
        cleaned_dir: Path to cleaned directory.
        logger: Logger instance.
        
    Returns:
        Dictionary with cleaning statistics.
    """
    try:
        pdf_dirs = [d for d in cleaned_dir.iterdir() if d.is_dir()]
        
        return {
            "total_cleaned": len(pdf_dirs),
            "has_text": sum(
                1 for d in pdf_dirs if (d / "text.txt").exists()
            ),
            "has_images": sum(
                1 for d in pdf_dirs if (d / "images").exists()
            ),
        }
    except Exception as e:
        logger.warning(f"Error getting cleaning summary: {str(e)}")
        return {}
