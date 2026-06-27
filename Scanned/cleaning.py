"""
Text cleaning and normalization module for Scanned PDFs.

Handles cleaning of extracted text from OCR to improve readability and
prepare content for downstream RAG indexing and embedding.

Cleaning operations include:
- Whitespace normalization
- Header/footer removal heuristics
- Blank line consolidation
- Paragraph preservation
- Special character handling (OCR artifacts)

Metadata is only updated on successful completion to ensure correct status.
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from utils import sanitize_filename


def clean_text_content(text: str) -> str:
    """
    Comprehensively clean extracted OCR text for readability.
    
    Performs the following transformations:
    - Normalizes line endings
    - Replaces PDF bullet characters with "-"
    - Replaces checkmark symbols with "[DONE]"
    - Removes unknown font artifacts
    - Normalizes whitespace while preserving structure
    - Preserves numbered lists (1., A., etc.)
    - Consolidates excessive blank lines
    - Preserves paragraph structure
    - Handles OCR-specific artifacts
    
    Args:
        text: Raw extracted text from OCR.
        
    Returns:
        Cleaned, readable text with proper formatting.
    """
    if not text:
        return ""
    
    # Step 1: Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    
    # Step 2: Replace bullet characters with "-"
    bullet_chars = [
        "•",
        "◦",
        "▪",
        "●",
        "○",
        "■",
        "◆",
        "►",
    ]
    for bullet in bullet_chars:
        # Replace bullet followed by space with "- "
        text = re.sub(
            rf"^\s*{re.escape(bullet)}\s+",
            "- ",
            text,
            flags=re.MULTILINE
        )
        # Replace standalone bullets
        text = text.replace(bullet, "-")
    
    # Step 3: Replace checkmark symbols with "[DONE]"
    checkmark_chars = [
        "✓",
        "✔",
        "☑",
        "✅",
    ]
    for checkmark in checkmark_chars:
        text = re.sub(rf"\s*{re.escape(checkmark)}\s*", "- ", text)
    
    # Step 4: Remove unknown font artifacts (only when standalone)
    artifact_chars = [
        "□",
        "▯",
    ]
    
    for artifact in artifact_chars:
        # Standalone artifact on its own line
        pattern = rf"^\s*{re.escape(artifact)}\s*$"
        text = re.sub(pattern, "", text, flags=re.MULTILINE)
        
        # Artifact surrounded by spaces within a line
        pattern = rf"\s+{re.escape(artifact)}\s+"
        text = re.sub(pattern, " ", text)
        
        # Artifact at beginning of line with spaces
        pattern = rf"^\s*{re.escape(artifact)}\s+"
        text = re.sub(pattern, "", text, flags=re.MULTILINE)
        
        # Artifact at end of line with spaces
        pattern = rf"\s+{re.escape(artifact)}\s*$"
        text = re.sub(pattern, "", text, flags=re.MULTILINE)
    
    # Step 5: Clean up spacing within lines
    lines = text.split("\n")
    cleaned_lines = []
    
    for line in lines:
        # Preserve leading spaces for indentation (lists, etc.)
        leading_spaces = len(line) - len(line.lstrip())
        content = line.lstrip()
        
        # Remove multiple spaces within content
        content = re.sub(r"  +", " ", content)
        
        # Remove trailing spaces
        content = content.rstrip()
        
        # Reconstruct line with preserved indentation
        if content:
            cleaned_line = " " * leading_spaces + content
        else:
            cleaned_line = ""
        
        cleaned_lines.append(cleaned_line)
    
    text = "\n".join(cleaned_lines)
    
    # Step 6: Normalize blank lines
    # Replace multiple consecutive blank lines with max 2 newlines (one blank line)
    text = re.sub(r"\n\n\n+", "\n\n", text)
    
    # Step 7: Remove spaces before newlines
    text = re.sub(r" +\n", "\n", text)
    
    # Step 8: Final trim
    text = text.strip()
    
    return text


def remove_repeated_headers_footers(text: str, threshold: int = 3) -> str:
    """
    Attempt to remove repeated headers and footers from text.
    
    Uses heuristic detection: lines that appear frequently in the same position
    (indicating headers/footers) are removed.
    
    Args:
        text: The text to clean.
        threshold: Minimum occurrences to consider a line as header/footer.
        
    Returns:
        Text with suspected headers/footers removed.
    """
    if not text:
        return ""
    
    lines = text.split("\n")
    
    if len(lines) < threshold:
        return text
    
    # Count line occurrences
    line_counts = {}
    for line in lines:
        stripped = line.strip()
        if stripped and len(stripped) < 100:  # Likely header/footer
            line_counts[stripped] = line_counts.get(stripped, 0) + 1
    
    # Find lines that appear too frequently (likely headers/footers)
    suspect_lines = {
        line for line, count in line_counts.items()
        if count >= threshold and len(line) < 80
    }
    
    # Filter out suspect lines
    filtered_lines = [
        line for line in lines
        if line.strip() not in suspect_lines
    ]
    
    # Rejoin and consolidate blank lines
    result = "\n".join(filtered_lines)
    result = re.sub(r"\n\n\n+", "\n\n", result)
    
    return result.strip()


def process_extracted_pdf(
    raw_pdf_dir: Path,
    cleaned_pdf_dir: Path,
    logger: logging.Logger
) -> bool:
    """
    Process a single extracted PDF directory through the cleaning pipeline.
    
    Reads from raw extraction output and produces cleaned output.
    Metadata is only updated on successful completion.
    
    Args:
        raw_pdf_dir: Path to the extracted raw PDF directory.
        cleaned_pdf_dir: Path to the output cleaned PDF directory.
        logger: Logger instance.
        
    Returns:
        True if cleaning successful, False otherwise.
    """
    pdf_name = raw_pdf_dir.name
    cleaning_errors = []
    
    logger.info(f"Starting cleaning: {pdf_name}")
    
    try:
        # Read raw metadata
        raw_metadata_path = raw_pdf_dir / "metadata.json"
        if not raw_metadata_path.exists():
            error_msg = f"No metadata found for {pdf_name}"
            logger.error(error_msg)
            cleaning_errors.append(error_msg)
            return False
        
        try:
            with open(raw_metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in metadata for {pdf_name}: {str(e)}"
            logger.error(error_msg)
            cleaning_errors.append(error_msg)
            return False
        
        # Read raw text
        raw_text_path = raw_pdf_dir / "text.txt"
        if not raw_text_path.exists():
            logger.warning(f"No text file found for {pdf_name}")
            text_content = ""
        else:
            try:
                text_content = raw_text_path.read_text(encoding="utf-8")
            except IOError as e:
                error_msg = f"Could not read text file for {pdf_name}: {str(e)}"
                logger.error(error_msg)
                cleaning_errors.append(error_msg)
                return False
        
        # Clean text
        logger.debug(f"Applying text cleaning: {pdf_name}")
        cleaned_text = clean_text_content(text_content)
        
        # Optional: Remove repeated headers/footers
        logger.debug(f"Removing suspected headers/footers: {pdf_name}")
        cleaned_text = remove_repeated_headers_footers(cleaned_text, threshold=3)
        
        # Create output directory
        cleaned_pdf_dir.mkdir(parents=True, exist_ok=True)
        
        # Save cleaned text
        cleaned_text_path = cleaned_pdf_dir / "text.txt"
        try:
            cleaned_text_path.write_text(cleaned_text, encoding="utf-8")
            logger.info(f"Cleaned text saved: {cleaned_text_path}")
        except IOError as e:
            error_msg = f"Could not write cleaned text for {pdf_name}: {str(e)}"
            logger.error(error_msg)
            cleaning_errors.append(error_msg)
            return False
        
        # Copy images if they exist
        raw_images_dir = raw_pdf_dir / "images"
        if raw_images_dir.exists():
            import shutil
            cleaned_images_dir = cleaned_pdf_dir / "images"
            try:
                if cleaned_images_dir.exists():
                    shutil.rmtree(cleaned_images_dir)
                shutil.copytree(raw_images_dir, cleaned_images_dir)
                logger.info(f"Copied images: {cleaned_images_dir}")
            except Exception as e:
                error_msg = f"Failed to copy images for {pdf_name}: {str(e)}"
                logger.warning(error_msg)
                cleaning_errors.append(error_msg)
                # Don't fail on image copy errors
        
        # Update and save metadata (only on success)
        metadata["cleaned"] = True
        metadata["processing_status"] = "completed"
        metadata["cleaned_at"] = datetime.now().isoformat()
        metadata["last_updated"] = datetime.now().isoformat()
        metadata["raw_text_size"] = len(text_content)
        metadata["cleaned_text_size"] = len(cleaned_text)
        metadata["errors"] = cleaning_errors
        
        cleaned_metadata_path = cleaned_pdf_dir / "metadata.json"
        try:
            with open(cleaned_metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Metadata updated: {cleaned_metadata_path}")
        except IOError as e:
            error_msg = f"Could not write metadata for {pdf_name}: {str(e)}"
            logger.error(error_msg)
            # This is a critical error - cleaning succeeded but metadata failed
            return False
        
        logger.info(f"Completed cleaning: {pdf_name}")
        return True
    
    except IOError as e:
        error_msg = f"File I/O error processing {pdf_name}: {str(e)}"
        logger.error(error_msg)
        cleaning_errors.append(error_msg)
        return False
    except Exception as e:
        error_msg = f"Unexpected error cleaning {pdf_name}: {str(e)}"
        logger.error(error_msg)
        cleaning_errors.append(error_msg)
        return False


def process_all_extracted(
    raw_dir: Path,
    cleaned_dir: Path,
    logger: logging.Logger
) -> tuple[int, int]:
    """
    Process all extracted PDFs from raw to cleaned state.
    
    Only processes PDFs not yet successfully cleaned (incremental processing).
    Uses proper metadata-based checks to allow retry of failed operations.
    
    Args:
        raw_dir: Path to extracted_raw directory.
        cleaned_dir: Path to cleaned directory.
        logger: Logger instance.
        
    Returns:
        Tuple of (total_processed, successful_cleaned).
    """
    if not raw_dir.exists():
        logger.warning(f"Raw extraction directory not found: {raw_dir}")
        return 0, 0
    
    # Find all extracted PDFs
    pdf_dirs = sorted([d for d in raw_dir.iterdir() if d.is_dir()])
    
    if not pdf_dirs:
        logger.info("No extracted PDFs found to clean")
        return 0, 0
    
    total = len(pdf_dirs)
    successful = 0
    
    logger.info(f"Found {total} extracted PDF(s) to process")
    
    for idx, pdf_dir in enumerate(pdf_dirs, 1):
        cleaned_pdf_dir = cleaned_dir / pdf_dir.name
        
        # Skip if already successfully cleaned
        cleaned_metadata = cleaned_pdf_dir / "metadata.json"
        if cleaned_metadata.exists():
            try:
                with open(cleaned_metadata, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    # Check if cleaning was successful
                    if meta.get("cleaned") and meta.get("processing_status") == "completed":
                        logger.info(f"[{idx}/{total}] Already cleaned: {pdf_dir.name}")
                        successful += 1
                        continue
            except Exception as e:
                logger.debug(f"Error checking cleaned status: {str(e)}")
                # If we can't read metadata, assume we need to reprocess
        
        logger.info(f"[{idx}/{total}] Processing: {pdf_dir.name}")
        
        if process_extracted_pdf(pdf_dir, cleaned_pdf_dir, logger):
            successful += 1
            logger.info(f"✓ Cleaned: {pdf_dir.name}")
        else:
            logger.error(f"✗ Failed to clean: {pdf_dir.name}")
    
    return total, successful
