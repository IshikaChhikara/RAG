"""
Unified PDF extraction module for non-scanned PDFs.
Handles text, table, and image extraction with a modular design.

This module provides functions to extract different content types from PDFs
and is designed to be extended for additional extraction methods in the future.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pdfplumber
import pandas as pd
import fitz

from utils import sanitize_filename, is_pdf_searchable, route_pdf


# ============================================================================
# TEXT EXTRACTION
# ============================================================================

def extract_text(pdf_path: Path, logger: logging.Logger) -> Optional[str]:
    """
    Extract text content from a non-scanned PDF using pdfplumber.
    
    Extracts text page by page with page markers to preserve document structure.
    
    Args:
        pdf_path: Path to the PDF file.
        logger: Logger instance for recording extraction progress.
        
    Returns:
        Extracted and cleaned text content, or None if extraction fails.
        
    Raises:
        Handled internally with logging. Returns None on error.
    """
    text_content = ""
    
    try:
        logger.debug(f"Starting text extraction: {pdf_path.name}")
        
        with pdfplumber.open(str(pdf_path)) as pdf:
            page_count = len(pdf.pages)
            
            if page_count == 0:
                logger.warning(f"PDF has no pages: {pdf_path.name}")
                return None
            
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    page_text = page.extract_text()
                    
                    if page_text:
                        text_content += f"\n{'='*60}\nPAGE {page_num}\n{'='*60}\n\n"
                        text_content += page_text
                        text_content += "\n"
                        logger.debug(f"Extracted text from page {page_num}")
                    
                except Exception as e:
                    logger.warning(
                        f"Failed to extract text from page {page_num} "
                        f"in {pdf_path.name}: {str(e)}"
                    )
                    continue
        
        if not text_content.strip():
            logger.warning(f"No text extracted from {pdf_path.name}")
            return None
        
        logger.info(
            f"Successfully extracted text from {pdf_path.name} "
            f"({page_count} pages)"
        )
        
        return text_content.strip()
    
    except pdfplumber.exceptions.PDFException as e:
        logger.error(
            f"PDFPlumber error while processing {pdf_path.name}: {str(e)}"
        )
        return None
    except PermissionError:
        logger.error(f"Permission denied reading {pdf_path.name}")
        return None
    except Exception as e:
        logger.error(
            f"Unexpected error extracting text from {pdf_path.name}: {str(e)}"
        )
        return None


# ============================================================================
# TABLE EXTRACTION
# ============================================================================

def extract_tables(
    pdf_path: Path, logger: logging.Logger
) -> dict[int, list[pd.DataFrame]]:
    """
    Extract tables from a non-scanned PDF using pdfplumber.
    
    Detects and extracts all tables across all pages with error resilience.
    Returns a dictionary mapping page numbers to lists of tables found on that page.
    
    Args:
        pdf_path: Path to the PDF file.
        logger: Logger instance for recording extraction progress.
        
    Returns:
        Dictionary mapping page_num (int) to list of DataFrames for that page.
        Empty dict if no tables found or extraction fails.
        
    Raises:
        Handled internally with logging. Returns empty dict on error.
    """
    tables_by_page = {}
    
    try:
        logger.debug(f"Starting table extraction: {pdf_path.name}")
        
        with pdfplumber.open(str(pdf_path)) as pdf:
            total_pages = len(pdf.pages)
            
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    page_tables = page.extract_tables()
                    
                    if page_tables:
                        tables_for_page = []
                        for table_idx, table in enumerate(page_tables):
                            df = _convert_table_to_dataframe(
                                table, page_num, table_idx, logger, pdf_path
                            )
                            
                            if df is not None and not df.empty:
                                tables_for_page.append(df)
                                logger.debug(
                                    f"Extracted table {table_idx + 1} from page {page_num}"
                                )
                        
                        if tables_for_page:
                            tables_by_page[page_num] = tables_for_page
                    
                except Exception as e:
                    logger.warning(
                        f"Failed to extract tables from page {page_num} "
                        f"in {pdf_path.name}: {str(e)}"
                    )
                    continue
        
        if tables_by_page:
            total_tables = sum(len(tables) for tables in tables_by_page.values())
            logger.info(f"Found {total_tables} table(s) in {pdf_path.name}")
        else:
            logger.info(f"No tables found in {pdf_path.name}")
        
        return tables_by_page
    
    except pdfplumber.exceptions.PDFException as e:
        logger.error(
            f"PDFPlumber error while processing {pdf_path.name}: {str(e)}"
        )
        return {}
    except PermissionError:
        logger.error(f"Permission denied reading {pdf_path.name}")
        return {}
    except Exception as e:
        logger.error(
            f"Unexpected error extracting tables from {pdf_path.name}: {str(e)}"
        )
        return {}


def _convert_table_to_dataframe(
    table: list[list],
    page_num: int,
    table_idx: int,
    logger: logging.Logger,
    pdf_path: Path
) -> Optional[pd.DataFrame]:
    """
    Convert raw PDF table data to a cleaned pandas DataFrame.
    
    Internal helper function for table extraction. Cleans empty rows/columns
    and applies minimal validation.
    
    Args:
        table: Raw table data as list of rows (list of lists).
        page_num: Page number where the table was found.
        table_idx: Index of the table on that page.
        logger: Logger instance.
        pdf_path: Path to the original PDF (for error messages).
        
    Returns:
        Cleaned DataFrame or None if conversion fails or table is empty.
    """
    try:
        if not table or len(table) < 1:
            return None
        
        # Use first row as headers
        headers = table[0]
        data_rows = table[1:]
        
        # Filter out completely empty rows
        data_rows = [
            row for row in data_rows
            if any(cell for cell in row)
        ]
        
        if not data_rows:
            return None
        
        df = pd.DataFrame(data_rows, columns=headers)
        
        # Remove completely empty columns
        df = df.dropna(axis=1, how="all")
        
        # Remove completely empty rows
        df = df.dropna(axis=0, how="all")
        
        return df if not df.empty else None
    
    except Exception as e:
        logger.warning(
            f"Failed to convert table on page {page_num} "
            f"(index {table_idx}) in {pdf_path.name} to DataFrame: {str(e)}"
        )
        return None


# ============================================================================
# IMAGE EXTRACTION
# ============================================================================

def extract_images(
    pdf_path: Path,
    output_images_dir: Path,
    logger: logging.Logger
) -> int:
    """
    Extract all images from a non-scanned PDF using PyMuPDF (fitz).
    
    Saves images with page and index information in the provided directory.
    Automatically handles color space conversions.
    
    Args:
        pdf_path: Path to the PDF file.
        output_images_dir: Directory where images will be saved.
        logger: Logger instance for recording extraction progress.
        
    Returns:
        Number of images successfully extracted. 0 if no images found or error.
        
    Raises:
        Handled internally with logging. Returns 0 on error.
    """
    image_count = 0
    
    try:
        logger.debug(f"Starting image extraction: {pdf_path.name}")
        
        doc = fitz.open(str(pdf_path))
        
        try:
            total_pages = len(doc)
            
            for page_num in range(total_pages):
                try:
                    page = doc[page_num]
                    image_list = page.get_images()
                    
                    for img_idx, img in enumerate(image_list):
                        try:
                            xref = img[0]
                            pix = fitz.Pixmap(doc, xref)
                            
                            # Convert CMYK to RGB if needed
                            if pix.n - pix.alpha > 3:
                                pix = fitz.Pixmap(fitz.csRGB, pix)
                            
                            # Create output directory for this PDF
                            output_dir = output_images_dir
                            output_dir.mkdir(parents=True, exist_ok=True)
                            
                            # Save image with descriptive naming
                            filename = (
                                f"{pdf_path.stem}_"
                                f"page_{page_num + 1}_img_{img_idx + 1}.png"
                            )
                            image_path = output_dir / filename
                            
                            pix.save(str(image_path))
                            image_count += 1
                            
                            logger.debug(f"Extracted image: {filename}")
                        
                        except Exception as e:
                            logger.warning(
                                f"Failed to extract image {img_idx + 1} "
                                f"from page {page_num + 1} in "
                                f"{pdf_path.name}: {str(e)}"
                            )
                            continue
                    
                except Exception as e:
                    logger.warning(
                        f"Failed to process page {page_num + 1} "
                        f"in {pdf_path.name}: {str(e)}"
                    )
                    continue
            
            if image_count > 0:
                logger.info(
                    f"Extracted {image_count} image(s) from {pdf_path.name}"
                )
            else:
                logger.info(f"No images found in {pdf_path.name}")
        
        finally:
            doc.close()
        
        return image_count
    
    except fitz.FileError:
        logger.error(f"Cannot open PDF file: {pdf_path.name}")
        return 0
    except PermissionError:
        logger.error(f"Permission denied reading {pdf_path.name}")
        return 0
    except Exception as e:
        logger.error(
            f"Unexpected error extracting images from {pdf_path.name}: {str(e)}"
        )
        return 0


# ============================================================================
# TABLE FORMATTING
# ============================================================================

def _format_table_as_markdown(df: pd.DataFrame) -> str:
    """
    Format a DataFrame as a markdown table.
    
    Creates a clean, human-readable markdown table suitable for RAG chunking.
    
    Args:
        df: DataFrame to format.
        
    Returns:
        Markdown-formatted table string.
    """
    try:
        # Convert DataFrame to markdown format
        markdown_lines = []
        
        # Header row
        header = "| " + " | ".join(str(col) for col in df.columns) + " |"
        markdown_lines.append(header)
        
        # Separator row
        separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"
        markdown_lines.append(separator)
        
        # Data rows
        for _, row in df.iterrows():
            row_str = "| " + " | ".join(str(val) for val in row) + " |"
            markdown_lines.append(row_str)
        
        return "\n".join(markdown_lines)
    
    except Exception:
        # Fallback to to_string if markdown conversion fails
        return df.to_string()


def _format_tables_by_page(
    tables_by_page: dict[int, list[pd.DataFrame]]
) -> str:
    """
    Format extracted tables organized by page into readable output.
    
    Internal helper to convert page-wise tables to text format with
    proper page markers and markdown formatting.
    
    Args:
        tables_by_page: Dictionary mapping page_num to list of DataFrames.
        
    Returns:
        Formatted tables as text, or empty string if no tables.
    """
    if not tables_by_page:
        return ""
    
    tables_text = ""
    
    for page_num in sorted(tables_by_page.keys()):
        tables = tables_by_page[page_num]
        
        for table_idx, df in enumerate(tables, 1):
            tables_text += f"\n--- Table {table_idx} (from Page {page_num}) ---\n\n"
            tables_text += _format_table_as_markdown(df)
            tables_text += "\n"
    
    return tables_text.strip()


# ============================================================================
# COMBINED PDF PROCESSING
# ============================================================================

def process_pdf(
    pdf_path: Path,
    output_raw_dir: Path,
    logger: logging.Logger
) -> bool:
    """
    Process a single PDF: extract text, tables, and images with page-wise output.
    
    Creates a structured output directory with:
    - text.txt: Combined text and table content organized by page
    - images/: Extracted images
    - metadata.json: Extraction metadata (only on success)
    
    Metadata is only written upon successful extraction to ensure correct status.
    Failed extractions do not create metadata, allowing retry on next run.
    
    Args:
        pdf_path: Path to the PDF file.
        output_raw_dir: Base directory for raw extracted output.
        logger: Logger instance.
        
    Returns:
        True if processing successful, False otherwise.
    """
    pdf_name = pdf_path.name
    sanitized_name = sanitize_filename(pdf_name)
    
    # Route PDF to appropriate pipeline
    routing = route_pdf(pdf_path, logger)
    if routing == "scanned":
        logger.warning(f"Skipping {pdf_name} - OCR support not yet implemented")
        return False
    
    # Create PDF-specific output directory
    pdf_output_dir = output_raw_dir / sanitized_name
    pdf_output_dir.mkdir(parents=True, exist_ok=True)
    
    images_output_dir = pdf_output_dir / "images"
    
    logger.info(f"Starting processing: {pdf_name}")
    
    extraction_errors = []
    
    try:
        # Extract text
        logger.debug(f"Text extraction starting: {pdf_name}")
        text_content = extract_text(pdf_path, logger)
        text_extracted = bool(text_content)
        
        # Extract tables (returns dict with page_num: [tables])
        logger.debug(f"Table extraction starting: {pdf_name}")
        tables_by_page = extract_tables(pdf_path, logger)
        tables_extracted = len(tables_by_page) > 0
        
        # Format tables for output (page-wise)
        tables_text = _format_tables_by_page(tables_by_page)
        
        # Combine text and tables (tables appear after text, organized by page)
        combined_content = text_content or ""
        if tables_text:
            combined_content += "\n\n" + tables_text
        
        # Save combined content
        text_output_path = pdf_output_dir / "text.txt"
        if combined_content.strip():
            text_output_path.write_text(combined_content, encoding="utf-8")
            logger.info(f"Text saved: {text_output_path}")
        else:
            extraction_errors.append("No text content extracted")
            logger.warning(f"No text content to save for {pdf_name}")
        
        # Extract images
        logger.debug(f"Image extraction starting: {pdf_name}")
        image_count = extract_images(pdf_path, images_output_dir, logger)
        images_extracted = image_count > 0
        
        # Get page count
        page_count = _get_page_count(pdf_path, logger)
        
        # Determine if extraction should be considered successful
        # Success = at least text or tables were extracted
        extraction_successful = text_extracted or tables_extracted
        
        if not extraction_successful:
            extraction_errors.append("No text or tables extracted")
            logger.warning(f"No extractable content found in {pdf_name}")
            return False
        
        # Only write metadata on successful extraction
        metadata = {
            "pdf_name": pdf_name,
            "pdf_path": str(pdf_path),
            "pages": page_count,
            "text_extracted": text_extracted,
            "tables_extracted": tables_extracted,
            "images_extracted": images_extracted,
            "is_searchable": is_pdf_searchable(pdf_path, logger),
            "processing_status": "completed",
            "processed_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "cleaned": False,
            "errors": extraction_errors
        }
        
        # Save metadata (only on success)
        metadata_path = pdf_output_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        logger.info(f"Metadata saved: {metadata_path}")
        
        logger.info(f"Completed processing: {pdf_name}")
        return True
    
    except Exception as e:
        error_msg = f"Unexpected error processing {pdf_name}: {str(e)}"
        logger.error(error_msg)
        extraction_errors.append(error_msg)
        
        # Cleanup on failure - remove partial extraction
        # This ensures we don't have incomplete output
        try:
            if pdf_output_dir.exists():
                # Only remove if it contains partial extraction
                if (pdf_output_dir / "text.txt").exists():
                    (pdf_output_dir / "text.txt").unlink()
                logger.debug(f"Cleaned up partial extraction for {pdf_name}")
        except Exception as cleanup_err:
            logger.warning(f"Error cleaning up partial extraction: {str(cleanup_err)}")
        
        return False


def _get_page_count(pdf_path: Path, logger: logging.Logger) -> int:
    """
    Get the total page count from a PDF.
    
    Internal helper to safely retrieve page count.
    
    Args:
        pdf_path: Path to the PDF file.
        logger: Logger instance.
        
    Returns:
        Number of pages, or 0 if error.
    """
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            return len(pdf.pages)
    except Exception as e:
        logger.warning(f"Failed to get page count for {pdf_path.name}: {str(e)}")
        return 0
