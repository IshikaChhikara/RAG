"""
OCR-based PDF extraction module for scanned PDFs.
Handles text, image, and table extraction from image-based PDFs using PaddleOCR.

This module provides functions to extract content from scanned PDFs
and maintains the same architecture as the non_scanned extraction module.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF for PDF to image conversion
import easyocr
from PIL import Image
import numpy as np

from utils import sanitize_filename


# ============================================================================
# OCR SETUP
# ============================================================================

def initialize_ocr(logger, lang_list=['en']):
    try:
        logger.info(f"Initializing EasyOCR with {lang_list}...")
        reader = easyocr.Reader(lang_list, gpu=False)
        logger.info("EasyOCR initialized successfully")
        return reader
    except Exception as e:
        logger.error(f"Failed to initialize OCR: {e}")
        return None


# ============================================================================
# PDF TO IMAGE CONVERSION
# ============================================================================

def convert_pdf_to_images(
    pdf_path: Path,
    logger: logging.Logger,
    dpi: int = 300
) -> list[np.ndarray]:
    """
    Convert PDF pages to image arrays using PyMuPDF.
    
    Converts each page of the PDF to a high-resolution image for
    optimal OCR accuracy.
    
    Args:
        pdf_path: Path to the PDF file.
        logger: Logger instance for recording progress.
        dpi: Resolution in DPI for image conversion (default 300 for OCR).
        
    Returns:
        List of numpy arrays representing PDF pages as images.
        Empty list if conversion fails.
        
    Raises:
        Handled internally with logging. Returns empty list on error.
    """
    images = []
    
    try:
        logger.debug(f"Converting PDF to images: {pdf_path.name}")
        
        # Open PDF
        pdf_document = fitz.open(str(pdf_path))
        page_count = len(pdf_document)
        
        if page_count == 0:
            logger.warning(f"PDF has no pages: {pdf_path.name}")
            return []
        
        # Convert factor for DPI (72 DPI is PyMuPDF default)
        zoom_factor = dpi / 72
        matrix = fitz.Matrix(zoom_factor, zoom_factor)
        
        for page_num, page in enumerate(pdf_document, 1):
            try:
                # Render page to image
                pix = page.get_pixmap(matrix=matrix)
                
                # Convert to numpy array
                img_array = np.frombuffer(pix.samples, dtype=np.uint8)
                img_array = img_array.reshape((pix.height, pix.width, pix.n))
                
                # Convert RGBA to RGB if necessary
                if img_array.shape[2] == 4:
                    # Remove alpha channel
                    img_array = img_array[:, :, :3]
                
                images.append(img_array)
                logger.debug(f"Converted page {page_num}/{page_count}")
            
            except Exception as e:
                logger.warning(
                    f"Failed to convert page {page_num} in {pdf_path.name}: {str(e)}"
                )
                continue
        
        pdf_document.close()
        
        if images:
            logger.info(
                f"Converted {len(images)}/{page_count} pages to images: {pdf_path.name}"
            )
        else:
            logger.warning(f"No pages successfully converted from {pdf_path.name}")
        
        return images
    
    except Exception as e:
        logger.error(f"Error converting PDF to images: {str(e)}")
        return []


# ============================================================================
# OCR TEXT EXTRACTION
# ============================================================================

def extract_text_from_images(
    images: list[np.ndarray],
    ocr,
    logger: logging.Logger
) -> str:
    """
    Extract text from image arrays using PaddleOCR.
    
    Performs OCR on each image with page markers to preserve document structure.
    
    Args:
        images: List of image arrays (numpy arrays) representing PDF pages.
        ocr: Initialized PaddleOCR instance.
        logger: Logger instance for recording extraction progress.
        
    Returns:
        Extracted text with page markers, or empty string if extraction fails.
        
    Raises:
        Handled internally with logging. Returns empty string on error.
    """
    text_content = ""
    
    try:
        logger.debug(f"Starting OCR text extraction for {len(images)} page(s)")
        
        for page_num, image in enumerate(images, 1):
            try:
                # Perform OCR on image
                results = ocr.readtext(image)
                
                if results:
                    # Extract text from OCR results
                    page_text = ""
                    for line in results:
                        text = line[1]
                        confidence = line[2]

                        if confidence > 0.3:
                            page_text += text + " "
                    
                    if page_text.strip():
                        text_content += f"\n{'='*60}\nPAGE {page_num}\n{'='*60}\n\n"
                        text_content += page_text.strip()
                        text_content += "\n"
                        logger.debug(f"Extracted text from page {page_num}")
                    else:
                        logger.debug(f"No text detected on page {page_num}")
                else:
                    logger.debug(f"No OCR results for page {page_num}")
            
            except Exception as e:
                logger.warning(
                    f"Failed to perform OCR on page {page_num}: {str(e)}"
                )
                continue
        
        if not text_content.strip():
            logger.warning("No text extracted from any page")
            return ""
        
        logger.info(f"Successfully extracted text from {len(images)} page(s) via OCR")
        return text_content.strip()
    
    except Exception as e:
        logger.error(f"Error in OCR text extraction: {str(e)}")
        return ""


# ============================================================================
# IMAGE EXTRACTION
# ============================================================================

def extract_images(
    pdf_path: Path,
    output_dir: Path,
    logger: logging.Logger
) -> int:
    """
    Extract images embedded in the PDF.
    
    Saves all images found in the PDF to the output directory.
    Also converts rendered pages to images for preservation.
    
    Args:
        pdf_path: Path to the PDF file.
        output_dir: Directory to save extracted images.
        logger: Logger instance for recording extraction progress.
        
    Returns:
        Number of images successfully extracted.
        
    Raises:
        Handled internally with logging. Returns 0 on error.
    """
    image_count = 0
    
    try:
        logger.debug(f"Starting image extraction: {pdf_path.name}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            pdf_document = fitz.open(str(pdf_path))
        except Exception as e:
            logger.error(f"Failed to open PDF: {str(e)}")
            return 0
        
        # Extract embedded images from PDF
        image_index = 1
        
        for page_num, page in enumerate(pdf_document, 1):
            try:
                # Get embedded images
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list, 1):
                    try:
                        # Extract image
                        xref = img[0]
                        pix = fitz.Pixmap(pdf_document, xref)
                        
                        # Save image
                        if pix.n - pix.alpha < 4:
                            image_path = output_dir / f"image_{image_index}.png"
                            pix.save(str(image_path))
                        else:
                            # Convert CMYK to RGB
                            pix = fitz.Pixmap(fitz.csRGB, pix)
                            image_path = output_dir / f"image_{image_index}.png"
                            pix.save(str(image_path))
                        
                        image_count += 1
                        logger.debug(f"Extracted image {image_index} from page {page_num}")
                        image_index += 1
                    
                    except Exception as e:
                        logger.warning(
                            f"Failed to extract image {img_index} from page {page_num}: {str(e)}"
                        )
                        continue
            
            except Exception as e:
                logger.warning(
                    f"Failed to extract images from page {page_num}: {str(e)}"
                )
                continue
        
        pdf_document.close()
        
        if image_count > 0:
            logger.info(f"Extracted {image_count} image(s) from {pdf_path.name}")
        else:
            logger.info(f"No images found in {pdf_path.name}")
        
        return image_count
    
    except Exception as e:
        logger.error(f"Unexpected error extracting images: {str(e)}")
        return 0


# ============================================================================
# COMBINED PDF PROCESSING
# ============================================================================

def process_pdf(
    pdf_path: Path,
    output_raw_dir: Path,
    logger: logging.Logger,
    ocr=None
) -> bool:
    """
    Process a single scanned PDF: extract text via OCR, images with page-wise output.
    
    Creates a structured output directory with:
    - text.txt: Combined OCR text organized by page
    - images/: Extracted images
    - metadata.json: Extraction metadata (only on success)
    
    Metadata is only written upon successful extraction to ensure correct status.
    Failed extractions do not create metadata, allowing retry on next run.
    
    Args:
        pdf_path: Path to the PDF file.
        output_raw_dir: Base directory for raw extracted output.
        logger: Logger instance.
        ocr: Optional pre-initialized PaddleOCR instance (will be initialized if None).
        
    Returns:
        True if processing successful, False otherwise.
    """
    pdf_name = pdf_path.name
    sanitized_name = sanitize_filename(pdf_name)
    
    # Create PDF-specific output directory
    pdf_output_dir = output_raw_dir / sanitized_name
    pdf_output_dir.mkdir(parents=True, exist_ok=True)
    
    images_output_dir = pdf_output_dir / "images"
    
    logger.info(f"Starting OCR processing: {pdf_name}")
    
    extraction_errors = []
    
    try:
        # Initialize OCR if not provided
        if ocr is None:
            pdf_lower = pdf_path.name.lower()
        
        if "hindi" in pdf_lower:
            ocr = initialize_ocr(logger, ['en', 'hi'])
            
        elif "chinese" in pdf_lower:
            ocr = initialize_ocr(logger, ['en', 'ch_sim'])
            
        else:
            ocr = initialize_ocr(logger, ['en'])
            
        if ocr is None:
            error_msg = "Failed to initialize OCR engine"
            logger.error(error_msg)
            extraction_errors.append(error_msg)
            import shutil
            
            if pdf_output_dir.exists():
                shutil.rmtree(pdf_output_dir, ignore_errors=True)
            return False
        
        # Convert PDF pages to images
        logger.debug(f"PDF to image conversion starting: {pdf_name}")
        images = convert_pdf_to_images(pdf_path, logger, dpi=300)
        
        if not images:
            error_msg = "No images created from PDF"
            logger.error(error_msg)
            extraction_errors.append(error_msg)

            if pdf_output_dir.exists():
                shutil.rmtree(pdf_output_dir, ignore_errors=True)

            return False
        
        page_count = len(images)
        
        # Extract text via OCR
        logger.debug(f"OCR text extraction starting: {pdf_name}")
        text_content = extract_text_from_images(images, ocr, logger)
        text_extracted = bool(text_content)
        
        if not text_extracted:
            error_msg = "No text extracted via OCR"
            logger.warning(error_msg)
            extraction_errors.append(error_msg)
            # Don't fail immediately - may still have images
        
        # Save combined content
        if text_content.strip():
            text_output_path = pdf_output_dir / "text.txt"
            text_output_path.write_text(text_content, encoding="utf-8")
            logger.info(f"Text saved: {text_output_path}")
        else:
            logger.warning(f"No text content to save for {pdf_name}")
        
        # Extract images
        logger.debug(f"Image extraction starting: {pdf_name}")
        image_count = extract_images(pdf_path, images_output_dir, logger)
        images_extracted = image_count > 0
        
        # Determine if extraction should be considered successful
        # Success = at least text was extracted
        extraction_successful = text_extracted
        
        if not extraction_successful:
            extraction_errors.append("No text extracted from scanned PDF")
            logger.warning(f"No extractable text found in {pdf_name}")

            if pdf_output_dir.exists():
                shutil.rmtree(pdf_output_dir, ignore_errors=True)

            return False
        
        # Only write metadata on successful extraction
        metadata = {
            "pdf_name": pdf_name,
            "pdf_path": str(pdf_path),
            "pages": page_count,
            "text_extracted": text_extracted,
            "images_extracted": images_extracted,
            "ocr_engine": "EasyOCR",
            "is_searchable": False,  # Scanned PDFs are never searchable
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
        
        logger.info(f"Completed OCR processing: {pdf_name}")
        return True
    
    except Exception as e:
        error_msg = f"Unexpected error processing {pdf_name}: {str(e)}"
        logger.error(error_msg)
        extraction_errors.append(error_msg)
        
        # Cleanup on failure - remove partial extraction
        try:
            if pdf_output_dir.exists():
                import shutil
                if pdf_output_dir.exists():
                    shutil.rmtree(pdf_output_dir, ignore_errors=True)
                logger.debug(f"Cleaned up partial extraction for {pdf_name}")
        except Exception as cleanup_err:
            logger.warning(f"Error cleaning up partial extraction: {str(cleanup_err)}")
        
        return False
