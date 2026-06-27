"""
Main orchestration module for the RAG PDF extraction pipeline.

Controls the complete workflow:
1. Automatic PDF discovery from input_pdfs/
2. Routing (searchable vs scanned PDFs)
3. Extraction (text, tables, images) with incremental processing
4. Cleaning with incremental processing
5. Metadata tracking for debugging and future RAG indexing

Designed to be modular and extensible for:
- OCR support for scanned PDFs
- Custom chunking strategies
- Embedding generation
- Vector database integration
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from utils import (
    setup_logging,
    create_pipeline_directories,
    get_pdf_files,
    validate_pdf_path,
    is_extraction_complete,
    is_cleaning_complete,
    get_extraction_summary,
    get_cleaning_summary,
    route_pdf,
)
from extract import process_pdf as extract_pdf_non_scanned
from cleaning import process_all_extracted


# ============================================================================
# DYNAMIC SCANNED MODULE IMPORT
# ============================================================================

def import_scanned_module():
    """
    Dynamically import the Scanned module for OCR processing.
    
    This function safely imports the Scanned extraction module if available.
    If not available (optional dependency), returns None and operations
    proceed with non_scanned only.
    
    Returns:
        Tuple of (extract_function, cleaning_function, utils_module) or (None, None, None)
    """
    try:
        # Import Scanned module from parent directory
        scanned_path = Path(__file__).parent.parent / "Scanned"
        
        if not scanned_path.exists():
            return None, None, None
        
        # Dynamically import Scanned modules
        import importlib.util
        
        # Import extract module
        extract_spec = importlib.util.spec_from_file_location(
            "scanned_extract", scanned_path / "extract.py"
        )
        scanned_extract = importlib.util.module_from_spec(extract_spec)
        extract_spec.loader.exec_module(scanned_extract)
        
        # Import cleaning module
        cleaning_spec = importlib.util.spec_from_file_location(
            "scanned_cleaning", scanned_path / "cleaning.py"
        )
        scanned_cleaning = importlib.util.module_from_spec(cleaning_spec)
        cleaning_spec.loader.exec_module(scanned_cleaning)
        
        # Import utils module
        utils_spec = importlib.util.spec_from_file_location(
            "scanned_utils", scanned_path / "utils.py"
        )
        scanned_utils = importlib.util.module_from_spec(utils_spec)
        utils_spec.loader.exec_module(scanned_utils)
        
        return (
            scanned_extract.process_pdf,
            scanned_cleaning.process_all_extracted,
            scanned_utils
        )
    
    except Exception as e:
        print(f"⚠️  Scanned module not available: {str(e)}")
        return None, None, None


class PDFExtractionPipeline:
    """
    Main orchestrator for the PDF extraction pipeline.
    
    Handles:
    - Automatic PDF discovery
    - PDF routing (searchable vs scanned)
    - Incremental extraction (skip already successfully processed)
    - Incremental cleaning (retry failed operations)
    - Error resilience
    - Comprehensive logging and metadata
    """
    
    def __init__(self, base_path: Path) -> None:
        """
        Initialize the PDF extraction pipeline.
        
        Args:
            base_path: Base directory for the RAG pipeline.
        """
        self.base_path = base_path
        
        # Create directory structure
        self.dirs = create_pipeline_directories(base_path)
        
        # Setup logging
        log_path = self.dirs["logs"] / "pipeline.log"
        self.logger = setup_logging(log_path)
        
        # Import Scanned module (optional)
        (
            self.extract_pdf_scanned,
            self.clean_scanned,
            self.scanned_utils
        ) = import_scanned_module()
        
        # Flag for Scanned module availability
        self.scanned_available = self.extract_pdf_scanned is not None
        
        self.logger.info("=" * 70)
        self.logger.info("PDF EXTRACTION PIPELINE INITIALIZED")
        self.logger.info(f"Base directory: {base_path}")
        if self.scanned_available:
            self.logger.info("Scanned (OCR) module available: YES")
        else:
            self.logger.info("Scanned (OCR) module available: NO")
        self.logger.info("=" * 70)
    
    def run_extraction(self) -> tuple[int, int, int]:
        """
        Run the extraction phase on all PDFs.
        
        Automatically discovers PDFs in input_pdfs/, routes them appropriately,
        and processes each one. Uses incremental processing: skips PDFs already
        successfully extracted.
        
        Returns:
            Tuple of (total_found, successfully_processed, routed_to_ocr)
        """
        self.logger.info("\n" + "=" * 70)
        self.logger.info("PHASE 1: EXTRACTION")
        self.logger.info("=" * 70)
        
        # Find all PDFs
        pdf_files = get_pdf_files(self.dirs["input"])
        
        if not pdf_files:
            self.logger.warning(f"No PDF files found in {self.dirs['input']}")
            print("\n⚠️  No PDF files found in input_pdfs/")
            return 0, 0, 0
        
        total_pdfs = len(pdf_files)
        successful_pdfs = 0
        skipped_pdfs = 0
        routed_scanned = 0
        
        self.logger.info(f"Found {total_pdfs} PDF file(s)")
        print(f"\n📄 Found {total_pdfs} PDF file(s) to process")
        
        for idx, pdf_path in enumerate(pdf_files, 1):
            pdf_name = pdf_path.name
            
            # Validate PDF
            if not validate_pdf_path(pdf_path):
                self.logger.error(f"Invalid PDF path: {pdf_path}")
                print(f"[{idx}/{total_pdfs}] ✗ Invalid: {pdf_name}")
                continue
            
            # Route PDF to appropriate pipeline
            routing = route_pdf(pdf_path, self.logger)
            
            if routing == "scanned":
                if self.scanned_available:
                    # Process with Scanned (OCR) module
                    print(f"[{idx}/{total_pdfs}] Processing with OCR: {pdf_name}")
                    
                    # Setup Scanned directories
                    scanned_base = self.base_path.parent / "Scanned"
                    scanned_dirs = self.scanned_utils.create_pipeline_directories(scanned_base)
                    
                    # Check if already extracted
                    if self.scanned_utils.is_extraction_complete(
                        pdf_name, scanned_dirs["extracted_raw"], self.logger
                    ):
                        self.logger.info(f"[{idx}/{total_pdfs}] Already extracted (Scanned): {pdf_name}")
                        print(f"[{idx}/{total_pdfs}] ⊘ Skipped (already extracted): {pdf_name}")
                        skipped_pdfs += 1
                        continue
                    
                    try:
                        if self.extract_pdf_scanned(
                            pdf_path, scanned_dirs["extracted_raw"], self.logger
                        ):
                            successful_pdfs += 1
                            routed_scanned += 1
                            print(f"[{idx}/{total_pdfs}] ✓ Extracted (OCR): {pdf_name}")
                        else:
                            print(f"[{idx}/{total_pdfs}] ✗ Failed (OCR): {pdf_name}")
                    except Exception as e:
                        self.logger.error(f"Error in Scanned extraction for {pdf_name}: {str(e)}")
                        print(f"[{idx}/{total_pdfs}] ✗ Error (OCR): {pdf_name}")
                else:
                    # Scanned module not available
                    routed_scanned += 1
                    print(f"[{idx}/{total_pdfs}] ⊘ Routed to OCR (not yet installed): {pdf_name}")
                    self.logger.info(f"PDF routed to OCR but Scanned module not available: {pdf_name}")
                continue
            
            # Process as non-scanned PDF
            # Check if already successfully extracted (incremental processing)
            if is_extraction_complete(pdf_name, self.dirs["extracted_raw"], self.logger):
                self.logger.info(f"[{idx}/{total_pdfs}] Already extracted: {pdf_name}")
                print(f"[{idx}/{total_pdfs}] ⊘ Skipped (already extracted): {pdf_name}")
                skipped_pdfs += 1
                continue
            
            # Process the PDF
            print(f"[{idx}/{total_pdfs}] Processing: {pdf_name}")
            
            try:
                if extract_pdf_non_scanned(pdf_path, self.dirs["extracted_raw"], self.logger):
                    successful_pdfs += 1
                    print(f"[{idx}/{total_pdfs}] ✓ Extracted: {pdf_name}")
                else:
                    print(f"[{idx}/{total_pdfs}] ✗ Failed: {pdf_name}")
            
            except Exception as e:
                self.logger.error(f"Unexpected error processing {pdf_name}: {str(e)}")
                print(f"[{idx}/{total_pdfs}] ✗ Error: {pdf_name}")
        
        self.logger.info(
            f"Extraction complete: {successful_pdfs}/{total_pdfs} successful, "
            f"{skipped_pdfs} skipped, {routed_scanned} routed to OCR pipeline"
        )
        
        return total_pdfs, successful_pdfs, routed_scanned
    
    def run_cleaning(self) -> tuple[int, int, int, int]:
        """
        Run the cleaning phase on all extracted PDFs (both modules).
        
        Processes files from extracted_raw/ and outputs to cleaned/.
        Uses incremental processing: skips PDFs already successfully cleaned,
        but retries failed cleaning operations.
        
        Returns:
            Tuple of (non_scanned_total, non_scanned_success, scanned_total, scanned_success)
        """
        self.logger.info("\n" + "=" * 70)
        self.logger.info("PHASE 2: CLEANING")
        self.logger.info("=" * 70)
        
        # Clean non_scanned PDFs
        total_non_scanned, successful_non_scanned = process_all_extracted(
            self.dirs["extracted_raw"],
            self.dirs["cleaned"],
            self.logger
        )
        
        # Clean Scanned PDFs (if available)
        total_scanned = 0
        successful_scanned = 0
        
        if self.scanned_available:
            try:
                scanned_base = self.base_path.parent / "Scanned"
                scanned_dirs = self.scanned_utils.create_pipeline_directories(scanned_base)
                
                total_scanned, successful_scanned = self.clean_scanned(
                    scanned_dirs["extracted_raw"],
                    scanned_dirs["cleaned"],
                    self.logger
                )
            except Exception as e:
                self.logger.warning(f"Error cleaning Scanned PDFs: {str(e)}")
        
        return total_non_scanned, successful_non_scanned, total_scanned, successful_scanned
    
    def run_complete_pipeline(self) -> bool:
        """
        Run the complete extraction and cleaning pipeline.
        
        Returns:
            True if pipeline completed successfully, False otherwise.
        """
        try:
            # Phase 1: Extraction
            extraction_total, extraction_success, routed_scanned = self.run_extraction()
            
            if extraction_total == 0:
                self.logger.warning("No PDFs to process")
                return False
            
            # Phase 2: Cleaning
            total_ns, success_ns, total_sc, success_sc = self.run_cleaning()
            
            # Print summary
            self.print_summary(
                extraction_total,
                extraction_success,
                routed_scanned,
                total_ns,
                success_ns,
                total_sc,
                success_sc
            )
            
            return True
        
        except KeyboardInterrupt:
            self.logger.warning("Pipeline interrupted by user")
            print("\n\n⚠️  Pipeline interrupted by user")
            return False
        
        except Exception as e:
            self.logger.error(f"Fatal error in pipeline: {str(e)}")
            print(f"\n❌ Fatal error: {str(e)}")
            return False
    
    def print_summary(
        self,
        extraction_total: int,
        extraction_success: int,
        routed_scanned: int,
        cleaning_ns_total: int,
        cleaning_ns_success: int,
        cleaning_sc_total: int,
        cleaning_sc_success: int
    ) -> None:
        """
        Print a comprehensive summary of the pipeline execution.
        
        Args:
            extraction_total: Total PDFs found for extraction.
            extraction_success: Successfully extracted PDFs.
            routed_scanned: PDFs routed to OCR pipeline.
            cleaning_ns_total: Non-scanned PDFs to clean.
            cleaning_ns_success: Successfully cleaned non-scanned PDFs.
            cleaning_sc_total: Scanned PDFs to clean.
            cleaning_sc_success: Successfully cleaned scanned PDFs.
        """
        print("\n" + "=" * 70)
        print("PIPELINE SUMMARY")
        print("=" * 70)
        
        print("\n📊 EXTRACTION PHASE")
        print(f"  Total found:              {extraction_total}")
        print(f"  Non-scanned processed:    {extraction_success - routed_scanned}")
        print(f"  Scanned (OCR) routed:     {routed_scanned}")
        
        if extraction_total > 0:
            extraction_rate = (extraction_success / extraction_total) * 100
            print(f"  Success rate:             {extraction_rate:.1f}%")
        
        print("\n📊 CLEANING PHASE")
        print(f"  Non-scanned to clean:     {cleaning_ns_total}")
        print(f"  Non-scanned cleaned:      {cleaning_ns_success}")
        
        if cleaning_ns_total > 0:
            ns_rate = (cleaning_ns_success / cleaning_ns_total) * 100
            print(f"  Non-scanned success:      {ns_rate:.1f}%")
        
        if self.scanned_available:
            print(f"\n  Scanned to clean:         {cleaning_sc_total}")
            print(f"  Scanned cleaned:          {cleaning_sc_success}")
            
            if cleaning_sc_total > 0:
                sc_rate = (cleaning_sc_success / cleaning_sc_total) * 100
                print(f"  Scanned success:          {sc_rate:.1f}%")
        
        print("\n📁 OUTPUT LOCATIONS")
        print(f"  Non-scanned extraction:   {self.dirs['extracted_raw']}")
        print(f"  Non-scanned cleaned:      {self.dirs['cleaned']}")
        
        if self.scanned_available:
            scanned_base = self.base_path.parent / "Scanned"
            print(f"  Scanned extraction:       {scanned_base / 'data' / 'extracted_raw'}")
            print(f"  Scanned cleaned:          {scanned_base / 'data' / 'cleaned'}")
        
        print(f"  Logs:                     {self.dirs['logs']}")
        
        # Print statistics
        print("\n📈 STATISTICS")
        extraction_stats = get_extraction_summary(self.dirs["extracted_raw"], self.logger)
        print(f"  Non-scanned with text:    {extraction_stats.get('has_text', 0)}")
        print(f"  Non-scanned with images:  {extraction_stats.get('has_images', 0)}")
        
        if self.scanned_available:
            try:
                scanned_base = self.base_path.parent / "Scanned"
                scanned_dirs = self.scanned_utils.create_pipeline_directories(scanned_base)
                scanned_stats = self.scanned_utils.get_extraction_summary(
                    scanned_dirs["extracted_raw"], self.logger
                )
                print(f"  Scanned with text:        {scanned_stats.get('has_text', 0)}")
                print(f"  Scanned with images:      {scanned_stats.get('has_images', 0)}")
            except Exception:
                pass
        
        print("\n📋 OUTPUT STRUCTURE")
        print("  Non-scanned/extracted_raw/")
        print("    ├── pdf_name_1/")
        print("    │   ├── text.txt")
        print("    │   ├── images/")
        print("    │   └── metadata.json")
        print("    └── pdf_name_2/")
        print("        └── ...\n")
        
        if self.scanned_available:
            print("  Scanned/extracted_raw/")
            print("    ├── pdf_name_1/")
            print("    │   ├── text.txt (OCR-extracted)")
            print("    │   ├── images/")
            print("    │   └── metadata.json")
            print("    └── pdf_name_2/")
            print("        └── ...\n")
        
        print("\n💡 METADATA INFORMATION")
        print("  ✓ Incremental processing enabled")
        print("  ✓ Failed extractions are retried")
        print("  ✓ Metadata only updated on success")
        print("  ✓ Automatic PDF routing (searchable vs scanned)")
        if self.scanned_available:
            print("  ✓ OCR support enabled for scanned PDFs")
        else:
            print("  ⚠ OCR support not available (Scanned module not installed)")
        
        print("\n" + "=" * 70)
        print("✅ Pipeline execution complete!")
        print("=" * 70)
        
        self.logger.info(
            f"Pipeline complete: "
            f"Extraction {extraction_success}/{extraction_total}, "
            f"Cleaning {cleaning_ns_success + cleaning_sc_success}/"
            f"{cleaning_ns_total + cleaning_sc_total}"
        )


def main() -> int:
    """
    Main entry point for the PDF extraction pipeline.
    
    Returns:
        Exit code (0 for success, 1 for failure).
    """
    # Determine base directory (current working directory or specified)
    base_path = Path.cwd()
    
    print("\n" + "=" * 70)
    print("RAG PIPELINE - PDF EXTRACTION AND CLEANING")
    print("=" * 70)
    print(f"Working directory: {base_path}\n")
    
    try:
        # Initialize and run pipeline
        pipeline = PDFExtractionPipeline(base_path)
        success = pipeline.run_complete_pipeline()
        
        return 0 if success else 1
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
