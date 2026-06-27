# RAG PDF Extraction Pipeline

## Overview
This project extracts text, images, and tables from both scanned and non-scanned PDF documents. It automatically detects the PDF type and routes it to the appropriate extraction pipeline.

## Features
- Automatic PDF type detection
- Non-scanned PDF text extraction
- Scanned PDF OCR extraction
- Hindi OCR support
- Chinese OCR support
- Image extraction
- Table extraction
- Logging
- Incremental processing
- Separate scanned and non-scanned pipelines

## Project Structure

RAG/
├── Scanned/
├── non_scanned/
├── .gitignore
└── README.md

## Requirements

- Python 3.12+
- EasyOCR
- PyMuPDF
- pdfplumber
- pandas
- openpyxl

Install dependencies:

pip install -r requirements.txt

## Run

For non-scanned pipeline:

python non_scanned/main.py

For scanned pipeline:

python Scanned/main.py

## Output

Extracted files are stored in:

- data/extracted_raw/
- data/cleaned/

## Technologies Used

- Python
- EasyOCR
- PyMuPDF (fitz)
- pdfplumber
- Pandas
- OpenPyXL
- Git

## Author

Ishika Chhikara
