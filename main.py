from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from typing import Optional
import sqlite3
import os
import uuid
from pdf2image import convert_from_bytes
from PIL import Image
import io

app = FastAPI()

# Database connection function
def get_db_connection():
    conn = sqlite3.connect('contracts.db')
    conn.row_factory = sqlite3.Row
    return conn

# Ensure upload directory exists
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload-contract")
async def upload_contract(
    file: UploadFile = File(...),
    title: str = Form(...),
    category: str = Form(...),
):
    # Generate a unique filename
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)

    # Save the uploaded file
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # # Convert to PDF if not already a PDF
    pdf_path = file_path
    # if file_extension.lower() != ".pdf":
    #     pdf_path = os.path.join(UPLOAD_DIR, f"{os.path.splitext(unique_filename)[0]}.pdf")
    #     images = convert_from_bytes(open(file_path, "rb").read())
    #     images[0].save(pdf_path, save_all=True, append_images=images[1:])
    #     os.remove(file_path)  # Remove the original non-PDF file

    # Save document details to the database
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO documents (title, category)
    VALUES (?, ?)
    ''', (title, category))
    document_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return JSONResponse(content={
        "message": "Contract uploaded successfully",
        "document_id": document_id,
        "title": title,
        "category": category,
        "file_path": pdf_path
    }, status_code=201)
