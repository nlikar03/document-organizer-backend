import os
import json
import fitz
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from openai import OpenAI

from google.cloud import vision
from google.oauth2 import service_account

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://nlikar03.github.io", 
        "http://localhost:3000",            
        "http://localhost:5173"            
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI()

# Initialize Google Vision with credentials from environment variable
GOOGLE_CREDENTIALS_JSON = os.getenv("GOOGLE_CREDENTIALS_JSON")
if GOOGLE_CREDENTIALS_JSON:
    credentials_dict = json.loads(GOOGLE_CREDENTIALS_JSON)
    credentials = service_account.Credentials.from_service_account_info(credentials_dict)
    vision_client = vision.ImageAnnotatorClient(credentials=credentials)
else:
    vision_client = vision.ImageAnnotatorClient()

APP_PASSWORD = os.getenv("APP_PASSWORD", "default_password")

# MODELS
class AIRequest(BaseModel):
    text: str
    structure: List[Dict[str, Any]]


class AIBatchRequest(BaseModel):
    texts: List[str]
    fileNames: List[str]
    structure: List[Dict[str, Any]]


class ExcelRequest(BaseModel):
    results: List[Dict[str, Any]]
    structure: List[Dict[str, Any]]


async def verify_password(x_password: str = Header(None)):
    if x_password != APP_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid password")
    return True


# GOOGLE VISION OCR

def extract_text_with_vision(image_bytes: bytes) -> str:
    """
    OCR using Google Cloud Vision API - much faster than OpenAI.
    """
    try:
        image = vision.Image(content=image_bytes)
        response = vision_client.text_detection(image=image)
        
        if response.error.message:
            raise Exception(f"Vision API error: {response.error.message}")
        
        texts = response.text_annotations
        if texts:
            return texts[0].description
        return ""
    
    except Exception as e:
        return f"Vision OCR error: {str(e)}"


def extract_text_from_pdf_batch(pdf_data: bytes, max_pages: int = 5) -> str:
    """
    Extract text from PDF using Google Vision with parallel processing.
    Processes first 4 pages and last page of the document.
    """
    try:
        doc = fitz.open(stream=pdf_data, filetype="pdf")
        total_pages = len(doc)
        
        # Determine which pages to process
        pages_to_process_indices = []
        if total_pages <= max_pages:
            # If document has 5 or fewer pages, process all
            pages_to_process_indices = list(range(total_pages))
        else:
            # First 4 pages
            pages_to_process_indices.extend(range(4))
            # Last page
            pages_to_process_indices.append(total_pages - 1)
        
        page_images = []
        for page_num in pages_to_process_indices:
            page = doc[page_num]
            pix = page.get_pixmap(dpi=150)
            img_bytes = pix.tobytes("png")
            page_images.append((page_num, img_bytes))
        
        full_text = {}
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_page = {
                executor.submit(extract_text_with_vision, img_bytes): page_num 
                for page_num, img_bytes in page_images
            }
            
            for future in as_completed(future_to_page):
                page_num = future_to_page[future]
                try:
                    text = future.result()
                    full_text[page_num] = text
                except Exception as e:
                    full_text[page_num] = f"Error on page {page_num + 1}: {str(e)}"
        
        doc.close()
        
        sorted_pages = sorted(full_text.keys())
        return "\n\n".join([full_text[page_num] for page_num in sorted_pages])
    
    except Exception as e:
        return f"PDF processing error: {str(e)}"


def extract_text_from_image(image_bytes: bytes) -> str:
    """
    OCR a single image using Google Vision.
    """
    return extract_text_with_vision(image_bytes)


def classify_single_document(text: str, structure: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Classify a single document synchronously.
    """
    simplified = [
        {"id": f["id"], "name": f["name"], "level": f["level"]}
        for f in structure
    ]

    system_content = f"""
You are a professional document organizer assistant for Kolektor. 
Your goal is to categorize document text into the provided folder structure.

FOLDER STRUCTURE:
{json.dumps(simplified, ensure_ascii=False)}

TASK:
1. Analyze the document text provided by the user.
2. Select the most appropriate folder from the structure.
3. Extract metadata (Title, Issuer, Document Number, Date).

Always return ONLY a valid JSON object with this structure:
{{
  "suggestedFolder": {{ "id": "folder_id", "name": "folder_name" }},
  "documentTitle": "brief description",
  "issuer": "who created it",
  "documentNumber": "id string or empty",
  "date": "DD.MM.YYYY or empty"
}}"""

    prompt = f"""
DOCUMENT TEXT:
{text[:8000]}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)
        folder_data = result.get("suggestedFolder", {})
        folder_id = folder_data.get("id", "0")

        return {
            "suggestedFolder": {
                "id": folder_id,
                "name": folder_data.get("name", "Unknown"),
                "fullPath": build_folder_path(folder_id, structure)
            },
            "documentTitle": result.get("documentTitle", ""),
            "issuer": result.get("issuer", ""),
            "documentNumber": result.get("documentNumber", ""),
            "date": result.get("date", "")
        }

    except Exception as e:
        return {
            "suggestedFolder": {
                "id": "0",
                "name": "Unknown",
                "fullPath": "0 PODATKI O POGODBI"
            },
            "documentTitle": "",
            "issuer": "",
            "documentNumber": "",
            "date": ""
        }


# HELPERJI

def build_folder_path(folder_id: str, structure: List[Dict[str, Any]]) -> str:
    folder_map = {f["id"]: f["name"] for f in structure}
    parts = folder_id.split(".")
    path = []

    for i in range(len(parts)):
        pid = ".".join(parts[:i + 1])
        if pid in folder_map:
            path.append(folder_map[pid])

    return " / ".join(path)


# ROUTES

@app.post("/api/verify-password")
async def verify_password_endpoint(valid: bool = Depends(verify_password)):
    return {"valid": True}


@app.post("/api/ocr", dependencies=[Depends(verify_password)])
async def process_ocr(file: UploadFile = File(...)):
    contents = await file.read()
    name = file.filename.lower()

    if name.endswith(".pdf"):
        text = extract_text_from_pdf_batch(contents)
    elif name.endswith((".png", ".jpg", ".jpeg", ".bmp")):
        text = extract_text_from_image(contents)
    else:
        raise HTTPException(400, "Unsupported file format")

    return {"fileName": file.filename, "text": text}


@app.post("/api/classify", dependencies=[Depends(verify_password)])
async def classify_document(request: AIRequest):
    result = classify_single_document(request.text, request.structure)
    return result

@app.post("/api/extract-metadata", dependencies=[Depends(verify_password)])
async def extract_metadata_only(file: UploadFile = File(...)):
    """
    Extract metadata from a document without classifying it.
    Used for files that are directly uploaded to specific folders.
    """
    contents = await file.read()
    name = file.filename.lower()

    if name.endswith(".pdf"):
        text = extract_text_from_pdf_batch(contents)
    elif name.endswith((".png", ".jpg", ".jpeg", ".bmp")):
        text = extract_text_from_image(contents)
    else:
        raise HTTPException(400, "Unsupported file format")

    try:
        system_content = """
You are a document metadata extractor. Analyze the document text and extract key metadata.

Return ONLY a valid JSON object with this structure:
{
  "documentTitle": "brief description of what this document is",
  "issuer": "who created or issued this document",
  "documentNumber": "document ID/number if present, otherwise empty string",
  "date": "DD.MM.YYYY format if date found, otherwise empty string"
}
"""

        prompt = f"""
DOCUMENT TEXT:
{text[:8000]}

Extract metadata from this document.
"""

        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)
        
        return {
            "documentTitle": result.get("documentTitle", ""),
            "issuer": result.get("issuer", ""),
            "documentNumber": result.get("documentNumber", ""),
            "date": result.get("date", "")
        }

    except Exception as e:
        print(f"Metadata extraction error: {str(e)}")
        return {
            "documentTitle": "",
            "issuer": "",
            "documentNumber": "",
            "date": "",
            "error": str(e)
        }


@app.post("/api/classify-batch", dependencies=[Depends(verify_password)])
async def classify_batch(request: AIBatchRequest):
    """
    Classify multiple documents in parallel (max 5 concurrent).
    """
    try:
        results = []
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_index = {
                executor.submit(classify_single_document, text, request.structure): idx
                for idx, text in enumerate(request.texts)
            }
            
            results = [None] * len(request.texts)
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    result = future.result()
                    results[idx] = {
                        "fileName": request.fileNames[idx],
                        "classification": result
                    }
                except Exception as e:  
                    results[idx] = {
                        "fileName": request.fileNames[idx],
                        "classification": {
                            "suggestedFolder": {
                                "id": "0",
                                "name": "Unknown",
                                "fullPath": "0 PODATKI O POGODBI"
                            },
                            "documentTitle": "",
                            "issuer": "",
                            "documentNumber": "",
                            "date": "",
                            "error": str(e)
                        }
                    }
        
        return {"results": results}
        
    except Exception as e:
        raise HTTPException(500, str(e))
    
@app.post("/api/extract-metadata-batch", dependencies=[Depends(verify_password)])
async def extract_metadata_batch(files: List[UploadFile] = File(...)):
    """
    Extract metadata from multiple files in parallel (max 5 concurrent).
    Much faster than one-by-one requests.
    """
    try:
        # First, do OCR on all files
        ocr_results = []
        for file in files:
            contents = await file.read()
            name = file.filename.lower()

            if name.endswith(".pdf"):
                text = extract_text_from_pdf_batch(contents)
            elif name.endswith((".png", ".jpg", ".jpeg", ".bmp")):
                text = extract_text_from_image(contents)
            else:
                text = ""
            
            ocr_results.append({
                "fileName": file.filename,
                "text": text
            })
        
        # Then extract metadata in parallel
        def extract_metadata_from_text(text: str) -> Dict[str, Any]:
            try:
                system_content = """
You are a document metadata extractor. Analyze the document text and extract key metadata.

Return ONLY a valid JSON object with this structure:
{
  "documentTitle": "brief description of what this document is",
  "issuer": "who created or issued this document",
  "documentNumber": "document ID/number if present, otherwise empty string",
  "date": "DD.MM.YYYY format if date found, otherwise empty string"
}
"""
                prompt = f"""
DOCUMENT TEXT:
{text[:8000]}

Extract metadata from this document.
"""

                response = client.chat.completions.create(
                    model="gpt-5-mini",
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"}
                )

                result = json.loads(response.choices[0].message.content)
                return {
                    "documentTitle": result.get("documentTitle", ""),
                    "issuer": result.get("issuer", ""),
                    "documentNumber": result.get("documentNumber", ""),
                    "date": result.get("date", "")
                }
            except Exception as e:
                print(f"Metadata extraction error: {str(e)}")
                return {
                    "documentTitle": "",
                    "issuer": "",
                    "documentNumber": "",
                    "date": ""
                }
        
        results = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_index = {
                executor.submit(extract_metadata_from_text, ocr_result["text"]): idx
                for idx, ocr_result in enumerate(ocr_results)
            }
            
            results = [None] * len(ocr_results)
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    metadata = future.result()
                    results[idx] = {
                        "fileName": ocr_results[idx]["fileName"],
                        **metadata
                    }
                except Exception as e:
                    results[idx] = {
                        "fileName": ocr_results[idx]["fileName"],
                        "documentTitle": "",
                        "issuer": "",
                        "documentNumber": "",
                        "date": "",
                        "error": str(e)
                    }
        
        return {"results": results}
        
    except Exception as e:
        raise HTTPException(500, str(e))
        
    
def iter_file(path, chunk_size=1024 * 1024):
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            yield chunk


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)