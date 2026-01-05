import os
import json
import zipfile
import io
import fitz
import base64
from typing import List, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from PIL import Image
from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from openai import OpenAI



# ======================
# APP SETUP
# ======================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "*"  # tighten later if you want
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI()  # uses OPENAI_API_KEY from env


# ======================
# MODELS
# ======================

class AIRequest(BaseModel):
    text: str
    structure: List[Dict[str, Any]]


# ======================
# OCR (OPENAI)
# ======================

def extract_text_from_image(image_bytes: bytes, image_type="png") -> str:
    """
    OCR an image using OpenAI Responses API with Base64 data URL.
    image_type: "png" or "jpeg"
    """
    try:
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:image/{image_type};base64,{image_b64}"

        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Extract all readable text from this image."},
                    {"type": "input_image", "image_url": data_url}
                ]
            }]
        )
        return response.output_text
    except Exception as e:
        return f"OCR image error: {str(e)}"


def extract_text_from_pdf(pdf_data: bytes) -> str:
    try:
        doc = fitz.open(stream=pdf_data, filetype="pdf")
        full_text = ""

        for page in doc:
            # Render page to PNG image in memory
            pix = page.get_pixmap()
            img_bytes = pix.tobytes("png")

            # Send image to OpenAI OCR
            text = extract_text_from_image(img_bytes)
            full_text += text + "\n"

        return full_text

    except Exception as e:
        return f"OCR PDF error: {str(e)}"


# ======================
# PDF WATERMARK
# ======================

def add_watermark_to_pdf(pdf_bytes: bytes, watermark_text: str) -> bytes:
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        if not reader.pages:
            return pdf_bytes

        first_page = reader.pages[0]
        width = float(first_page.mediabox.width)
        height = float(first_page.mediabox.height)

        packet = io.BytesIO()
        can = canvas.Canvas(packet, pagesize=(width, height))
        can.setFillColorRGB(1, 0, 0)
        can.setFont("Helvetica-Bold", 20)

        margin = 30
        can.drawRightString(width - margin, height - margin, watermark_text)
        can.save()

        packet.seek(0)
        watermark_pdf = PdfReader(packet)

        writer = PdfWriter()
        first_page.merge_page(watermark_pdf.pages[0])
        writer.add_page(first_page)

        for i in range(1, len(reader.pages)):
            writer.add_page(reader.pages[i])

        output = io.BytesIO()
        writer.write(output)
        output.seek(0)
        return output.read()

    except Exception:
        return pdf_bytes


# ======================
# HELPERS
# ======================

def build_folder_path(folder_id: str, structure: List[Dict[str, Any]]) -> str:
    folder_map = {f["id"]: f["name"] for f in structure}
    parts = folder_id.split(".")
    path = []

    for i in range(len(parts)):
        pid = ".".join(parts[:i + 1])
        if pid in folder_map:
            path.append(folder_map[pid])

    return " / ".join(path)


# ======================
# ROUTES
# ======================

@app.post("/api/ocr")
async def process_ocr(file: UploadFile = File(...)):
    contents = await file.read()
    name = file.filename.lower()

    if name.endswith(".pdf"):
        text = extract_text_from_pdf(contents)
    elif name.endswith((".png", ".jpg", ".jpeg", ".bmp")):
        text = extract_text_from_image(contents)
    else:
        raise HTTPException(400, "Unsupported file format")

    return {"fileName": file.filename, "text": text}


@app.post("/api/classify")
async def classify_document(request: AIRequest):
    simplified = [
        {"id": f["id"], "name": f["name"], "level": f["level"]}
        for f in request.structure
    ]

    prompt = f"""
    You are a document organizer assistant.

    DOCUMENT TEXT:
    {request.text[:2000]}

    FOLDER STRUCTURE:
    {json.dumps(simplified, ensure_ascii=False)}

    Return ONLY JSON:
    {{ "id": "folder_id", "name": "folder_name" }}
    """

    try:
        response = client.responses.create(
            model="gpt-5-mini",
            input=prompt,
            response_format={"type": "json_object"}
        )

        result = json.loads(response.output_text)
        folder_id = result.get("id", "0")

        return {
            "id": folder_id,
            "name": result.get("name", "Unknown"),
            "fullPath": build_folder_path(folder_id, request.structure)
        }

    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/generate-zip")
async def generate_zip(
    files: List[UploadFile] = File(...),
    metadata: UploadFile = File(...)
):
    try:
        meta = json.loads((await metadata.read()).decode())
        structure = meta["structure"]
        file_mapping = meta["fileMapping"]

        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
            folder_paths = {}

            for folder in structure:
                parts = []
                ids = folder["id"].split(".")
                for i in range(len(ids)):
                    fid = ".".join(ids[:i + 1])
                    f = next((x for x in structure if x["id"] == fid), None)
                    if f:
                        parts.append(f["name"])
                path = os.path.join(*parts)
                folder_paths[folder["id"]] = path
                zipf.writestr(path + "/", "")

            for f in files:
                info = file_mapping.get(f.filename)
                if not info:
                    continue

                content = await f.read()

                if f.filename.lower().endswith(".pdf"):
                    content = add_watermark_to_pdf(content, info["docCode"])

                base, ext = os.path.splitext(f.filename)
                new_name = f"{info['fileNumber']:03d}_{base}{ext}"
                path = os.path.join(folder_paths[info["folderId"]], new_name)

                zipf.writestr(path, content)

        zip_buffer.seek(0)
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=DZO_Dokumenti.zip"}
        )

    except Exception as e:
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
