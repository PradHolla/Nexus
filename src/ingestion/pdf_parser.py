import os
import json
import re
import fitz  # PyMuPDF
import pymupdf4llm
import boto3
from typing import List, Dict, Any
import concurrent.futures
import io
from PIL import Image
import uuid
from src.job_status_store import append_job_log

bedrock_client = boto3.client('bedrock-runtime', region_name=os.getenv("AWS_REGION", "us-east-1"))
VISION_MODEL_ID = "moonshotai.kimi-k2.5" # Or zai.glm-5

CHARACTER_THRESHOLD = 300 
ADMIN_KEYWORDS = ["office hours", "grading policy", "zoom link", "late submission", "prerequisites", "textbooks", "ta information", "course work", "course overview"]

def clean_json_response(content: str) -> dict:
    try:
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        return {}
    except Exception as e:
        print(f"JSON Parsing Error: {e}")
        return {}

def sanitize_extracted_text(text: str) -> str:
    """Removes massive unbroken strings (like Base64 image leaks) that crash tokenizers."""
    cleaned = re.sub(r'\S{200,}', '[GARBAGE_DATA_REMOVED]', text)
    return cleaned.strip()

def analyze_diagram_with_kimi(image_bytes: bytes, image_ext: str = "png", job_id: str = None) -> Dict[str, str]:
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
            
        img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        
        compressed_bytes = buffer.getvalue()
        fmt = "jpeg"
    except Exception as e:
        msg = f"Warning: Image compression failed. Error: {e}"
        if job_id:
            append_job_log(job_id, msg)
        else:
            print(msg)
        compressed_bytes = image_bytes
        fmt = "png" if "png" in image_ext.lower() else "jpeg"

    prompt = """
    You are an expert academic tutor. Analyze this diagram, chart, or visual slide. 
    1. Describe the educational concept shown in high detail so it can be used for a quiz.
    2. Provide a 1-3 word topic tag for what this image represents.
    
    Return ONLY valid JSON. Format:
    {
        "description": "A detailed explanation...",
        "topic_tag": "Neural Networks"
    }
    """
    
    try:
        response = bedrock_client.converse(
            modelId=VISION_MODEL_ID,
            messages=[{
                "role": "user",
                "content": [
                    {"image": {"format": fmt, "source": {"bytes": compressed_bytes}}},
                    {"text": prompt}
                ]
            }],
            inferenceConfig={"maxTokens": 512, "temperature": 0.1}
        )
        content = response['output']['message']['content'][0]['text']
        
        parsed_json = clean_json_response(content)
        return parsed_json if parsed_json else {"description": "Visual could not be parsed.", "topic_tag": "Diagram"}
    except Exception as e:
        msg = f"Vision fallback failed: {e}"
        if job_id:
            append_job_log(job_id, msg)
        else:
            print(msg)
        return {"description": "Vision API failed.", "topic_tag": "Diagram"}

def parse_pdf(file_path: str, filename: str, course_id: str, lecture_number: int, job_id: str = None) -> List[Dict[str, Any]]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find {file_path}")

    msg = f"Processing PDF: {filename} (Semantic Grouping Enabled)"
    if job_id:
        append_job_log(job_id, msg)
    else:
        print(msg)
    
    doc_chunks = pymupdf4llm.to_markdown(file_path, page_chunks=True)
    doc_pdf = fitz.open(file_path) 
    
    chunks = []
    vision_tasks = [] 
    
    # Helper to generate a fresh semantic chunk wrapper
    def _create_new_chunk(topic: str, start_page: int) -> dict:
        return {
            "chunk_id": f"{course_id}_{filename}_p{start_page}_{uuid.uuid4().hex[:6]}",
            "file_name": filename,
            "course_id": course_id,
            "lecture_number": lecture_number,
            "page_number": start_page, # Tracks where the concept started
            "is_administrative": False,
            "has_math": False,
            "used_vision": False,
            "text": "",
            "topic": topic
        }

    current_chunk = _create_new_chunk("General Context", 1)
    
    for index, page_data in enumerate(doc_chunks):
        page_num = index + 1 
        clean_text = sanitize_extracted_text(page_data.get("text", ""))
        
        is_admin = any(kw in clean_text.lower() for kw in ADMIN_KEYWORDS)
        
        if is_admin:
            admin_chunk = _create_new_chunk("Course Administration", page_num)
            admin_chunk["is_administrative"] = True
            admin_chunk["text"] = clean_text
            chunks.append(admin_chunk)
            continue

        if len(clean_text) < CHARACTER_THRESHOLD:
            page = doc_pdf.load_page(page_num - 1) 
            pix = page.get_pixmap(dpi=150)
            image_bytes = pix.tobytes("png")
            
            current_chunk["used_vision"] = True
            
            # Pass the running chunk reference to the vision task array
            vision_tasks.append({
                "chunk_ref": current_chunk, 
                "image_bytes": image_bytes,
            })
            
            if clean_text:
                current_chunk["text"] += f"\n{clean_text}\n"
                
        else:
            # SEMANTIC SPLITTING: Look for Markdown headers (##, ###)
            headers = re.findall(r'^(#{1,3})\s+([A-Za-z].+)$', clean_text, re.MULTILINE)
            
            if headers:
                # We found a new topic! If the current chunk has data, save it and start a new one
                if current_chunk["text"].strip() or current_chunk["used_vision"]:
                    chunks.append(current_chunk)
                    new_topic = headers[-1][1].replace("**", "").strip()
                    current_chunk = _create_new_chunk(new_topic, page_num)
                else:
                    # It's empty, just rename the topic
                    current_chunk["topic"] = headers[-1][1].replace("**", "").strip()
            
            if bool(re.search(r'[\$\\]', clean_text)):
                current_chunk["has_math"] = True
                
            current_chunk["text"] += f"\n{clean_text}\n"
            
    # Flush the final chunk at the end of the document
    if current_chunk["text"].strip() or current_chunk["used_vision"]:
        chunks.append(current_chunk)
        
    doc_pdf.close()

    if vision_tasks:
        msg = f"Concurrently processing {len(vision_tasks)} vision fallbacks via Kimi 2.5..."
        if job_id:
            append_job_log(job_id, msg)
        else:
            print(msg)
        
        def _fetch_vision(task):
            result = analyze_diagram_with_kimi(task["image_bytes"], job_id=job_id)
            return task, result

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(_fetch_vision, task) for task in vision_tasks]
            
            for future in concurrent.futures.as_completed(futures):
                task, vision_result = future.result()
                chunk_ref = task["chunk_ref"]
                
                original_text = chunk_ref["text"]
                # The visual description gets injected right into the middle of the semantic chunk
                chunk_ref["text"] = f"{original_text}\n\n[Visual Description]: {vision_result.get('description', '')}"
                
                if vision_result.get("topic_tag") and vision_result.get("topic_tag") != "Diagram":
                    chunk_ref["topic"] = vision_result.get("topic_tag")
                    
        msg = "Vision processing complete."
        if job_id:
            append_job_log(job_id, msg)
        else:
            print(msg)

    return chunks

if __name__ == "__main__":
    sample_pdf_path = "1-introduction.pdf"
    if os.path.exists(sample_pdf_path):
        print(f"Extracting chunks from {sample_pdf_path}...\n")
        extracted_chunks = parse_pdf(sample_pdf_path, "1-introduction.pdf", "CS584", lecture_number=1)
        academic_chunks = [c for c in extracted_chunks if not c.get("is_administrative")]
        
        with open("extracted_chunks_semantic.json", "w") as f:
            json.dump(academic_chunks, f, indent=2)
            
        print(f"Saved {len(academic_chunks)} academic chunks.")
