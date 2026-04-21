import os
import json
import re
import fitz  # PyMuPDF
import pymupdf4llm
import boto3
from typing import List, Dict, Any
import concurrent.futures

bedrock_client = boto3.client('bedrock-runtime', region_name=os.getenv("AWS_REGION", "us-east-1"))
VISION_MODEL_ID = "moonshotai.kimi-k2.5"

# Increased threshold: Academic slides often have a title, one bullet, and a huge diagram.
# 300 characters is a safer net for "sparse" pages.
CHARACTER_THRESHOLD = 300 

ADMIN_KEYWORDS = ["office hours", "grading policy", "zoom link", "late submission", "prerequisites", "textbooks", "ta information", "course work", "course overview"]

def clean_json_response(content: str) -> dict:
    """Extracts JSON from an LLM response, ignoring conversational padding."""
    try:
        # Find everything between the first { and last }
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        return {}
    except Exception as e:
        print(f"JSON Parsing Error: {e}")
        return {}

def analyze_diagram_with_kimi(image_bytes: bytes) -> Dict[str, str]:
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
                    {"image": {"format": "png", "source": {"bytes": image_bytes}}},
                    {"text": prompt}
                ]
            }],
            inferenceConfig={"maxTokens": 512, "temperature": 0.1}
        )
        content = response['output']['message']['content'][0]['text']
        
        parsed_json = clean_json_response(content)
        return parsed_json if parsed_json else {"description": "Visual could not be parsed.", "topic_tag": "Diagram"}
        
    except Exception as e:
        print(f"Vision fallback failed: {e}")
        return {"description": "Vision API failed.", "topic_tag": "Diagram"}

def parse_pdf(file_path: str, filename: str, course_id: str, lecture_number: int) -> List[Dict[str, Any]]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find {file_path}")

    print(f"Processing PDF: {filename}")
    
    doc_chunks = pymupdf4llm.to_markdown(file_path, page_chunks=True)
    doc_pdf = fitz.open(file_path) 
    
    chunks = []
    current_topic = "General Context"
    vision_tasks = [] # We will collect images here to process in parallel
    
    for index, page_data in enumerate(doc_chunks):
        page_num = index + 1 
        clean_text = page_data.get("text", "").strip()
        
        is_admin = any(kw in clean_text.lower() for kw in ADMIN_KEYWORDS)
        
        chunk_data = {
            "chunk_id": f"{course_id}_{filename}_p{page_num}",
            "file_name": filename,
            "course_id": course_id,
            "lecture_number": lecture_number,
            "page_number": page_num,
            "is_administrative": is_admin,
            "has_math": bool(re.search(r'[\$\\]', clean_text)),
            "used_vision": False
        }

        if is_admin:
            chunk_data["text"] = clean_text
            chunk_data["topic"] = "Course Administration"
            chunks.append(chunk_data)
            continue

        if len(clean_text) < CHARACTER_THRESHOLD:
            # We flag it, grab the image, but DO NOT call Bedrock yet.
            page = doc_pdf.load_page(page_num - 1) 
            pix = page.get_pixmap(dpi=150)
            image_bytes = pix.tobytes("png")
            
            chunk_data["text"] = clean_text
            chunk_data["topic"] = current_topic # Inherit temporarily
            chunk_data["used_vision"] = True
            
            vision_tasks.append({
                "chunk_ref": chunk_data, 
                "image_bytes": image_bytes,
                "page_num": page_num
            })
            
        else:
            headers = re.findall(r'^(#{1,2})\s+([A-Za-z].+)$', clean_text, re.MULTILINE)
            if headers:
                current_topic = headers[-1][1].replace("**", "").strip()
                
            chunk_data["text"] = clean_text
            chunk_data["topic"] = current_topic
            
        chunks.append(chunk_data)
        
    doc_pdf.close()

    # --- BATCH PROCESS ALL IMAGES CONCURRENTLY ---
    if vision_tasks:
        print(f"Concurrently processing {len(vision_tasks)} vision fallbacks via Kimi 2.5...")
        
        def _fetch_vision(task):
            result = analyze_diagram_with_kimi(task["image_bytes"])
            return task, result

        # 10 workers ensures we don't hit Bedrock rate limits while still being 10x faster
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(_fetch_vision, task) for task in vision_tasks]
            
            for future in concurrent.futures.as_completed(futures):
                task, vision_result = future.result()
                chunk_ref = task["chunk_ref"]
                
                # Inject the results back into the dictionary reference
                original_text = chunk_ref["text"]
                chunk_ref["text"] = f"{original_text}\n\n[Visual Description]: {vision_result.get('description', '')}"
                
                # Override the inherited topic if Kimi found a better one
                if vision_result.get("topic_tag") and vision_result.get("topic_tag") != "Diagram":
                    chunk_ref["topic"] = vision_result.get("topic_tag")
                    
        print("Vision processing complete.")

    return chunks

if __name__ == "__main__":
    sample_pdf_path = "1-introduction.pdf"
    if os.path.exists(sample_pdf_path):
        print(f"Extracting chunks from {sample_pdf_path}...\n")
        # Suggestion 8: Pass lecture number dynamically
        extracted_chunks = parse_pdf(sample_pdf_path, "1-introduction.pdf", "CS584", lecture_number=1)
        
        # Filter out administrative chunks just to see what the Vector DB would actually get
        academic_chunks = [c for c in extracted_chunks if not c.get("is_administrative")]
        
        with open("extracted_chunks_v2.json", "w") as f:
            json.dump(academic_chunks, f, indent=2)
            
        print(f"Saved {len(academic_chunks)} academic chunks (filtered out {len(extracted_chunks) - len(academic_chunks)} admin pages).")