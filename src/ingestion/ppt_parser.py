import os
import json
import re
import boto3
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from typing import List, Dict, Any
import concurrent.futures

bedrock_client = boto3.client('bedrock-runtime', region_name=os.getenv("AWS_REGION", "us-east-1"))
VISION_MODEL_ID = "moonshotai.kimi-k2.5"

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
    """Removes massive unbroken strings (like Base64 image leaks or hex dumps) that crash tokenizers."""
    # If a 'word' is over 200 characters long with no spaces, it is not human text. 
    # This strips out Base64 leaks while preserving real paragraphs.
    cleaned = re.sub(r'\S{200,}', '[GARBAGE_DATA_REMOVED]', text)
    return cleaned.strip()

def analyze_diagram_with_kimi(image_bytes: bytes, image_ext: str) -> Dict[str, str]:
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
    
    # Bedrock requires specific formats like 'png' or 'jpeg'
    fmt = "png" if "png" in image_ext.lower() else "jpeg"
    
    try:
        response = bedrock_client.converse(
            modelId=VISION_MODEL_ID,
            messages=[{
                "role": "user",
                "content": [
                    {"image": {"format": fmt, "source": {"bytes": image_bytes}}},
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

def parse_ppt(file_path: str, filename: str, course_id: str, lecture_number: int) -> List[Dict[str, Any]]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find {file_path}")

    print(f"Processing PPTX: {filename}")
    
    prs = Presentation(file_path)
    chunks = []
    current_topic = "General Context"
    vision_tasks = [] 
    
    for slide_idx, slide in enumerate(prs.slides):
        page_num = slide_idx + 1
        
        # 1. Extract all text from the slide shapes
        text_runs = []
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text:
                text_runs.append(shape.text)
                
        clean_text = sanitize_extracted_text("\n".join(text_runs).strip())
        is_admin = any(kw in clean_text.lower() for kw in ADMIN_KEYWORDS)
        
        chunk_data = {
            "chunk_id": f"{course_id}_{filename}_s{page_num}",
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

        # 2. Update the running topic if there is a title shape
        if slide.shapes.title and slide.shapes.title.text:
            current_topic = slide.shapes.title.text.strip()

        # 3. Vision Fallback Check
        if len(clean_text) < CHARACTER_THRESHOLD:
            # Find the largest image on the slide to send to Bedrock
            image_blobs = []
            for shape in slide.shapes:
                if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                    image_blobs.append((shape.image.blob, shape.image.ext))
                elif shape.shape_type == MSO_SHAPE_TYPE.GROUP:
                    # Sometimes diagrams are grouped shapes containing pictures
                    for sub_shape in shape.shapes:
                        if sub_shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                            image_blobs.append((sub_shape.image.blob, sub_shape.image.ext))
            
            if image_blobs:
                # Grab the largest file size, assuming it's the main diagram and not a tiny logo
                largest_image, ext = max(image_blobs, key=lambda x: len(x[0]))
                
                chunk_data["text"] = clean_text
                chunk_data["topic"] = current_topic
                chunk_data["used_vision"] = True
                
                vision_tasks.append({
                    "chunk_ref": chunk_data, 
                    "image_bytes": largest_image,
                    "ext": ext,
                    "page_num": page_num
                })
            else:
                # Sparse slide, but no images found
                chunk_data["text"] = clean_text
                chunk_data["topic"] = current_topic
        else:
            chunk_data["text"] = clean_text
            chunk_data["topic"] = current_topic
            
        chunks.append(chunk_data)

    # 4. Batch Process Images
    if vision_tasks:
        print(f"Concurrently processing {len(vision_tasks)} PPTX vision fallbacks via Kimi 2.5...")
        
        def _fetch_vision(task):
            result = analyze_diagram_with_kimi(task["image_bytes"], task["ext"])
            return task, result

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(_fetch_vision, task) for task in vision_tasks]
            
            for future in concurrent.futures.as_completed(futures):
                task, vision_result = future.result()
                chunk_ref = task["chunk_ref"]
                
                original_text = chunk_ref["text"]
                chunk_ref["text"] = f"{original_text}\n\n[Visual Description]: {vision_result.get('description', '')}"
                
                if vision_result.get("topic_tag") and vision_result.get("topic_tag") != "Diagram":
                    chunk_ref["topic"] = vision_result.get("topic_tag")
                    
        print("PPTX Vision processing complete.")

    return chunks