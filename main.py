import os
import glob
import cv2
import numpy as np
import google.generativeai as genai
from PIL import Image
import time
import json
from dotenv import load_dotenv

from utils.image_tools import resize_image

# --- Configuration ---
TEST_FOLDER = './dataset'
DOG_REF_SINGLE = './reference/master_dog.jpg' 
CAT_REF_SINGLE = './reference/master_cat.jpg'

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel(
    model_name='gemini-2.0-flash-exp',
    generation_config={"response_mime_type": "application/json"}
)

def gemini_classify_fixed_ref():
    print("📦 Loading Master References...")
    img_dog_ref = cv2.imread(DOG_REF_SINGLE)
    img_cat_ref = cv2.imread(CAT_REF_SINGLE)

    if img_dog_ref is None or img_cat_ref is None:
        print("❌ Error: Cannot find master reference images.")
        return []

    img_dog_ref = resize_image(img_dog_ref)
    img_cat_ref = resize_image(img_cat_ref)

    test_images = glob.glob(os.path.join(TEST_FOLDER, '*.[jp][pg]*'))
    print(f"🔍 Found {len(test_images)} images. Starting...\n")

    all_results = []
    
    for test_img_path in test_images:
        filename = os.path.basename(test_img_path)
        
        try:
            img_test = cv2.imread(test_img_path)
            if img_test is None: continue
            img_test = resize_image(img_test)

            combined = np.hstack((img_dog_ref, img_cat_ref, img_test))
            combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(combined_rgb)

            instruction = """
            Identify the image in the RIGHT panel by comparing it with the DOG (LEFT) and CAT (MIDDLE) references.
            Return a JSON object with keys: "result" (either "DOG" or "CAT") and "confidence" (High, Medium, or Low).
            """

            response = model.generate_content([instruction, pil_img])
            
            try:
                data = json.loads(response.text)
                all_results.append({
                    'filename': filename,
                    'result': data.get('result'),
                    'confidence': data.get('confidence')
                })
            except json.JSONDecodeError:
                print(f"⚠️ Failed to parse JSON for {filename}")

            time.sleep(5)

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

    return all_results

if __name__ == '__main__':
    results = gemini_classify_fixed_ref()
    
    # พิมพ์ผลลัพธ์แบบสวยงาม
    import pprint
    pprint.pprint(results)