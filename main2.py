import os
import glob
import cv2
import numpy as np
import google.generativeai as genai  # เปลี่ยนจาก ollama
from PIL import Image  # Gemini รองรับ PIL Image โดยตรง
import io

from utils.images import get_random_image, resize_image

# --- Configuration ---
DOG_FOLDER = './dog'
CAT_FOLDER = './cat'
TEST_FOLDER = './test'
# 🔑 นำ API Key จาก Google AI Studio มาใส่ที่นี่
GEMINI_API_KEY = "AIzaSyCO4KeK_p5TrkTfkMipevbxE0M37IxafvE"

# ตั้งค่าโมเดล
genai.configure(api_key=GEMINI_API_KEY)
# แนะนำ 1.5-flash สำหรับงานวนลูปจำนวนมาก เพราะฟรีและเร็ว
model = genai.GenerativeModel('gemini-2.0-flash-exp')

def classify_images():
    # 1. Get all test images
    test_images = []
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']
    for ext in extensions:
        test_images.extend(glob.glob(os.path.join(TEST_FOLDER, ext)))
    
    if not test_images:
        print(f"❌ No images found in {TEST_FOLDER}")
        return

    print(f"🔍 Found {len(test_images)} test images. Starting classification with Gemini...\n")

    for test_img_path in test_images:
        filename = os.path.basename(test_img_path)
        print(f"🖼️  Processing: {filename}")

        # 2. Select Reference Images
        dog_ref_path = get_random_image(DOG_FOLDER)
        cat_ref_path = get_random_image(CAT_FOLDER)

        if not dog_ref_path or not cat_ref_path:
            print("❌ Error: Missing reference images.")
            continue

        try:
            # 3. Load and Preprocess
            img_test = cv2.imread(test_img_path)
            img_dog = cv2.imread(dog_ref_path)
            img_cat = cv2.imread(cat_ref_path)

            if img_test is None or img_dog is None or img_cat is None:
                continue

            img_test = resize_image(img_test)
            img_dog = resize_image(img_dog)
            img_cat = resize_image(img_cat)

            # 4. Construct Visual Prompt (Horizontal Stack)
            combined_img_cv2 = np.hstack((img_dog, img_cat, img_test))
            
            # แปลง OpenCV image (BGR) เป็น PIL Image (RGB) เพื่อส่งให้ Gemini
            combined_img_rgb = cv2.cvtColor(combined_img_cv2, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(combined_img_rgb)

        except Exception as e:
            print(f"❌ Error processing images: {e}")
            continue

        # 5. Send to Gemini
        instruction = """
        You are an expert image classifier. The image provided has three panels:
        - LEFT: Reference DOG
        - MIDDLE: Reference CAT
        - RIGHT: TEST image
        
        Compare the features of the TEST image with the DOG and CAT references.
        Return ONLY this format:
        RESULT: [DOG or CAT]
        CONFIDENCE: [High/Medium/Low]
        """

        try:
            # Gemini ส่งทั้ง Prompt และ Image เข้าไปพร้อมกันได้เลย
            response = model.generate_content([instruction, pil_img])
            
            print(f"✅ Result for {filename}:")
            print(response.text.strip())
            print("-" * 50)
            
        except Exception as e:
            print(f"❌ Gemini API Error: {e}")

if __name__ == '__main__':
    classify_images()