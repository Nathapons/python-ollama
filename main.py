import os
import cv2
import numpy as np
import random
import glob
from ollama import Client
from utils.images import get_random_image, resize_image

# Configuration
DOG_FOLDER = './dog'
CAT_FOLDER = './cat'
TEST_FOLDER = './test'
OLLAMA_HOST = 'http://localhost:11434'
MODEL_NAME = 'llama3.2-vision'


def classify_images():
    # 1. Get all test images
    test_images = []
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']
    for ext in extensions:
        test_images.extend(glob.glob(os.path.join(TEST_FOLDER, ext)))
    
    if not test_images:
        print(f"❌ No images found in {TEST_FOLDER}")
        return

    print(f"🔍 Found {len(test_images)} test images. Starting classification...\n")

    client = Client(host=OLLAMA_HOST)

    for test_img_path in test_images:
        filename = os.path.basename(test_img_path)
        print(f"🖼️  Processing: {filename}")

        # 2. Select Random Reference Images
        dog_ref_path = get_random_image(DOG_FOLDER)
        cat_ref_path = get_random_image(CAT_FOLDER)

        if not dog_ref_path or not cat_ref_path:
            print("❌ Error: Could not find reference images in Dog or Cat folders.")
            return

        # 3. Load and Preprocess Images
        try:
            img_test = cv2.imread(test_img_path)
            img_dog = cv2.imread(dog_ref_path)
            img_cat = cv2.imread(cat_ref_path)

            if img_test is None or img_dog is None or img_cat is None:
                print(f"❌ Error loading images for {filename}. Skipping.")
                continue

            # Resize to standard size
            img_test = resize_image(img_test)
            img_dog = resize_image(img_dog)
            img_cat = resize_image(img_cat)

            # 4. Construct Visual Prompt (Horizontal Stack: Dog Ref | Cat Ref | Test Image)
            # This helps the model compare features side-by-side
            combined_img = np.hstack((img_dog, img_cat, img_test))
            
            _, buffer = cv2.imencode('.jpg', combined_img)
            image_bytes = buffer.tobytes()

        except Exception as e:
            print(f"❌ Error processing images: {e}")
            continue

        # 5. Send to Ollama
        instruction = """
        You are an expert image classifier. The image provided corresponds to three panels composed horizontally:
        - LEFT panel: A reference image of a DOG.
        - MIDDLE panel: A reference image of a CAT.
        - RIGHT panel: The TEST image to classify.

        Compare the visual features (ear shape, snout, fur texture, posture) of the TEST image (Right) with the DOG (Left) and CAT (Middle).
        Determine if the TEST image is a DOG or a CAT.

        Return response in this format:
        RESULT: [DOG or CAT]
        CONFIDENCE: [High/Medium/Low]
        REASON: [Brief explanation of matching features]
        """

        try:
            response = client.chat(
                model=MODEL_NAME,
                messages=[{
                    'role': 'user',
                    'content': instruction,
                    'images': [image_bytes]
                }],
                options={
                    'temperature': 0.1, # Lower temperature for more deterministic results
                    'num_predict': 128
                }
            )
            
            print(f"✅ Result for {filename}:")
            print(response['message']['content'].strip())
            print("-" * 50)
            
        except Exception as e:
            print(f"❌ API Error: {e}")

if __name__ == '__main__':
    classify_images()