import os
import base64
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

load_dotenv()

# 1. Initialize Model
# LangChain จะเรียกใช้ API ผ่านคลาส ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0
)

def encode_image(image_path):
    """แปลงรูปภาพเป็น Base64 เพื่อส่งให้ LangChain"""
    with Image.open(image_path) as img:
        # ปรับขนาดภาพเล็กน้อยเพื่อประหยัด Token (Optional)
        img.thumbnail((512, 512))
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

def classify_with_langchain():
    # 2. เตรียมข้อมูลรูปภาพ
    dog_refs = [f"./reference/dog/{num}.jpg" for num in range(1, 10)]
    cat_refs = [f"./reference/cat/{num}.jpg" for num in range(1, 10)]
    test_img_path = "./dataset/test1.jpg"

    # 3. สร้างรายการ Content สำหรับ HumanMessage
    content_list = [{"type": "text", "text": "Identify the last image based on these references."}]

    # ใส่ Reference Dogs
    content_list.append({"type": "text", "text": "--- DOG REFERENCES ---"})
    for path in dog_refs:
        content_list.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encode_image(path)}"}
        })

    # ใส่ Reference Cats
    content_list.append({"type": "text", "text": "--- CAT REFERENCES ---"})
    for path in cat_refs:
        content_list.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encode_image(path)}"}
        })

    # ใส่รูปที่ต้องการทดสอบ (Task)
    content_list.append({"type": "text", "text": "--- QUESTION: WHAT IS THIS? ---"})
    content_list.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{encode_image(test_img_path)}"}
    })
    
    content_list.append({"type": "text", "text": "Return JSON: { 'result': 'DOG'/'CAT', 'confidence': 'High'/'Low' }"})

    # 4. ส่ง Message ไปยัง LLM
    message = HumanMessage(content=content_list)
    response = llm.invoke([message])

    print("\n--- AI Response ---")
    print(response.content)

if __name__ == "__main__":
    classify_with_langchain()