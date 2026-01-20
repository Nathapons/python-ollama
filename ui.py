from dotenv import load_dotenv
import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
import google.generativeai as genai
import threading
import os
from dotenv import load_dotenv
import numpy as np
import json

load_dotenv()

# --- Configuration ---
API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash-exp')

class VisionApp:
    def __init__(self, window):
        # โหลดภาพ Reference เตรียมไว้เลยเพื่อความเร็ว
        DOG_REF_PATH = "./reference/dog/1.jpg"
        CAT_REF_PATH = "./reference/cat/1.jpg"
        self.ref_dog = self.load_reference(DOG_REF_PATH)
        self.ref_cat = self.load_reference(CAT_REF_PATH)

        self.window = window
        self.window.title("Gemini Vision Control")
        # กำหนดขนาดหน้าต่างโปรแกรมเริ่มต้นให้ใหญ่หน่อย
        self.window.geometry("850x700") 

        self.cap = None
        self.is_running = False
        self.is_paused = False

        # --- ส่วนแสดงผลภาพ (Label) ---
        # สังเกตว่าผมเอา width, height ออก เพื่อให้ Label ขยายตามรูปภาพที่เราจะใส่
        self.label_vid = tk.Label(window, text="Camera is OFF", width=60, height=30, bg="black", fg="white")
        self.label_vid.pack(padx=10, pady=10, expand=True) # expand=True ช่วยจัดพื้นที่

        # --- ปุ่มควบคุม ---
        self.btn_toggle = tk.Button(window, text="Start Camera", command=self.toggle_camera, font=("Arial", 14), width=15)
        self.btn_toggle.pack(pady=5)

        self.btn_capture = tk.Button(window, text="Analyze Image", command=self.analyze_image, state=tk.DISABLED, font=("Arial", 14), width=15, bg="#dddddd")
        self.btn_capture.pack(pady=5)

        # กล่องข้อความผลลัพธ์
        self.result_text = tk.Text(window, height=6, width=60, font=("Arial", 12))
        self.result_text.pack(padx=10, pady=10)

    def toggle_camera(self):
        if not self.is_running:
            # เปิดกล้อง
            self.cap = cv2.VideoCapture(0)
            
            # พยายามขอความละเอียดสูงจากกล้อง
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open camera")
                return
            
            self.is_running = True
            self.btn_toggle.config(text="Stop Camera", bg="#ffcccc")
            self.btn_capture.config(state=tk.NORMAL, bg="#ccffcc")
            self.update_frame()
        else:
            # ปิดกล้อง
            self.is_running = False
            if self.cap:
                self.cap.release()
            self.btn_toggle.config(text="Start Camera", bg="SystemButtonFace")
            self.btn_capture.config(state=tk.DISABLED, bg="#dddddd")
            self.label_vid.config(image="", text="Camera is OFF", width=60, height=15) # รีเซ็ตขนาดตอนปิด

    def update_frame(self):
        if self.is_running:
            if not self.is_paused:
                ret, frame = self.cap.read()
                if ret:
                    # 1. แปลงสี BGR -> RGB
                    self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(self.current_frame)

                    # 2. หัวใจสำคัญ: ปรับขนาดภาพให้ใหญ่ตามต้องการ (เช่น 640x480)
                    # ยิ่งเลขเยอะ ภาพยิ่งใหญ่เต็มจอครับ
                    target_width = 640
                    target_height = 480
                    resized_img = img.resize((target_width, target_height), Image.LANCZOS)

                    # 3. ส่งเข้า Label
                    img_tk = ImageTk.PhotoImage(image=resized_img)
                    self.label_vid.img_tk = img_tk
                    self.label_vid.configure(image=img_tk, width=target_width, height=target_height) # บังคับ Label ให้เท่ารูป
                    self.label_vid.configure(text="") # ลบข้อความออก
            
            self.window.after(10, self.update_frame)

    def analyze_image(self):
        if not self.is_running: return

        # กรณีที่ 1: ถ้ากล้องกำลังทำงานปกติ -> ให้หยุดภาพและส่งวิเคราะห์
        if not self.is_paused:
            self.is_paused = True  # สั่งหยุดภาพ
            self.btn_capture.config(text="Resume Camera", bg="#ffffcc") # เปลี่ยนปุ่มเป็นสีเหลือง
            
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "Freezing image & Analyzing...\n")
            
            # ส่งไปวิเคราะห์ที่ Thread (เหมือนเดิม)
            threading.Thread(target=self.call_gemini_with_ref).start()

        # กรณีที่ 2: ถ้าภาพหยุดอยู่ (Pause) -> ให้กลับมาทำงานต่อ
        else:
            self.is_paused = False # สั่งเดินกล้องต่อ
            self.btn_capture.config(text="Compare & Identify", bg="#dddddd") # เปลี่ยนปุ่มกลับ
            self.result_text.delete(1.0, tk.END)

    def load_reference(self, path):
        """โหลดรูป Reference และแปลงเป็น RGB รอไว้"""
        if not os.path.exists(path):
            messagebox.showwarning("Warning", f"Reference image not found: {path}")
            return None
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # แปลง BGR -> RGB
        return img

    def resize_image_height(self, image, height=300):
        """ฟังก์ชันช่วยปรับขนาดภาพโดยยึดความสูงเป็นหลัก (Aspect Ratio คงเดิม)"""
        h, w = image.shape[:2]
        aspect = w / h
        new_w = int(height * aspect)
        return cv2.resize(image, (new_w, height))

    def call_gemini_with_ref(self):
        try:
            # 1. ตรวจสอบว่ามีภาพ Reference ครบไหม
            if self.ref_dog is None or self.ref_cat is None:
                self.result_text.insert(tk.END, "\nError: Missing reference images.")
                return

            # 2. ปรับขนาดภาพให้สูงเท่ากัน (เช่น 300px) เพื่อนำมาต่อกัน
            target_h = 300
            
            # Resize Reference
            r_dog = self.resize_image_height(self.ref_dog, target_h)
            r_cat = self.resize_image_height(self.ref_cat, target_h)
            
            # Resize Camera Frame (self.current_frame เป็น RGB อยู่แล้ว)
            r_cam = self.resize_image_height(self.current_frame, target_h)

            # 3. นำภาพมาต่อกันแนวนอน: [DOG] | [CAT] | [CAMERA]
            combined_img = np.hstack((r_dog, r_cat, r_cam))
            
            # แปลงเป็น PIL เพื่อส่ง Gemini
            pil_combined = Image.fromarray(combined_img)

            # 4. Prompt คำสั่ง
            instruction = """
            Look at the combined image.
            - Left: Reference Dog
            - Middle: Reference Cat
            - Right: Target Image to Analyze

            Task: Compare the Right image with the Left and Middle references.
            Rules:
            1. If the Right image looks like a DOG, return "result": "DOG".
            2. If the Right image looks like a CAT, return "result": "CAT".
            3. If the Right image is NEITHER a dog nor a cat (e.g., it's a person, a car, a blank wall, or unclear), return "result": "OTHER".

            Return ONLY a JSON object with keys: "result" and "confidence".
            """

            # 5. ส่ง API
            response = model.generate_content([instruction, pil_combined])
            
            # 6. แกะ JSON (Handle กรณีมี Markdown ติดมา)
            clean_text = response.text.replace('```json', '').replace('```', '').strip()
            data = json.loads(clean_text)

            # แสดงผล
            result_type = data.get('result', 'OTHER') # ถ้าแกะไม่ได้ ให้ตีว่าเป็น OTHER ไว้ก่อน
            confidence = data.get('confidence', 'Low')
            # --- 6. แสดงผลลัพธ์ตามที่คุณต้องการ ---
            self.result_text.delete(1.0, tk.END)

            if result_type == "OTHER":
                # แสดงข้อความแจ้งเตือนเมื่อไม่ใช่หมาหรือแมว
                msg = "ไม่สามารถวิเคราะห์ได้ เนื่องจากโปรแกรมนี้แยกได้แค่หมากับแมวเท่านั้น"
                self.result_text.insert(tk.END, msg)
            else:
                # กรณีเจอ หมา หรือ แมว
                msg = f"Result: {result_type}\nConfidence: {confidence}"
                self.result_text.insert(tk.END, msg)

        except Exception as e:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Error: {e}")

# --- Main ---
if __name__ == "__main__":
    root = tk.Tk()
    app = VisionApp(root)
    root.mainloop()