FROM ollama/ollama:latest

# สั่งให้ Ollama รันเป็นพื้นหลังชั่วคราวเพื่อ Pull โมเดลระหว่างขั้นตอน Build
RUN nohup bash -c "ollama serve &" && \
    sleep 5 && \
    ollama pull llama3.2-vision:11b

# เปิดพอร์ตมาตรฐาน
EXPOSE 11434