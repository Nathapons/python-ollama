FROM ollama/ollama:latest

# กำหนดที่อยู่ของโมเดลไว้ในโฟลเดอร์อื่น เพื่อไม่ให้โดน Volume ทับ
ENV OLLAMA_MODELS=/root/.ollama/models

# รัน Ollama Serve ชั่วคราว และทำการ Pull โมเดลที่ต้องการ
RUN nohup bash -c "ollama serve &" && \
    sleep 5 && \
    ollama pull llama3.2-vision:11b && \
    ollama pull moondream

EXPOSE 11434