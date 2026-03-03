import os
import uuid
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.llms import HuggingFaceHub # Or use OpenAI/Llama
from gtts import gTTS
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, ColorClip
import shutil

app = FastAPI()

# --- CONFIGURATION ---
# In a real app, you would use a real LLM API key here
# For this demo, we will simulate the LLM logic to keep it runnable without keys
class MockLLM:
    def __call__(self, prompt):
        return f"Generated video for: {prompt}. This is a simulated response."

# --- MEMORY SETUP ---
# This allows the bot to remember the conversation context
memory = ConversationBufferMemory()
# We use a mock LLM for this demo. Replace with: llm = OpenAI(temperature=0.7)
llm = MockLLM() 
conversation = ConversationChain(llm=llm, memory=memory)

# --- VIDEO GENERATION LOGIC ---
# NOTE: Real Text-to-Video requires heavy GPU models (like Stable Video Diffusion).
# This function simulates the generation process for the code structure.
def generate_video_clip(prompt: str, duration: int = 5):
    """
    Generates a video clip. 
    In production, you would call a local API or a GPU model here.
    """
    print(f"Generating video for prompt: {prompt}")
    
    # 1. Create a simple colored video as a placeholder
    # Width 640, Height 480, Color based on prompt hash
    color = (hash(prompt) % 255, (hash(prompt) // 255) % 255, (hash(prompt) // 65535) % 255)
    
    clip = ColorClip(size=(640, 480), color=color, duration=duration)
    clip = clip.set_fps(24)
    
    # Save to temp file
    video_path = f"temp_video_{uuid.uuid4()}.mp4"
    clip.write_videofile(video_path, codec='libx264', audio=False)
    return video_path

def generate_audio_clip(text: str):
    """
    Generates audio from text using Google TTS.
    """
    tts = gTTS(text=text, lang='en')
    audio_path = f"temp_audio_{uuid.uuid4()}.mp3"
    tts.save(audio_path)
    return audio_path

def combine_media(video_path: str, audio_path: str, output_path: str):
    """
    Combines video and audio using MoviePy.
    """
    video = VideoFileClip(video_path)
    audio = AudioFileClip(audio_path)
    
    # Set audio
    video = video.set_audio(audio)
    
    # Write final file
    video.write_videofile(output_path, codec='libx264', audio_codec='aac')
    
    # Cleanup temp files
    os.remove(video_path)
    os.remove(audio_path)

# --- API ENDPOINTS ---

class ChatRequest(BaseModel):
    message: str
    video_duration: int = 5  # Default 5 seconds, can be increased for "no limit"

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Handles the chat logic, generates memory, and triggers video generation.
    """
    try:
        # 1. Get response from LLM (with Memory)
        # We append the video request to the prompt so the bot knows what to do
        full_prompt = f"{request.message} (Generate a video for this)"
        response = conversation.predict(input=full_prompt)
        
        # 2. Generate Audio (The bot's voice)
        audio_path = generate_audio_clip(response)
        
        # 3. Generate Video (The visual)
        # To simulate "No Limit", we can loop the video clip
        video_path = generate_video_clip(request.message, request.video_duration)
        
        # 4. Combine them
        output_filename = f"output_{uuid.uuid4()}.mp4"
        combine_media(video_path, audio_path, output_filename)
        
        return {
            "response": response,
            "video_url": f"/videos/{output_filename}",
            "memory_context": memory.load_memory_variables({})
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/videos/{filename}")
async def get_video(filename: str):
    """
    Serves the generated video file.
    """
    if not os.path.exists(filename):
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(filename)

# --- CLEANUP SCHEDULER (Optional) ---
# In a real app, you'd want to delete old videos to save disk space
# This is a simple background task placeholder
@app.on_event("startup")
async def startup_event():
    print("Video Chatbot Server Started")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
