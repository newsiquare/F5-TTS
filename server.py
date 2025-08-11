#!/usr/bin/env python3
"""
F5-TTS FastAPI Server
A high-performance FastAPI server for F5-TTS text-to-speech synthesis with concurrent support
"""

import os
import sys
import time
import uuid
import logging
import asyncio
from typing import Optional, List
from pathlib import Path
import io

# Add the source directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    import torch
    import soundfile as sf
    from pydub import AudioSegment, silence
    from f5_tts.api import F5TTS
    
    from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel, Field
    import uvicorn
except ImportError as e:
    print(f"Missing required dependencies: {e}")
    print("Please install FastAPI and dependencies:")
    print("pip install fastapi uvicorn python-multipart")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="F5-TTS FastAPI Server",
    description="High-performance F5-TTS server with concurrent support",
    version="2.0.0"
)

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
device = None
output_dir = 'outputs'
resources_dir = 'resources'
default_ref_text = "Some call me nature, others call me mother nature."

# Ensure directories exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(resources_dir, exist_ok=True)

# Request/Response models
class TTSRequest(BaseModel):
    """TTS synthesis request model"""
    text: str = Field(..., description="Text to synthesize", min_length=1)
    voice: str = Field("default_en", description="Voice/speaker identifier")
    speed: float = Field(1.0, description="Speech speed multiplier", ge=0.1, le=3.0)
    remove_silence: bool = Field(True, description="Remove silence from output")

class SynthesizeRequest(BaseModel):
    """Advanced synthesis request model with reference audio support"""
    gen_text: str = Field(..., description="Text to synthesize", min_length=1)
    ref_audio_path: str = Field(..., description="Path to reference audio file")
    ref_text: str = Field(..., description="Reference text that matches the reference audio")
    model: str = Field("F5TTS_Base", description="Model to use for synthesis")
    vocoder_name: str = Field("vocos", description="Vocoder to use")
    speed: float = Field(1.0, description="Speech speed multiplier", ge=0.1, le=3.0)
    remove_silence: bool = Field(True, description="Remove silence from output")
    output_format: str = Field("wav", description="Output audio format")

class UploadResponse(BaseModel):
    """Upload response model"""
    message: str
    voice_id: str
    file_path: str

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    model_loaded: bool
    device: str
    timestamp: float


# Utility functions
async def initialize_model():
    """Initialize F5-TTS model asynchronously"""
    global model, device
    
    device = (
        "cuda" if torch.cuda.is_available()
        else "xpu" if torch.xpu.is_available() 
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    
    try:
        logger.info(f"Initializing F5-TTS model on device: {device}")
        # Run model initialization in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        model = await loop.run_in_executor(None, lambda: F5TTS(device=device))
        logger.info("Model initialized successfully")
        
        # Setup default reference audio
        await setup_default_audio()
        
        return True
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        return False


async def setup_default_audio():
    """Setup default reference audio file asynchronously"""
    try:
        from importlib.resources import files
        default_ref_audio = str(files("f5_tts").joinpath("infer/examples/basic/basic_ref_en.wav"))
        if os.path.exists(default_ref_audio):
            default_path = os.path.join(resources_dir, "default_en.wav")
            if not os.path.exists(default_path):
                import shutil
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, shutil.copy2, default_ref_audio, default_path)
                logger.info("Default reference audio copied")
    except Exception as e:
        logger.warning(f"Could not setup default reference audio: {str(e)}")


def convert_to_wav(input_path: str, output_path: str, target_sr: int = 24000) -> bool:
    """Convert audio file to WAV format"""
    try:
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_channels(1)  # Convert to mono
        audio = audio.set_frame_rate(target_sr)  # Set sample rate
        audio.export(output_path, format='wav')
        return True
    except Exception as e:
        logger.error(f"Failed to convert audio: {str(e)}")
        return False


def process_reference_audio(audio_path: str, max_duration: int = 15000) -> str:
    """Process reference audio to optimal length"""
    try:
        aseg = AudioSegment.from_file(audio_path)
        
        # Try to split on silence for natural clipping
        non_silent_segs = silence.split_on_silence(
            aseg, 
            min_silence_len=1000, 
            silence_thresh=-50, 
            keep_silence=1000, 
            seek_step=10
        )
        
        processed_audio = AudioSegment.silent(duration=0)
        for segment in non_silent_segs:
            if len(processed_audio) > 6000 and len(processed_audio + segment) > max_duration:
                break
            processed_audio += segment
        
        # If still too long, force clip
        if len(processed_audio) > max_duration:
            processed_audio = processed_audio[:max_duration]
        
        # Save processed audio
        processed_path = f"{output_dir}/processed_ref_{uuid.uuid4().hex[:8]}.wav"
        processed_audio.export(processed_path, format='wav')
        
        return processed_path
        
    except Exception as e:
        logger.error(f"Failed to process reference audio: {str(e)}")
        return audio_path


def find_voice_file(voice_id: str) -> Optional[str]:
    """Find voice file by ID"""
    # First try WAV files
    wav_files = [f for f in os.listdir(resources_dir) 
                 if f.startswith(voice_id) and f.lower().endswith('.wav')]
    
    if wav_files:
        return os.path.join(resources_dir, wav_files[0])
    
    # Then try other formats and convert
    other_files = [f for f in os.listdir(resources_dir) 
                   if f.startswith(voice_id)]
    
    if other_files:
        source_path = os.path.join(resources_dir, other_files[0])
        wav_path = os.path.join(resources_dir, f"{voice_id}.wav")
        if convert_to_wav(source_path, wav_path):
            return wav_path
        
    return None


async def synthesize_speech_async(text: str, voice_id: str = "default_en", speed: float = 1.0) -> str:
    """Synthesize speech asynchronously and return output file path"""
    global model
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    # Find reference audio
    if voice_id == "default_en":
        ref_file = os.path.join(resources_dir, "default_en.wav")
        ref_text = default_ref_text
        
        if not os.path.exists(ref_file):
            raise HTTPException(status_code=404, detail="Default reference audio not found")
    else:
        ref_file = find_voice_file(voice_id)
        if not ref_file:
            raise HTTPException(status_code=404, detail=f"Voice '{voice_id}' not found")
        
        # Process and transcribe in thread pool
        loop = asyncio.get_event_loop()
        processed_ref = await loop.run_in_executor(None, process_reference_audio, ref_file)
        ref_text = await loop.run_in_executor(None, model.transcribe, processed_ref)
        ref_file = processed_ref
    
    # Generate speech in thread pool to avoid blocking
    output_path = f"{output_dir}/output_{uuid.uuid4().hex[:8]}.wav"
    
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: model.infer(
                ref_file=ref_file,
                ref_text=ref_text,
                gen_text=text,
                speed=speed,
                remove_silence=True,
                file_wave=output_path
            )
        )
        
        return output_path
        
    except Exception as e:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise e


# Background task functions
def cleanup_file(file_path: str):
    """Clean up a single file"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        logger.warning(f"Failed to cleanup file {file_path}: {e}")


def cleanup_files(file_paths: List[str]):
    """Clean up multiple files"""
    for file_path in file_paths:
        cleanup_file(file_path)


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize model and perform warmup"""
    logger.info("Starting F5-TTS FastAPI server...")
    
    # Initialize model
    if not await initialize_model():
        logger.error("Failed to initialize model during startup")
        return
    
    # Warmup
    try:
        logger.info("Performing warmup inference...")
        warmup_path = await synthesize_speech_async("This is a warmup test.", "default_en", 1.0)
        os.remove(warmup_path)  # Clean up warmup file
        logger.info("Warmup completed successfully")
    except Exception as e:
        logger.warning(f"Warmup failed: {str(e)}")


# API Endpoints
@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information"""
    return {
        "name": "F5-TTS FastAPI Server",
        "version": "2.0.0",
        "status": "running",
        "device": device,
        "features": ["concurrent_processing", "async_inference", "voice_cloning"],
        "endpoints": {
            "GET /": "API information",
            "GET /health": "Health check",
            "GET /voices": "List available voices",
            "POST /tts": "Simple text-to-speech synthesis",
            "POST /synthesize": "Advanced synthesis with reference audio",
            "POST /upload_voice": "Upload reference voice",
            "POST /clone_voice": "Voice cloning with uploaded audio"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        device=device,
        timestamp=time.time()
    )


@app.get("/voices")
async def list_voices():
    """List available voice IDs"""
    try:
        voices = []
        for file in os.listdir(resources_dir):
            if file.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
                voice_id = os.path.splitext(file)[0]
                voices.append({
                    "voice_id": voice_id,
                    "filename": file
                })
        
        return {
            "voices": voices,
            "count": len(voices)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list voices: {str(e)}")


@app.post("/upload_voice", response_model=UploadResponse)
async def upload_voice(
    voice_id: str = Form(...),
    file: UploadFile = File(...)
):
    """Upload a reference voice file"""
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Check file extension
        allowed_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format. Allowed: {allowed_extensions}"
            )
        
        # Check file size (max 10MB)
        contents = await file.read()
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large (max 10MB)")
        
        # Save original file
        original_path = os.path.join(resources_dir, f"{voice_id}{file_ext}")
        with open(original_path, 'wb') as f:
            f.write(contents)
        
        # Convert to WAV for F5-TTS in thread pool
        wav_path = os.path.join(resources_dir, f"{voice_id}.wav")
        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(None, convert_to_wav, original_path, wav_path)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to convert audio to WAV")
        
        logger.info(f"Voice uploaded successfully: {voice_id}")
        
        return UploadResponse(
            message=f"Voice '{voice_id}' uploaded successfully",
            voice_id=voice_id,
            file_path=wav_path
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/tts")
async def text_to_speech(
    request: TTSRequest,
    background_tasks: BackgroundTasks
):
    """Synthesize speech from text using a specified voice"""
    try:
        start_time = time.time()
        logger.info(f"TTS request: text='{request.text[:50]}...', voice='{request.voice}', speed={request.speed}")
        
        output_path = await synthesize_speech_async(
            text=request.text,
            voice_id=request.voice,
            speed=request.speed
        )
        
        # Schedule file cleanup after response is sent
        background_tasks.add_task(cleanup_file, output_path)
        
        # Create streaming response
        def file_generator():
            with open(output_path, 'rb') as f:
                while chunk := f.read(8192):
                    yield chunk
        
        elapsed_time = time.time() - start_time
        
        return StreamingResponse(
            file_generator(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=synthesized_speech.wav",
                "X-Device-Used": device,
                "X-Voice-Used": request.voice,
                "X-Elapsed-Time": str(elapsed_time),
                "Access-Control-Allow-Origin": "*"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS synthesis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {str(e)}")


@app.post("/clone_voice")
async def clone_voice_with_file(
    background_tasks: BackgroundTasks,
    text: str = Form(...),
    ref_text: Optional[str] = Form(None),
    speed: float = Form(1.0),
    remove_silence: bool = Form(True),
    file: UploadFile = File(...)
):
    """Clone voice using uploaded reference audio"""
    global model
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    try:
        # Save uploaded reference audio
        contents = await file.read()
        temp_ref_path = f"{output_dir}/temp_ref_{uuid.uuid4().hex[:8]}.wav"
        
        # Save and convert to WAV
        temp_original = f"{output_dir}/temp_original_{uuid.uuid4().hex[:8]}{Path(file.filename).suffix}"
        with open(temp_original, 'wb') as f:
            f.write(contents)
        
        # Convert in thread pool
        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(None, convert_to_wav, temp_original, temp_ref_path)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to process reference audio")
        
        # Process reference audio in thread pool
        processed_ref_path = await loop.run_in_executor(None, process_reference_audio, temp_ref_path)
        
        # Get reference text
        if ref_text is None or ref_text.strip() == "":
            logger.info("Transcribing reference audio...")
            ref_text = await loop.run_in_executor(None, model.transcribe, processed_ref_path)
            logger.info(f"Transcribed text: {ref_text}")
        
        # Generate speech in thread pool
        output_path = f"{output_dir}/cloned_{uuid.uuid4().hex[:8]}.wav"
        
        await loop.run_in_executor(
            None,
            lambda: model.infer(
                ref_file=processed_ref_path,
                ref_text=ref_text,
                gen_text=text,
                speed=speed,
                remove_silence=remove_silence,
                file_wave=output_path
            )
        )
        
        # Schedule cleanup of temporary files
        background_tasks.add_task(cleanup_files, [temp_original, temp_ref_path, processed_ref_path, output_path])
        
        # Create streaming response
        def file_generator():
            with open(output_path, 'rb') as f:
                while chunk := f.read(8192):
                    yield chunk
        
        return StreamingResponse(
            file_generator(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=cloned_speech.wav",
                "X-Device-Used": device,
                "Access-Control-Allow-Origin": "*"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice cloning failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Voice cloning failed: {str(e)}")


@app.post("/synthesize")
async def synthesize_with_reference(
    request: SynthesizeRequest,
    background_tasks: BackgroundTasks
):
    """Advanced synthesis with custom reference audio and text"""
    try:
        start_time = time.time()
        logger.info(f"Synthesize request: text='{request.gen_text[:50]}...', ref_audio='{request.ref_audio_path}'")
        
        # Check if model is loaded
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Check if reference audio file exists
        ref_path = request.ref_audio_path
        if not os.path.isabs(ref_path):
            # Try relative to current directory
            ref_path = os.path.join(os.getcwd(), request.ref_audio_path)
        
        if not os.path.exists(ref_path):
            raise HTTPException(status_code=404, detail=f"Reference audio file not found: {request.ref_audio_path}")
        
        # Create output path
        output_filename = f"synthesized_{uuid.uuid4().hex[:8]}.{request.output_format}"
        output_path = os.path.join(output_dir, output_filename)
        
        # Process reference audio if needed
        processed_ref_path = ref_path
        if not ref_path.lower().endswith('.wav'):
            # Convert to WAV for better compatibility
            temp_wav = os.path.join(output_dir, f"temp_ref_{uuid.uuid4().hex[:8]}.wav")
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(None, convert_to_wav, ref_path, temp_wav)
            if success:
                processed_ref_path = temp_wav
            else:
                logger.warning(f"Failed to convert reference audio, using original: {ref_path}")
        
        # Run synthesis in thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: model.infer(
                ref_file=processed_ref_path,
                ref_text=request.ref_text,
                gen_text=request.gen_text,
                speed=request.speed,
                remove_silence=request.remove_silence,
                file_wave=output_path
            )
        )
        
        # Clean up temporary reference file if created
        if processed_ref_path != ref_path:
            background_tasks.add_task(cleanup_file, processed_ref_path)
        
        # Schedule output file cleanup
        background_tasks.add_task(cleanup_file, output_path)
        
        # Create streaming response
        def file_generator():
            with open(output_path, 'rb') as f:
                while chunk := f.read(8192):
                    yield chunk
        
        elapsed_time = time.time() - start_time
        
        return StreamingResponse(
            file_generator(),
            media_type=f"audio/{request.output_format}",
            headers={
                "Content-Disposition": f"attachment; filename=synthesized.{request.output_format}",
                "X-Device-Used": device,
                "X-Model-Used": request.model,
                "X-Vocoder-Used": request.vocoder_name,
                "X-Elapsed-Time": str(elapsed_time),
                "Access-Control-Allow-Origin": "*"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Synthesis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="F5-TTS FastAPI Server")
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--workers', type=int, default=1, help='Number of worker processes')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload for development')
    
    args = parser.parse_args()
    
    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level="info"
    )
