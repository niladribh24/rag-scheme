import os
import subprocess
import tempfile
import threading
from io import BytesIO
from pathlib import Path

import soundfile as sf
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from parler_tts import ParlerTTSForConditionalGeneration
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR.parent / ".env")
HF_TOKEN = os.environ["HF_TOKEN"]
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25MB, generous for a few minutes of speech
MAX_SPEAK_CHARS = 1000  # keeps generation time/VRAM bounded

app = FastAPI(title="Setu — Voice Service")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Both models loaded once at process startup, same pattern as api.py's index build.
asr_model = AutoModel.from_pretrained(
    "ARTPARK-IISc/SraVaani-1.0", trust_remote_code=True, token=HF_TOKEN
).to(DEVICE).eval()

tts_model = ParlerTTSForConditionalGeneration.from_pretrained(
    "ai4bharat/indic-parler-tts", token=HF_TOKEN
).to(DEVICE).eval()
tts_tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts", token=HF_TOKEN)
tts_description_tokenizer = AutoTokenizer.from_pretrained(
    tts_model.config.text_encoder._name_or_path, token=HF_TOKEN
)

DEFAULT_VOICE_DESCRIPTION = (
    "A female speaker delivers a clear, moderate-speed speech with a close "
    "recording and no background noise."
)

# Both models share one GPU with limited VRAM. Two requests running .generate()
# at the same time can race for memory and OOM, so inference is serialized here
# rather than left to FastAPI's default threadpool concurrency.
gpu_lock = threading.Lock()


class SpeakRequest(BaseModel):
    text: str
    description: str = DEFAULT_VOICE_DESCRIPTION


@app.get("/health")
def health_check():
    return {"status": "ok", "device": DEVICE}


def _convert_to_wav(src_path: str, dst_path: str) -> None:
    """Normalize arbitrary browser-recorded audio (webm/opus, ogg, mp3, ...) to
    16kHz mono WAV, since soundfile/libsndfile can't read most of those directly."""
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", src_path, "-ar", "16000", "-ac", "1", dst_path],
        capture_output=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise HTTPException(
            status_code=400,
            detail="Could not decode audio. Unsupported or corrupt file.",
        )


@app.post("/transcribe")
async def transcribe(audio: UploadFile):
    raw = await audio.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty audio upload.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Audio file too large.")

    suffix = Path(audio.filename or "audio.webm").suffix or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix) as src, \
         tempfile.NamedTemporaryFile(suffix=".wav") as wav:
        src.write(raw)
        src.flush()
        _convert_to_wav(src.name, wav.name)

        with gpu_lock:
            hyps = asr_model.transcribe(wav.name, return_hypotheses=True)

    if not hyps:
        raise HTTPException(status_code=422, detail="Could not transcribe audio.")
    return {"text": hyps[0].text}


@app.post("/speak")
def speak(req: SpeakRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must not be empty.")
    if len(text) > MAX_SPEAK_CHARS:
        raise HTTPException(
            status_code=413, detail=f"text exceeds {MAX_SPEAK_CHARS} character limit."
        )

    description_ids = tts_description_tokenizer(req.description, return_tensors="pt").to(DEVICE)
    prompt_ids = tts_tokenizer(text, return_tensors="pt").to(DEVICE)

    with gpu_lock:
        generation = tts_model.generate(
            input_ids=description_ids.input_ids,
            attention_mask=description_ids.attention_mask,
            prompt_input_ids=prompt_ids.input_ids,
            prompt_attention_mask=prompt_ids.attention_mask,
        )
    audio_arr = generation.cpu().numpy().squeeze()

    buffer = BytesIO()
    sf.write(buffer, audio_arr, tts_model.config.sampling_rate, format="WAV")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="audio/wav")
