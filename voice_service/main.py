import os
import tempfile
from io import BytesIO
from pathlib import Path

import soundfile as sf
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from parler_tts import ParlerTTSForConditionalGeneration
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR.parent / ".env")
HF_TOKEN = os.environ["HF_TOKEN"]
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

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


class SpeakRequest(BaseModel):
    text: str
    description: str = DEFAULT_VOICE_DESCRIPTION


@app.get("/health")
def health_check():
    return {"status": "ok", "device": DEVICE}


@app.post("/transcribe")
async def transcribe(audio: UploadFile):
    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp.flush()
        hyps = asr_model.transcribe(tmp.name, return_hypotheses=True)
    return {"text": hyps[0].text}


@app.post("/speak")
def speak(req: SpeakRequest):
    description_ids = tts_description_tokenizer(req.description, return_tensors="pt").to(DEVICE)
    prompt_ids = tts_tokenizer(req.text, return_tensors="pt").to(DEVICE)

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
