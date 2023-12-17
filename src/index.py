from fastapi import FastAPI, File, UploadFile, Form, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, pipeline
import requests
from PIL import Image
import io
import base64
from src.dtos.ISayHelloDto import ISayHelloDto

from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache
import os

# Add the directory containing ffmpeg to the PATH
ffmpeg_directory = "C:\\Users\\User\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-6.1-full_build\\bin"
os.environ["PATH"] = f"{ffmpeg_directory};{os.environ['PATH']}"

app = FastAPI()

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
transcriber = pipeline(model="openai/whisper-tiny")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed")
model = VisionEncoderDecoderModel.from_pretrained("src/model/captcha")

def read_image(file) -> Image.Image:
    image = Image.open(io.BytesIO(file))
    return image

def read_base64_image(base64_data) -> Image.Image:
    base64_bytes = base64.b64decode(base64_data)
    image = Image.open(io.BytesIO(base64_bytes))
    return image

# @cache()
def process_image(image):
    # prepare image
    pixel_values = processor(image, return_tensors="pt").pixel_values

    # generate (no beam search)
    generated_ids = model.generate(pixel_values)

    # decode
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return generated_text

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


@app.post("/hello")
async def hello_message(dto: ISayHelloDto):
    return {"message": f"Hello {dto.message}"}


@app.post("/extract-text")
async def extract_text(file: UploadFile = File(...)):
    try:
        image = read_image(await file.read())
        text = process_image(image)
        return JSONResponse(content={"text": text}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/extract-text/bs64")
async def extract_text_bs64(base64_data: str = Form(None)):
    try:
        image = read_base64_image(base64_data)
        text = process_image(image)
        return JSONResponse(content={"text": text}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    

@app.get("/transcribe/")
async def transcribe_audio(url: str = Query(..., title="Audio URL", description="URL of the audio file")):
    try:
        print(url)
        transcription_result = transcriber(url)
        print(transcription_result)
        return JSONResponse(content={"text": transcription_result['text']}, status_code=200)
    except Exception as e:
        print(e)
        return JSONResponse(content={"error": str(e)}, status_code=500)
    

# @app.on_event("startup")
# async def startup():
#     # redis = aioredis.from_url("redis://default:vUo6k9BQ5tac6S8LE9EG8recFi3DiwNy@redis-12209.c292.ap-southeast-1-1.ec2.cloud.redislabs.com:12209")
#     redis = aioredis.from_url("redis://localhost")
#     FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")