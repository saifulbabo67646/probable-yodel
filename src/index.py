from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import requests
from PIL import Image
import io
import base64
from src.dtos.ISayHelloDto import ISayHelloDto

app = FastAPI()

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed")
model = VisionEncoderDecoderModel.from_pretrained("src/model/captcha")

def read_image(file) -> Image.Image:
    image = Image.open(io.BytesIO(file))
    return image

def read_base64_image(base64_data) -> Image.Image:
    base64_bytes = base64.b64decode(base64_data)
    image = Image.open(io.BytesIO(base64_bytes))
    return image

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