from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import model_func
import base64
import io, json
from models.experimental import *
from utils.datasets import *

app = FastAPI()

CATEGORY_NAMES=['__background__', 'Document']

# define the Input class
class Input(BaseModel):
    base64str: str

def base64str_to_PILImage(base64str):
   base64_img_bytes = base64str.encode('utf-8')
   base64bytes = base64.b64decode(base64_img_bytes)
   bytesObj = io.BytesIO(base64bytes)
   img = Image.open(bytesObj)
   return img

@app.get("/predict")
def get_predictionbase64():
    return model_func.detect('runs/exp5_yolov5s_results/weights/best.pt')

# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1",port=8000)

# uvicorn detect_API:app --reload
