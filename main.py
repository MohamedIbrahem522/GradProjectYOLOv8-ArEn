from ultralytics import YOLO
from typing import Annotated
import io
import torch
import base64
from PIL import Image
import numpy as np
from fastapi import FastAPI, Body, HTTPException, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from googletrans import Translator

class ImageFilePath(BaseModel):
    image_data: str

app = FastAPI()

translator = Translator()

async def read_image(image_data: str) -> Image.Image:
    try:
        image_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail="Error reading image: unsupported image format or empty")

async def predict(img: Image, accept_language: str = Header(None)):
    def encode_image(predicted_img):
        image = Image.fromarray(predicted_img)
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        buffer.seek(0)
        encoded_img = base64.b64encode(buffer.read()).decode('utf-8')
        return encoded_img

    if img.mode != 'RGB':
        img = img.convert('RGB')

    image_np = np.array(img)

    model = YOLO('yolov8n.pt')

    result = model.predict(image_np, imgsz=640, conf=0.5, save=False)

    if result is None:  
        return encode_image(ploted_image), 

    classes = result[0].names
    predictions = result[0].boxes.cls

    ploted_image = result[0].plot()

    if len(predictions) == 0:
        if accept_language.lower() == "ar":
            return encode_image(ploted_image), "لم يتم العثور على معلومات" 
        else:
            return encode_image(ploted_image), "No information found"  

    class_names_list = []
    for n in predictions:
        class_name = classes[int(n)]
        if accept_language.lower() == "ar":
            arabic_translation = translator.translate(class_name, src='en', dest='ar').text
            class_names_list.append(f"{arabic_translation}")
        else:
            class_names_list.append(class_name)

    return encode_image(predicted_img=ploted_image), class_names_list

@app.post('/predict')
async def yolo_model_prediction(image_file_path: Annotated[ImageFilePath, Body(embed=False)], accept_language: str = Header(None)):
    try:
        img = await read_image(image_file_path.image_data)
        if img is None:
            raise HTTPException(status_code=400, detail="Image format not supported or empty")
        prediction_result = await predict(img, accept_language)
        content = {
            "class_names": prediction_result[1],
            "image_data": prediction_result[0]
        }
        return JSONResponse(status_code=200, content=content)

    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"cause": 400, "message": e.detail})

    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"cause": 400, "message": str(e)}
        )

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('app:app', host='0.0.0.0', port=8000, reload=True)



# uvicorn modelar-en:app --host 0.0.0.0 --port 8000
