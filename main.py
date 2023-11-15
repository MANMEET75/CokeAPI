####################################### IMPORT #################################
import json
import sys
import uvicorn
from io import BytesIO
from app import *
import pandas as pd
from PIL import Image
from fastapi import FastAPI, File, status
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from loguru import logger

from app import get_image_from_bytes



logger.remove()
logger.add(
    sys.stderr,
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>",
    level=10,
)
logger.add("log.log", rotation="1 MB", level="DEBUG", compression="zip")

###################### FastAPI Setup #############################

app = FastAPI(
    title="Coca Cola Inventory Detection",
    version="2023.1.31",
)

origins = [
    "http://localhost",
    "http://localhost:8008",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def save_openapi_json():
    """Save the OpenAPI documentation data of the FastAPI application to a JSON file."""
    openapi_data = app.openapi()
    with open("openapi.json", "w") as file:
        json.dump(openapi_data, file)

@app.get("/", include_in_schema=False)
async def redirect():
    return RedirectResponse("/docs")

@app.get('/healthcheck', status_code=status.HTTP_200_OK)
def perform_healthcheck():
    """Perform a healthcheck and return a response."""
    return {'healthcheck': 'Everything OK!'}

######################### Support Func #################################

def crop_image_by_predict(image: Image, predict: pd.DataFrame = pd.DataFrame(), crop_class_name: str = None) -> Image:
    """Crop an image based on the detection of a certain object.

    Args:
        image: Image to be cropped.
        predict (pd.DataFrame): Prediction results of object detection model.
        crop_class_name (str, optional): Object class to crop by. Returns the first object if not provided.

    Returns:
        Image: Cropped image or None.
    """
    crop_predicts = predict[predict['name'] == crop_class_name]

    if crop_predicts.empty:
        raise HTTPException(status_code=400, detail=f"{crop_class_name} not found in photo")

    if len(crop_predicts) > 1:
        crop_predicts = crop_predicts.sort_values(by=['confidence'], ascending=False)

    crop_bbox = crop_predicts[['xmin', 'ymin', 'xmax', 'ymax']].iloc[0].values
    return image.crop(crop_bbox)

######################### MAIN Func #################################

@app.post("/img_polluted_object_detection_to_json")
async def img_polluted_object_detection_to_json(file: bytes = File(...)):
    """Object Detection from an image and return detection results in JSON format.

    Args:
        file (bytes): The image file in bytes format.
    Returns:
        JSONResponse: JSON with object detection results and pollution status.
    """
    # Initializations
    result = {'detect_objects': None}
    input_image = get_image_from_bytes(file)
    predict = detect_pollutedItems_model(input_image)
    num_detected_objects = len(predict)
    result['detect_objects'] = num_detected_objects

    return JSONResponse(content=result)

@app.post("/img_pretrained_object_detection_to_json")
async def img_pretrained_object_detection_to_json(file: bytes = File(...)):
    """Object Detection from an image and return detection results in JSON format.

    Args:
        file (bytes): The image file in bytes format.
    Returns:
        JSONResponse: JSON with object detection results and pollution status.
    """
    # Initializations
    result = {'detect_objects': None}
    input_image = get_image_from_bytes(file)
    predict = detect_pretrained_model(input_image)
    num_detected_objects = len(predict)
    result['detect_objects'] = num_detected_objects

    return JSONResponse(content=result)



@app.post("/img_custom_object_detection_to_json")
async def img_object_detection_to_json(file: bytes = File(...)):
    """Object Detection from an image and return detection results in JSON format.

    Args:
        file (bytes): The image file in bytes format.
    Returns:
        JSONResponse: JSON with object detection results and pollution status.
    """
    # Initializations
    result = {'detect_objects': None}
    input_image = get_image_from_bytes(file)
    predict = detect_custom_model(input_image)
    num_detected_objects = len(predict)


    result['detect_objects'] = num_detected_objects

    return JSONResponse(content=result)

@app.post("/img_polluted_object_detection_to_img")
def img_polluted_object_detection_to_img(file: bytes = File(...)):
    """Object Detection from an image and display annotated images with pollution status.

    Args:
        file (bytes): The image file in bytes format.
    Returns:
        StreamingResponse: An image with bbox annotations.
    """
    input_image = get_image_from_bytes(file)
    polluted_model_results = detect_pollutedItems_model(input_image)
    num_detected_objects = len(polluted_model_results)


    final_polluted_image = add_bboxs_on_img(image=input_image, predict=polluted_model_results)

    final_polluted_image_bytes = get_bytes_from_image(final_polluted_image)

    return StreamingResponse(content=final_polluted_image_bytes, media_type="image/jpeg")

@app.post("/img_pretrained_detection_to_img")
def img_pretrained_detection_to_img(file: bytes = File(...)):
    """Object Detection from an image and display annotated images with pollution status.

    Args:
        file (bytes): The image file in bytes format.
    Returns:
        StreamingResponse: An image with bbox annotations.
    """
    input_image = get_image_from_bytes(file)
    pretrained_model_results = detect_pretrained_model(input_image)
    num_detected_objects = len(pretrained_model_results)



    # final_polluted_image = add_bboxs_on_img(image=input_image, predict=polluted_model_results)
    final_pretrained_image = add_bboxs_on_img(image=input_image, predict=pretrained_model_results)

    # final_polluted_image_bytes = get_bytes_from_image(final_polluted_image)
    final_pretrained_image_bytes = get_bytes_from_image(final_pretrained_image)

    return StreamingResponse(content=final_pretrained_image_bytes, media_type="image/jpeg")

@app.post("/img_custom_detection_to_img")
def img_custom_detection_to_img(file: bytes = File(...)):
    """Object Detection from an image and display annotated images with pollution status.

    Args:
        file (bytes): The image file in bytes format.
    Returns:
        StreamingResponse: An image with bbox annotations.
    """
    input_image = get_image_from_bytes(file)
    custom_model_results = detect_custom_model(input_image)
    num_detected_objects = len(custom_model_results)


    # final_polluted_image = add_bboxs_on_img(image=input_image, predict=polluted_model_results)
    final_custom_image = add_bboxs_on_img(image=input_image, predict=custom_model_results)

    # final_polluted_image_bytes = get_bytes_from_image(final_polluted_image)
    final_custom_image_bytes = get_bytes_from_image(final_custom_image)

    return StreamingResponse(content=final_custom_image_bytes, media_type="image/jpeg")


@app.post("/get_result")
async def get_result(file: bytes = File(...)):
    """Get the number of detected objects using all three object detection models.

    Args:
        file (bytes): The image file in bytes format.
    Returns:
        JSONResponse: JSON with the number of detected objects for each model.
    """
    input_image = get_image_from_bytes(file)
    
    # Detect objects using the polluted object detection model
    predict_polluted = detect_pollutedItems_model(input_image)
    num_detected_objects_polluted = len(predict_polluted)
    
    # Detect objects using the pretrained object detection model
    predict_pretrained = detect_pretrained_model(input_image)
    num_detected_objects_pretrained = len(predict_pretrained)
    
    # Detect objects using the custom object detection model
    predict_custom = detect_custom_model(input_image)
    num_detected_objects_custom = len(predict_custom)


    if num_detected_objects_polluted>0:
        result="Inventory is polluted"
    else:
        difference=num_detected_objects_pretrained-num_detected_objects_custom
        if difference>0:
            result="Inventory is polluted"
        else:
            result="Inventory is not polluted"
        


    result = {
        'result': result
    }
    
    return JSONResponse(content=result)

@app.post("/identify_blur_image")
async def check_blur(file: UploadFile):
    # Create a temporary file to save the uploaded image
    with open("temp_image.jpg", "wb") as temp_image:
        temp_image.write(file.file.read())

    # Check if the uploaded image is blurred
    is_blurred_image = is_blurred("temp_image.jpg")

    # Remove the temporary image file
    import os
    os.remove("temp_image.jpg")

    if is_blurred_image:
        return JSONResponse(content={"message": "Image is blurred"}, status_code=200)
    else:
        return JSONResponse(content={"message": "Image is not blurred"}, status_code=200)

   



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



