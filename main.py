# import asyncio
import logging
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
import io
from PIL import Image

# Set event loop policy for Windows
# asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


# Initialize the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ml_api")

# Initialize the FastAPI app
app = FastAPI()


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust to your needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
model = load_model('multiclassifier.keras')

# Define class names
CLASS_TYPES = ['glioma', 'meningioma', 'notumor', 'pituitary']
image_size = (150, 150)

# Preprocess the uploaded image
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(image_size)
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values to [0, 1]
    return img_array

@app.get("/")
async def health_check():
    return "The health check is successful!"

@app.get("/health")
async def health_check():
    return "The health check is successful!"

# Define the prediction endpoint
@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    print("received image")
    # Read the uploaded file
    image_bytes = await file.read()
    
    # Preprocess the image
    preprocessed_image = preprocess_image(image_bytes)
    
    # Make predictions
    predictions = model.predict(preprocessed_image)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_name = CLASS_TYPES[predicted_class_index]
    
    # Return the predicted class name
    return JSONResponse(content={"predicted_class": predicted_class_name})

# Run the app using `uvicorn` if executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
