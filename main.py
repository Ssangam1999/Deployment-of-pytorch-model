from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from PIL import Image
from io import BytesIO
import torchvision.transforms as transforms

model = torch.jit.load('model_scripted.pt')
model.eval()

app = FastAPI()

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((120, 120)),  # Resize to 120x120
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read image file
    image_data = await file.read()
    image = Image.open(BytesIO(image_data))

    # Preprocess the image
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)

    # Convert class index to label
    label = 'human' if predicted.item() == 0 else 'monkey'

    # Return the result as JSON
    return JSONResponse(content={"prediction": label})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

