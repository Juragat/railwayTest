import os
import io
import torch
import torch.nn as nn
import gdown
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from torchvision import transforms
from PIL import Image, ImageDraw
import uvicorn

from fastapi import Depends, Header, HTTPException

# Define your secret API key (can also be stored in environment variables for security)
API_KEY = "supersecretapikey"

# Dependency to check API key in header
def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid API Key")

# ---------- Config ----------
MODEL_PATH = "persist/best_model.pth"
GDRIVE_ID = "1rW6UfVvMkbAXOT9SNGLE6dFWVLLpbwfP"
os.makedirs("persist", exist_ok=True)

# ---------- Download model ----------
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?id={GDRIVE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
        print("Download complete.")
    else:
        print("Model already exists. Skipping download.")

# ---------- Model architecture ----------
class KeypointModel(nn.Module):
    def __init__(self):
        super(KeypointModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2)
        )

        self.regressor = nn.Sequential(
            nn.Linear(256 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.regressor(x)
        return x

# ---------- Init ----------
print("Setting device...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Downloading model if needed...")
download_model()

print("Loading model...")
model = KeypointModel()
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device).eval()
print("Model loaded successfully.")

# ---------- Image transform ----------
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Use the same size as in training
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---------- FastAPI ----------
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app with proper documentation
app = FastAPI()

# Add CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

@app.get("/")
def read_root():
    return {"message": "CelebA API is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring systems."""
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)):
    try:
        print("Received file:", file.filename)
        
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        original_size = image.size
        
        # Save original for drawing
        original = image.copy()
        
        # Transform for model
        tensor = transform(image).unsqueeze(0).to(device)
        print("Image transformed to size:", tensor.shape)
        
        # Run inference
        with torch.no_grad():
            print("Running model...")
            output = model(tensor).view(-1, 2)
            
            # Scale keypoints back to original image size
            scale_x = original_size[0] / 128
            scale_y = original_size[1] / 128
            
            # Scale the output keypoints
            scaled_output = output.clone()
            scaled_output[:, 0] *= scale_x
            scaled_output[:, 1] *= scale_y
            
            keypoints = scaled_output.cpu().numpy()
        
        # Draw keypoints on the image
        draw = ImageDraw.Draw(original)
        keypoint_names = ['Left Eye', 'Right Eye', 'Nose', 'Left Mouth', 'Right Mouth']
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        
        for i, (x, y) in enumerate(keypoints):
            r = max(5, int(min(original_size) * 0.01))  # Dynamic radius based on image size
            draw.ellipse((x - r, y - r, x + r, y + r), fill=colors[i % len(colors)])
            
            # Label keypoints
            if i < len(keypoint_names):
                draw.text((x + r + 5, y), keypoint_names[i], fill=colors[i % len(colors)])
        
        print("Keypoints drawn. Returning image.")
        
        # Return the image with keypoints
        buffer = io.BytesIO()
        original.save(buffer, format="PNG")
        buffer.seek(0)
        
        return StreamingResponse(buffer, media_type="image/png")
    
    except Exception as e:
        print("Error:", str(e))
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

# ---------- Main ----------
if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.environ.get("PORT", 8000))  # fallback to 8000 for local dev
    uvicorn.run("app:app", host="0.0.0.0", port=port)
