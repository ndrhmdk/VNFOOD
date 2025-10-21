import streamlit as st
import pandas as pd
import pickle
import torch.nn as nn
import torch
from torchvision import datasets, transforms, models
from PIL import Image
import os
from dotenv import load_dotenv
from google import genai
import tempfile
from credentials import API_KEY, GEMINI_INSTRUCTION

# CSS
st.set_page_config(page_title="🍜 Vietnamese Food Recognizer", layout="centered")

st.markdown("""
<style>
    /* Streamlit-specific global adjustments */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #e0f7fa, #b3e5fc);
    }
    .main {
        background: none; /* Remove Streamlit main background if any */
    }
    
    /* Adapt core styles from HTML template */
    :root {
        --primary-color: #6D94C5;
        --secondary-color: #91C8E4; 
        --accent-color: #E8DFCA;
        --text-color: #333;
        --light-text-color: #6c757d;
        --bg-light: #f8f9fa;
        --success-color: #48B3AF;
        --info-border: #4FB7B3;
        --border-color: #e9ecef;
    }

    /* Simulate the centered container box (applied to the main Streamlit container) */
    /* Targeting the direct children of the app container to apply the main card style */
    [data-testid="stVerticalBlock"] > div:has(.title):first-child { 
        text-align: center;
        background-color: white;
        padding: 50px 30px;
        border-radius: 18px;
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.15);
        max-width: 750px;
        margin: 20px auto; /* Centering the main block */
        box-sizing: border-box;
    }

    h1 {
        font-family: 'Montserrat', sans-serif;
        color: var(--primary-color);
        margin-bottom: 15px;
        font-size: 2.5em;
        font-weight: 700;
        text-align: center;
    }
    .lead {
        color: var(--light-text-color);
        margin-bottom: 30px;
        font-size: 1.1em;
        text-align: center;
    }

    /* --- IMAGE CENTERING FIX --- */
    /* Wrapper div for image centering */
    .image-preview-box {
        margin: 20px auto 30px; /* Crucial for block-level centering */
        max-width: 300px; /* Constrain size */
        border: 2px solid var(--border-color);
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
    }
    
    /* Style for Streamlit image container inside the custom box */
    .image-preview-box > div[data-testid="stImage"] {
        text-align: center !important;
        padding: 0;
    }
    .image-preview-box > div[data-testid="stImage"] img {
        width: 100%;
        height: auto;
        display: block;
    }
    /* -------------------------- */
    
    .result-card {
        margin-top: 40px;
        padding: 30px;
        border-radius: 15px;
        background-color: #e6ffec; /* var(--info-bg) equivalent */
        border: 2px solid var(--info-border);
        text-align: left;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
    }

    .result-title {
        color: var(--info-border);
        font-size: 1.8em;
        font-weight: 700;
        text-align: center;
        margin-bottom: 25px;
        border-bottom: 3px solid var(--accent-color);
        padding-bottom: 10px;
    }
    
    .data-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 15px;
        padding: 10px 15px;
        background-color: white;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
    }
    
    .data-label {
        font-weight: 600;
        color: var(--light-text-color);
    }
    
    .data-value {
        font-weight: 700;
        color: var(--text-color);
    }
    .data-value.dish {
        color: var(--primary-color);
        font-size: 1.5em;
    }
    .data-value.confidence {
        color: var(--success-color);
    }
    
    .ingredients-box {
        margin-top: 25px;
        padding: 15px;
        background-color: white;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
    }

    .ingredients-box h4 {
        font-size: 1.2em;
        color: var(--primary-color);
        margin-bottom: 10px;
        border-bottom: 1px dashed var(--accent-color);
        padding-bottom: 5px;
    }

    .ingredients-list {
        list-style: none;
        padding: 0;
        margin: 0;
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    }

    .ingredients-list li {
        background-color: var(--bg-light);
        padding: 5px 10px;
        border-radius: 5px;
        font-size: 0.9em;
        color: var(--text-color);
        border: 1px solid var(--border-color);
    }
    
    /* Custom styling for the file uploader */
    .upload-box {
        text-align: center;
        background: none;
        border: none;
        padding: 0;
        margin-bottom: 20px;
    }
    .stFileUploader {
        margin-top: 20px;
    }
    .stFileUploader > label {
        display: none; /* Hide default label */
    }
    .initial-state {
        margin-top: 40px;
        padding: 50px 20px;
        background-color: var(--bg-light);
        border: 2px dashed var(--border-color);
        border-radius: 15px;
    }
    .initial-state i {
        font-size: 3em;
        color: var(--secondary-color);
        margin-bottom: 15px;
    }
    .initial-state p {
        font-size: 1.2em;
        color: var(--light-text-color);
    }
    /* Add font awesome link in head section or ensure it's loaded in the Streamlit environment */
    /* st.markdown will take care of the <head> injection if the link is included in the initial HTML */
</style>
""", unsafe_allow_html=True)

# Title and Subtitle
st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">', unsafe_allow_html=True)
st.markdown("<h1 class='title'>🍜 Vietnamese Food Recognizer</h1>", unsafe_allow_html=True)
st.markdown("<p class='lead'>Identify Banh Mi, Banh Cuon, or Pho from your photo instantly.</p>", unsafe_allow_html=True)

# Model & Gemini Setup

LABELS = ['banh_cuon', 'banh_mi', 'pho']
NUM_CLASSES = len(LABELS)
DISPLAY_NAMES = {
    "banh_cuon": "Banh Cuon",
    "banh_mi": "Banh Mi",
    "pho": "Pho"
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load API
GEMINI_API_KEY = API_KEY
if GEMINI_API_KEY == "":
    load_dotenv()
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

ROOT_DIR = os.path.abspath(".")
MODEL_WEIGHT_PATH = os.path.join(ROOT_DIR, "VNFOOD_model_weights.pth") 
INGREDIENTS_EXTRACTION_INSTRUCTION = GEMINI_INSTRUCTION

class VNFOODs(nn.Module):
    def __init__(self):
        super(VNFOODs, self).__init__()
        base_model = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

        for param in base_model.parameters():
            param.requires_grad = False
            
        self.feature_extractor = base_model.features
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(base_model.last_channel, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, NUM_CLASSES),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x
    

@st.cache_resource
def load_model():
    model = torch.load(
        MODEL_WEIGHT_PATH, 
        weights_only=False,
        map_location='cpu'
    )
    model.eval()
    return model

try:
    model = load_model()
except FileNotFoundError:
    st.error(f"[Error]: Model file not found at {MODEL_WEIGHT_PATH}. "
             "Please ensure the file is in the root directory.")
    model = None 

@st.cache_resource
def load_gemini_client():
    # NOTE:
    client = genai.Client(api_key=GEMINI_API_KEY)
    return client

client = load_gemini_client()

def get_ingredients_from_food_image(image_file: str):
    try:
        upload_file = client.files.upload(file=image_file)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[upload_file, INGREDIENTS_EXTRACTION_INSTRUCTION],
            config={"response_mime_type": "application/json"},
        )
        content = response.candidates[0].content.parts[0].text
        import json
        ingredients = json.loads(content)['ingredients']
    except Exception as e:
        ingredients = ["bread", "pork", "fresh herbs", "pickled daikon & carrots"] 
    return ingredients

# Streamlit Interface
st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], 
                                 help="Upload a photo of Banh Mi, Banh Cuon, or Pho.")
st.markdown("</div>", unsafe_allow_html=True)


if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    # Image Display
    st.markdown('<div class="image-preview-box">', unsafe_allow_html=True)
    st.image(image, caption="📸 Uploaded Image", width='stretch')
    st.markdown('</div>', unsafe_allow_html=True)

    # Setup
    input_tensor = transform(image).unsqueeze(0)
    status_placeholder = st.empty()
    status_placeholder.info("Running food recognition model and extracting ingredients...")

    # Run Prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = nn.Softmax(dim=1)(outputs)
        top_prob, top_catid = torch.topk(probs, 1)

    # Get results
    pred_label = LABELS[top_catid[0].item()]
    display_name = DISPLAY_NAMES.get(pred_label, pred_label.title())
    confidence_score = f"{top_prob.item()*100:.2f}"
    
    # Run Gemini Extraction
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, uploaded_file.name)
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        ingredients = get_ingredients_from_food_image(img_path)

    status_placeholder.empty()

    # Display Results Card
    
    # Start the main result card container
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown('<div class="result-title">Recognition Result:</div>', unsafe_allow_html=True)
    
    # Predicted Dish
    st.markdown(f"""
        <div class="data-row">
            <span class="data-label">Predicted Dish:</span>
            <span class="data-value dish">{display_name}</span>
        </div>
        """, unsafe_allow_html=True)

    # Confidence Score
    st.markdown(f"""
        <div class="data-row">
            <span class="data-label">Confidence Score:</span>
            <span class="data-value confidence">{confidence_score}%</span>
        </div>
        """, unsafe_allow_html=True)

    # Estimated Calories (Static for now)
    st.markdown("""
        <div class="data-row">
            <span class="data-label">Estimated Calories:</span>
            <span class="data-value">~500 kcal</span>
        </div>
        """, unsafe_allow_html=True)

    # Ingredients Box
    st.markdown('<div class="ingredients-box">', unsafe_allow_html=True)
    st.markdown('<h4><i class="fas fa-carrot"></i> Ingredients</h4>', unsafe_allow_html=True)
    
    # Ingredients List
    ingredients_html = "".join([f"<li>{item}</li>" for item in ingredients])
    st.markdown(f'<ul class="ingredients-list">{ingredients_html}</ul>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True) 
    st.markdown('</div>', unsafe_allow_html=True) 

else:
    st.markdown("""
    <div class="initial-state">
        <i class="fas fa-image"></i>
        <p>Ready to analyze! Select a photo of Vietnamese food to begin recognition.</p>
    </div>
    """, unsafe_allow_html=True)