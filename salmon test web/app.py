"""
Streamlit App for Fish Origin Classification
Standalone application with integrated model inference
Supports: Data preview → Batch inference → Detailed results with Grad-CAM
"""

import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import io
import base64
from PIL import Image
import tifffile as tiff
from datetime import datetime
import time
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# ML/DL imports
import torch
import torch.nn.functional as F
from torchvision import transforms
import cv2

# Project imports
from src.config import IMG_SIZE, NORMALIZE_MEAN, NORMALIZE_STD
from src.model.simple_multimodal_cnn import create_simple_multimodal_cnn
from src.utils import get_device
from src.image_preprocessing import preprocess_img_pipeline

# ==================== Model Configuration ====================
CLASS_NAMES = ['Hatchery', 'Natural']
FL_APPROX_MEAN = 60.0
FL_APPROX_STD = 10.0

@st.cache_resource
def load_model():
    """Load the trained model (cached)"""
    device = get_device()
    
    # Path to checkpoint
    checkpoint_path = os.path.join(project_root, "best.ckpt")
    if not os.path.exists(checkpoint_path):
        st.error(f"❌ Model checkpoint not found: {checkpoint_path}")
        st.stop()
    
    # Create and load model
    model = create_simple_multimodal_cnn(num_classes=2, dropout_rate=0.4, use_year=False, num_layers=6)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model, device

def normalize_fl(fl_value):
    """Normalize FL value"""
    return (fl_value - FL_APPROX_MEAN) / FL_APPROX_STD

def preprocess_image(img_array):
    """Preprocess image using the same pipeline as training"""
    # Core preprocessing pipeline
    img_proc = preprocess_img_pipeline(img_array)  # H W 3, [0,1]
    
    # Convert to PIL RGB for torchvision transforms
    pil_img = Image.fromarray((img_proc * 255).astype(np.uint8), mode="RGB")
    
    # Apply transforms
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])
    
    img_tensor = transform(pil_img).unsqueeze(0)  # Add batch dimension
    return img_tensor

def grad_cam(model, image, tabular_features, device, target_class=None):
    """Generate Grad-CAM heatmap"""
    model.eval()
    
    # Find the last convolutional layer
    target_layer = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = name
    
    if target_layer is None:
        return None, None
    
    activations = {}
    gradients = {}
    
    def forward_hook(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook
    
    def backward_hook(name):
        def hook(module, grad_input, grad_output):
            gradients[name] = grad_output[0].detach()
        return hook
    
    # Register hooks
    handles = []
    for name, module in model.named_modules():
        if name == target_layer:
            handle_forward = module.register_forward_hook(forward_hook(name))
            handle_backward = module.register_full_backward_hook(backward_hook(name))
            handles = [handle_forward, handle_backward]
            break
    
    # Forward pass
    image = image.to(device)
    image.requires_grad = True
    
    output = model(image, tabular_features)
    
    if target_class is None:
        target_class = output.argmax(dim=1).item()
    
    # Backward pass
    model.zero_grad()
    output[0, target_class].backward()
    
    # Generate CAM
    if target_layer in gradients and target_layer in activations:
        grad = gradients[target_layer]
        act = activations[target_layer]
        
        # Global average pooling of gradients
        weights = grad.mean(dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        cam = (weights * act).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        # Resize to input size
        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        
        # Remove hooks
        for handle in handles:
            handle.remove()
        
        return cam, target_class
    
    # Remove hooks if failed
    for handle in handles:
        handle.remove()
    
    return None, None

def overlay_heatmap(img_array, heatmap):
    """Overlay heatmap on image"""
    h, w = img_array.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    
    # Convert heatmap to RGB
    heatmap_colored = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay with alpha blending
    overlay = cv2.addWeighted(img_array, 0.6, heatmap_colored, 0.4, 0)
    return overlay

# ==================== Page Configuration ====================
st.set_page_config(
    page_title="Fish Origin Classification",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== Session State Initialization ====================
def init_session_state():
    """Initialize all session state variables"""
    if 'page' not in st.session_state:
        st.session_state.page = 'upload'  # upload, preview, results, detail
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None  # {samples: list, csv_df: DataFrame}
    if 'inference_results' not in st.session_state:
        st.session_state.inference_results = None  # {sample_id: result}
    if 'selected_sample_id' not in st.session_state:
        st.session_state.selected_sample_id = None
    if 'model' not in st.session_state:
        st.session_state.model = None
        st.session_state.device = None

init_session_state()

# Load model on startup
if st.session_state.model is None:
    with st.spinner("Loading model..."):
        st.session_state.model, st.session_state.device = load_model()

# ==================== Custom Styling ====================
st.markdown("""
    <style>
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 15px;
        border-radius: 5px;
        color: #155724;
    }
    .info-box {
        background: #d1ecf1;
        border: 1px solid #bee5eb;
        padding: 15px;
        border-radius: 5px;
        color: #0c5460;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== Utility Functions ====================
def load_zip_data(zip_file):
    """
    Load ZIP file containing CSV and TIFF images
    CSV format: Scale_Id, Image_Id, Sort_Id, FL, Sex
    Images matched by Image_Id (e.g., Image_Id='6_2X' matches '6_2X.tiff')
    Returns: {samples: [list of sample dicts], csv_df: DataFrame}
    """
    try:
        zip_obj = zipfile.ZipFile(zip_file)
        files = zip_obj.namelist()
        
        # Filter out macOS system files
        # __MACOSX/ folder and ._* files are generated by macOS compression
        files = [f for f in files if '__MACOSX' not in f and not f.split('/')[-1].startswith('._')]
        
        # Find CSV file
        csv_file = [f for f in files if f.endswith('.csv')]
        if not csv_file:
            st.error("❌ No CSV file found in ZIP")
            return None
        
        # Read and fix CSV with inconsistent trailing commas
        csv_bytes = zip_obj.read(csv_file[0])
        csv_text = csv_bytes.decode('utf-8')
        
        # Remove all trailing commas from each line to ensure consistency
        fixed_lines = []
        for line in csv_text.split('\n'):
            line_stripped = line.rstrip()
            if line_stripped.endswith(','):
                line_stripped = line_stripped[:-1]
            fixed_lines.append(line_stripped)
        
        csv_text_fixed = '\n'.join(fixed_lines)
        
        # Load the fixed CSV
        csv_df = pd.read_csv(
            io.StringIO(csv_text_fixed),
            skipinitialspace=True,
            skip_blank_lines=True
        )
        
        # Clean column names - remove leading/trailing spaces and quotes
        csv_df.columns = csv_df.columns.str.strip().str.strip('"').str.strip("'")
        
        # Check and fix column alignment if needed
        # FL should have numbers, not Male/Female
        if 'FL' in csv_df.columns and 'Sex' in csv_df.columns:
            fl_sample = csv_df['FL'].dropna().head(3).astype(str).tolist()
            
            if any(val.lower() in ['male', 'female'] for val in fl_sample):
                # Column shift detected - fix it silently
                temp_scale_id = csv_df['Scale_Id'].copy()
                temp_image_id = csv_df['Image_Id'].copy()
                temp_sort_id = csv_df['Sort_Id'].copy()
                temp_fl = csv_df['FL'].copy()
                
                csv_df['Image_Id'] = temp_scale_id
                csv_df['Sort_Id'] = temp_image_id
                csv_df['FL'] = temp_sort_id
                csv_df['Sex'] = temp_fl
        
        # Remove Unnamed columns
        unnamed_cols = [col for col in csv_df.columns if str(col).startswith('Unnamed')]
        if unnamed_cols:
            csv_df = csv_df.drop(columns=unnamed_cols)
        
        # Validate required columns
        required_cols = ['Image_Id', 'FL', 'Sex']
        missing_cols = [col for col in required_cols if col not in csv_df.columns]
        if missing_cols:
            st.error(f"❌ Missing required CSV columns: {', '.join(missing_cols)}")
            st.info("Expected columns: Scale_Id, Image_Id, Sort_Id, FL, Sex")
            return None
        
        # Find TIFF images (skip system files)
        tiff_files = [f for f in files if f.lower().endswith(('.tiff', '.tif')) and not f.split('/')[-1].startswith('._')]
        
        # Load images and match with CSV by Image_Id
        samples = []
        for tiff_path in tiff_files:
            try:
                img_data = tiff.imread(io.BytesIO(zip_obj.read(tiff_path)))
                
                # Extract filename without extension
                filename = tiff_path.split('/')[-1]
                image_id_no_ext = filename.rsplit('.', 1)[0]
                
                # Match with CSV by Image_Id
                # Try exact match first (case-sensitive)
                csv_row = csv_df[csv_df['Image_Id'].astype(str).str.strip() == image_id_no_ext.strip()]
                
                # If no match, try case-insensitive match
                if csv_row.empty:
                    csv_row = csv_df[csv_df['Image_Id'].astype(str).str.strip().str.lower() == image_id_no_ext.strip().lower()]
                
                if csv_row.empty:
                    csv_row_data = None
                else:
                    csv_row_data = csv_row.iloc[0]
                
                samples.append({
                    'sample_id': image_id_no_ext,
                    'filename': filename,
                    'image_path': tiff_path,
                    'image_data': img_data,
                    'csv_row': csv_row_data
                })
            except Exception as e:
                st.warning(f"⚠️ Error loading image {tiff_path}: {e}")
        
        return {
            'samples': samples,
            'csv_df': csv_df,
            'num_samples': len(samples),
            'num_images_matched': len([s for s in samples if s['csv_row'] is not None])
        }
    
    except Exception as e:
        st.error(f"❌ Error loading ZIP file: {e}")
        return None

def image_to_base64(img_array):
    """Convert image array to base64 string"""
    if img_array.ndim == 2:
        # Grayscale - convert to RGB
        img_array = np.stack([img_array] * 3, axis=-1)
    
    # Normalize to 0-255 if needed
    if img_array.dtype != np.uint8:
        if img_array.max() > 1.0:
            img_array = (img_array / img_array.max() * 255).astype(np.uint8)
        else:
            img_array = (img_array * 255).astype(np.uint8)
    
    pil_img = Image.fromarray(img_array)
    buffer = io.BytesIO()
    pil_img.save(buffer, format='TIFF')
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return img_str

def run_inference(sample):
    """Run model inference on a single sample"""
    try:
        model = st.session_state.model
        device = st.session_state.device
        
        # Get image data
        img_array = sample['image_data']
        
        # Convert to RGB if grayscale
        if img_array.ndim == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.ndim == 3 and img_array.shape[2] == 1:
            img_array = np.repeat(img_array, 3, axis=2)
        
        # Normalize to uint8 if needed
        if img_array.dtype != np.uint8:
            if img_array.max() > 1.0:
                img_array = (img_array / img_array.max() * 255).astype(np.uint8)
            else:
                img_array = (img_array * 255).astype(np.uint8)
        
        # Extract features from CSV row
        sex = 2  # Default: Unknown
        fl = 60.0  # Default: 60cm
        
        if sample['csv_row'] is not None:
            csv_row = sample['csv_row']
            
            # Extract FL value
            try:
                fl = float(csv_row['FL'])
            except (ValueError, KeyError, TypeError):
                fl = 60.0
            
            # Extract Sex value
            try:
                sex_val = csv_row['Sex']
                if pd.isna(sex_val) or sex_val is None or str(sex_val).strip() == '':
                    sex = 2
                else:
                    sex_raw = str(sex_val).strip().lower()
                    if sex_raw == 'male':
                        sex = 0
                    elif sex_raw == 'female':
                        sex = 1
                    elif sex_raw == 'nan':
                        sex = 2
                    else:
                        sex = 2
            except (ValueError, KeyError, TypeError):
                sex = 2
        
        # Preprocess image
        img_tensor = preprocess_image(img_array)
        
        # Prepare tabular features
        fl_normalized = normalize_fl(fl)
        tabular_features = {
            'sex': torch.tensor([sex], dtype=torch.long).to(device),
            'fl': torch.tensor([fl_normalized], dtype=torch.float32).to(device),
            'year': torch.tensor([0], dtype=torch.long).to(device)
        }
        
        # Make prediction
        with torch.no_grad():
            output = model(img_tensor.to(device), tabular_features)
            probabilities = F.softmax(output, dim=1)
            prediction = output.argmax(dim=1).item()
            confidence = probabilities[0, prediction].item()
        
        # Generate Grad-CAM
        heatmap, _ = grad_cam(model, img_tensor, tabular_features, device)
        
        # Prepare response
        result = {
            'prediction': prediction,
            'class_name': CLASS_NAMES[prediction],
            'confidence': float(confidence),
            'probabilities': [
                float(probabilities[0, 0].item()),
                float(probabilities[0, 1].item())
            ],
            'original_image': image_to_base64(img_array),
            'heatmap_image': image_to_base64(overlay_heatmap(img_array, heatmap)) if heatmap is not None else None
        }
        
        return result
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def base64_to_image(base64_str):
    """Convert base64 string to PIL Image"""
    try:
        img_data = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(img_data))
    except:
        return None

# ==================== Page: Upload ====================
def page_upload():
    st.title("🐟 Fish Origin Classification System")
    st.markdown("*Hatchery vs. Natural Salmon Classification using Multimodal CNN with Grad-CAM*")
    
    st.markdown("---")
    
    st.subheader("📤 Step 1: Upload Data")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        Upload a ZIP file containing:
        - **CSV file** with required columns: Image_Id, FL, Sex
        - **TIFF images** matching the Image_Id in the CSV
        
        Example structure:
        ```
        data.zip
        ├── metadata.csv (Image_Id, FL, Sex, ...)
        ├── 6_2X.tiff
        ├── 14_2X.tiff
        └── ...
        ```
        
        **Required CSV columns:**
        - Image_Id: must match image filename
        - FL: Fork Length (numeric, cm)
        - Sex: Male, Female, or empty
        """)
    
    with col2:
        st.info("💡 Tips:\n- ZIP filename doesn't matter\n- Images matched by Image_Id column\n- Supports .tif and .tiff formats\n- Sex can be Male/Female or empty")
    
    uploaded_zip = st.file_uploader(
        "Choose ZIP file",
        type=['zip'],
        key="zip_upload"
    )
    
    if uploaded_zip:
        with st.spinner("📦 Loading ZIP file..."):
            data = load_zip_data(uploaded_zip)
        
        if data:
            st.session_state.uploaded_data = data
            
            st.success(f"✓ ZIP loaded successfully!")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Images", data['num_samples'])
            with col2:
                st.metric("Images Matched", data['num_images_matched'])
            with col3:
                st.metric("CSV Records", len(data['csv_df']))
            
            # Show unmatched images if any
            unmatched_samples = [s for s in data['samples'] if s['csv_row'] is None]
            if unmatched_samples:
                st.warning(f"⚠️ {len(unmatched_samples)} images have no CSV match (will use default FL=60, Sex=Unknown)")
            
            if st.button("➡️ Next: Preview Data", key="to_preview", use_container_width=True):
                st.session_state.page = 'preview'
                st.rerun()

# ==================== Page: Preview ====================
def page_preview():
    st.title("📋 Data Preview")
    
    if not st.session_state.uploaded_data:
        st.error("❌ No data loaded. Please upload a ZIP file first.")
        if st.button("⬅️ Back"):
            st.session_state.page = 'upload'
            st.rerun()
        return
    
    data = st.session_state.uploaded_data
    
    # Show CSV preview
    st.subheader("📊 CSV Data Preview")
    
    # Show data quality info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(data['csv_df']))
    with col2:
        sex_missing = data['csv_df']['Sex'].isna().sum()
        st.metric("Sex Missing", sex_missing)
    with col3:
        fl_missing = data['csv_df']['FL'].isna().sum()
        st.metric("FL Missing", fl_missing)
    with col4:
        sex_counts = data['csv_df']['Sex'].value_counts(dropna=False)
        male_count = sex_counts.get('Male', 0) + sex_counts.get('male', 0)
        female_count = sex_counts.get('Female', 0) + sex_counts.get('female', 0)
        st.metric("M/F Ratio", f"{male_count}/{female_count}")
    
    st.dataframe(data['csv_df'], use_container_width=True)
    
    st.markdown("---")
    
    # Show image previews
    st.subheader("🖼️ Image Preview")
    
    cols_per_row = 4
    num_samples = min(len(data['samples']), 12)  # Preview max 12 images
    
    cols = st.columns(cols_per_row)
    for idx, sample in enumerate(data['samples'][:num_samples]):
        col = cols[idx % cols_per_row]
        
        with col:
            try:
                img = sample['image_data']
                if img.ndim == 2:
                    # Grayscale
                    st.image(img, use_column_width=True, caption=sample['sample_id'])
                else:
                    # Color
                    st.image(img[:, :, :3] if img.shape[2] >= 3 else img, use_column_width=True, caption=sample['sample_id'])
            except Exception as e:
                st.warning(f"Error displaying {sample['sample_id']}")
    
    if len(data['samples']) > num_samples:
        st.info(f"ℹ️ Showing {num_samples} of {len(data['samples'])} images")
    
    st.markdown("---")
    
    # Navigation
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("⬅️ Back", use_container_width=True):
            st.session_state.page = 'upload'
            st.rerun()
    
    with col3:
        if st.button("➡️ Start Inference", key="to_inference", use_container_width=True):
            st.session_state.page = 'inference'
            st.rerun()

# ==================== Page: Inference ====================
def page_inference():
    st.title("🔄 Batch Inference")
    
    if not st.session_state.uploaded_data:
        st.error("❌ No data loaded.")
        if st.button("⬅️ Back to Upload"):
            st.session_state.page = 'upload'
            st.rerun()
        return
    
    data = st.session_state.uploaded_data
    samples = data['samples']
    
    st.info(f"Processing {len(samples)} samples...")
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_placeholder = st.empty()
    
    results = {}
    failed = []
    
    for idx, sample in enumerate(samples):
        status_text.text(f"Processing: {idx + 1}/{len(samples)} - {sample['sample_id']}")
        
        # Call API
        result = run_inference(sample)
        
        result['sample_id'] = sample['sample_id']
        result['csv_data'] = sample['csv_row'].to_dict() if sample['csv_row'] is not None else {}
        
        results[sample['sample_id']] = result
        
        if 'error' in result:
            failed.append(sample['sample_id'])
        
        progress_bar.progress((idx + 1) / len(samples))
    
    status_text.empty()
    progress_bar.empty()
    
    # Save results
    st.session_state.inference_results = results
    
    # Show summary
    st.success(f"✓ Inference complete!")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Samples", len(samples))
    with col2:
        st.metric("Successful", len(samples) - len(failed))
    with col3:
        st.metric("Failed", len(failed))
    
    if failed:
        st.warning(f"⚠️ Failed samples: {', '.join(failed[:5])}")
    
    st.markdown("---")
    
    # Navigation
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("⬅️ Back", use_container_width=True):
            st.session_state.page = 'preview'
            st.rerun()
    
    with col3:
        if st.button("➡️ View Results", key="to_results", use_container_width=True):
            st.session_state.page = 'results'
            st.rerun()

# ==================== Page: Results Table ====================
def page_results():
    st.title("📊 Inference Results")
    
    if not st.session_state.inference_results:
        st.error("❌ No inference results available.")
        if st.button("⬅️ Back"):
            st.session_state.page = 'upload'
            st.rerun()
        return
    
    results = st.session_state.inference_results
    
    # Build results dataframe
    results_list = []
    for sample_id, result in results.items():
        row = {
            'Sample ID': sample_id,
            'Class': result.get('class_name', 'ERROR'),
            'Confidence': result.get('confidence', 0.0),
            'Hatchery %': result.get('probabilities', [0, 0])[0] * 100,
            'Natural %': result.get('probabilities', [0, 0])[1] * 100,
            'Status': 'Success' if 'class_name' in result else 'Failed'
        }
        results_list.append(row)
    
    df_results = pd.DataFrame(results_list)
    
    # Sorting and filtering
    st.markdown("### Filter & Sort")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sort_by = st.selectbox(
            "Sort by",
            ["Sample ID", "Class", "Confidence"],
            help="Choose how to sort the results"
        )
    
    with col2:
        status_filter = st.selectbox(
            "Filter by Status",
            ["All", "Success", "Failed"],
            help="All=show all results, Success=only successful predictions, Failed=only failed predictions"
        )
    
    with col3:
        min_confidence = st.slider(
            "Minimum Confidence",
            0.0, 1.0, 0.0, 0.05,
            help="Only show results with confidence ≥ this value. Example: set to 0.7 to show only predictions with ≥ 70% confidence"
        )
        if min_confidence > 0:
            st.caption(f"💡 Showing only results with confidence ≥ {min_confidence:.0%}")
    
    # Apply filters
    df_filtered = df_results.copy()
    
    if status_filter != "All":
        df_filtered = df_filtered[df_filtered['Status'] == status_filter]
    
    df_filtered = df_filtered[df_filtered['Confidence'] >= min_confidence]
    
    # Sort
    if sort_by == "Confidence":
        df_filtered = df_filtered.sort_values('Confidence', ascending=False)
    else:
        df_filtered = df_filtered.sort_values(sort_by)
    
    # Display table with clickable rows
    st.subheader("Results Table - Click Sample ID to view details")
    
    # Table header
    col1, col2, col3, col4, col5, col6 = st.columns([2, 1.5, 1.5, 1.5, 1.5, 1])
    with col1:
        st.markdown("**Sample ID**")
    with col2:
        st.markdown("**Class**")
    with col3:
        st.markdown("**Confidence**")
    with col4:
        st.markdown("**Hatchery %**")
    with col5:
        st.markdown("**Natural %**")
    with col6:
        st.markdown("**Status**")
    
    st.markdown("---")
    
    # Create clickable table - each row with a view button
    for idx, row in df_filtered.iterrows():
        sample_id = row['Sample ID']
        
        col1, col2, col3, col4, col5, col6 = st.columns([2, 1.5, 1.5, 1.5, 1.5, 1])
        
        with col1:
            # Make Sample ID a clickable button
            if st.button(f"📊 {sample_id}", key=f"view_{sample_id}", use_container_width=True):
                st.session_state.selected_sample_id = sample_id
                st.session_state.page = 'detail'
                st.rerun()
        
        with col2:
            st.write(row['Class'])
        
        with col3:
            st.write(f"{row['Confidence']:.1%}")
        
        with col4:
            st.write(f"{row['Hatchery %']:.1f}%")
        
        with col5:
            st.write(f"{row['Natural %']:.1f}%")
        
        with col6:
            status_emoji = "✅" if row['Status'] == 'Success' else "❌"
            st.write(status_emoji)
    
    st.markdown("---")
    
    # Export options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv_data = df_results.to_csv(index=False)
        st.download_button(
            "📥 Download CSV",
            csv_data,
            f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv"
        )
    
    with col2:
        json_data = df_results.to_json(orient='records', indent=2)
        st.download_button(
            "📥 Download JSON",
            json_data,
            f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "application/json"
        )
    
    with col3:
        if st.button("⬅️ Back to Upload", use_container_width=True):
            st.session_state.page = 'upload'
            st.rerun()

# ==================== Page: Detail ====================
def page_detail():
    st.title("🔍 Sample Details")
    
    if not st.session_state.selected_sample_id or not st.session_state.inference_results:
        st.error("❌ No sample selected.")
        if st.button("⬅️ Back to Results"):
            st.session_state.page = 'results'
            st.rerun()
        return
    
    sample_id = st.session_state.selected_sample_id
    result = st.session_state.inference_results.get(sample_id)
    
    if not result:
        st.error(f"❌ No results found for {sample_id}")
        if st.button("⬅️ Back"):
            st.session_state.page = 'results'
            st.rerun()
        return
    
    # Check for errors
    if 'error' in result:
        st.error(f"❌ Inference failed: {result['error']}")
        if st.button("⬅️ Back"):
            st.session_state.page = 'results'
            st.rerun()
        return
    
    # Header
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Sample ID", sample_id)
    with col2:
        st.metric("Classification", result.get('class_name', 'N/A'))
    with col3:
        st.metric("Confidence", f"{result.get('confidence', 0):.1%}")
    
    st.markdown("---")
    
    # Two column layout: Image + Grad-CAM on left, details on right
    col_left, col_right = st.columns(2)
    
    # ===== LEFT COLUMN: Images =====
    with col_left:
        st.subheader("🖼️ Original Image & Grad-CAM")
        
        tabs = st.tabs(["Original Image", "Grad-CAM Heatmap"])
        
        with tabs[0]:
            if 'original_image' in result:
                img = base64_to_image(result['original_image'])
                if img:
                    st.image(img, use_column_width=True, caption="Original Scale Image")
                else:
                    st.warning("Could not decode image")
            else:
                st.warning("No original image")
        
        with tabs[1]:
            if 'heatmap_image' in result and result['heatmap_image']:
                img = base64_to_image(result['heatmap_image'])
                if img:
                    st.image(img, use_column_width=True, caption="Grad-CAM: Feature Importance Map (Overlaid on Original)")
                else:
                    st.warning("Could not decode heatmap")
            else:
                st.warning("Grad-CAM not available")
    
    # ===== RIGHT COLUMN: Details =====
    with col_right:
        st.subheader("📊 Prediction Details")
        
        # Probabilities
        st.markdown("**Class Probabilities**")
        prob_hatchery = result.get('probabilities', [0, 0])[0]
        prob_natural = result.get('probabilities', [0, 0])[1]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Hatchery", f"{prob_hatchery:.1%}")
        with col2:
            st.metric("Natural", f"{prob_natural:.1%}")
        
        st.markdown("---")
        
        st.subheader("📋 Sample Metadata")
        
        if result.get('csv_data'):
            csv_data = result['csv_data']
            for key, value in csv_data.items():
                st.write(f"**{key}**: {value}")
        else:
            st.info("No metadata available")
        
        st.markdown("---")
        
        # Interpretation
        st.subheader("💡 Interpretation")
        
        confidence = result.get('confidence', 0)
        class_name = result.get('class_name', 'Unknown')
        
        if confidence > 0.9:
            st.success(f"**High Confidence**: Strong indication of {class_name} salmon")
        elif confidence > 0.7:
            st.info(f"**Moderate Confidence**: Likely {class_name} salmon")
        elif confidence > 0.6:
            st.warning(f"**Low Confidence**: Likely {class_name} salmon, but consider manual review")
        else:
            st.error(f"**Very Low Confidence**: Manual review strongly recommended")
        
        st.markdown("""
        **Grad-CAM Interpretation:**
        - Red/hot regions indicate areas the model considered important
        - Focus on salmon scale features (circuli, focus, age bands)
        - Compare heatmap across samples to understand decision patterns
        """)
    
    st.markdown("---")
    
    # Navigation
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("⬅️ Back to Results", use_container_width=True):
            st.session_state.page = 'results'
            st.rerun()
    
    # Move to next/previous sample
    samples_list = list(st.session_state.inference_results.keys())
    current_idx = samples_list.index(sample_id)
    
    with col2:
        if current_idx > 0:
            if st.button("⬅ Previous Sample", use_container_width=True):
                st.session_state.selected_sample_id = samples_list[current_idx - 1]
                st.rerun()
    
    with col3:
        if current_idx < len(samples_list) - 1:
            if st.button("Next Sample ➡", use_container_width=True):
                st.session_state.selected_sample_id = samples_list[current_idx + 1]
                st.rerun()

# ==================== Main App Router ====================
def main():
    # Sidebar
    with st.sidebar:
        st.title("🐟 Navigation")
        page = st.session_state.page
        
        st.markdown(f"""
        **Current Page**: {page.upper()}
        
        **Workflow**:
        1. Upload ZIP file
        2. Preview data
        3. Run inference
        4. View results
        5. Inspect details
        """)
        
        st.divider()
        
        # Quick navigation
        if st.button("🏠 Start Over", use_container_width=True):
            st.session_state.page = 'upload'
            st.session_state.uploaded_data = None
            st.session_state.inference_results = None
            st.session_state.selected_sample_id = None
            st.rerun()
        
        st.divider()
        
        # Stats
        if st.session_state.uploaded_data:
            st.markdown("**Loaded Data**")
            st.write(f"Samples: {st.session_state.uploaded_data['num_samples']}")
            st.write(f"Matched: {st.session_state.uploaded_data['num_images_matched']}")
        
        if st.session_state.inference_results:
            st.markdown("**Inference Results**")
            successful = sum(1 for r in st.session_state.inference_results.values() if 'class_name' in r)
            st.write(f"Successful: {successful}")
            st.write(f"Total: {len(st.session_state.inference_results)}")
    
    # Route to appropriate page
    if st.session_state.page == 'upload':
        page_upload()
    elif st.session_state.page == 'preview':
        page_preview()
    elif st.session_state.page == 'inference':
        page_inference()
    elif st.session_state.page == 'results':
        page_results()
    elif st.session_state.page == 'detail':
        page_detail()
    else:
        page_upload()

if __name__ == '__main__':
    main()