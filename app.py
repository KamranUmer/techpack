import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image as RLImage
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
# Conditional import of canvas to prevent crashes on cloud deployment
# Force disable canvas on any cloud deployment
import os
IS_CLOUD = (
    os.getenv("STREAMLIT_SHARING_MODE") == "true" or 
    "streamlit.app" in str(os.getenv("STREAMLIT_SERVER_URL", "")) or
    "share.streamlit.io" in str(os.getenv("STREAMLIT_SERVER_URL", "")) or
    "STREAMLIT_CLOUD" in os.environ
)

if IS_CLOUD:
    CANVAS_AVAILABLE = False
    st_canvas = None
else:
    try:
        from streamlit_drawable_canvas import st_canvas
        CANVAS_AVAILABLE = True
    except Exception:
        CANVAS_AVAILABLE = False
        st_canvas = None

from ai_part import ai_generate_description, generate_pdf_report
from opencv_logic import apply_logo_realistic


def load_image(path):
    try:
        return Image.open(path).convert("RGBA")
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None


def fetch_key_value_table(file_path, start_row, end_row, column1, column2):
    df = pd.read_excel(file_path, header=None)
    subset = df.iloc[start_row - 1 : end_row, [1, 2]].dropna()
    subset.columns = [column1, column2]
    return subset.values.tolist()


st.set_page_config(page_title="Logo Placement Tool", layout="wide")

st.title("🧢 Tech Pack Logo Placement Tool")

# Debug information
st.sidebar.write(f"🔧 Debug Info:")
st.sidebar.write(f"IS_CLOUD: {IS_CLOUD}")
st.sidebar.write(f"CANVAS_AVAILABLE: {CANVAS_AVAILABLE}")
if IS_CLOUD:
    st.sidebar.success("🌐 Cloud mode - Canvas disabled for stability")


if "results" not in st.session_state:
    st.session_state.results = []


if "logo_path" not in st.session_state:
    st.session_state.logo_path = None


if "w_cm" not in st.session_state:
    st.session_state.w_cm = 5.0


if "h_cm" not in st.session_state:
    st.session_state.h_cm = 5.0

if "retry" not in st.session_state:
    st.session_state.retry = False


st.subheader("Step 0: Upload Excel & Select Data Range")

excel_file = st.file_uploader("Upload Excel File", type=["xlsx", "xls"], key="excel_upload")


if excel_file:
    df = pd.read_excel(excel_file, header=None)

    total_rows = len(df)
    st.write(f"📊 Total rows detected: {total_rows}")

    key_col_input = st.text_input("Enter column name for Keys (renamed)").strip()
    value_col_input = st.text_input("Enter column name for Values (renamed)").strip()

    start_row = st.number_input("Start Row (1-indexed)", min_value=1, max_value=total_rows, value=1, step=1)
    end_row = st.number_input("End Row (1-indexed)", min_value=1, value=total_rows, step=1)

    if st.button("📥 Fetch Data from Excel"):
        subset = df.iloc[start_row - 1 : end_row, [1, 2]].dropna()

        if key_col_input and value_col_input:
            subset.columns = [key_col_input, value_col_input]
        else:
            subset.columns = ["Key", "Value"]

        st.success(f"✅ Fetched {len(subset)} rows.")
        st.dataframe(subset)

        design_data = subset.values.tolist()
        design_table = Table(design_data, colWidths=[7 * cm, 8 * cm])
        design_table.setStyle(
            TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        st.info("Table ready for PDF export ✅")


st.subheader("Step 1: Upload Logo Image")

logo_file = st.file_uploader("Upload Logo Image", type=["png", "jpg", "jpeg"], key="logo_upload")

if logo_file:
    logo_filename = f"logo_{logo_file.name}"

    if st.session_state.logo_path is None or os.path.basename(st.session_state.logo_path) != logo_filename:
        os.makedirs("uploads", exist_ok=True)

        logo_path = os.path.join("uploads", logo_filename)

        logo = Image.open(logo_file).convert("RGBA")

        if logo_path.lower().endswith((".jpg", ".jpeg")):
            logo = logo.convert("RGB")

        logo.save(logo_path)

        st.session_state.logo_path = logo_path

        st.success("✅ Logo uploaded.")


st.subheader("Step 2: Define Approximate Logo Size (for PDF Report)")

st.info(
    "This size is for the text description in the final report. The visual size is determined by the area you draw."
)

col_w, col_h = st.columns(2)

with col_w:
    st.session_state.w_cm = st.number_input("Width (cm)", min_value=1.0, value=st.session_state.w_cm, step=0.5)

with col_h:
    st.session_state.h_cm = st.number_input("Height (cm)", min_value=1.0, value=st.session_state.h_cm, step=0.5)


st.subheader("Step 3: Upload and Place Logo on Cap")


st.info(
    "**HOW TO USE:** 1. Click 4 corners in clockwise order (Top-Left → Top-Right → Bottom-Right → Bottom-Left). **2. Double-click the 4th point to finalize the shape.** A preview will then appear."
)


cap_file = st.file_uploader(
    "Upload Cap/Base Image", type=["png", "jpg", "jpeg"], key=f"cap_{len(st.session_state.get('results', []))}"
)


if cap_file:
    try:
        st.write(f"🔍 Processing cap file: {cap_file.name}")
        cap_filename = f"cap_{cap_file.name}"
        
        # Try to work directly with uploaded file first (better for cloud)
        try:
            cap_image = Image.open(cap_file).convert("RGBA")
            st.write(f"✅ Loaded image directly from upload")
        except Exception as direct_error:
            st.write(f"⚠️ Direct load failed: {direct_error}, trying file save method...")
            
            # Fallback to saving file (for local compatibility)
            cap_path = os.path.join("uploads", cap_filename)
            if not os.path.exists(cap_path):
                os.makedirs("uploads", exist_ok=True)
                cap_image = Image.open(cap_file).convert("RGBA")

                if cap_path.lower().endswith((".jpg", ".jpeg")):
                    cap_image = cap_image.convert("RGB")

                cap_image.save(cap_path)
            else:
                cap_image = load_image(cap_path)
            
        if cap_image is None:
            st.error("Failed to load cap image. Please try uploading again.")
            st.stop()
            
    except Exception as e:
        st.error(f"Error processing cap image: {e}")
        st.stop()

    max_width = 600

    scale = max_width / cap_image.width

    display_size = (max_width, int(cap_image.height * scale))

    cap_resized = cap_image.resize(display_size)
    st.write(f"✅ Image loaded successfully. Displaying canvas...")

    # Force use fallback mode always for now to prevent crashes
    st.info("🎯 Using optimized placement mode for stability")
    canvas_result = None

    # Show a simple image preview if canvas doesn't work
    if not canvas_result or not hasattr(canvas_result, 'json_data'):
        st.image(cap_resized, caption="Cap Image Preview")
        st.info("🎯 Logo Placement Mode: Choose from the placement options below")
        
        # Enhanced fallback with multiple placement options
        st.subheader("📍 Choose Logo Placement")
        
        col1, col2, col3 = st.columns(3)
        
        placement_choice = None
        
        with col1:
            if st.button("🎯 Center", key=f"center_{len(st.session_state.get('results', []))}"):
                placement_choice = "center"
        
        with col2:
            if st.button("⬆️ Top Center", key=f"top_{len(st.session_state.get('results', []))}"):
                placement_choice = "top"
                
        with col3:
            if st.button("⬇️ Bottom Center", key=f"bottom_{len(st.session_state.get('results', []))}"):
                placement_choice = "bottom"
        
        # Logo size selector
        logo_size_percent = st.slider("Logo Size (% of image)", min_value=10, max_value=50, value=25, key=f"size_{len(st.session_state.get('results', []))}")
        
        if placement_choice and st.session_state.logo_path:
            img_height, img_width = cap_image.height, cap_image.width
            logo_size = min(img_width, img_height) * (logo_size_percent / 100)
            
            # Calculate placement positions
            if placement_choice == "center":
                center_x, center_y = img_width // 2, img_height // 2
                placement_desc = "Center"
            elif placement_choice == "top":
                center_x, center_y = img_width // 2, img_height // 4
                placement_desc = "Top Center"
            elif placement_choice == "bottom":
                center_x, center_y = img_width // 2, img_height * 3 // 4
                placement_desc = "Bottom Center"
            
            dest_points = [
                (center_x - logo_size//2, center_y - logo_size//2),  # top-left
                (center_x + logo_size//2, center_y - logo_size//2),  # top-right  
                (center_x + logo_size//2, center_y + logo_size//2),  # bottom-right
                (center_x - logo_size//2, center_y + logo_size//2),  # bottom-left
            ]
            
            st.write(f"🔄 Applying logo to {placement_desc.lower()} with {logo_size_percent}% size...")
            
            try:
                # Ensure we have a saved image file for opencv processing
                cap_path = os.path.join("uploads", cap_filename)
                if not os.path.exists(cap_path):
                    os.makedirs("uploads", exist_ok=True)
                    cap_image.save(cap_path)
                
                os.makedirs("output2", exist_ok=True)
                out_path = os.path.join("output2", os.path.splitext(cap_filename)[0] + "_with_logo.png")
                applied = apply_logo_realistic(cap_path, st.session_state.logo_path, dest_points, out_path)
                
                if applied:
                    st.image(applied, caption=f"Preview with Logo - {placement_desc}", width=400)
                    
                    # Add placement description input
                    placement_input = st.text_input(
                        "Placement description (e.g., Front Panel)",
                        placement_desc,
                        key=f"placement_desc_{len(st.session_state.get('results', []))}",
                    )
                    
                    if st.button("✅ Save This Cap", key=f"save_fallback_{len(st.session_state.get('results', []))}"):
                        ai_desc = ai_generate_description(
                            placement_input, (st.session_state.w_cm, st.session_state.h_cm), cap_file.name
                        )
                        
                        st.session_state.results.append(
                            {
                                "image": cap_path,
                                "logo": st.session_state.logo_path,
                                "size_cm": (st.session_state.w_cm, st.session_state.h_cm),
                                "placement": placement_input,
                                "description": ai_desc,
                                "orig_width": 600,
                                "orig_height": 600,
                                "output": out_path,
                            }
                        )
                        
                        st.success("✅ Cap saved successfully!")
                        st.rerun()
                        
                else:
                    st.error("Failed to apply logo. Please try again.")
                    
            except Exception as e:
                st.error(f"Error applying logo: {e}")
        
    elif canvas_result and canvas_result.json_data and canvas_result.json_data["objects"]:
        last_object = canvas_result.json_data["objects"][-1]

        if last_object["type"] == "path" and len(last_object["path"]) == 5:
            points = last_object["path"]

            dest_points = [(p[1] / scale, p[2] / scale) for p in points[:4]]

            if st.session_state.logo_path:
                try:
                    os.makedirs("output2", exist_ok=True)
                    out_path = os.path.join("output2", os.path.splitext(cap_filename)[0] + "_with_logo.png")
                    applied = apply_logo_realistic(cap_path, st.session_state.logo_path, dest_points, out_path)
                except Exception as e:
                    st.error(f"Error applying logo: {e}")
                    applied = None

                if applied:
                    st.image(applied, caption="Preview", width=400)

                    placement = st.text_input(
                        "Placement description (e.g., Front Panel)",
                        "Front Panel",
                        key=f"placement_{len(st.session_state.get('results', []))}",
                    )

                    if st.button("✅ Save This Cap", key=f"save_{len(st.session_state.get('results', []))}"):
                        ai_desc = ai_generate_description(
                            placement, (st.session_state.w_cm, st.session_state.h_cm), cap_file.name
                        )
                        #orig_width, orig_height = 600
                        st.session_state.results.append(
                            {
                                "image": cap_path,
                                "logo": st.session_state.logo_path,
                                "size_cm": (st.session_state.w_cm, st.session_state.h_cm),
                                "placement": placement,
                                "description": ai_desc,
                                "orig_width": 600,
                                "orig_height": 600,
                                "output": out_path,
                            }
                        )

                        st.success("Cap saved! Upload another image or generate the report below.")

                        st.rerun()

if st.session_state.results:
    st.markdown("---")

    st.header("Final Report")

    st.write(f"📦 You have added **{len(st.session_state.results)}** cap views so far.")

    cols = st.columns(min(len(st.session_state.results), 4))

    for i, result in enumerate(st.session_state.results):
        with cols[i % 4]:
            st.image(result["output"], caption=result["placement"], use_column_width=200)

    if st.button("📄 Generate PDF Report"):
        generate_pdf_report(st.session_state.results, "logo_techpack.pdf")

        with open("logo_techpack.pdf", "rb") as f:
            st.download_button("⬇️ Download Techpack PDF", f, file_name="logo_techpack.pdf")
