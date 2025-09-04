import os
import sys
from PIL import Image
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer,
    Image as RLImage
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from openai import OpenAI

# --- OpenAI Setup ---
def make_openai_client():
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception:
        return None

client = make_openai_client()

# --- AI helpers ---
def ai_generate_description(placement: str, size_cm: tuple, image_name: str) -> str:
    fallback = f"Logo placed on the {placement}, approximately {size_cm[0]:.1f}×{size_cm[1]:.1f} cm."
    if client is None:
        return fallback
    try:
        prompt = (
            "You are a tech pack maker. Write a precise, professional one-sentence description "
            f"for this placement.\nPlacement: {placement}\nSize: {size_cm[0]}×{size_cm[1]} cm\nFile: {image_name}"
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You write brief production notes for apparel tech packs."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=80,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return fallback

def ai_generate_summary(items: list) -> str:
    if not items:
        return "No items were processed."
    fallback = "\n".join(
        f"- {os.path.basename(i['image'])}: {i['placement']} @ {i['size_cm'][0]}×{i['size_cm'][1]} cm"
        for i in items
    )
    if client is None:
        return "Report summary:\n" + fallback
    try:
        bullet = "\n".join(
            f"- File: {os.path.basename(i['image'])}, Placement: {i['placement']}, Size: {i['size_cm'][0]}×{i['size_cm'][1]} cm"
            for i in items
        )
        prompt = (
            "You are a tech pack maker. Write a short professional summary (2–4 sentences) for this report. Which should include what we want to make based on the data and dont include name of the images\n"
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You write summaries for apparel tech packs."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=200,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return fallback

def ai_ask(question: str) -> str:
    """Ask user a question as if AI is conducting the conversation."""
    if client:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a tech pack maker. Ask questions to collect logo, cap, size, and placement."},
                {"role": "user", "content": question},
            ],
            temperature=0.5,
            max_tokens=50,
        )
        return resp.choices[0].message.content.strip()
    else:
        return question

# --- Image helpers ---
def resize_logo(logo_path, width_px, height_px):
    logo_img = Image.open(logo_path).convert("RGBA")
    return logo_img.resize((width_px, height_px))

def get_click_coordinates(image_path):
    coords = {}
    img = Image.open(image_path).convert("RGBA")

    def onclick(event):
        if event.xdata and event.ydata:
            coords["center"] = (int(event.xdata), int(event.ydata))
            print(f"✅ Clicked {coords['center']} on {os.path.basename(image_path)}")
            plt.close()

    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title(f"Click logo placement → {os.path.basename(image_path)}")
    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()
    return coords.get("center")

def apply_logo(cap_path, logo_path, width_px, height_px, out_path):
    logo_resized = resize_logo(logo_path, width_px, height_px)
    center = get_click_coordinates(cap_path)
    if not center:
        print("⚠️ No click registered, skipping.")
        return None
    cx, cy = center
    w, h = logo_resized.size
    pos = (int(cx - w/2), int(cy - h/2))
    cap_img = Image.open(cap_path).convert("RGBA")
    composite = cap_img.copy()
    composite.paste(logo_resized, pos, logo_resized)
    composite.save(out_path)
    print(f"✅ Saved {out_path}")
    return out_path

# --- PDF Report ---
def generate_pdf_report(results, pdf_path="logo_report.pdf"):
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    normal = ParagraphStyle("NormalWrap", parent=styles["Normal"], fontSize=9, wordWrap="CJK")
    story = []
    story.append(Paragraph("<b>Logo Placement Report</b>", styles["Title"]))
    story.append(Spacer(1, 6))

    ai_summary = ai_generate_summary(results)
    story.append(Paragraph("<b>Summary</b>", styles["Heading3"]))
    story.append(Paragraph(ai_summary.replace("\n", "<br/>"), normal))
    story.append(Spacer(1, 12))

    table_data = [["Logo", "Size (cm)", "Placement", "AI Description"]]
    for item in results:
        size_cm = f"{item['size_cm'][0]}×{item['size_cm'][1]} cm"
        logo_preview = RLImage(item["logo"], width=2*cm, height=2*cm)
        table_data.append([
            logo_preview,
            Paragraph(size_cm, normal),
            Paragraph(item["placement"], normal),
            Paragraph(item["description"], normal),
        ])
    table = Table(table_data, colWidths=[3*cm, 3*cm, 4*cm, 6*cm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 1), (0, -1), 'CENTER'),   # Center only logo column
        ('ALIGN', (1, 1), (-1, -1), 'LEFT'),    # Left align others
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))

    story.append(table)
    story.append(Spacer(1, 14))

    # --- Cap Images Grid (2x2 layout) ---
    MAX_WIDTH = 10 * cm
    MAX_HEIGHT = 10 * cm

    cap_rows = []
    row = []
    for i, item in enumerate(results, 1):
        pil_img = Image.open(item["output"])
        w, h = pil_img.size
        aspect = w / h

        # Convert from pixels to cm using ~37.8 px/cm (96 DPI ≈ 37.8 px/cm)
        if w > h:
            display_w = min(MAX_WIDTH, w / 37.8 * cm)
            display_h = display_w / aspect
        else:
            display_h = min(MAX_HEIGHT, h / 37.8 * cm)
            display_w = display_h * aspect

        img = RLImage(item["output"], width=display_w, height=display_h)
        row.append(img)

        if len(row) == 2:  # two images per row
            cap_rows.append(row)
            row = []

    if row:  # leftover if odd number
        cap_rows.append(row)

    # Add grid table of cap images
    cap_table = Table(cap_rows, colWidths=[8*cm, 8*cm])
    cap_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(cap_table)

    doc.build(story)
    print(f"📄 PDF report saved as {pdf_path}")

# --- Main flow ---
def main():
    results = []
    while True:
        logo_path = input(ai_ask("🖼️ Please enter the path to your logo image: ")).strip()
        if not os.path.exists(logo_path):
            print("⚠️ Logo not found.")
            continue

        cap_path = input(ai_ask("🧢 Enter path to the cap/base image: ")).strip()
        if not os.path.exists(cap_path):
            print("⚠️ Cap not found.")
            continue

        try:
            size_in = input(ai_ask("👉 Enter logo width and height (cm, separated by space): "))
            w_cm, h_cm = map(float, size_in.split())
            w, h = int(w_cm * 37.8), int(h_cm * 37.8)
        except Exception:
            print("⚠️ Invalid size. Using 3×3 cm.")
            w_cm, h_cm = 3, 3
            w, h = int(3*37.8), int(3*37.8)

        placement = input(ai_ask("📍 Where should I place the logo? (front, side, back, etc.): ")).strip()

        out_dir = "output1"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, os.path.splitext(os.path.basename(cap_path))[0] + "_with_logo.png")

        applied = apply_logo(cap_path, logo_path, w, h, out_path)
        if applied:
            ai_desc = ai_generate_description(placement, (w_cm, h_cm), os.path.basename(cap_path))
            results.append({
                "image": cap_path,
                "logo": logo_path,
                "size_cm": (w_cm, h_cm),
                "placement": placement,
                "description": ai_desc,
                "output": out_path,
            })

        cont = input(ai_ask("➕ Do you want to add another logo? (yes/no): ")).strip().lower()
        if cont != "yes":
            break

    if results:
        generate_pdf_report(results)
    else:
        print("⚠️ No logos applied.")

if __name__ == "__main__":
    main()
