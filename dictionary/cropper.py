"""
cropper.py - High-precision PDF page column extraction and segmentation
Using OpenCV vertical projection histogram for pixel-perfect column gutter detection
and unsharp masking for enhanced diacritic recognition.
"""

import os
from typing import Tuple, List, Dict, Any, Optional
import fitz
from PIL import Image, ImageFilter, ImageEnhance
import cv2
import numpy as np

def detect_column_gutter(img_bgr: np.ndarray, y_top: int, y_bot: int) -> int:
    """Find the exact x coordinate of the vertical gutter between Column 1 and Column 2."""
    h, w, _ = img_bgr.shape
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY_INV)

    x_search_start = int(0.35 * w)
    x_search_end = int(0.65 * w)

    roi = binary[y_top:y_bot, x_search_start:x_search_end]

    v_proj = np.sum(roi, axis=0).astype(np.float32)
    smoothed = cv2.GaussianBlur(v_proj.reshape(1, -1), (25, 1), 0).flatten()

    gutter_rel_x = int(np.argmin(smoothed))
    gutter_x = x_search_start + gutter_rel_x
    return gutter_x


def enhance_crop_for_ocr(img: Image.Image) -> Image.Image:
    """Enhance sharpness and contrast to ensure tiny diacritics (cedillas, tildes, dots) are crisp."""
    # Unsharp mask
    sharp = img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=150, threshold=1))
    # Slight contrast boost
    enhancer = ImageEnhance.Contrast(sharp)
    enhanced = enhancer.enhance(1.15)
    return enhanced


def get_page_layout_bounds(page: fitz.Page, dpi: int = 300) -> Dict[str, Any]:
    """
    Analyze page layout to determine exact bounding boxes for Column 1 and Column 2.
    """
    pix = page.get_pixmap(dpi=dpi)
    img_bgr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
    h, w, _ = img_bgr.shape
    scale = dpi / 72.0

    td = page.get_text("dict")

    is_section_start = False
    body_y0_list = []
    body_y1_list = []
    body_x0_list = []
    body_x1_list = []
    footer_top_pt = page.rect.height

    for b in td.get("blocks", []):
        if "lines" in b:
            for l in b["lines"]:
                for s in l["spans"]:
                    text = s["text"].strip()
                    bbox = s["bbox"]

                    if s["size"] > 25 and bbox[1] < page.rect.height * 0.40:
                        is_section_start = True

                    if "KEY" in text or "proverb" in text or "< derived from" in text:
                        footer_top_pt = min(footer_top_pt, bbox[1])
                    elif bbox[1] > page.rect.height * 0.92 and text.isdigit():
                        footer_top_pt = min(footer_top_pt, bbox[1])

                    elif 8 <= s["size"] <= 14 and len(text) >= 1:
                        if bbox[1] > page.rect.height * 0.06:
                            body_y0_list.append(bbox[1])
                            body_y1_list.append(bbox[3])
                            body_x0_list.append(bbox[0])
                            body_x1_list.append(bbox[2])

    if body_y0_list:
        min_body_y = min(body_y0_list)
        y_top_px = max(0, int((min_body_y - 4) * scale))
    else:
        y_top_px = int(0.082 * h)

    if footer_top_pt < page.rect.height:
        y_bot_px = min(h, int((footer_top_pt - 4) * scale))
    elif body_y1_list:
        max_body_y = max(body_y1_list)
        y_bot_px = min(h, int((max_body_y + 4) * scale))
    else:
        y_bot_px = int(0.885 * h)

    gutter_x = detect_column_gutter(img_bgr, y_top_px, y_bot_px)

    left_margin_pt = min(body_x0_list) if body_x0_list else 15.0
    right_margin_pt = max(body_x1_list) if body_x1_list else page.rect.width - 15.0

    x_left = max(0, int((left_margin_pt - 6) * scale))
    x_right = min(w, int((right_margin_pt + 6) * scale))

    col1_box = (x_left, y_top_px, gutter_x - 6, y_bot_px)
    col2_box = (gutter_x + 6, y_top_px, x_right, y_bot_px)

    # Extract images / illustrations info
    img_infos = page.get_images()
    illustrations = []
    for info in img_infos:
        xref = info[0]
        try:
            rects = page.get_image_rects(xref)
            for r in rects:
                if r.width > 30 and r.height > 30 and (r.width < page.rect.width * 0.8 or r.height < page.rect.height * 0.8):
                    illustrations.append({
                        "xref": xref,
                        "bbox_pt": (r.x0, r.y0, r.x1, r.y1),
                        "bbox_px": (int(r.x0 * scale), int(r.y0 * scale), int(r.x1 * scale), int(r.y1 * scale))
                    })
        except Exception:
            pass

    pil_img = Image.frombytes("RGB", [w, h], pix.samples)
    return {
        "width": w,
        "height": h,
        "scale": scale,
        "is_section_start": is_section_start,
        "gutter_x": gutter_x,
        "col1_box": col1_box,
        "col2_box": col2_box,
        "illustrations": illustrations,
        "image": pil_img
    }


def crop_page_columns(page: fitz.Page, output_dir: str, page_num: int, dpi: int = 300) -> Dict[str, Any]:
    """Crop Column 1 and Column 2 of a given PDF page and save to disk with enhancement."""
    os.makedirs(output_dir, exist_ok=True)
    layout = get_page_layout_bounds(page, dpi=dpi)
    img = layout["image"]

    col1_img = enhance_crop_for_ocr(img.crop(layout["col1_box"]))
    col2_img = enhance_crop_for_ocr(img.crop(layout["col2_box"]))

    col1_path = os.path.join(output_dir, f"page_{page_num}_col1.png")
    col2_path = os.path.join(output_dir, f"page_{page_num}_col2.png")

    col1_img.save(col1_path)
    col2_img.save(col2_path)

    return {
        "page_pdf": page_num,
        "page_book": page_num - 25,
        "col1_path": col1_path,
        "col2_path": col2_path,
        "col1_box": layout["col1_box"],
        "col2_box": layout["col2_box"],
        "gutter_x": layout["gutter_x"],
        "is_section_start": layout["is_section_start"],
        "illustrations": layout["illustrations"]
    }
