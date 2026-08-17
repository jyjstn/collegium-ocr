import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger('easyocr').setLevel(logging.ERROR)

import easyocr
import numpy as np
from PIL import Image

def read_stat_column(reader, cropped_img):
    # Upscale 2x in-memory for high accuracy on single digit dim numbers
    upscaled = cropped_img.resize((cropped_img.width * 2, cropped_img.height * 2), Image.Resampling.LANCZOS)
    res = reader.readtext(np.array(upscaled), detail=0, mag_ratio=2, text_threshold=0.1, low_text=0.1)
    if res:
        digits = ''.join(c for c in res[0] if c.isdigit())
        return digits if digits else '0'
    return '0'

def extract_valorant_scoreboard(image_path):
    img = Image.open(image_path)
    width, height = img.size

    # Initialize EasyOCR reader with verbose=False to suppress warning messages
    reader = easyocr.Reader(['en'], gpu=True, verbose=False)

    # --- 1. CROP & EXTRACT MATCH HEADER & RESULT (IN-MEMORY) ---
    match_header_box = (int(width * 0.35), int(height * 0.06), int(width * 0.65), int(height * 0.16))
    cropped_header = img.crop(match_header_box)
    header_results = reader.readtext(np.array(cropped_header), detail=0)

    print("=== MATCH RESULT ===")
    print("Match Header/Score:", " ".join(header_results))

    match_info_box = (int(width * 0.01), int(height * 0.09), int(width * 0.16), int(height * 0.18))
    cropped_info = img.crop(match_info_box)
    info_results = reader.readtext(np.array(cropped_info), detail=0)

    print("Match Info:", " ".join(info_results))
    print("-" * 65)

    # --- 2. CROP & EXTRACT 10 PLAYER STATS ROWS (IN-MEMORY) ---
    row_top_start = 0.295
    row_height = 0.0498

    parsed_players = []

    print("\n=== PLAYER SCOREBOARD (FULL STATS: NAME, ACS, KDA, ECON, FB, PLANTS, DEFUSES) ===")
    for i in range(10):
        top = int(height * (row_top_start + (i * row_height)))
        bottom = int(height * (row_top_start + ((i + 1) * row_height)))

        # 1. Main row data: Player info, ACS, KDA, ECON (x: 0.14 to 0.585)
        main_crop = img.crop((int(width * 0.14), top, int(width * 0.585), bottom))
        main_stats = reader.readtext(np.array(main_crop), detail=0, text_threshold=0.3, low_text=0.3)

        # 2. First Bloods Column (x: 0.585 to 0.67)
        fb_crop = img.crop((int(width * 0.585), top, int(width * 0.67), bottom))
        fb = read_stat_column(reader, fb_crop)

        # 3. Plants Column (x: 0.67 to 0.745)
        plant_crop = img.crop((int(width * 0.67), top, int(width * 0.745), bottom))
        plants = read_stat_column(reader, plant_crop)

        # 4. Defuses Column (x: 0.745 to 0.825)
        defuse_crop = img.crop((int(width * 0.745), top, int(width * 0.825), bottom))
        defuses = read_stat_column(reader, defuse_crop)

        print(f"Player {i+1:02d}: {main_stats} | FirstBlood: {fb} | Plants: {plants} | Defuses: {defuses}")

        parsed_players.append({
            "player_index": i + 1,
            "main_stats": main_stats,
            "first_bloods": fb,
            "plants": plants,
            "defuses": defuses
        })

    return parsed_players

if __name__ == "__main__":
    extract_valorant_scoreboard("valMatch4.png")
