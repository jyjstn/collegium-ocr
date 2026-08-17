import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger('easyocr').setLevel(logging.ERROR)

import easyocr
import numpy as np
from PIL import Image

def extract_ml_scoreboard(image_path):
    # Load original image
    img = Image.open(image_path)
    width, height = img.size

    # Initialize EasyOCR reader with verbose=False to suppress warning messages
    reader = easyocr.Reader(['en'], gpu=True, verbose=False)

    # --- 1. CROP & EXTRACT MATCH HEADER & RESULT (IN-MEMORY) ---
    # Top-center header region: "41 VICTORY 40"
    match_header_box = (int(width * 0.30), int(height * 0.01), int(width * 0.70), int(height * 0.14))
    cropped_header = img.crop(match_header_box)
    header_results = reader.readtext(np.array(cropped_header), detail=0)

    # Top-right duration region: e.g. "Duration 18:54"
    duration_box = (int(width * 0.70), int(height * 0.10), int(width * 0.88), int(height * 0.16))
    cropped_duration = img.crop(duration_box)
    duration_results = reader.readtext(np.array(cropped_duration), detail=0)

    print("=== MATCH RESULT ===")
    print("Match Header/Score:", " ".join(header_results))
    print("Match Duration:", " ".join(duration_results))
    print("-" * 65)

    # --- 2. CROP & EXTRACT PLAYER STATS TABLE (5 BLUE VS 5 RED) ---
    # Vertical bounds for the 5 player rows (22% to 86% down)
    row_top_start = 0.22
    row_height = 0.128

    teams = {
        "Blue Team (Left)": {"x_start": 0.05, "x_end": 0.495},
        "Red Team (Right)": {"x_start": 0.505, "x_end": 0.95}
    }

    parsed_players = []

    for team_name, coords in teams.items():
        print(f"\n=== {team_name.upper()} ===")
        for i in range(5):
            top = int(height * (row_top_start + (i * row_height)))
            bottom = int(height * (row_top_start + ((i + 1) * row_height)))
            left = int(width * coords["x_start"])
            right = int(width * coords["x_end"])

            # Crop player row in-memory
            row_crop = img.crop((left, top, right, bottom))
            row_text = reader.readtext(np.array(row_crop), detail=0, text_threshold=0.3, low_text=0.3)

            print(f"Player {i+1} Raw OCR:", row_text)
            parsed_players.append({
                "team": team_name,
                "row": i + 1,
                "raw_data": row_text
            })

    return {
        "match_header": header_results,
        "match_duration": duration_results,
        "players": parsed_players
    }

if __name__ == "__main__":
    # Specify your Mobile Legends screenshot path below
    extract_ml_scoreboard("ml1.jpg")
