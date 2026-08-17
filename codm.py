import warnings
warnings.filterwarnings('ignore')

import easyocr
import numpy as np
from PIL import Image

def extract_codm_scoreboard(image_path):
    # Load original image
    img = Image.open(image_path)
    width, height = img.size

    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'], gpu=True)

    # --- 1. CROP & EXTRACT MATCH FINAL SCORE (IN-MEMORY) ---
    match_score_box = (int(width * 0.02), int(height * 0.02), int(width * 0.45), int(height * 0.22))
    cropped_match_score = img.crop(match_score_box)
    match_score_results = reader.readtext(np.array(cropped_match_score), detail=0)

    print("=== MATCH RESULT ===")
    print("Match Header/Score:", " ".join(match_score_results))
    print("-" * 40)

    # --- 2. CROP & EXTRACT PLAYER STATS TABLE (IN-MEMORY) ---
    row_top_start = 0.30
    row_height = 0.11

    teams = {
        "Blue Team": {"x_start": 0.02, "x_end": 0.495},
        "Red Team": {"x_start": 0.505, "x_end": 0.98}
    }

    parsed_players = []

    for team_name, coords in teams.items():
        print(f"\n=== {team_name.upper()} ===")
        for i in range(5):
            top = int(height * (row_top_start + (i * row_height)))
            bottom = int(height * (row_top_start + ((i + 1) * row_height)))
            left = int(width * coords["x_start"])
            right = int(width * coords["x_end"])

            # Crop row in-memory
            row_crop = img.crop((left, top, right, bottom))
            row_text = reader.readtext(np.array(row_crop), detail=0, text_threshold=0.3, low_text=0.3)

            print(f"Player {i+1} Raw OCR:", row_text)
            parsed_players.append({
                "team": team_name,
                "row": i + 1,
                "raw_data": row_text
            })

    return parsed_players

if __name__ == "__main__":
    extract_codm_scoreboard("codm2.jpg")