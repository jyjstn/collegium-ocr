import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger('easyocr').setLevel(logging.ERROR)

import easyocr
import numpy as np
from PIL import Image

def clean_lol_player_data(raw_data):
    cleaned = []
    for item in raw_data:
        s = item.strip()
        # Filter out level badge artifacts like '@ 18', 'X 14', '7 18', '0 15', '16', '% 13'
        if (s.startswith(('@', 'X', '7 ', '0 ', '%', 'T ')) and len(s) <= 5) or (s.isdigit() and len(s) <= 2):
            continue
        cleaned.append(s)
    return cleaned

def extract_lol_scoreboard(image_path):
    # Load original image
    img = Image.open(image_path)
    width, height = img.size

    # Initialize EasyOCR reader with verbose=False to suppress warning messages
    reader = easyocr.Reader(['en'], gpu=True, verbose=False)

    # --- 1. MATCH OUTCOME HEADER (TOP LEFT) ---
    header_crop = img.crop((int(width * 0.02), int(height * 0.02), int(width * 0.35), int(height * 0.12)))
    match_outcome = reader.readtext(np.array(header_crop), detail=0)

    print("=== MATCH OUTCOME ===")
    print("Header Info:", " ".join(match_outcome))
    print("-" * 65)

    # --- 2. TEAM 1 SCOREBOARD (TOP HALF - EXCLUDING SOCIAL LIST) ---
    t1_header_crop = img.crop((int(width * 0.02), int(height * 0.155), int(width * 0.80), int(height * 0.205)))
    t1_header = reader.readtext(np.array(t1_header_crop), detail=0)
    print("\n=== BLUE TEAM (TEAM 1) ===")
    print("Team 1 Overview:", " ".join(t1_header))

    t1_players = []
    row_start_t1 = 0.205
    row_height = 0.058

    for i in range(5):
        top = int(height * (row_start_t1 + (i * row_height)))
        bottom = int(height * (row_start_t1 + ((i + 1) * row_height)))
        left = int(width * 0.02)
        right = int(width * 0.80)  # Excludes far-right social list

        crop = img.crop((left, top, right, bottom))
        raw_text = reader.readtext(np.array(crop), detail=0, text_threshold=0.3, low_text=0.3)
        row_text = clean_lol_player_data(raw_text)

        print(f"Team 1 Player {i+1}:", row_text)
        t1_players.append({"team": "Team 1", "slot": i + 1, "raw_data": row_text})

    # --- 3. TEAM 2 SCOREBOARD (BOTTOM HALF - EXCLUDING SOCIAL LIST) ---
    t2_header_crop = img.crop((int(width * 0.02), int(height * 0.505), int(width * 0.80), int(height * 0.555)))
    t2_header = reader.readtext(np.array(t2_header_crop), detail=0)
    print("\n=== RED TEAM (TEAM 2) ===")
    print("Team 2 Overview:", " ".join(t2_header))

    t2_players = []
    row_start_t2 = 0.555

    for i in range(5):
        top = int(height * (row_start_t2 + (i * row_height)))
        bottom = int(height * (row_start_t2 + ((i + 1) * row_height)))
        left = int(width * 0.02)
        right = int(width * 0.80)  # Excludes far-right social list

        crop = img.crop((left, top, right, bottom))
        raw_text = reader.readtext(np.array(crop), detail=0, text_threshold=0.3, low_text=0.3)
        row_text = clean_lol_player_data(raw_text)

        print(f"Team 2 Player {i+1}:", row_text)
        t2_players.append({"team": "Team 2", "slot": i + 1, "raw_data": row_text})

    return {
        "match_outcome": match_outcome,
        "team1_header": t1_header,
        "team1_players": t1_players,
        "team2_header": t2_header,
        "team2_players": t2_players
    }

if __name__ == "__main__":
    extract_lol_scoreboard("lol1.jpg")
