import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import re
import json
from paddleocr import PaddleOCR
from fuzzywuzzy import fuzz

# ─────────────────────────── PAGE CONFIG ───────────────────────────
st.set_page_config(
    page_title="MAG ESPORTS — FF Points Calculator",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────── THEME CSS ───────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@600;700&family=Exo+2:wght@400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Exo 2', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0a0a0f 0%, #0f1117 50%, #0a0a0f 100%);
    color: #e2e8f0;
}

/* Hero banner */
.hero-banner {
    background: linear-gradient(90deg, #ff4500 0%, #ff6b00 40%, #1a1a2e 100%);
    padding: 18px 32px;
    border-radius: 12px;
    margin-bottom: 24px;
    display: flex; align-items: center; gap: 16px;
    box-shadow: 0 4px 32px rgba(255,70,0,0.35);
}
.hero-title { font-family: 'Rajdhani', sans-serif; font-size: 2.2rem; font-weight: 700; color: #fff; margin: 0; letter-spacing: 2px; }
.hero-sub   { font-size: 0.95rem; color: #ffd580; margin: 0; }

/* Metric cards */
.metric-row { display: flex; gap: 12px; margin-bottom: 20px; flex-wrap: wrap; }
.metric-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid #ff4500;
    border-radius: 10px;
    padding: 14px 22px;
    flex: 1; min-width: 120px; text-align: center;
    box-shadow: 0 2px 12px rgba(255,70,0,0.15);
}
.metric-value { font-family: 'Rajdhani', sans-serif; font-size: 2rem; font-weight: 700; color: #ff4500; }
.metric-label { font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }

/* Section headers */
.section-header {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.3rem; font-weight: 700;
    color: #ff6b00;
    border-left: 4px solid #ff4500;
    padding-left: 12px;
    margin: 20px 0 12px;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* Leaderboard table */
.lb-table { width: 100%; border-collapse: collapse; }
.lb-table th {
    background: #ff4500; color: #fff;
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px;
    padding: 10px 14px; text-align: center;
}
.lb-table td { padding: 9px 14px; text-align: center; border-bottom: 1px solid #1e293b; font-size: 0.9rem; }
.lb-table tr:nth-child(even) td { background: #0f1926; }
.lb-table tr:nth-child(odd)  td { background: #0a1020; }
.lb-table tr:hover td { background: #1a2840; }
.rank-1 td { background: linear-gradient(90deg,#3d2b00,#1a1200) !important; color: #ffd700; font-weight: 700; }
.rank-2 td { background: linear-gradient(90deg,#2a2a2a,#111) !important; color: #c0c0c0; font-weight: 700; }
.rank-3 td { background: linear-gradient(90deg,#2a1500,#110800) !important; color: #cd7f32; font-weight: 700; }
.team-name-cell { text-align: left !important; font-weight: 600; }

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #ff4500, #ff6b00) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    padding: 10px 20px !important;
    box-shadow: 0 2px 12px rgba(255,70,0,0.3) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover { box-shadow: 0 4px 20px rgba(255,70,0,0.5) !important; transform: translateY(-1px) !important; }

/* Sidebar */
[data-testid="stSidebar"] { background: #0a0a15 !important; border-right: 1px solid #1e293b; }

/* Expander */
.streamlit-expanderHeader {
    background: #0f1926 !important;
    border: 1px solid #1e293b !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    font-family: 'Rajdhani', sans-serif !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: #0f1117; gap: 4px; }
.stTabs [data-baseweb="tab"] {
    background: #1a1a2e !important;
    color: #94a3b8 !important;
    border-radius: 8px 8px 0 0 !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
}
.stTabs [aria-selected="true"] { background: #ff4500 !important; color: #fff !important; }

/* Status pills */
.pill-success { background:#14532d; color:#4ade80; border-radius:4px; padding:2px 8px; font-size:0.75rem; }
.pill-warn    { background:#451a03; color:#fb923c; border-radius:4px; padding:2px 8px; font-size:0.75rem; }
.pill-info    { background:#1e3a5f; color:#60a5fa; border-radius:4px; padding:2px 8px; font-size:0.75rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────── OCR INIT ───────────────────────────
@st.cache_resource
def load_ocr():
    return PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

ocr_engine = load_ocr()

# ─────────────────────────── CONSTANTS ───────────────────────────
PLACEMENT_PTS = {1:12, 2:9, 3:8, 4:7, 5:6, 6:5, 7:4, 8:3, 9:2, 10:1, 11:0, 12:0}
KILL_PT = 1  # 1 point per kill

# ─────────────────────────── IMAGE PREPROCESSING ───────────────────────────
def preprocess_image(img_array, mode="match"):
    """Strong OpenCV pipeline before OCR."""
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_array.copy()

    # Upscale 2x
    h, w = gray.shape
    upscaled = cv2.resize(gray, (w*2, h*2), interpolation=cv2.INTER_CUBIC)

    # Gaussian blur to remove noise
    blurred = cv2.GaussianBlur(upscaled, (3, 3), 0)

    # Binarization: try Otsu first
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # If too dark (text is white on dark bg), invert
    white_px = np.sum(otsu == 255)
    black_px = np.sum(otsu == 0)
    if black_px > white_px:
        otsu = cv2.bitwise_not(otsu)

    # Adaptive threshold for uneven regions
    adaptive = cv2.adaptiveThreshold(blurred, 255,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 15, 8)
    if np.sum(adaptive == 0) > np.sum(adaptive == 255):
        adaptive = cv2.bitwise_not(adaptive)

    # Pick better result (more white background = better for OCR)
    if np.sum(otsu == 255) >= np.sum(adaptive == 255):
        result = otsu
    else:
        result = adaptive

    # Morphological close to connect broken strokes
    kernel = np.ones((2, 2), np.uint8)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)

    return result

def crop_roi(img_array, mode="match"):
    """Crop irrelevant edges for match result / lobby screenshots."""
    h, w = img_array.shape[:2]
    if mode == "match":
        # Remove top ~12%, bottom ~10%, right ~8% (character panel)
        top = int(h * 0.12)
        bottom = int(h * 0.90)
        right = int(w * 0.92)
        return img_array[top:bottom, 0:right]
    elif mode == "lobby":
        # Remove top ~8%, bottom ~8%
        top = int(h * 0.08)
        bottom = int(h * 0.92)
        return img_array[top:bottom, :]
    return img_array

# ─────────────────────────── OCR HELPERS ───────────────────────────
def run_ocr(img_array):
    """Run PaddleOCR on preprocessed image, return list of (text, confidence, bbox)."""
    # Convert grayscale→RGB for PaddleOCR
    if len(img_array.shape) == 2:
        rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    
    result = ocr_engine.ocr(rgb, cls=True)
    lines = []
    if result and result[0]:
        for item in result[0]:
            if item and len(item) >= 2:
                bbox, (text, conf) = item[0], item[1]
                lines.append({
                    "text": text.strip(),
                    "conf": conf,
                    "bbox": bbox,
                    "y": bbox[0][1]   # top-left y for vertical sort
                })
    # Sort top-to-bottom
    lines.sort(key=lambda x: x["y"])
    return lines

# ─────────────────────────── LOBBY SCREENSHOT PARSER ───────────────────────────
def parse_lobby_screenshot(img_pil):
    """
    Extract (slot_number, player_name) pairs from a Free Fire lobby screenshot.
    In lobby: each row shows slot number + player IGN.
    Returns dict: {slot_int: player_name_str}
    """
    arr = np.array(img_pil.convert("RGB"))
    arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    
    roi = crop_roi(arr_bgr, mode="lobby")
    processed = preprocess_image(roi, mode="lobby")
    lines = run_ocr(processed)
    
    slot_map = {}  # {slot: player_name}
    
    # Pattern: number (1-12) followed by player name on same row
    # or alternating lines: "1" then "PlayerName"
    texts = [l["text"] for l in lines]
    ys    = [l["y"]    for l in lines]
    
    i = 0
    while i < len(texts):
        t = texts[i].strip()
        # Check if this line is a slot number
        num_match = re.match(r'^(\d{1,2})$', t)
        if num_match:
            slot = int(num_match.group(1))
            if 1 <= slot <= 12:
                # Look for player name on same Y level or next line
                # Check if next token on same line or next line is a name
                name = None
                if i+1 < len(texts):
                    next_t = texts[i+1].strip()
                    # If next line y is close (same row) or slightly below
                    y_diff = abs(ys[i+1] - ys[i]) if len(ys) > i+1 else 999
                    if y_diff < 30 and not re.match(r'^\d{1,2}$', next_t) and len(next_t) >= 2:
                        name = next_t
                        i += 1  # consume name too
                if name:
                    slot_map[slot] = name
        else:
            # Try "slot name" in one line like "1 PlayerIGN"
            combined = re.match(r'^(\d{1,2})\s+([A-Za-z0-9_\-\.]{2,20})$', t)
            if combined:
                slot = int(combined.group(1))
                name = combined.group(2)
                if 1 <= slot <= 12:
                    slot_map[slot] = name
        i += 1
    
    return slot_map, lines  # also return raw lines for display

# ─────────────────────────── MATCH RESULT PARSER ───────────────────────────
def parse_match_result(img_pil):
    """
    Extract (rank, kills) for 12 teams from a Free Fire match result screenshot.
    
    Free Fire match result layout:
    - Left panel: Ranks 1-6 (or 1-5)
    - Right panel: Ranks 6-12 (or 6-12)
    - Each rank block: 2-4 player rows
    - Each player row: PlayerName  + N Elimination(s)
    
    Strategy:
    1. Split image into left & right halves
    2. Extract all OCR text with Y coordinates
    3. Group player rows into rank blocks using "rank number" markers
    4. Sum kills per block = team kills
    
    Returns: list of {rank, kills, raw_text}
    """
    arr = np.array(img_pil.convert("RGB"))
    arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    
    roi = crop_roi(arr_bgr, mode="match")
    processed = preprocess_image(roi, mode="match")
    lines = run_ocr(processed)
    
    raw_texts = [l["text"] for l in lines]
    
    # Extract rank and kill info
    rank_blocks = {}  # {rank: kill_count}
    
    # Method 1: Find rank markers (#1, #2, 1st, 2nd, or plain numbers in rank position)
    # Method 2: Find "N Elimination" patterns and group by proximity
    
    # Parse kills — look for patterns like "5 Elim", "3 Kills", "2 ELIM", or just number before keyword
    kill_pattern = re.compile(r'(\d{1,2})\s*(?:elim|kill|frag|排除|击倒)', re.IGNORECASE)
    
    # Also look for rank patterns
    rank_pattern = re.compile(r'^#?(\d{1,2})$|^(\d{1,2})(?:st|nd|rd|th)?$', re.IGNORECASE)
    
    # Group OCR lines by Y proximity into "blocks"
    # Each block = one team rank section
    blocks = []
    current_block = []
    last_y = None
    BLOCK_GAP = 50  # pixels gap between rank blocks (in 2x upscaled image)
    
    for l in lines:
        y = l["y"]
        if last_y is None or (y - last_y) < BLOCK_GAP:
            current_block.append(l)
        else:
            if current_block:
                blocks.append(current_block)
            current_block = [l]
        last_y = y
    if current_block:
        blocks.append(current_block)
    
    # For each block, determine rank and total kills
    extracted = []
    found_rank = None
    
    # Simpler approach: scan all lines sequentially
    # When we see a rank number, start a new entry
    # Accumulate kills until next rank number
    
    current_rank = None
    current_kills = 0
    
    for l in lines:
        text = l["text"].strip()
        
        # Check if it's a rank marker
        rm = rank_pattern.match(text)
        if rm:
            r = int(rm.group(1) or rm.group(2))
            if 1 <= r <= 12:
                if current_rank is not None:
                    extracted.append({"rank": current_rank, "kills": current_kills})
                current_rank = r
                current_kills = 0
                continue
        
        # Check for kill count
        km = kill_pattern.search(text)
        if km:
            k = int(km.group(1))
            if k <= 20:  # sanity cap per player
                current_kills += k
    
    # Save last
    if current_rank is not None:
        extracted.append({"rank": current_rank, "kills": current_kills})
    
    # If we didn't find rank markers, fall back to positional approach
    if len(extracted) < 6:
        extracted = positional_fallback(lines)
    
    # Deduplicate and sort
    seen = set()
    final = []
    for e in sorted(extracted, key=lambda x: x["rank"]):
        if e["rank"] not in seen:
            seen.add(e["rank"])
            final.append(e)
    
    # Fill missing ranks with 0 kills
    existing_ranks = {e["rank"] for e in final}
    for r in range(1, 13):
        if r not in existing_ranks:
            final.append({"rank": r, "kills": 0})
    
    final = sorted(final, key=lambda x: x["rank"])
    return final, lines

def positional_fallback(lines):
    """
    If rank markers not found, assume Free Fire layout:
    Left half = ranks 1-6, Right half = ranks 7-12
    Divide Y space equally.
    """
    if not lines:
        return [{"rank": r, "kills": 0} for r in range(1, 13)]
    
    kill_pattern = re.compile(r'(\d{1,2})\s*(?:elim|kill|frag)', re.IGNORECASE)
    
    max_y = max(l["y"] for l in lines)
    min_y = min(l["y"] for l in lines)
    span = max(max_y - min_y, 1)
    
    # 6 equal bands on each "half" — but we only have merged image
    # Divide into 12 equal Y bands
    band_h = span / 12
    
    rank_kills = {r: 0 for r in range(1, 13)}
    
    for l in lines:
        km = kill_pattern.search(l["text"])
        if km:
            k = int(km.group(1))
            if 1 <= k <= 20:
                y_pos = l["y"] - min_y
                rank = min(12, max(1, int(y_pos / band_h) + 1))
                rank_kills[rank] += k
    
    return [{"rank": r, "kills": rank_kills[r]} for r in range(1, 13)]

# ─────────────────────────── PLAYER→TEAM MATCHING ───────────────────────────
def match_players_to_teams(lobby_slot_map, team_slot_config):
    """
    lobby_slot_map: {slot: player_ign}  (from lobby screenshot OCR)
    team_slot_config: {slot: team_name}  (from user sidebar input)
    Returns: {slot: {team_name, player_ign}}
    """
    result = {}
    for slot in range(1, 13):
        team = team_slot_config.get(slot, f"Team {slot}")
        player = lobby_slot_map.get(slot, "Unknown")
        result[slot] = {"team": team, "player": player}
    return result

# ─────────────────────────── POINTS CALCULATION ───────────────────────────
def calculate_points(rank, kills):
    placement = PLACEMENT_PTS.get(rank, 0)
    kill_pts = kills * KILL_PT
    return placement + kill_pts

def build_aggregate(matches, team_slot_config):
    """
    matches: list of {match_name, results: [{slot, rank, kills}]}
    team_slot_config: {slot: team_name}
    Returns sorted aggregate list.
    """
    agg = {}
    for slot in range(1, 13):
        team = team_slot_config.get(slot, f"Team {slot}")
        agg[slot] = {
            "team": team,
            "total_pts": 0,
            "total_kills": 0,
            "total_placement_pts": 0,
            "matches": []
        }
    
    for m in matches:
        match_name = m["match_name"]
        for r in m["results"]:
            slot = r["slot"]
            rank = r["rank"]
            kills = r["kills"]
            pts = calculate_points(rank, kills)
            place_pts = PLACEMENT_PTS.get(rank, 0)
            
            agg[slot]["total_pts"] += pts
            agg[slot]["total_kills"] += kills
            agg[slot]["total_placement_pts"] += place_pts
            agg[slot]["matches"].append({
                "match": match_name,
                "rank": rank,
                "kills": kills,
                "pts": pts
            })
    
    rows = list(agg.values())
    # Sort: total_pts desc, tiebreak by total_kills
    rows.sort(key=lambda x: (-x["total_pts"], -x["total_kills"]))
    for i, row in enumerate(rows):
        row["position"] = i + 1
    return rows

# ─────────────────────────── IMAGE GENERATION ───────────────────────────
def generate_leaderboard_image(agg_rows, matches, tournament_name="MAG ESPORTS", round_label="Overall Standings"):
    """Generate a professional leaderboard image."""
    num_matches = len(matches)
    cols = 5 + num_matches  # pos, team, total_kills, place_pts, total | per match
    
    row_h = 46
    header_h = 90
    footer_h = 50
    col_w = 160
    pos_w = 60
    team_w = 220
    pts_w = 100
    
    total_w = pos_w + team_w + (num_matches * pts_w) + pts_w + pts_w + pts_w
    total_h = header_h + len(agg_rows) * row_h + footer_h
    
    img = Image.new("RGB", (total_w, total_h), (10, 10, 20))
    draw = ImageDraw.Draw(img)
    
    # Background grid pattern
    for x in range(0, total_w, 40):
        draw.line([(x, 0), (x, total_h)], fill=(20, 20, 35), width=1)
    for y in range(0, total_h, 40):
        draw.line([(0, y), (total_w, y)], fill=(20, 20, 35), width=1)
    
    # Header bar
    draw.rectangle([(0, 0), (total_w, header_h)], fill=(180, 40, 0))
    draw.rectangle([(0, header_h-4), (total_w, header_h)], fill=(255, 100, 0))
    
    # Fonts
    try:
        font_big   = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        font_med   = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        font_tiny  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        font_big = font_med = font_small = font_tiny = ImageFont.load_default()
    
    # Title
    draw.text((20, 15), tournament_name.upper(), fill=(255, 255, 255), font=font_big)
    draw.text((20, 52), round_label, fill=(255, 200, 100), font=font_small)
    draw.text((total_w - 200, 15), "FREE FIRE", fill=(255, 200, 100), font=font_med)
    draw.text((total_w - 200, 45), "POINTS CALCULATOR", fill=(200, 200, 200), font=font_tiny)
    
    # Column headers
    y_hdr = header_h + 4
    draw.rectangle([(0, header_h), (total_w, header_h + 34)], fill=(30, 30, 50))
    
    x = 0
    hdrs = ["#", "TEAM"] + [m["match_name"][:4].upper() for m in matches] + ["KILLS", "PLACE", "TOTAL"]
    widths = [pos_w, team_w] + [pts_w]*num_matches + [pts_w, pts_w, pts_w]
    
    for hdr, w in zip(hdrs, widths):
        draw.text((x + w//2, y_hdr), hdr, fill=(255, 150, 50), font=font_tiny, anchor="mt")
        x += w
    
    # Data rows
    rank_colors = {1: (255, 215, 0), 2: (192, 192, 192), 3: (205, 127, 50)}
    bg_colors = [(15, 20, 35), (12, 16, 28)]
    
    for i, row in enumerate(agg_rows):
        y_row = header_h + 34 + i * row_h
        bg = (35, 20, 0) if row["position"] == 1 else \
             (25, 25, 25) if row["position"] == 2 else \
             (30, 15, 0) if row["position"] == 3 else bg_colors[i % 2]
        draw.rectangle([(0, y_row), (total_w, y_row + row_h - 1)], fill=bg)
        
        txt_color = rank_colors.get(row["position"], (220, 220, 220))
        
        x = 0
        # Position
        medal = {1:"🥇", 2:"🥈", 3:"🥉"}.get(row["position"], str(row["position"]))
        draw.text((x + pos_w//2, y_row + row_h//2), str(row["position"]),
                  fill=txt_color, font=font_med, anchor="mm")
        x += pos_w
        
        # Team name
        draw.text((x + 10, y_row + row_h//2), row["team"][:20],
                  fill=txt_color, font=font_med, anchor="lm")
        x += team_w
        
        # Per match points
        match_pts_by_name = {m_data["match"]: m_data["pts"] for m_data in row["matches"]}
        for m in matches:
            pts = match_pts_by_name.get(m["match_name"], 0)
            draw.text((x + pts_w//2, y_row + row_h//2), str(pts),
                      fill=(180, 220, 255), font=font_small, anchor="mm")
            x += pts_w
        
        # Total kills
        draw.text((x + pts_w//2, y_row + row_h//2), str(row["total_kills"]),
                  fill=(100, 220, 100), font=font_small, anchor="mm")
        x += pts_w
        
        # Placement pts
        draw.text((x + pts_w//2, y_row + row_h//2), str(row["total_placement_pts"]),
                  fill=(200, 180, 100), font=font_small, anchor="mm")
        x += pts_w
        
        # Total pts (big)
        draw.text((x + pts_w//2, y_row + row_h//2), str(row["total_pts"]),
                  fill=txt_color, font=font_med, anchor="mm")
    
    # Footer
    y_foot = header_h + 34 + len(agg_rows)*row_h
    draw.rectangle([(0, y_foot), (total_w, total_h)], fill=(180, 40, 0))
    draw.text((total_w//2, y_foot + 14), "MAG ESPORTS  •  FREE FIRE TOURNAMENT  •  POWERED BY POINTCALC",
              fill=(255, 255, 255), font=font_tiny, anchor="mt")
    
    return img

# ─────────────────────────── SESSION STATE ───────────────────────────
def init_state():
    defaults = {
        "tournament_name": "MAG ESPORTS FF Tournament",
        "round_label": "Day 1 — Overall",
        "teams": {i: f"Team {i}" for i in range(1, 13)},
        "matches": [],            # [{match_name, results:[{slot,rank,kills}]}]
        "lobby_maps": {},         # {match_name: {slot: player_ign}}
        "active_match_data": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ─────────────────────────── SIDEBAR ───────────────────────────
with st.sidebar:
    st.markdown('<p style="font-family:Rajdhani;font-size:1.4rem;font-weight:700;color:#ff4500;text-transform:uppercase;letter-spacing:2px;">⚙️ Settings</p>', unsafe_allow_html=True)
    
    st.session_state.tournament_name = st.text_input("🏆 Tournament Name", st.session_state.tournament_name)
    st.session_state.round_label = st.text_input("📅 Round / Day Label", st.session_state.round_label)
    
    st.markdown('<p class="section-header">👥 Team Slots</p>', unsafe_allow_html=True)
    st.caption("Slot = room position (1–12). Enter team names once, use for all matches.")
    
    for i in range(1, 13):
        st.session_state.teams[i] = st.text_input(
            f"Slot {i}", st.session_state.teams[i], key=f"team_{i}", label_visibility="visible"
        )
    
    st.markdown("---")
    st.markdown('<p style="color:#ff6b00;font-size:0.85rem;">📊 Placement Points Table</p>', unsafe_allow_html=True)
    for rank, pts in PLACEMENT_PTS.items():
        if pts > 0:
            st.markdown(f'<span style="color:#94a3b8;font-size:0.8rem;">Rank {rank} → <b style="color:#ff6b00">{pts} pts</b></span>', unsafe_allow_html=True)
    st.markdown('<span style="color:#94a3b8;font-size:0.8rem;">Kill → <b style="color:#4ade80">1 pt each</b></span>', unsafe_allow_html=True)
    
    st.markdown("---")
    if st.button("🗑️ Reset All Data", use_container_width=True):
        st.session_state.matches = []
        st.session_state.lobby_maps = {}
        st.rerun()

# ─────────────────────────── MAIN UI ───────────────────────────
st.markdown("""
<div class="hero-banner">
  <div>
    <p class="hero-title">🔥 FF POINTS CALCULATOR</p>
    <p class="hero-sub">MAG ESPORTS — Tournament Management System</p>
  </div>
</div>
""", unsafe_allow_html=True)

# Aggregate calc
agg_rows = build_aggregate(st.session_state.matches, st.session_state.teams)

# Metrics
leader = agg_rows[0]["team"] if agg_rows else "—"
leader_pts = agg_rows[0]["total_pts"] if agg_rows else 0
num_matches = len(st.session_state.matches)

st.markdown(f"""
<div class="metric-row">
  <div class="metric-card"><div class="metric-value">{num_matches}</div><div class="metric-label">Matches</div></div>
  <div class="metric-card"><div class="metric-value">12</div><div class="metric-label">Teams</div></div>
  <div class="metric-card"><div class="metric-value" style="font-size:1.2rem">{leader}</div><div class="metric-label">Current Leader</div></div>
  <div class="metric-card"><div class="metric-value">{leader_pts}</div><div class="metric-label">Leader Points</div></div>
</div>
""", unsafe_allow_html=True)

# ─────── TABS ───────
tab_add, tab_results, tab_export = st.tabs(["➕  ADD MATCH", "📊  LEADERBOARD", "🖼️  EXPORT IMAGE"])

# ═══════════════════════════ TAB 1: ADD MATCH ═══════════════════════════
with tab_add:
    st.markdown('<p class="section-header">Add New Match</p>', unsafe_allow_html=True)
    
    match_name = st.text_input("Match Name", f"Match {num_matches+1}", key="new_match_name")
    
    col_lobby, col_match = st.columns(2)
    
    with col_lobby:
        st.markdown('<p class="section-header">📸 Lobby Screenshot (Optional)</p>', unsafe_allow_html=True)
        st.caption("Upload lobby SS to automatically extract player IGNs for each slot.")
        lobby_file = st.file_uploader("Lobby Screenshot", type=["jpg","jpeg","png","webp"], key="lobby_upload")
        
        lobby_map = {}  # {slot: player_ign}
        
        if lobby_file:
            lobby_img = Image.open(lobby_file)
            st.image(lobby_img, caption="Uploaded Lobby SS", use_container_width=True)
            
            if st.button("🔍 Extract Players from Lobby SS", use_container_width=True):
                with st.spinner("OCR running on lobby screenshot..."):
                    arr_bgr = cv2.cvtColor(np.array(lobby_img.convert("RGB")), cv2.COLOR_RGB2BGR)
                    roi = crop_roi(arr_bgr, "lobby")
                    processed = preprocess_image(roi, "lobby")
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.image(roi[:,:,::-1], caption="Cropped ROI", use_container_width=True)
                    with col_b:
                        st.image(processed, caption="Pre-processed (OCR input)", use_container_width=True)
                    
                    lobby_map_raw, raw_lines = parse_lobby_screenshot(lobby_img)
                    st.session_state[f"lobby_map_{match_name}"] = lobby_map_raw
                    
                    if lobby_map_raw:
                        st.success(f"✅ {len(lobby_map_raw)} players extracted!")
                        st.json({f"Slot {k}": v for k, v in sorted(lobby_map_raw.items())})
                    else:
                        st.warning("⚠️ No slot-player pairs found. Enter manually below.")
                    
                    with st.expander("📝 Raw OCR Lines"):
                        for l in raw_lines:
                            st.text(f"[conf:{l['conf']:.2f}] {l['text']}")
    
    with col_match:
        st.markdown('<p class="section-header">📸 Match Result Screenshot</p>', unsafe_allow_html=True)
        st.caption("Upload match result SS to auto-extract rank & kills.")
        match_file = st.file_uploader("Match Result Screenshot", type=["jpg","jpeg","png","webp"], key="match_upload")
        
        if match_file:
            match_img = Image.open(match_file)
            st.image(match_img, caption="Uploaded Match Result", use_container_width=True)
            
            if st.button("🔍 Extract Ranks & Kills from Match SS", use_container_width=True):
                with st.spinner("OCR running on match result..."):
                    arr_bgr = cv2.cvtColor(np.array(match_img.convert("RGB")), cv2.COLOR_RGB2BGR)
                    roi = crop_roi(arr_bgr, "match")
                    processed = preprocess_image(roi, "match")
                    
                    col_a2, col_b2 = st.columns(2)
                    with col_a2:
                        st.image(roi[:,:,::-1], caption="Cropped ROI", use_container_width=True)
                    with col_b2:
                        st.image(processed, caption="Pre-processed (OCR input)", use_container_width=True)
                    
                    extracted, raw_lines2 = parse_match_result(match_img)
                    st.session_state[f"match_extract_{match_name}"] = extracted
                    st.success(f"✅ {len(extracted)} rank entries found!")
                    
                    with st.expander("📝 Raw OCR Lines"):
                        for l in raw_lines2:
                            st.text(f"[conf:{l['conf']:.2f}] {l['text']}")
    
    # ─── MANUAL ENTRY TABLE ───
    st.markdown('<p class="section-header">📋 Enter / Verify Match Data</p>', unsafe_allow_html=True)
    st.caption("OCR auto-fills rank & kills. Verify and correct. Slot = team room position.")
    
    # Build default data
    ocr_data = st.session_state.get(f"match_extract_{match_name}", None)
    lobby_data = st.session_state.get(f"lobby_map_{match_name}", {})
    
    # Prepare initial values
    slot_ranks  = {}
    slot_kills  = {}
    if ocr_data:
        for i, e in enumerate(ocr_data[:12]):
            slot = i + 1  # assume rank order = slot order (can be corrected)
            slot_ranks[slot] = e["rank"]
            slot_kills[slot] = e["kills"]
    
    entry_rows = []
    for slot in range(1, 13):
        team_name = st.session_state.teams.get(slot, f"Team {slot}")
        player_ign = lobby_data.get(slot, "—")
        default_rank  = slot_ranks.get(slot, slot)
        default_kills = slot_kills.get(slot, 0)
        entry_rows.append({
            "slot": slot,
            "team": team_name,
            "player_ign": player_ign,
            "rank": default_rank,
            "kills": default_kills,
        })
    
    # Display editable columns
    header_cols = st.columns([0.5, 2, 2, 1, 1, 1.5])
    header_cols[0].markdown("**Slot**")
    header_cols[1].markdown("**Team**")
    header_cols[2].markdown("**Player IGN**")
    header_cols[3].markdown("**Rank**")
    header_cols[4].markdown("**Kills**")
    header_cols[5].markdown("**Points**")
    
    final_results = []
    for row in entry_rows:
        c0, c1, c2, c3, c4, c5 = st.columns([0.5, 2, 2, 1, 1, 1.5])
        c0.markdown(f"**{row['slot']}**")
        c1.markdown(f'<span style="color:#e2e8f0">{row["team"]}</span>', unsafe_allow_html=True)
        c2.markdown(f'<span style="color:#60a5fa;font-size:0.85rem">{row["player_ign"]}</span>', unsafe_allow_html=True)
        
        rank  = c3.number_input("", min_value=1, max_value=12, value=row["rank"],  key=f"rank_{match_name}_{row['slot']}", label_visibility="collapsed")
        kills = c4.number_input("", min_value=0, max_value=99, value=row["kills"], key=f"kills_{match_name}_{row['slot']}", label_visibility="collapsed")
        pts   = calculate_points(rank, kills)
        c5.markdown(f'<span style="color:#ff6b00;font-weight:700;font-size:1.1rem">{pts} pts</span>', unsafe_allow_html=True)
        
        final_results.append({"slot": row["slot"], "rank": rank, "kills": kills})
    
    st.markdown("---")
    if st.button(f"💾 SAVE MATCH — {match_name}", use_container_width=True):
        # Remove existing match with same name
        st.session_state.matches = [m for m in st.session_state.matches if m["match_name"] != match_name]
        st.session_state.matches.append({
            "match_name": match_name,
            "results": final_results
        })
        st.success(f"✅ {match_name} saved! {len(st.session_state.matches)} match(es) total.")
        st.balloons()
        st.rerun()

# ═══════════════════════════ TAB 2: LEADERBOARD ═══════════════════════════
with tab_results:
    st.markdown('<p class="section-header">📊 Aggregate Leaderboard</p>', unsafe_allow_html=True)
    
    if not st.session_state.matches:
        st.info("No matches saved yet. Go to ➕ ADD MATCH tab.")
    else:
        # Build match columns for header
        match_cols = [m["match_name"] for m in st.session_state.matches]
        
        # HTML table
        match_headers = "".join(f"<th>{mc}</th>" for mc in match_cols)
        table_html = f"""
        <table class="lb-table">
        <thead><tr>
          <th>#</th><th>TEAM</th><th>PLAYER IGN</th>
          {match_headers}
          <th>KILLS</th><th>PLACE PTS</th><th>TOTAL</th>
        </tr></thead><tbody>
        """
        
        # Get player maps
        all_lobby_maps = {}
        for slot in range(1, 13):
            best_ign = "—"
            for mn in match_cols:
                m = st.session_state.get(f"lobby_map_{mn}", {})
                if slot in m:
                    best_ign = m[slot]
                    break
            all_lobby_maps[slot] = best_ign
        
        for row in agg_rows:
            pos = row["position"]
            cls = f"rank-{pos}" if pos <= 3 else ""
            medal = {1:"🥇",2:"🥈",3:"🥉"}.get(pos, str(pos))
            
            team_name = row["team"]
            # Find slot for this team
            slot_for_team = next((s for s, t in st.session_state.teams.items() if t == team_name), None)
            ign = all_lobby_maps.get(slot_for_team, "—") if slot_for_team else "—"
            
            match_pts_html = ""
            match_pts_by_name = {m_data["match"]: m_data["pts"] for m_data in row["matches"]}
            for mc in match_cols:
                pts = match_pts_by_name.get(mc, 0)
                match_pts_html += f"<td>{pts}</td>"
            
            table_html += f"""<tr class="{cls}">
              <td>{medal}</td>
              <td class="team-name-cell">{team_name}</td>
              <td style="color:#60a5fa;font-size:0.85rem">{ign}</td>
              {match_pts_html}
              <td style="color:#4ade80">{row['total_kills']}</td>
              <td style="color:#fbbf24">{row['total_placement_pts']}</td>
              <td style="color:#ff6b00;font-weight:700;font-size:1.1rem">{row['total_pts']}</td>
            </tr>"""
        
        table_html += "</tbody></table>"
        st.markdown(table_html, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown('<p class="section-header">📁 Per-Match Breakdown</p>', unsafe_allow_html=True)
        for m in st.session_state.matches:
            with st.expander(f"📋 {m['match_name']}"):
                rows_data = []
                for r in m["results"]:
                    slot = r["slot"]
                    team = st.session_state.teams.get(slot, f"Team {slot}")
                    pts = calculate_points(r["rank"], r["kills"])
                    rows_data.append({
                        "Slot": slot,
                        "Team": team,
                        "Rank": r["rank"],
                        "Kills": r["kills"],
                        "Place Pts": PLACEMENT_PTS.get(r["rank"], 0),
                        "Kill Pts": r["kills"],
                        "Total": pts
                    })
                rows_data.sort(key=lambda x: x["Rank"])
                
                for rd in rows_data:
                    rank_medal = {1:"🥇",2:"🥈",3:"🥉"}.get(rd["Rank"],"")
                    st.markdown(
                        f'`Rank {rd["Rank"]}{rank_medal}` &nbsp; **{rd["Team"]}** &nbsp;&nbsp; '
                        f'Kills: `{rd["Kills"]}` &nbsp; Place Pts: `{rd["Place Pts"]}` &nbsp; '
                        f'**Total: `{rd["Total"]} pts`**',
                        unsafe_allow_html=True
                    )

# ═══════════════════════════ TAB 3: EXPORT ═══════════════════════════
with tab_export:
    st.markdown('<p class="section-header">🖼️ Generate Leaderboard Image</p>', unsafe_allow_html=True)
    
    if not st.session_state.matches:
        st.info("No matches saved yet.")
    else:
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            exp_tournament = st.text_input("Tournament Name on Image", st.session_state.tournament_name)
        with col_e2:
            exp_round = st.text_input("Round Label on Image", st.session_state.round_label)
        
        if st.button("🖼️ GENERATE IMAGE", use_container_width=True):
            with st.spinner("Generating leaderboard image..."):
                img = generate_leaderboard_image(
                    agg_rows,
                    st.session_state.matches,
                    tournament_name=exp_tournament,
                    round_label=exp_round
                )
                
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                buf.seek(0)
                
                st.image(img, caption="Generated Leaderboard", use_container_width=True)
                
                st.download_button(
                    label="⬇️ DOWNLOAD PNG",
                    data=buf,
                    file_name=f"leaderboard_{exp_round.replace(' ','_')}.png",
                    mime="image/png",
                    use_container_width=True
                )
