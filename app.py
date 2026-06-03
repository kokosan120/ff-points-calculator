import os
# Crucial System Fixes for PaddleOCR
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["FLAGS_use_mkldnn"] = "0"

import streamlit as st
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import numpy as np
import re
from rapidfuzz import process, fuzz
# Lazy load PaddleOCR to avoid slowing down initial Streamlit render
from paddleocr import PaddleOCR
import io

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
PLACEMENT_POINTS = {
    1: 12, 2: 9, 3: 8, 4: 7, 5: 6,
    6: 5, 7: 4, 8: 3, 9: 2, 10: 1,
    11: 0, 12: 0
}

# Dummy Slotlist
DEFAULT_SLOTLIST = [
    "MAG ESPORTS", "TEAM ELITE", "TOTAL GAMING", "DESI GAMERS",
    "BLIND ESPORTS", "GODLIKE", "ORANGUTAN", "TSG ARMY",
    "GALAXY RACER", "UB ESPORTS", "EVL ESPORTS", "NIGMA GALAXY"
]

@st.cache_resource
def load_ocr_engine():
    # Configure PaddleOCR strictly as requested
    return PaddleOCR(use_gpu=False, lang='en', enable_mkldnn=False)

def clean_ocr_text(text: str) -> str:
    text = re.sub(r'(?<!\w)[Oo](?!\w)', '0', text)
    text = re.sub(r'(?<!\w)[lI](?!\w)', '1', text)
    text = re.sub(r'[꧁꧂☠✦★༺༻]', '', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

def calculate_points(rank, kills):
    rank_pts = PLACEMENT_POINTS.get(rank, 0)
    # 1 Point per Kill
    total_pts = rank_pts + kills
    return rank_pts, total_pts

def parse_match_result(ocr_data, slotlist, img_width, img_height):
    if not ocr_data or not ocr_data[0]:
        return []

    # Step 1: Collect all blocks with position + clean text
    blocks = []
    for line in ocr_data[0]:
        bbox, (text, conf) = line
        if conf < 0.25:
            continue
        cx = (bbox[0][0] + bbox[2][0]) / 2
        cy = (bbox[0][1] + bbox[2][1]) / 2
        clean = clean_ocr_text(text)
        if len(clean.strip()) < 1:
            continue
        blocks.append({'text': clean, 'x': cx, 'y': cy})

    # Step 2: Split into LEFT column (rank 1-5) and RIGHT (6-11)
    left  = [b for b in blocks if b['x'] < img_width * 0.52]
    right = [b for b in blocks if b['x'] >= img_width * 0.52]

    def group_into_rank_blocks(col_blocks, start_rank):
        if not col_blocks:
            return []
        
        col_blocks.sort(key=lambda b: b['y'])
        
        # Dynamic tolerance based on column height spread
        ys = [b['y'] for b in col_blocks]
        spread = max(ys) - min(ys)
        tolerance = spread * 0.07  # 7% of column height
        
        # Group into rows
        rows = []
        cur = [col_blocks[0]]
        for b in col_blocks[1:]:
            if abs(b['y'] - cur[0]['y']) <= tolerance:
                cur.append(b)
            else:
                rows.append(cur)
                cur = [b]
        rows.append(cur)
        
        # Each rank block = 2 consecutive player rows
        # Group pairs of rows into one rank block
        rank_blocks = []
        i = 0
        rank = start_rank
        while i < len(rows):
            # Merge 2 rows = 1 team (duo layout)
            block_rows = rows[i:i+2]
            combined_text = ' '.join(
                b['text'] for row in block_rows for b in row)
            
            # Extract kills: all numbers before "Eliminat"
            kill_matches = re.findall(
                r'(\d+)\s*Eliminat', combined_text, re.IGNORECASE)
            # Fallback: if no "Eliminat" found, find all 1-2 digit nums
            if not kill_matches:
                kill_matches = re.findall(r'\b(\d{1,2})\b', combined_text)
            total_kills = sum(int(k) for k in kill_matches)
            
            # Fuzzy match against slotlist
            from rapidfuzz import process, fuzz
            best = process.extractOne(
                combined_text, slotlist, scorer=fuzz.WRatio)
            team = best[0] if best and best[1] > 35 else "UNKNOWN"
            
            rank_blocks.append({
                'Rank': rank,
                'Team Name': team,
                'Kills': total_kills,
                'Raw OCR': combined_text,
                'Source': 'match_result_v3'
            })
            rank += 1
            i += 2  # jump 2 rows = next team
        
        return rank_blocks

    left_teams  = group_into_rank_blocks(left, 1)
    right_teams = group_into_rank_blocks(right, 6)
    all_teams   = left_teams + right_teams

    # Deduplicate
    seen = set()
    final = []
    for t in all_teams:
        key = t['Team Name'].upper().strip()
        if key not in seen:
            seen.add(key)
            final.append(t)
        else:
            t['Team Name'] = f"DUPLICATE? {t['Team Name']}"
            final.append(t)

    return final[:12]

def parse_ocr_results(ocr_data, slotlist, image_width):
    if not ocr_data or not ocr_data[0]:
        return []

    blocks = []
    # Extract text and calculate bounding box centers
    for line in ocr_data[0]:
        bbox, (text, conf) = line
        # bbox is [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        x_coords = [point[0] for point in bbox]
        y_coords = [point[1] for point in bbox]
        center_x = sum(x_coords) / 4
        center_y = sum(y_coords) / 4
        
        col = 'left' if center_x < (image_width / 2) else 'right'
        
        blocks.append({
            'text': clean_ocr_text(text),
            'x': center_x,
            'y': center_y,
            'col': col
        })

    left_col = sorted([b for b in blocks if b['col'] == 'left'], key=lambda b: b['y'])
    right_col = sorted([b for b in blocks if b['col'] == 'right'], key=lambda b: b['y'])

    def extract_from_column(col_blocks, starting_rank):
        if not col_blocks:
            return []
            
        rows = []
        current_row = [col_blocks[0]]
        
        Y_TOLERANCE = 50 
        
        for block in col_blocks[1:]:
            if abs(block['y'] - current_row[0]['y']) < Y_TOLERANCE:
                current_row.append(block)
            else:
                rows.append(current_row)
                current_row = [block]
        rows.append(current_row)
        
        teams_data = []
        rank_counter = starting_rank
        
        for row in rows:
            row_text = " ".join([b['text'] for b in row])
            
            kills_list = re.findall(r'(\d+)\s*Eliminati', row_text, re.IGNORECASE)
            total_kills = sum([int(k) for k in kills_list])
            
            best_match = process.extractOne(row_text, slotlist, scorer=fuzz.WRatio)
            team_name = best_match[0] if best_match and best_match[1] > 50 else "UNKNOWN"
            
            if len(row_text) > 3:
                teams_data.append({
                    "Rank": rank_counter,
                    "Team Name": team_name,
                    "Kills": total_kills,
                    "Raw OCR": row_text,
                    "Source": "coordinate_grouping"
                })
                rank_counter += 1
                
        return teams_data

    left_teams = extract_from_column(left_col, 1)
    right_teams = extract_from_column(right_col, 6 if left_teams else 1) 
    
    teams_data = left_teams + right_teams

    # --- FALLBACK REGEX PARSING ---
    if len(teams_data) < 10:
        existing_ranks = {t["Rank"] for t in teams_data}
        raw_text_flat = "\n".join([b['text'] for b in sorted(blocks, key=lambda b: b['y'])])
        pattern = r'#?(\d{1,2})\s+([A-Z][A-Z0-9 \-\.]{2,25})\s+.*?(\d{1,2})\s*(?:Eliminati|Kill)'
        matches = re.findall(pattern, raw_text_flat, re.IGNORECASE | re.MULTILINE)
        
        for match in matches:
            rank = int(match[0])
            if rank not in existing_ranks:
                team_str = match[1].strip()
                kills = int(match[2])
                
                best_match = process.extractOne(team_str, slotlist, scorer=fuzz.WRatio)
                if best_match and best_match[1] > 45:
                    team_name = best_match[0]
                else:
                    team_name = "UNKNOWN"
                    
                teams_data.append({
                    "Rank": rank,
                    "Team Name": team_name,
                    "Kills": kills,
                    "Raw OCR": team_str,
                    "Source": "fallback_regex"
                })
                existing_ranks.add(rank)
                
        teams_data = sorted(teams_data, key=lambda x: x["Rank"])

    seen = set()
    deduped = []
    for team in teams_data:
        key = team["Team Name"].upper().strip()
        if key not in seen and key != "UNKNOWN":
            seen.add(key)
            deduped.append(team)
        else:
            team["Team Name"] = f"DUPLICATE? {team['Team Name']}"
            deduped.append(team)
    return deduped


# ==========================================
# STREAMLIT UI
# ==========================================
st.set_page_config(page_title="MAG ESPORTS - FF Points Calculator", layout="wide")
st.title("🏆 MAG ESPORTS: Free Fire Points Calculator")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("1. Settings & Upload")
    
    screen_type = st.radio("Image Type", ["Match Result", "Lobby"])
    
    slotlist_df = pd.DataFrame({
        "Slot": list(range(1, 13)),
        "Team Name": DEFAULT_SLOTLIST
    })
    edited_slotlist = st.data_editor(
        slotlist_df,
        num_rows="fixed",
        use_container_width=True,
        column_config={
            "Slot": st.column_config.NumberColumn(
                "Slot #", disabled=True, width="small"),
            "Team Name": st.column_config.TextColumn(
                "Team Name", width="large")
        },
        key="slotlist_editor"
    )
    slotlist = edited_slotlist["Team Name"].dropna().str.strip().tolist()
    
    uploaded_files = st.file_uploader(
        "Upload Screenshots",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True
    )

if uploaded_files:
    all_match_results = []
    
    with col2:
        for i, uploaded_file in enumerate(uploaded_files):
            st.markdown(f"#### Match {i+1}: {uploaded_file.name}")
            image = Image.open(uploaded_file).convert("RGB")
            
            st.write(f"Image loaded: {image.size}, mode: {image.mode}")
            
            if screen_type == "Match Result":
                # 1. Crop: remove top 12% (FREE FIRE logo) 
                #          remove bottom 10% (BACK button)
                #          remove right 8% (game character covers right panel)
                w, h = image.size
                image = image.crop((0, int(h*0.12), int(w*0.92), int(h*0.90)))
                
                # 2. Enhance for dark background with gold/white text
                image = image.convert('RGB')
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(2.0)
                enhancer2 = ImageEnhance.Sharpness(image)
                image = enhancer2.enhance(2.0)
                
            temp_path = f"temp_screenshot_{i}.png"
            # 3. Save for OCR
            w2, h2 = image.size
            image.save(temp_path)
            
            # Update dimensions for parser
            img_width, img_height = w2, h2
            
            with st.spinner("Processing OCR (Local CPU Mode)..."):
                ocr = load_ocr_engine()
                try:
                    result = ocr.ocr(temp_path, cls=False)
                except Exception as e:
                    st.error(f"OCR Engine Error: {str(e)}")
                    st.stop()
                
            st.success("OCR Extraction Complete!")
            
            if screen_type == "Match Result":
                with st.expander(f"Raw OCR Output ({len(result[0]) if result and result[0] else 0} blocks)"):
                    st.write(result)
            else:
                st.write(f"OCR raw result type: {type(result)}")
                st.write(f"OCR result length: {len(result) if result else 'NONE'}")
                if result and result[0]:
                    st.write(f"First block sample: {result[0][0]}")
                else:
                    st.error("OCR returned empty or None — this is the failure point")
                
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            if screen_type == "Match Result":
                extracted_data = parse_match_result(result, slotlist, img_width, img_height)
            else:
                extracted_data = parse_ocr_results(result, slotlist, img_width)
            
            st.write(f"Teams extracted: {len(extracted_data)}")
            st.write(extracted_data)
            
            df = pd.DataFrame(extracted_data)
            
            if not df.empty:
                df[['Place Pts', 'Total Pts']] = df.apply(
                    lambda row: calculate_points(row['Rank'], row['Kills']), axis=1, result_type="expand"
                )
                
                if "Source" in df.columns:
                    df = df[['Rank', 'Team Name', 'Kills', 'Place Pts', 'Total Pts', 'Raw OCR', 'Source']]
                else:
                    df = df[['Rank', 'Team Name', 'Kills', 'Place Pts', 'Total Pts', 'Raw OCR']]
                    
                all_match_results.append(df)
                
                leaderboard = df.sort_values(by=['Total Pts', 'Place Pts'], ascending=[False, False]).reset_index(drop=True)
                leaderboard['Final Rank'] = leaderboard.index + 1
                
                show_raw = st.toggle("Show Raw OCR Debug Data", value=False, key=f"raw_{i}")
                display_df = leaderboard if show_raw else leaderboard.drop(
                    columns=["Raw OCR", "Source"], errors='ignore')

                edited_df = st.data_editor(display_df, num_rows="dynamic", use_container_width=True, key=f"editor_{i}")
                
                st.info("Review and fix the table above, then generate image.")
                if edited_df["Team Name"].str.upper().str.strip().str.startswith("DUPLICATE?").any():
                    st.warning("Fix duplicate team names before generating image.")
                else:
                    if st.button("Generate Final Leaderboard Image", key=f"generate_btn_{i}"):
                        try:
                            template = Image.open("template.png")
                        except FileNotFoundError:
                            template = Image.new('RGB', (1080, 1920), color=(30, 30, 30))
                        
                        draw = ImageDraw.Draw(template)
                        
                        try:
                            font = ImageFont.truetype("arial.ttf", 32)
                        except IOError:
                            font = ImageFont.load_default()
                        
                        START_X = 150
                        START_Y = 300
                        ROW_HEIGHT = 80
                        
                        for idx, row in edited_df.iterrows():
                            y_pos = START_Y + (idx * ROW_HEIGHT)
                            
                            rank_str = str(row['Final Rank'])
                            team_str = str(row['Team Name'])
                            kills_str = str(row['Kills'])
                            pts_str = str(row['Total Pts'])
                            
                            draw.text((START_X, y_pos), rank_str, fill="white", font=font)
                            draw.text((START_X + 150, y_pos), team_str, fill="gold", font=font)
                            draw.text((START_X + 500, y_pos), kills_str, fill="white", font=font)
                            draw.text((START_X + 700, y_pos), pts_str, fill="cyan", font=font)
                        
                        st.image(template, caption="Generated Leaderboard", use_container_width=True)
                        
                        img_byte_arr = io.BytesIO()
                        template.save(img_byte_arr, format='PNG')
                        st.download_button(
                            label="Download Final Image",
                            data=img_byte_arr.getvalue(),
                            file_name=f"final_leaderboard_{i}.png",
                            mime="image/png",
                            key=f"dl_btn_{i}"
                        )
            else:
                st.error("No teams found. Please check the image formatting and OCR.")
                
        if len(all_match_results) > 1:
            st.header("Aggregate Leaderboard")
            combined = pd.concat(all_match_results, ignore_index=True)
            agg = combined.groupby("Team Name", as_index=False).agg(
                Total_Kills=("Kills","sum"),
                Total_Place_Pts=("Place Pts","sum"),
                Total_Points=("Total Pts","sum"),
                Matches=("Rank","count")
            ).sort_values("Total_Points", ascending=False).reset_index(drop=True)
            agg["Overall Rank"] = agg.index + 1
            st.dataframe(agg, use_container_width=True)
