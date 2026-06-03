import os
import streamlit as st
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import numpy as np
import re
from rapidfuzz import process, fuzz
import io

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
PLACEMENT_POINTS = {
    1: 12, 2: 9, 3: 8, 4: 7, 5: 6,
    6: 5, 7: 4, 8: 3, 9: 2, 10: 1,
    11: 0, 12: 0
}

DEFAULT_SLOTLIST = [
    "MAG ESPORTS", "TEAM ELITE", "TOTAL GAMING", "DESI GAMERS",
    "BLIND ESPORTS", "GODLIKE", "ORANGUTAN", "TSG ARMY",
    "GALAXY RACER", "UB ESPORTS", "EVL ESPORTS", "NIGMA GALAXY"
]

@st.cache_resource
def load_ocr_engine():
    import easyocr
    return easyocr.Reader(['en'], gpu=False)

def clean_ocr_text(text: str) -> str:
    text = re.sub(r'(?<!\w)[Oo](?!\w)', '0', text)
    text = re.sub(r'(?<!\w)[lI](?!\w)', '1', text)
    text = re.sub(r'[꧁꧂☠✦★༺༻]', '', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

def calculate_points(rank, kills):
    rank_pts = PLACEMENT_POINTS.get(rank, 0)
    total_pts = rank_pts + kills
    return rank_pts, total_pts

def run_ocr(image):
    """Run EasyOCR and return result in standard format"""
    ocr = load_ocr_engine()
    temp_path = "temp_ocr_img.png"
    image.save(temp_path)
    try:
        raw = ocr.readtext(temp_path)
        # Convert EasyOCR format to PaddleOCR-compatible format
        # EasyOCR: (bbox, text, conf)
        # Our format: (bbox, (text, conf))
        result = [[(item[0], (item[1], item[2])) for item in raw]]
    except Exception as e:
        st.error(f"OCR Error: {str(e)}")
        result = [[]]
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    return result

def parse_match_result(ocr_data, slotlist, img_width, img_height):
    if not ocr_data or not ocr_data[0]:
        return []

    blocks = []
    for line in ocr_data[0]:
        bbox, (text, conf) = line
        if conf < 0.20:
            continue
        # EasyOCR bbox is [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        cx = sum(xs) / 4
        cy = sum(ys) / 4
        clean = clean_ocr_text(text)
        if len(clean.strip()) < 1:
            continue
        blocks.append({'text': clean, 'x': cx, 'y': cy, 'conf': conf})

    if not blocks:
        return []

    # Split LEFT (rank 1-5) and RIGHT (rank 6-11)
    left  = [b for b in blocks if b['x'] < img_width * 0.52]
    right = [b for b in blocks if b['x'] >= img_width * 0.52]

    def group_into_rank_blocks(col_blocks, start_rank):
        if not col_blocks:
            return []

        col_blocks.sort(key=lambda b: b['y'])

        ys = [b['y'] for b in col_blocks]
        spread = max(ys) - min(ys)
        tolerance = max(spread * 0.07, 15)

        rows = []
        cur = [col_blocks[0]]
        for b in col_blocks[1:]:
            if abs(b['y'] - cur[0]['y']) <= tolerance:
                cur.append(b)
            else:
                rows.append(cur)
                cur = [b]
        rows.append(cur)

        rank_blocks = []
        i = 0
        rank = start_rank
        while i < len(rows):
            block_rows = rows[i:i+2]
            combined_text = ' '.join(
                b['text'] for row in block_rows for b in row)

            kill_matches = re.findall(
                r'(\d+)\s*Eliminat', combined_text, re.IGNORECASE)
            if not kill_matches:
                kill_matches = re.findall(r'\b(\d{1,2})\b', combined_text)
            total_kills = sum(int(k) for k in kill_matches)

            best = process.extractOne(
                combined_text, slotlist, scorer=fuzz.WRatio)
            team = best[0] if best and best[1] > 35 else "UNKNOWN"

            rank_blocks.append({
                'Rank': rank,
                'Team Name': team,
                'Kills': total_kills,
                'Raw OCR': combined_text,
                'Source': 'match_result'
            })
            rank += 1
            i += 2

        return rank_blocks

    left_teams  = group_into_rank_blocks(left, 1)
    right_teams = group_into_rank_blocks(right, 6)
    all_teams   = left_teams + right_teams

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

def parse_lobby_result(ocr_data, slotlist, image_width):
    if not ocr_data or not ocr_data[0]:
        return []

    blocks = []
    for line in ocr_data[0]:
        bbox, (text, conf) = line
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        center_x = sum(xs) / 4
        center_y = sum(ys) / 4
        col = 'left' if center_x < (image_width / 2) else 'right'
        blocks.append({
            'text': clean_ocr_text(text),
            'x': center_x,
            'y': center_y,
            'col': col
        })

    left_col  = sorted([b for b in blocks if b['col'] == 'left'],  key=lambda b: b['y'])
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
                    "Source": "lobby"
                })
                rank_counter += 1
        return teams_data

    left_teams  = extract_from_column(left_col, 1)
    right_teams = extract_from_column(right_col, 6 if left_teams else 1)
    teams_data  = left_teams + right_teams

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

    screen_type = st.radio("Screenshot Type", ["Match Result", "Lobby"], horizontal=True)

    st.markdown("**Slotlist — Edit Team Names:**")
    slotlist_df = pd.DataFrame({
        "Slot": list(range(1, 13)),
        "Team Name": DEFAULT_SLOTLIST
    })
    edited_slotlist = st.data_editor(
        slotlist_df,
        num_rows="fixed",
        use_container_width=True,
        column_config={
            "Slot": st.column_config.NumberColumn("Slot #", disabled=True, width="small"),
            "Team Name": st.column_config.TextColumn("Team Name", width="large")
        },
        key="slotlist_editor"
    )
    slotlist = edited_slotlist["Team Name"].dropna().str.strip().tolist()

    uploaded_files = st.file_uploader(
        "Upload Screenshots (Select Multiple)",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True
    )

if uploaded_files:
    all_match_results = []

    with col2:
        for i, uploaded_file in enumerate(uploaded_files):
            st.markdown(f"---\n#### Match {i+1}: `{uploaded_file.name}`")
            image = Image.open(uploaded_file).convert("RGB")
            img_width, img_height = image.size
            st.write(f"📐 Image Size: `{img_width} x {img_height}`")

            if screen_type == "Match Result":
                # Crop: remove FREE FIRE logo top, BACK button bottom, character right
                w, h = image.size
                image = image.crop((0, int(h*0.12), int(w*0.92), int(h*0.90)))
                # Enhance contrast + sharpness for dark FF background
                image = ImageEnhance.Contrast(image).enhance(2.0)
                image = ImageEnhance.Sharpness(image).enhance(2.0)
                img_width, img_height = image.size

            with st.spinner(f"Running OCR on Match {i+1}..."):
                result = run_ocr(image)

            # Debug expander - always visible
            with st.expander(f"🔍 Raw OCR Blocks — Match {i+1} ({len(result[0]) if result and result[0] else 0} blocks found)"):
                if result and result[0]:
                    debug_rows = []
                    for line in result[0]:
                        bbox, (text, conf) = line
                        xs = [p[0] for p in bbox]
                        ys = [p[1] for p in bbox]
                        cx = round(sum(xs)/4)
                        cy = round(sum(ys)/4)
                        debug_rows.append({
                            "Text": text,
                            "Confidence": round(conf, 2),
                            "Center X": cx,
                            "Center Y": cy,
                            "Side": "LEFT" if cx < img_width*0.52 else "RIGHT"
                        })
                    st.dataframe(pd.DataFrame(debug_rows), use_container_width=True)
                else:
                    st.error("❌ OCR returned 0 blocks — try a clearer screenshot")

            # Parse based on type
            if screen_type == "Match Result":
                extracted_data = parse_match_result(result, slotlist, img_width, img_height)
            else:
                extracted_data = parse_lobby_result(result, slotlist, img_width)

            st.write(f"✅ Teams extracted: **{len(extracted_data)}**")

            df = pd.DataFrame(extracted_data)

            if not df.empty:
                df[['Place Pts', 'Total Pts']] = df.apply(
                    lambda row: calculate_points(row['Rank'], row['Kills']),
                    axis=1, result_type="expand"
                )
                df = df[['Rank', 'Team Name', 'Kills', 'Place Pts', 'Total Pts', 'Raw OCR', 'Source']]
                all_match_results.append(df)

                leaderboard = df.sort_values(
                    by=['Total Pts', 'Place Pts'], ascending=[False, False]
                ).reset_index(drop=True)
                leaderboard['Final Rank'] = leaderboard.index + 1

                show_raw = st.toggle("🔍 Show Raw OCR Debug Columns", value=False, key=f"raw_{i}")
                display_df = leaderboard if show_raw else leaderboard.drop(
                    columns=["Raw OCR", "Source"], errors='ignore')

                st.markdown("**✏️ Edit table below if needed, then generate image:**")
                edited_df = st.data_editor(
                    display_df, num_rows="dynamic",
                    use_container_width=True, key=f"editor_{i}")

                st.info("✅ Review table above. Fix any errors, then click Generate.")

                has_dupes = edited_df["Team Name"].str.upper().str.strip().str.startswith("DUPLICATE?").any()
                if has_dupes:
                    st.warning("⚠️ Duplicate team names found! Fix them before generating.")
                else:
                    if st.button(f"🎨 Generate Leaderboard Image — Match {i+1}", key=f"gen_{i}"):
                        try:
                            template = Image.open("template.png")
                        except FileNotFoundError:
                            template = Image.new('RGB', (1080, 1920), color=(20, 20, 30))

                        draw = ImageDraw.Draw(template)
                        try:
                            font_big  = ImageFont.truetype("arial.ttf", 40)
                            font_med  = ImageFont.truetype("arial.ttf", 32)
                        except IOError:
                            font_big = font_med = ImageFont.load_default()

                        START_X    = 80
                        START_Y    = 300
                        ROW_HEIGHT = 100

                        # Header
                        draw.text((540, 100), "MAG ESPORTS", fill="#FF4C29", font=font_big, anchor="mm")
                        draw.text((540, 160), "MATCH RESULTS", fill="white", font=font_med, anchor="mm")

                        for idx, row in edited_df.iterrows():
                            y_pos = START_Y + (idx * ROW_HEIGHT)
                            draw.text((START_X,        y_pos), str(row.get('Final Rank', idx+1)), fill="white",  font=font_med)
                            draw.text((START_X + 150,  y_pos), str(row['Team Name']),              fill="gold",   font=font_med)
                            draw.text((START_X + 600,  y_pos), str(row['Kills']),                  fill="white",  font=font_med)
                            draw.text((START_X + 800,  y_pos), str(row['Total Pts']),              fill="#00FFFF",font=font_med)

                        st.image(template, caption="Generated Leaderboard", use_container_width=True)

                        img_byte_arr = io.BytesIO()
                        template.save(img_byte_arr, format='PNG')
                        st.download_button(
                            label="⬇️ Download Leaderboard Image",
                            data=img_byte_arr.getvalue(),
                            file_name=f"leaderboard_match_{i+1}.png",
                            mime="image/png",
                            key=f"dl_{i}"
                        )
            else:
                st.error("❌ No teams found. Check debug expander above for raw OCR output.")

        # Aggregate section
        if len(all_match_results) > 1:
            st.markdown("---")
            st.header("📊 Aggregate Leaderboard — All Matches")
            combined = pd.concat(all_match_results, ignore_index=True)
            agg = combined.groupby("Team Name", as_index=False).agg(
                Total_Kills=("Kills", "sum"),
                Total_Place_Pts=("Place Pts", "sum"),
                Total_Points=("Total Pts", "sum"),
                Matches_Played=("Rank", "count")
            ).sort_values("Total_Points", ascending=False).reset_index(drop=True)
            agg["Overall Rank"] = agg.index + 1
            st.dataframe(agg, use_container_width=True)
