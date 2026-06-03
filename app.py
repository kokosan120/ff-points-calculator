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

# ==========================================
# OCR ENGINE
# ==========================================
@st.cache_resource
def load_ocr_engine():
    import easyocr
    return easyocr.Reader(['en'], gpu=False)

def run_ocr(image):
    ocr = load_ocr_engine()
    temp_path = "temp_ocr_img.png"
    image.save(temp_path)
    try:
        raw = ocr.readtext(temp_path)
        result = [[(item[0], (item[1], item[2])) for item in raw]]
    except Exception as e:
        st.error(f"OCR Error: {str(e)}")
        result = [[]]
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    return result

# ==========================================
# UTILITIES
# ==========================================
def clean_ocr_text(text: str) -> str:
    text = re.sub(r'(?<!\w)[Oo](?!\w)', '0', text)
    text = re.sub(r'(?<!\w)[lI](?!\w)', '1', text)
    text = re.sub(r'[꧁꧂☠✦★༺༻]', '', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

def calculate_points(rank, kills):
    rank_pts = PLACEMENT_POINTS.get(int(rank), 0)
    return rank_pts, rank_pts + int(kills)

# ==========================================
# LOBBY PARSER
# ==========================================
def parse_lobby(ocr_data, slotlist, image_width):
    if not ocr_data or not ocr_data[0]:
        return []
    blocks = []
    for line in ocr_data[0]:
        bbox, (text, conf) = line
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        cx = sum(xs) / 4
        cy = sum(ys) / 4
        col = 'left' if cx < (image_width / 2) else 'right'
        blocks.append({'text': clean_ocr_text(text), 'x': cx, 'y': cy, 'col': col})

    left_col  = sorted([b for b in blocks if b['col'] == 'left'],  key=lambda b: b['y'])
    right_col = sorted([b for b in blocks if b['col'] == 'right'], key=lambda b: b['y'])

    def extract_col(col_blocks, start_rank):
        if not col_blocks:
            return []
        rows, cur = [], [col_blocks[0]]
        for b in col_blocks[1:]:
            if abs(b['y'] - cur[0]['y']) < 50:
                cur.append(b)
            else:
                rows.append(cur)
                cur = [b]
        rows.append(cur)
        out = []
        rank = start_rank
        for row in rows:
            row_text = " ".join(b['text'] for b in row)
            if len(row_text) > 3:
                best = process.extractOne(row_text, slotlist, scorer=fuzz.WRatio)
                team = best[0] if best and best[1] > 45 else "UNKNOWN"
                out.append({"Slot": rank, "Team Name": team, "Raw OCR": row_text})
                rank += 1
        return out

    left_teams  = extract_col(left_col, 1)
    right_teams = extract_col(right_col, 7 if left_teams else 1)
    return left_teams + right_teams

# ==========================================
# MATCH RESULT PARSER
# ==========================================
def parse_match_result(ocr_data, slotlist, img_width, img_height):
    if not ocr_data or not ocr_data[0]:
        return []

    blocks = []
    for line in ocr_data[0]:
        bbox, (text, conf) = line
        if conf < 0.20:
            continue
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        cx = sum(xs) / 4
        cy = sum(ys) / 4
        clean = clean_ocr_text(text)
        if len(clean.strip()) < 1:
            continue
        blocks.append({'text': clean, 'x': cx, 'y': cy})

    if not blocks:
        return []

    left  = [b for b in blocks if b['x'] < img_width * 0.52]
    right = [b for b in blocks if b['x'] >= img_width * 0.52]

    def group_col(col_blocks, start_rank):
        if not col_blocks:
            return []
        col_blocks.sort(key=lambda b: b['y'])
        ys     = [b['y'] for b in col_blocks]
        spread = max(ys) - min(ys)
        tol    = max(spread * 0.07, 15)
        rows, cur = [], [col_blocks[0]]
        for b in col_blocks[1:]:
            if abs(b['y'] - cur[0]['y']) <= tol:
                cur.append(b)
            else:
                rows.append(cur)
                cur = [b]
        rows.append(cur)

        out = []
        rank = start_rank
        i = 0
        while i < len(rows):
            merged = rows[i:i+2]
            text   = ' '.join(b['text'] for row in merged for b in row)
            kills  = re.findall(r'(\d+)\s*Eliminat', text, re.IGNORECASE)
            if not kills:
                kills = re.findall(r'\b(\d{1,2})\b', text)
            total_kills = sum(int(k) for k in kills)
            best = process.extractOne(text, slotlist, scorer=fuzz.WRatio)
            team = best[0] if best and best[1] > 35 else "UNKNOWN"
            out.append({
                'Rank': rank, 'Team Name': team,
                'Kills': total_kills, 'Raw OCR': text
            })
            rank += 1
            i += 2
        return out

    all_teams = group_col(left, 1) + group_col(right, 6)

    # Dedup
    seen, final = set(), []
    for t in all_teams:
        key = t['Team Name'].upper().strip()
        if key not in seen:
            seen.add(key)
            final.append(t)
        else:
            t['Team Name'] = f"DUPLICATE? {t['Team Name']}"
            final.append(t)
    return final[:12]


# ==========================================
# STREAMLIT UI
# ==========================================
st.set_page_config(
    page_title="MAG ESPORTS - FF Points Calculator",
    layout="wide",
    page_icon="🏆"
)

st.markdown("""
<h1 style='text-align:center; color:#FF4C29;'>
    🏆 MAG ESPORTS — Free Fire Points Calculator
</h1>
""", unsafe_allow_html=True)

# ── SESSION STATE INIT ──────────────────────────────────────────
if "slotlist"         not in st.session_state:
    st.session_state.slotlist = DEFAULT_SLOTLIST.copy()
if "match_results"    not in st.session_state:
    st.session_state.match_results = []       # list of DataFrames
if "match_names"      not in st.session_state:
    st.session_state.match_names = []         # match labels

# ══════════════════════════════════════════════════════════
# STEP 1 — SLOTLIST
# ══════════════════════════════════════════════════════════
st.markdown("---")
st.header("📋 Step 1 — Set Slotlist")

slot_col1, slot_col2 = st.columns([1, 2])

with slot_col1:
    st.markdown("**Manually enter / edit team names:**")
    slotlist_df = pd.DataFrame({
        "Slot": list(range(1, 13)),
        "Team Name": st.session_state.slotlist
    })
    edited_slot = st.data_editor(
        slotlist_df,
        num_rows="fixed",
        use_container_width=True,
        column_config={
            "Slot": st.column_config.NumberColumn("Slot #", disabled=True, width="small"),
            "Team Name": st.column_config.TextColumn("Team Name", width="large")
        },
        key="slotlist_editor"
    )
    st.session_state.slotlist = edited_slot["Team Name"].dropna().str.strip().tolist()

with slot_col2:
    st.markdown("**OR auto-fill slotlist from Lobby Screenshot:**")
    lobby_file = st.file_uploader(
        "Upload Lobby Screenshot",
        type=['png','jpg','jpeg'],
        key="lobby_uploader"
    )
    if lobby_file:
        lobby_img = Image.open(lobby_file).convert("RGB")
        w, h = lobby_img.size
        st.image(lobby_img, caption="Lobby Screenshot", use_container_width=True)
        if st.button("🔍 Extract Teams from Lobby SS"):
            with st.spinner("Running OCR on Lobby..."):
                lobby_result = run_ocr(lobby_img)
            lobby_teams = parse_lobby(lobby_result, st.session_state.slotlist, w)
            if lobby_teams:
                names = ["UNKNOWN"] * 12
                for t in lobby_teams:
                    idx = int(t["Slot"]) - 1
                    if 0 <= idx < 12:
                        names[idx] = t["Team Name"]
                st.session_state.slotlist = names
                st.success(f"✅ Extracted {len(lobby_teams)} teams! Slotlist updated — check left table.")
                st.rerun()
            else:
                st.error("❌ No teams found in lobby screenshot.")

slotlist = st.session_state.slotlist

# ══════════════════════════════════════════════════════════
# STEP 2 — MATCH RESULTS UPLOAD
# ══════════════════════════════════════════════════════════
st.markdown("---")
st.header("🎮 Step 2 — Upload Match Result Screenshots")

match_files = st.file_uploader(
    "Upload Match Result Screenshots (select multiple for bulk)",
    type=['png','jpg','jpeg'],
    accept_multiple_files=True,
    key="match_uploader"
)

if match_files:
    if st.button("⚡ Process All Match Screenshots"):
        st.session_state.match_results = []
        st.session_state.match_names   = []

        for i, mf in enumerate(match_files):
            with st.spinner(f"Processing Match {i+1}: {mf.name}..."):
                img = Image.open(mf).convert("RGB")
                w, h = img.size

                # Preprocess for match result
                img = img.crop((0, int(h*0.12), int(w*0.92), int(h*0.90)))
                img = ImageEnhance.Contrast(img).enhance(2.0)
                img = ImageEnhance.Sharpness(img).enhance(2.0)
                iw, ih = img.size

                result = run_ocr(img)
                extracted = parse_match_result(result, slotlist, iw, ih)

            if extracted:
                df = pd.DataFrame(extracted)
                df[['Place Pts', 'Total Pts']] = df.apply(
                    lambda r: calculate_points(r['Rank'], r['Kills']),
                    axis=1, result_type="expand"
                )
                st.session_state.match_results.append(df)
                st.session_state.match_names.append(mf.name)
            else:
                st.warning(f"⚠️ Match {i+1} ({mf.name}): No teams extracted.")

        st.success(f"✅ {len(st.session_state.match_results)} match(es) processed!")
        st.rerun()

# ══════════════════════════════════════════════════════════
# STEP 3 — VIEW + EDIT MATCH DATA
# ══════════════════════════════════════════════════════════
if st.session_state.match_results:
    st.markdown("---")
    st.header("✏️ Step 3 — Review & Edit Match Data")

    tabs = st.tabs([f"Match {i+1}" for i in range(len(st.session_state.match_results))])

    for i, (tab, df, name) in enumerate(zip(
        tabs,
        st.session_state.match_results,
        st.session_state.match_names
    )):
        with tab:
            st.markdown(f"**File:** `{name}`")
            show_raw = st.toggle("Show Raw OCR Debug", value=False, key=f"raw_{i}")

            display_cols = ['Rank','Team Name','Kills','Place Pts','Total Pts']
            if show_raw:
                display_cols.append('Raw OCR')

            edited = st.data_editor(
                df[display_cols],
                num_rows="dynamic",
                use_container_width=True,
                key=f"match_editor_{i}"
            )
            # Save edits back
            for col in display_cols:
                if col in edited.columns:
                    st.session_state.match_results[i][col] = edited[col].values

    # ══════════════════════════════════════════════════════════
    # STEP 4 — AGGREGATE POINTS TABLE
    # ══════════════════════════════════════════════════════════
    st.markdown("---")
    st.header("📊 Step 4 — Aggregate Points Table")

    combined = pd.concat(st.session_state.match_results, ignore_index=True)

    agg = combined.groupby("Team Name", as_index=False).agg(
        Total_Kills    = ("Kills",      "sum"),
        Total_Place_Pts= ("Place Pts",  "sum"),
        Total_Points   = ("Total Pts",  "sum"),
        Matches_Played = ("Rank",       "count")
    ).sort_values("Total_Points", ascending=False).reset_index(drop=True)
    agg["Overall Rank"] = agg.index + 1
    agg = agg[["Overall Rank","Team Name","Matches_Played",
               "Total_Kills","Total_Place_Pts","Total_Points"]]
    agg.columns = ["#","Team","Matches","Total Kills","Place Pts","Total Points"]

    st.dataframe(
        agg,
        use_container_width=True,
        hide_index=True
    )

    # ══════════════════════════════════════════════════════════
    # STEP 5 — GENERATE LEADERBOARD IMAGE
    # ══════════════════════════════════════════════════════════
    st.markdown("---")
    st.header("🎨 Step 5 — Generate Leaderboard Image")

    if st.button("🏆 Generate Final Leaderboard Image"):
        try:
            template = Image.open("template.png")
        except FileNotFoundError:
            template = Image.new('RGB', (1080, 1920), color=(15, 15, 25))

        draw = ImageDraw.Draw(template)
        try:
            font_title = ImageFont.truetype("arial.ttf", 52)
            font_sub   = ImageFont.truetype("arial.ttf", 36)
            font_row   = ImageFont.truetype("arial.ttf", 30)
            font_hdr   = ImageFont.truetype("arial.ttf", 28)
        except IOError:
            font_title = font_sub = font_row = font_hdr = ImageFont.load_default()

        # Header
        draw.rectangle([(0,0),(1080,80)], fill="#FF4C29")
        draw.text((540, 40),  "MAG ESPORTS",    fill="white",  font=font_title, anchor="mm")
        draw.text((540, 130), "AGGREGATE LEADERBOARD", fill="#FF4C29", font=font_sub, anchor="mm")

        # Column headers
        START_Y = 220
        ROW_H   = 85
        draw.text((60,  START_Y), "#",           fill="#94a3b8", font=font_hdr)
        draw.text((130, START_Y), "TEAM",         fill="#94a3b8", font=font_hdr)
        draw.text((620, START_Y), "KILLS",        fill="#94a3b8", font=font_hdr)
        draw.text((750, START_Y), "PLACE PTS",    fill="#94a3b8", font=font_hdr)
        draw.text((900, START_Y), "TOTAL",        fill="#94a3b8", font=font_hdr)

        # Divider
        draw.line([(50, START_Y+45),(1040, START_Y+45)], fill="#2A2E3A", width=2)

        for idx, row in agg.iterrows():
            y = START_Y + 80 + (idx * ROW_H)
            # Alternate row bg
            if idx % 2 == 0:
                draw.rectangle([(50, y-15),(1040, y+55)], fill="#1a1a2e")
            draw.text((60,  y), str(row["#"]),           fill="white",   font=font_row)
            draw.text((130, y), str(row["Team"]),         fill="#FFD700", font=font_row)
            draw.text((620, y), str(row["Total Kills"]),  fill="white",   font=font_row)
            draw.text((750, y), str(row["Place Pts"]),    fill="#94a3b8", font=font_row)
            draw.text((900, y), str(row["Total Points"]), fill="#00FFFF", font=font_row)

        st.image(template, caption="Final Leaderboard", use_container_width=True)

        buf = io.BytesIO()
        template.save(buf, format='PNG')
        st.download_button(
            label="⬇️ Download Leaderboard Image",
            data=buf.getvalue(),
            file_name="mag_esports_leaderboard.png",
            mime="image/png"
        )

    # Clear all matches button
    st.markdown("---")
    if st.button("🗑️ Clear All Match Data & Start Fresh"):
        st.session_state.match_results = []
        st.session_state.match_names   = []
        st.rerun()
