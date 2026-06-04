import os
import streamlit as st
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import numpy as np
import cv2
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

# ==========================================
# ✅ OPENCV PRE-PROCESSING PIPELINE
# ==========================================
def preprocess_image(pil_image, screen_type="Match Result"):
    """
    Strong OpenCV preprocessing pipeline for Free Fire screenshots.
    Steps:
      1. Crop ROI (remove header/footer/character overlay)
      2. Convert to Grayscale
      3. Upscale 2x for clearer text
      4. Binarize using Adaptive Threshold (removes noisy background)
    Returns:
      processed_pil  : PIL image ready for OCR
      debug_pil      : PIL image for UI preview
    """
    # Convert PIL → OpenCV (BGR)
    img_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    h, w   = img_cv.shape[:2]

    # ── Step 1: ROI Crop ──────────────────────────────────────
    if screen_type == "Match Result":
        # Remove: top 12% (FREE FIRE logo), bottom 10% (BACK button),
        #         right 8% (game character covers right panel)
        y1 = int(h * 0.12)
        y2 = int(h * 0.90)
        x2 = int(w * 0.92)
        img_cv = img_cv[y1:y2, 0:x2]
    else:
        # Lobby — minimal crop, just remove edges
        y1 = int(h * 0.08)
        y2 = int(h * 0.92)
        img_cv = img_cv[y1:y2, 0:w]

    # ── Step 2: Grayscale ─────────────────────────────────────
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # ── Step 3: Upscale 2x (INTER_CUBIC for quality) ─────────
    scale  = 2
    gh, gw = gray.shape
    gray   = cv2.resize(gray, (gw * scale, gh * scale),
                        interpolation=cv2.INTER_CUBIC)

    # ── Step 4: Denoise (removes pixel noise before threshold) ─
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # ── Step 5: Binarization ──────────────────────────────────
    # Try Otsu first (best for high-contrast panels like Free Fire rank panels)
    _, otsu = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Adaptive threshold as fallback (better for uneven lighting areas)
    adaptive = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,   # large block = handles big background gradients
        C=15
    )

    # Blend: use Otsu where confident, adaptive elsewhere
    # Simple rule: invert if most pixels are black (dark text on white bg preferred)
    if np.mean(otsu) < 127:
        # Image is mostly dark → invert so text is BLACK on WHITE
        binarized = cv2.bitwise_not(otsu)
    else:
        binarized = otsu

    # ── Step 6: Morphological cleanup (connect broken text strokes) ──
    kernel    = np.ones((2, 2), np.uint8)
    binarized = cv2.morphologyEx(binarized, cv2.MORPH_CLOSE, kernel)

    # Convert back to PIL
    processed_pil = Image.fromarray(binarized).convert("RGB")

    # Debug preview: side-by-side (original crop | processed)
    orig_pil      = Image.fromarray(
        cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    debug_w       = orig_pil.width + processed_pil.width + 20
    debug_h       = max(orig_pil.height, processed_pil.height) + 50
    debug_canvas  = Image.new("RGB", (debug_w, debug_h), (30, 30, 30))
    debug_canvas.paste(orig_pil,       (0,   25))
    debug_canvas.paste(processed_pil,  (orig_pil.width + 20, 25))
    try:
        from PIL import ImageDraw as ID, ImageFont as IF
        dd  = ID.Draw(debug_canvas)
        fnt = IF.load_default()
        dd.text((10,  5), "ORIGINAL (cropped)", fill="yellow", font=fnt)
        dd.text((orig_pil.width + 30, 5), "PRE-PROCESSED (OCR input)",
                fill="lime", font=fnt)
    except Exception:
        pass

    return processed_pil, debug_canvas


# ==========================================
# OCR RUNNER
# ==========================================
def run_ocr(processed_pil_image):
    """Run EasyOCR on a preprocessed PIL image."""
    ocr       = load_ocr_engine()
    temp_path = "temp_ocr_processed.png"
    processed_pil_image.save(temp_path)
    try:
        raw    = ocr.readtext(temp_path)
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
        blocks.append({'text': clean_ocr_text(text),
                       'x': cx, 'y': cy, 'col': col})

    left_col  = sorted([b for b in blocks if b['col'] == 'left'],
                       key=lambda b: b['y'])
    right_col = sorted([b for b in blocks if b['col'] == 'right'],
                       key=lambda b: b['y'])

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
        out  = []
        rank = start_rank
        for row in rows:
            row_text = " ".join(b['text'] for b in row)
            if len(row_text) > 3:
                best = process.extractOne(
                    row_text, slotlist, scorer=fuzz.WRatio)
                team = best[0] if best and best[1] > 45 else "UNKNOWN"
                out.append({"Slot": rank, "Team Name": team,
                            "Raw OCR": row_text})
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
        if conf < 0.15:
            continue
        xs  = [p[0] for p in bbox]
        ys  = [p[1] for p in bbox]
        cx  = sum(xs) / 4
        cy  = sum(ys) / 4
        cln = clean_ocr_text(text)
        if len(cln.strip()) < 1:
            continue
        blocks.append({'text': cln, 'x': cx, 'y': cy})

    if not blocks:
        return []

    left  = sorted([b for b in blocks if b['x'] < img_width * 0.50],
                   key=lambda b: b['y'])
    right = sorted([b for b in blocks if b['x'] >= img_width * 0.50],
                   key=lambda b: b['y'])

    def extract_kills_from_col(col_blocks, start_rank, slotlist):
        if not col_blocks:
            return []
        ys     = [b['y'] for b in col_blocks]
        spread = max(ys) - min(ys) if len(ys) > 1 else 100
        tol    = max(spread * 0.055, 12)

        rows, cur = [], [col_blocks[0]]
        for b in col_blocks[1:]:
            if abs(b['y'] - cur[0]['y']) <= tol:
                cur.append(b)
            else:
                rows.append(cur)
                cur = [b]
        rows.append(cur)

        results = []
        rank    = start_rank
        i       = 0
        while i < len(rows) and rank <= (start_rank + 5):
            block    = rows[i:i+2]
            combined = ' '.join(b['text'] for r in block for b in r)

            # STRICT: only count numbers immediately before Eliminat
            kill_matches = re.findall(r'(\d{1,2})\s*[Ee]liminat', combined)
            total_kills  = min(sum(int(k) for k in kill_matches), 30)

            team = slotlist[rank - 1] if rank - 1 < len(slotlist) \
                   else f"Team {rank}"

            results.append({'Rank': rank, 'Team Name': team,
                            'Kills': total_kills, 'Raw OCR': combined})
            rank += 1
            i    += 2
        return results

    left_teams  = extract_kills_from_col(left,  1, slotlist)
    right_teams = extract_kills_from_col(right, 6, slotlist)
    all_teams   = sorted(left_teams + right_teams, key=lambda x: x['Rank'])
    return all_teams[:12]


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

# Session state
for key, default in [
    ("slotlist",      DEFAULT_SLOTLIST.copy()),
    ("match_results", []),
    ("match_names",   []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ══════════════════════════════════════════
# STEP 1 — SLOTLIST
# ══════════════════════════════════════════
st.markdown("---")
st.header("📋 Step 1 — Set Slotlist")

slot_col1, slot_col2 = st.columns([1, 2])

with slot_col1:
    slotlist_df = pd.DataFrame({
        "Slot":      list(range(1, 13)),
        "Team Name": st.session_state.slotlist
    })
    edited_slot = st.data_editor(
        slotlist_df, num_rows="fixed", use_container_width=True,
        column_config={
            "Slot":      st.column_config.NumberColumn(
                "Slot #", disabled=True, width="small"),
            "Team Name": st.column_config.TextColumn(
                "Team Name", width="large"),
        },
        key="slotlist_editor"
    )
    st.session_state.slotlist = (
        edited_slot["Team Name"].dropna().str.strip().tolist()
    )

with slot_col2:
    st.markdown("**OR auto-fill from Lobby Screenshots:**")
    lobby_files = st.file_uploader(
        "Upload Lobby Screenshots (select all at once)",
        type=['png','jpg','jpeg'],
        accept_multiple_files=True,
        key="lobby_uploader"
    )
    if lobby_files:
        prev_cols = st.columns(min(len(lobby_files), 4))
        for ci, lf in enumerate(lobby_files[:4]):
            with prev_cols[ci]:
                st.image(Image.open(lf),
                         caption=f"Lobby {ci+1}",
                         use_container_width=True)

        if st.button("🔍 Extract Teams from Lobby Screenshots"):
            all_lobby_teams = []
            for ci, lf in enumerate(lobby_files):
                lf.seek(0)
                raw_img   = Image.open(lf).convert("RGB")
                proc_img, debug_img = preprocess_image(raw_img, "Lobby")
                w, _      = proc_img.size
                with st.spinner(f"OCR on Lobby {ci+1}..."):
                    lob_result = run_ocr(proc_img)

                with st.expander(f"🔬 Pre-processed View — Lobby {ci+1}"):
                    st.image(debug_img, use_container_width=True,
                             caption="Left: Original Crop | Right: Pre-processed (OCR input)")

                teams = parse_lobby(lob_result, st.session_state.slotlist, w)
                st.write(f"Lobby {ci+1} → {len(teams)} teams found")
                all_lobby_teams.extend(teams)

            if all_lobby_teams:
                names = ["UNKNOWN"] * 12
                for t in all_lobby_teams:
                    idx = int(t.get("Slot", 1)) - 1
                    if 0 <= idx < 12 and names[idx] == "UNKNOWN":
                        names[idx] = t["Team Name"]
                if sum(1 for n in names if n != "UNKNOWN") < 6:
                    for si, t in enumerate(all_lobby_teams[:12]):
                        names[si] = t["Team Name"]
                st.session_state.slotlist = names
                st.success(
                    f"✅ {len(all_lobby_teams)} teams from "
                    f"{len(lobby_files)} screenshots!")
                st.rerun()
            else:
                st.error("❌ No teams found.")

slotlist = st.session_state.slotlist

# ══════════════════════════════════════════
# STEP 2 — MATCH RESULTS
# ══════════════════════════════════════════
st.markdown("---")
st.header("🎮 Step 2 — Upload Match Result Screenshots")

match_files = st.file_uploader(
    "Upload Match Screenshots (select multiple for bulk)",
    type=['png','jpg','jpeg'],
    accept_multiple_files=True,
    key="match_uploader"
)

if match_files:
    if st.button("⚡ Process All Matches"):
        st.session_state.match_results = []
        st.session_state.match_names   = []
        progress = st.progress(0)

        for i, mf in enumerate(match_files):
            st.markdown(f"**Processing Match {i+1}: `{mf.name}`**")
            raw_img  = Image.open(mf).convert("RGB")

            # ── Pre-process ──────────────────────────────────
            proc_img, debug_img = preprocess_image(raw_img, "Match Result")
            iw, ih = proc_img.size

            # ── Show pre-processed image in expander ─────────
            with st.expander(
                f"🔬 Pre-processed Image — Match {i+1} "
                f"(click to verify text clarity)"
            ):
                st.image(
                    debug_img,
                    use_container_width=True,
                    caption="Left: Original Crop  |  "
                            "Right: Pre-processed (what OCR sees) — "
                            "text should be BLACK on WHITE"
                )

            # ── Run OCR on processed image ────────────────────
            with st.spinner(f"Running OCR on Match {i+1}..."):
                result    = run_ocr(proc_img)
                extracted = parse_match_result(
                    result, slotlist, iw, ih)

            if extracted:
                df = pd.DataFrame(extracted)
                df[['Place Pts', 'Total Pts']] = df.apply(
                    lambda r: calculate_points(r['Rank'], r['Kills']),
                    axis=1, result_type="expand"
                )
                st.session_state.match_results.append(df)
                st.session_state.match_names.append(mf.name)
                st.success(
                    f"✅ Match {i+1}: {len(extracted)} teams extracted")
            else:
                st.warning(
                    f"⚠️ Match {i+1}: No teams found — "
                    "check pre-processed image above")

            progress.progress((i+1) / len(match_files))

        st.rerun()

# ══════════════════════════════════════════
# STEP 3 — EDIT
# ══════════════════════════════════════════
if st.session_state.match_results:
    st.markdown("---")
    st.header("✏️ Step 3 — Review & Edit Each Match")

    tabs = st.tabs([
        f"📋 Match {i+1}"
        for i in range(len(st.session_state.match_results))
    ])

    for i, (tab, df, name) in enumerate(zip(
        tabs,
        st.session_state.match_results,
        st.session_state.match_names
    )):
        with tab:
            st.caption(f"File: `{name}`")
            show_raw = st.toggle(
                "Show Raw OCR Debug", value=False, key=f"raw_{i}")
            cols = ['Rank','Team Name','Kills','Place Pts','Total Pts']
            if show_raw:
                cols.append('Raw OCR')

            edited = st.data_editor(
                df[cols], num_rows="fixed",
                use_container_width=True,
                column_config={
                    "Rank":      st.column_config.NumberColumn(
                        "Rank", width="small"),
                    "Team Name": st.column_config.TextColumn("Team Name"),
                    "Kills":     st.column_config.NumberColumn(
                        "Kills", width="small"),
                    "Place Pts": st.column_config.NumberColumn(
                        "Place Pts", disabled=True, width="small"),
                    "Total Pts": st.column_config.NumberColumn(
                        "Total Pts", disabled=True, width="small"),
                },
                key=f"editor_{i}"
            )
            if not edited.empty:
                edited[['Place Pts','Total Pts']] = edited.apply(
                    lambda r: calculate_points(r['Rank'], r['Kills']),
                    axis=1, result_type="expand"
                )
                for col in ['Rank','Team Name','Kills',
                            'Place Pts','Total Pts']:
                    st.session_state.match_results[i][col] = \
                        edited[col].values

    # ══════════════════════════════════════
    # STEP 4 — AGGREGATE
    # ══════════════════════════════════════
    st.markdown("---")
    st.header("📊 Step 4 — Aggregate Points Table")
    st.caption(
        f"Combined results from "
        f"{len(st.session_state.match_results)} match(es)")

    combined = pd.concat(
        st.session_state.match_results, ignore_index=True)
    agg = combined.groupby("Team Name", as_index=False).agg(
        Total_Kills     = ("Kills",     "sum"),
        Total_Place_Pts = ("Place Pts", "sum"),
        Total_Points    = ("Total Pts", "sum"),
        Matches_Played  = ("Rank",      "count")
    ).sort_values(
        "Total_Points", ascending=False
    ).reset_index(drop=True)
    agg["Overall Rank"] = agg.index + 1
    agg = agg[[
        "Overall Rank","Team Name","Matches_Played",
        "Total_Kills","Total_Place_Pts","Total_Points"
    ]]
    agg.columns = [
        "#","Team","Matches","Total Kills","Place Pts","Total Points"]

    st.dataframe(agg, use_container_width=True, hide_index=True)

    # ══════════════════════════════════════
    # STEP 5 — IMAGE GENERATION
    # ══════════════════════════════════════
    st.markdown("---")
    st.header("🎨 Step 5 — Generate Leaderboard Image")

    if st.button("🏆 Generate Final Leaderboard Image"):
        W, H = 1080, 1920
        try:
            template = Image.open("template.png").resize((W, H))
        except FileNotFoundError:
            template = Image.new('RGB', (W, H), color=(10, 10, 20))

        draw = ImageDraw.Draw(template)
        try:
            f_title = ImageFont.truetype("arial.ttf", 56)
            f_sub   = ImageFont.truetype("arial.ttf", 34)
            f_hdr   = ImageFont.truetype("arial.ttf", 26)
            f_row   = ImageFont.truetype("arial.ttf", 28)
        except IOError:
            f_title = f_sub = f_hdr = f_row = ImageFont.load_default()

        draw.rectangle([(0,0),(W,100)], fill="#FF4C29")
        draw.text((W//2,50),  "MAG ESPORTS",
                  fill="white",   font=f_title, anchor="mm")
        draw.text((W//2,145), "AGGREGATE LEADERBOARD",
                  fill="#FF4C29", font=f_sub,   anchor="mm")
        draw.text((W//2,195),
                  f"Total Matches: {len(st.session_state.match_results)}",
                  fill="#94a3b8", font=f_hdr,   anchor="mm")

        SY  = 250
        ROW = 118
        X   = [55, 130, 590, 730, 880, 990]
        draw.line([(40,SY+40),(W-40,SY+40)], fill="#FF4C29", width=2)
        for txt, x in zip(
            ["#","TEAM","MATCHES","KILLS","PLACE","TOTAL"], X
        ):
            draw.text((x, SY+10), txt, fill="#94a3b8", font=f_hdr)
        draw.line([(40,SY+50),(W-40,SY+50)], fill="#333355", width=1)

        rank_colors = {1:"#FFD700", 2:"#C0C0C0", 3:"#CD7F32"}
        for idx, row in agg.iterrows():
            y  = SY + 80 + (idx * ROW)
            bg = "#12122a" if idx % 2 == 0 else "#0d0d1f"
            draw.rectangle([(40,y-18),(W-40,y+80)], fill=bg)
            rc = rank_colors.get(int(row["#"]), "white")
            draw.text((X[0],y), str(row["#"]),
                      fill=rc,        font=f_row)
            draw.text((X[1],y), str(row["Team"]),
                      fill="#FFD700", font=f_row)
            draw.text((X[2],y), str(row["Matches"]),
                      fill="#94a3b8", font=f_row)
            draw.text((X[3],y), str(row["Total Kills"]),
                      fill="white",   font=f_row)
            draw.text((X[4],y), str(row["Place Pts"]),
                      fill="#94a3b8", font=f_row)
            draw.text((X[5],y), str(row["Total Points"]),
                      fill="#00FFFF", font=f_row)
            draw.line([(40,y+88),(W-40,y+88)],
                      fill="#1e1e3a", width=1)

        st.image(template, use_container_width=True)
        buf = io.BytesIO()
        template.save(buf, format='PNG')
        st.download_button(
            "⬇️ Download Leaderboard Image",
            data=buf.getvalue(),
            file_name="mag_esports_leaderboard.png",
            mime="image/png"
        )

    st.markdown("---")
    if st.button("🗑️ Clear All & Start Fresh", type="secondary"):
        st.session_state.match_results = []
        st.session_state.match_names   = []
        st.rerun()
