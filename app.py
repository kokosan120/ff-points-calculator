import os, io, re
import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from rapidfuzz import process, fuzz

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
PLACEMENT_POINTS = {
    1:12, 2:9, 3:8, 4:7, 5:6,
    6:5,  7:4, 8:3, 9:2, 10:1, 11:0, 12:0
}
DEFAULT_TEAMS = [
    "MAG ESPORTS","TEAM ELITE","TOTAL GAMING","DESI GAMERS",
    "BLIND ESPORTS","GODLIKE","ORANGUTAN","TSG ARMY",
    "GALAXY RACER","UB ESPORTS","EVL ESPORTS","NIGMA GALAXY"
]

st.set_page_config(
    page_title="MAG ESPORTS | FF Points Calculator",
    page_icon="🔥", layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"]{
  background:linear-gradient(135deg,#080810 0%,#0f0f1a 60%,#080810 100%);
  color:#e0e0e0;
}
[data-testid="stHeader"]{background:transparent;}
section[data-testid="stSidebar"]{display:none;}
.hero{
  background:linear-gradient(90deg,#1a0500,#FF4C29 35%,#FF6B00 65%,#1a0500);
  border-radius:14px; padding:30px 0 16px 0; text-align:center;
  margin-bottom:10px; box-shadow:0 0 50px #FF4C2955;
  border:1px solid #FF4C2933;
}
.hero-t{font-size:3rem;font-weight:900;color:#fff;
  letter-spacing:8px;text-transform:uppercase;
  text-shadow:0 0 30px #FF4C29,0 2px 6px #000;}
.hero-s{font-size:.9rem;color:#FFD700;letter-spacing:5px;
  text-transform:uppercase;margin-top:6px;}
.stab{
  background:#12122a; border:1px solid #FF4C2933;
  border-radius:12px; padding:20px; margin-bottom:14px;
}
.sh{
  font-size:1rem;font-weight:800;color:#FF4C29;
  text-transform:uppercase;letter-spacing:3px;
  border-left:4px solid #FF4C29;padding-left:10px;
  margin-bottom:12px;
}
.agg-table{width:100%;border-collapse:collapse;font-size:.93rem;}
.agg-table th{
  background:#FF4C29;color:#fff;padding:10px 12px;
  text-align:center;font-weight:800;letter-spacing:2px;
  text-transform:uppercase;font-size:.82rem;
}
.agg-table td{
  padding:9px 12px;text-align:center;
  border-bottom:1px solid #1e1e3a;
}
.agg-table tr:nth-child(even) td{background:#0f0f22;}
.agg-table tr:nth-child(odd)  td{background:#0a0a1a;}
.agg-table tr:hover td{background:#FF4C2915;}
.rg{color:#FFD700;font-weight:900;font-size:1.1rem;}
.rs{color:#C0C0C0;font-weight:900;}
.rb{color:#CD7F32;font-weight:900;}
.rn{color:#e0e0e0;font-weight:600;}
.tn{color:#FFD700;font-weight:700;text-align:left!important;}
.tc{color:#00FFFF;font-weight:900;font-size:1.05rem;}
.tk{color:#ff9944;font-weight:700;}
.tp{color:#94a3b8;}
.tm{color:#aaa;}
[data-testid="stButton"]>button{
  background:linear-gradient(90deg,#FF4C29,#FF6B00)!important;
  color:#fff!important;font-weight:800!important;
  border:none!important;border-radius:8px!important;
  box-shadow:0 0 18px #FF4C2944!important;
  transition:all .2s!important;
}
[data-testid="stButton"]>button:hover{
  box-shadow:0 0 30px #FF4C29bb!important;
  transform:translateY(-1px)!important;
}
[data-testid="stMetric"]{
  background:#12122a;border:1px solid #FF4C2933;
  border-radius:10px;padding:12px;
}
[data-testid="stMetricValue"]{color:#FFD700!important;font-weight:900;}
[data-testid="stMetricLabel"]{color:#94a3b8!important;}
hr{border-color:#FF4C2922!important;}
[data-testid="stFileUploader"]{
  border:2px dashed #FF4C2966!important;
  border-radius:10px!important;background:#0d0d1f!important;
}
[data-testid="stDataEditor"]{
  border:1px solid #FF4C2933!important;border-radius:8px!important;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════
def ss(k,v):
    if k not in st.session_state: st.session_state[k]=v

ss("teams",   DEFAULT_TEAMS.copy())          # 12 team names
ss("matches", [])                             # list of dicts {name, data: df}
ss("adding",  False)                          # adding new match?
ss("edit_idx",None)                           # which match to edit

# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════
def blank_match_df(teams):
    return pd.DataFrame({
        "Slot": list(range(1,13)),
        "Team": teams,
        "Rank": [0]*12,
        "Kills":[0]*12,
    })

def calc_df(df):
    df=df.copy()
    df["Rank"]  = pd.to_numeric(df["Rank"],  errors='coerce').fillna(0).astype(int)
    df["Kills"] = pd.to_numeric(df["Kills"], errors='coerce').fillna(0).astype(int)
    df["Kills"] = df["Kills"].clip(0,99)
    df["Rank"]  = df["Rank"].clip(0,12)
    df["Place Pts"] = df["Rank"].map(lambda r: PLACEMENT_POINTS.get(r,0))
    df["Kill Pts"]  = df["Kills"]
    df["Total Pts"] = df["Place Pts"] + df["Kill Pts"]
    return df

def build_aggregate():
    if not st.session_state.matches: return None
    rows=[]
    for m in st.session_state.matches:
        d=m["data"].copy()
        d["_match"]=m["name"]
        rows.append(d)
    all_data=pd.concat(rows,ignore_index=True)
    agg=all_data.groupby("Team",as_index=False).agg(
        Matches   =("Total Pts","count"),
        Total_Kill_Pts =("Kill Pts","sum"),
        Total_Place_Pts=("Place Pts","sum"),
        Total_Points   =("Total Pts","sum"),
        Total_Kills    =("Kills","sum"),
    ).sort_values(
        ["Total_Points","Total_Kills"],
        ascending=[False,False]
    ).reset_index(drop=True)
    agg["#"]=agg.index+1
    return agg

# ══════════════════════════════════════════════════════════════
# OCR (assist only)
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def load_ocr():
    import easyocr
    return easyocr.Reader(['en'],gpu=False)

def preprocess(pil_img):
    img=cv2.cvtColor(np.array(pil_img),cv2.COLOR_RGB2BGR)
    h,w=img.shape[:2]
    img=img[int(h*.12):int(h*.90), 0:int(w*.92)]
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gh,gw=gray.shape
    gray=cv2.resize(gray,(gw*2,gh*2),interpolation=cv2.INTER_CUBIC)
    gray=cv2.GaussianBlur(gray,(3,3),0)
    _,bw=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    if np.mean(bw)<127: bw=cv2.bitwise_not(bw)
    k=np.ones((2,2),np.uint8)
    bw=cv2.morphologyEx(bw,cv2.MORPH_CLOSE,k)
    orig=Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    proc=Image.fromarray(bw).convert("RGB")
    return proc, orig

def run_ocr_assist(pil_img, teams):
    """OCR → tries to fill rank+kills. Returns partial df for human to verify."""
    proc, orig = preprocess(pil_img)
    ocr = load_ocr()
    tmp="tmp_ocr.png"; proc.save(tmp)
    try:
        raw=ocr.readtext(tmp)
    except: raw=[]
    finally:
        if os.path.exists(tmp): os.remove(tmp)

    iw,ih=proc.size
    blocks=[]
    for item in raw:
        bbox,text,conf=item
        if conf<0.15: continue
        xs=[p[0] for p in bbox]; ys=[p[1] for p in bbox]
        cx=sum(xs)/4; cy=sum(ys)/4
        t=re.sub(r'[꧁꧂☠✦★༺༻]','',text).strip()
        if t: blocks.append({'text':t,'x':cx,'y':cy})

    left =sorted([b for b in blocks if b['x']<iw*.50],key=lambda b:b['y'])
    right=sorted([b for b in blocks if b['x']>=iw*.50],key=lambda b:b['y'])

    def extract_col(col,start_rank):
        if not col: return []
        ys=[b['y'] for b in col]
        tol=max((max(ys)-min(ys))*.055,12) if len(ys)>1 else 12
        rows,cur=[],[col[0]]
        for b in col[1:]:
            if abs(b['y']-cur[0]['y'])<=tol: cur.append(b)
            else: rows.append(cur); cur=[b]
        rows.append(cur)
        out=[]; rank=start_rank; i=0
        while i<len(rows) and rank<=(start_rank+5):
            combined=' '.join(b['text'] for r in rows[i:i+2] for b in r)
            kills=re.findall(r'(\d{1,2})\s*[Ee]liminat',combined)
            total_kills=min(sum(int(k) for k in kills),30)
            out.append({'rank':rank,'kills':total_kills,'raw':combined})
            rank+=1; i+=2
        return out

    left_data=extract_col(left,1)
    right_data=extract_col(right,6)
    all_data=sorted(left_data+right_data,key=lambda x:x['rank'])

    df=blank_match_df(teams)
    for item in all_data:
        idx=item['rank']-1
        if 0<=idx<12:
            df.at[idx,'Rank'] =item['rank']
            df.at[idx,'Kills']=item['kills']
    return df, proc, orig

# ══════════════════════════════════════════════════════════════
# UI — HERO
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div class="hero-t">🔥 MAG ESPORTS 🔥</div>
  <div class="hero-s">Free Fire · Points Calculator · Pro Edition</div>
</div>
""",unsafe_allow_html=True)

# ── LIVE STATS ──
agg_now=build_aggregate()
m1,m2,m3,m4=st.columns(4)
m1.metric("🎮 Matches",    len(st.session_state.matches))
m2.metric("👥 Teams",      len([t for t in st.session_state.teams if t]))
if agg_now is not None and not agg_now.empty:
    m3.metric("🏆 Leader",  str(agg_now.iloc[0]["Team"]))
    m4.metric("⚡ Points",  int(agg_now.iloc[0]["Total_Points"]))
else:
    m3.metric("🏆 Leader","—")
    m4.metric("⚡ Points","0")

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# SIDEBAR-STYLE LEFT COL — Team Setup
# ══════════════════════════════════════════════════════════════
left_col, right_col = st.columns([1,2], gap="large")

with left_col:
    st.markdown('<div class="sh">👥 Team Setup</div>',unsafe_allow_html=True)
    st.caption("Set all 12 team names. Slot = match result rank position.")
    tdf=pd.DataFrame({"Slot":list(range(1,13)),
                      "Team Name":st.session_state.teams})
    edited_t=st.data_editor(
        tdf, num_rows="fixed", use_container_width=True,
        column_config={
            "Slot":      st.column_config.NumberColumn("Slot",disabled=True,width="small"),
            "Team Name": st.column_config.TextColumn("Team Name"),
        }, key="team_ed", height=460
    )
    st.session_state.teams=edited_t["Team Name"].dropna().str.strip().tolist()
    teams=st.session_state.teams

    st.markdown("---")
    st.markdown('<div class="sh">🔥 Placement Points</div>',unsafe_allow_html=True)
    pts_df=pd.DataFrame({
        "Rank":  list(PLACEMENT_POINTS.keys()),
        "Points":[PLACEMENT_POINTS[r] for r in PLACEMENT_POINTS]
    })
    st.dataframe(pts_df, hide_index=True,
                 use_container_width=True, height=180)
    st.caption("Kill Points: 1 pt per kill")

# ══════════════════════════════════════════════════════════════
# RIGHT COL — Match Entry + Results
# ══════════════════════════════════════════════════════════════
with right_col:

    # ── ADD MATCH BUTTON ──
    hc1,hc2=st.columns([2,1])
    hc1.markdown('<div class="sh">📊 Match Manager</div>',
                 unsafe_allow_html=True)
    if hc2.button("➕ ADD NEW MATCH", use_container_width=True):
        st.session_state.adding=True
        st.session_state.edit_idx=None

    # ════════════════════════════════════════════
    # ADD / EDIT MATCH PANEL
    # ════════════════════════════════════════════
    if st.session_state.adding or st.session_state.edit_idx is not None:
        is_edit = st.session_state.edit_idx is not None
        panel_title = f"✏️ Edit Match {st.session_state.edit_idx+1}" \
                      if is_edit else "➕ Add New Match"

        with st.container():
            st.markdown(f'<div class="sh">{panel_title}</div>',
                        unsafe_allow_html=True)

            # Match name
            default_name = (st.session_state.matches[st.session_state.edit_idx]["name"]
                            if is_edit
                            else f"Match {len(st.session_state.matches)+1}")
            match_name=st.text_input("Match Name", default_name, key="mname")

            # OCR Assist toggle
            with st.expander("📸 OCR Assist — Auto-fill from Screenshot (optional)"):
                st.caption("Upload match result SS → OCR fills rank & kills. You MUST verify manually.")
                ss_file=st.file_uploader("Match Result Screenshot",
                                         type=['png','jpg','jpeg'],
                                         key="ss_assist")
                if ss_file:
                    raw_pil=Image.open(ss_file).convert("RGB")
                    if st.button("🤖 Run OCR Assist"):
                        with st.spinner("Running OCR..."):
                            ocr_df,proc_img,orig_img=run_ocr_assist(
                                raw_pil, teams)
                        c1,c2=st.columns(2)
                        c1.image(orig_img, caption="Original (cropped)",
                                 use_container_width=True)
                        c2.image(proc_img, caption="Pre-processed",
                                 use_container_width=True)
                        st.session_state["ocr_prefill"]=ocr_df
                        st.success("✅ OCR done! Data pre-filled below — VERIFY before saving.")

            # Entry table
            st.markdown("**Enter Rank & Kills for each team:**")
            st.caption("Rank = final placement in this match (1=Winner). Kills = total team kills.")

            if "ocr_prefill" in st.session_state and not is_edit:
                init_df=st.session_state["ocr_prefill"]
            elif is_edit:
                raw=st.session_state.matches[st.session_state.edit_idx]["data"]
                init_df=raw[["Slot","Team","Rank","Kills"]].copy()
            else:
                init_df=blank_match_df(teams)

            # Sync team names
            init_df["Team"]=teams[:12] if len(teams)>=12 else teams+[""]*(12-len(teams))

            entry=st.data_editor(
                init_df, num_rows="fixed",
                use_container_width=True,
                column_config={
                    "Slot": st.column_config.NumberColumn("Slot",disabled=True,width="small"),
                    "Team": st.column_config.TextColumn("Team",disabled=True),
                    "Rank": st.column_config.NumberColumn(
                        "Rank (1-12)", min_value=0, max_value=12,
                        step=1, width="small",
                        help="0 = did not finish / disqualified"),
                    "Kills":st.column_config.NumberColumn(
                        "Kills",min_value=0,max_value=99,
                        step=1,width="small"),
                }, key="match_entry", height=490
            )

            # Live preview
            preview=calc_df(entry)
            st.markdown("**Live Points Preview:**")
            st.dataframe(
                preview[["Team","Rank","Kills",
                          "Place Pts","Kill Pts","Total Pts"]],
                hide_index=True, use_container_width=True,
                column_config={
                    "Place Pts":st.column_config.NumberColumn(style="color:#94a3b8"),
                    "Kill Pts": st.column_config.NumberColumn(style="color:#ff9944"),
                    "Total Pts":st.column_config.NumberColumn(style="color:#00FFFF"),
                }
            )

            bc1,bc2,bc3=st.columns(3)
            if bc1.button("💾 SAVE MATCH",use_container_width=True):
                final=calc_df(entry)
                if is_edit:
                    st.session_state.matches[st.session_state.edit_idx]={
                        "name":match_name,"data":final}
                    st.session_state.edit_idx=None
                else:
                    st.session_state.matches.append(
                        {"name":match_name,"data":final})
                    st.session_state.adding=False
                if "ocr_prefill" in st.session_state:
                    del st.session_state["ocr_prefill"]
                st.success("✅ Match saved!")
                st.rerun()

            if bc2.button("❌ Cancel",use_container_width=True):
                st.session_state.adding=False
                st.session_state.edit_idx=None
                if "ocr_prefill" in st.session_state:
                    del st.session_state["ocr_prefill"]
                st.rerun()

            if is_edit:
                if bc3.button("🗑️ Delete Match",use_container_width=True):
                    st.session_state.matches.pop(st.session_state.edit_idx)
                    st.session_state.edit_idx=None
                    st.rerun()

    st.markdown("---")

    # ════════════════════════════════════════════
    # SAVED MATCHES LIST
    # ════════════════════════════════════════════
    if st.session_state.matches:
        st.markdown('<div class="sh">📋 Saved Matches</div>',
                    unsafe_allow_html=True)
        for mi,m in enumerate(st.session_state.matches):
            d=m["data"]
            winner=d.loc[d["Rank"]==1,"Team"].values
            wname=winner[0] if len(winner) else "—"
            total_kills=int(d["Kills"].sum())
            mc1,mc2,mc3,mc4=st.columns([2,2,1,1])
            mc1.markdown(f"**{m['name']}**")
            mc2.caption(f"🏆 Winner: {wname} | 💀 Total Kills: {total_kills}")
            if mc3.button("✏️",key=f"edit_{mi}",use_container_width=True):
                st.session_state.edit_idx=mi
                st.session_state.adding=False
                st.rerun()
            if mc4.button("🗑️",key=f"del_{mi}",use_container_width=True):
                st.session_state.matches.pop(mi)
                st.rerun()
            with st.expander(f"View {m['name']} details"):
                st.dataframe(
                    d[["Slot","Team","Rank","Kills",
                       "Place Pts","Kill Pts","Total Pts"]],
                    hide_index=True, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# AGGREGATE TABLE
# ══════════════════════════════════════════════════════════════
if st.session_state.matches:
    st.markdown("---")
    st.markdown('<div class="sh">🏆 AGGREGATE POINTS TABLE</div>',
                unsafe_allow_html=True)
    st.caption(f"Combined from {len(st.session_state.matches)} match(es) | "
               "Tiebreaker: Total Kills")

    agg=build_aggregate()

    # Per-match breakdown columns
    all_rows=[]
    for team in teams:
        row={"Team":team}
        grand_kills=0; grand_place=0; grand_total=0
        for mi,m in enumerate(st.session_state.matches):
            t_row=m["data"][m["data"]["Team"]==team]
            if not t_row.empty:
                tp=int(t_row["Total Pts"].values[0])
                tk=int(t_row["Kills"].values[0])
                tpl=int(t_row["Place Pts"].values[0])
                row[f"M{mi+1}"]=tp
                grand_kills+=tk; grand_place+=tpl; grand_total+=tp
            else:
                row[f"M{mi+1}"]=0
        row["Total Kills"]=grand_kills
        row["Place Pts"]=grand_place
        row["TOTAL"]=grand_total
        all_rows.append(row)

    full_df=(pd.DataFrame(all_rows)
             .sort_values(["TOTAL","Total Kills"],ascending=[False,False])
             .reset_index(drop=True))
    full_df.insert(0,"#",full_df.index+1)

    # HTML table
    rank_cls={1:"rg",2:"rs",3:"rb"}
    match_headers="".join(
        f"<th>M{i+1}</th>" for i in range(len(st.session_state.matches)))
    rows_html=""
    for _,row in full_df.iterrows():
        rn=int(row["#"])
        rc=rank_cls.get(rn,"rn")
        medal={1:"🥇",2:"🥈",3:"🥉"}.get(rn,str(rn))
        match_tds="".join(
            f"<td class='tm'>{int(row.get(f'M{i+1}',0))}</td>"
            for i in range(len(st.session_state.matches)))
        rows_html+=f"""<tr>
          <td class="{rc}">{medal}</td>
          <td class="tn">{row['Team']}</td>
          {match_tds}
          <td class="tk">{int(row['Total Kills'])}</td>
          <td class="tp">{int(row['Place Pts'])}</td>
          <td class="tc">{int(row['TOTAL'])}</td>
        </tr>"""

    st.markdown(f"""
    <div style="overflow-x:auto;">
    <table class="agg-table">
      <thead><tr>
        <th>#</th><th style="text-align:left">TEAM</th>
        {match_headers}
        <th>KILLS</th><th>PLACE PTS</th><th>TOTAL PTS</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    </div>
    """,unsafe_allow_html=True)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════
    # IMAGE GENERATOR
    # ══════════════════════════════════════════════════════════
    st.markdown('<div class="sh">🎨 GENERATE LEADERBOARD IMAGE</div>',
                unsafe_allow_html=True)
    gc1,gc2,gc3=st.columns(3)
    t_name=gc1.text_input("Tournament Name","MAG ESPORTS OPEN")
    t_day =gc2.text_input("Day / Round","Day 1 — Overall Standings")
    t_org =gc3.text_input("Organizer","MAG ESPORTS")

    if st.button("🏆 GENERATE FINAL LEADERBOARD IMAGE",
                 use_container_width=True):
        W,H=1200,1900
        img=Image.new("RGB",(W,H),(7,7,16))
        draw=ImageDraw.Draw(img)

        # Grid bg
        for y in range(0,H,55):
            draw.line([(0,y),(W,y)],fill=(18,18,32),width=1)
        for x in range(0,W,55):
            draw.line([(x,0),(x,H)],fill=(18,18,32),width=1)

        try:
            fb =ImageFont.truetype("arialbd.ttf",74)
            ft =ImageFont.truetype("arialbd.ttf",40)
            fs =ImageFont.truetype("arial.ttf",  28)
            fh =ImageFont.truetype("arialbd.ttf",24)
            fr =ImageFont.truetype("arial.ttf",  28)
            frb=ImageFont.truetype("arialbd.ttf",30)
        except:
            fb=ft=fs=fh=fr=frb=ImageFont.load_default()

        # Top bar
        draw.rectangle([(0,0),(W,115)],fill=(220,60,30))
        for xi in range(0,W+120,45):
            draw.polygon([(xi,0),(xi+35,0),(xi+12,115),(xi-23,115)],
                         fill=(255,107,0))
        draw.text((W//2,57),"MAG ESPORTS",fill="white",
                  font=fb,anchor="mm")

        # Sub bars
        draw.rectangle([(0,115),(W,178)],fill=(20,20,45))
        draw.text((W//2,146),t_name.upper(),
                  fill="#FFD700",font=ft,anchor="mm")
        draw.rectangle([(0,178),(W,218)],fill=(13,13,30))
        draw.text((W//2,198),t_day,fill="#94a3b8",font=fs,anchor="mm")
        draw.text((20,198),f"Matches: {len(st.session_state.matches)}",
                  fill="#FF4C29",font=fs,anchor="lm")

        # Col headers
        nm=len(st.session_state.matches)
        SY=245; ROW=105
        COLS={"#":40,"TEAM":120,"KILLS":820,"PLACE":940,"TOTAL":1080}
        mcol_start=680; mcol_w=max(1,min(130, (COLS["KILLS"]-mcol_start)//max(nm,1)))

        draw.rectangle([(20,SY),(W-20,SY+46)],fill=(220,60,30))
        draw.text((COLS["#"]+5,SY+23),"#",fill="white",font=fh,anchor="lm")
        draw.text((COLS["TEAM"],SY+23),"TEAM",fill="white",font=fh,anchor="lm")
        for mi in range(nm):
            draw.text((mcol_start+mi*mcol_w+mcol_w//2,SY+23),
                      f"M{mi+1}",fill="white",font=fh,anchor="mm")
        draw.text((COLS["KILLS"],SY+23),"KILLS",fill="white",font=fh,anchor="lm")
        draw.text((COLS["PLACE"],SY+23),"PLACE",fill="white",font=fh,anchor="lm")
        draw.text((COLS["TOTAL"],SY+23),"TOTAL",fill="white",font=fh,anchor="lm")
        draw.line([(20,SY+47),(W-20,SY+47)],fill="#FFD700",width=2)

        rc_map={1:(255,215,0),2:(192,192,192),3:(205,127,50)}
        medals={1:"1ST",2:"2ND",3:"3RD"}

        for idx,row in full_df.iterrows():
            y=SY+58+idx*ROW
            if y+ROW>H-100: break
            bg=(14,14,32) if idx%2==0 else (10,10,24)
            draw.rectangle([(20,y-6),(W-20,y+ROW-14)],fill=bg)
            rn=int(row["#"])
            rc=rc_map.get(rn,(180,180,200))
            if rn<=3:
                draw.rectangle([(20,y-6),(30,y+ROW-14)],fill=rc)
            medal=medals.get(rn,f"#{rn}")
            draw.text((COLS["#"]+5,y+ROW//2-8),medal,fill=rc,font=frb,anchor="lm")
            draw.text((COLS["TEAM"],y+ROW//2-8),str(row["Team"]),
                      fill=(255,215,0),font=frb,anchor="lm")
            for mi in range(nm):
                mv=int(row.get(f"M{mi+1}",0))
                mc=(0,255,180) if mv>0 else (80,80,100)
                draw.text((mcol_start+mi*mcol_w+mcol_w//2,y+ROW//2-8),
                          str(mv),fill=mc,font=fr,anchor="mm")
            draw.text((COLS["KILLS"],y+ROW//2-8),
                      str(int(row["Total Kills"])),
                      fill=(255,153,68),font=frb,anchor="lm")
            draw.text((COLS["PLACE"],y+ROW//2-8),
                      str(int(row["Place Pts"])),
                      fill=(148,163,184),font=fr,anchor="lm")
            draw.text((COLS["TOTAL"],y+ROW//2-8),
                      str(int(row["TOTAL"])),
                      fill=(0,255,255),font=frb,anchor="lm")
            draw.line([(20,y+ROW-15),(W-20,y+ROW-15)],
                      fill=(25,25,50),width=1)

        # Footer
        draw.rectangle([(0,H-70),(W,H)],fill=(220,60,30))
        draw.text((W//2,H-35),
                  f"{t_org} · FREE FIRE POINTS CALCULATOR",
                  fill="white",font=fs,anchor="mm")

        st.image(img,use_container_width=True)
        buf=io.BytesIO(); img.save(buf,format='PNG')
        st.download_button("⬇️ DOWNLOAD LEADERBOARD",
                           buf.getvalue(),
                           "mag_leaderboard.png","image/png",
                           use_container_width=True)

    st.markdown("---")
    if st.button("🗑️ Clear ALL Data",type="secondary"):
        st.session_state.matches=[]
        st.rerun()
