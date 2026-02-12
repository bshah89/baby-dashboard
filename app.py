import json
from datetime import datetime, date, time, timedelta

import pandas as pd
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials


# =========================================================
# Page
# =========================================================
st.set_page_config(page_title="New Parent Hub", page_icon="👶🏻", layout="wide")


# =========================================================
# 4 digit PIN lock gate
# Secrets required
# [app_security]
# pin = "1234"
# =========================================================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False


def pin_gate():
    st.markdown("<div class='container'>", unsafe_allow_html=True)
    st.markdown("## 🔒 Baby Hub")
    st.caption("Enter 4 digit PIN to continue")
    pin_input = st.text_input("PIN", type="password", max_chars=4, placeholder="••••")
    if pin_input:
        expected = str(st.secrets["app_security"]["pin"]).strip()
        if pin_input.strip() == expected:
            st.session_state.authenticated = True
            st.success("Unlocked")
            st.rerun()
        else:
            st.error("Incorrect PIN")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()


if not st.session_state.authenticated:
    pin_gate()


# =========================================================
# Style, mobile first
# =========================================================
CSS = """
<style>
  .container {
    max-width: 460px;
    margin: 0 auto;
    padding-bottom: 120px;
  }

  .title {
    font-size: 30px;
    font-weight: 900;
    letter-spacing: -0.02em;
    margin-bottom: 6px;
  }

  .subtle { color: rgba(0,0,0,0.55); font-size: 13px; line-height: 1.35; }
  @media (prefers-color-scheme: dark) {
    .subtle { color: rgba(255,255,255,0.65); }
  }

  .card {
    border: 1px solid rgba(0,0,0,0.08);
    border-radius: 18px;
    padding: 14px 14px;
    background: rgba(255,255,255,0.72);
    box-shadow: 0 10px 24px rgba(0,0,0,0.06);
    margin-bottom: 12px;
  }
  @media (prefers-color-scheme: dark) {
    .card {
      border: 1px solid rgba(255,255,255,0.10);
      background: rgba(20,20,22,0.55);
      box-shadow: 0 10px 24px rgba(0,0,0,0.35);
    }
  }

  .kpiGrid2 {
    display: grid;
    grid-template-columns: repeat(2, minmax(0,1fr));
    gap: 10px;
  }

  .kpiGrid3 {
    display: grid;
    grid-template-columns: repeat(3, minmax(0,1fr));
    gap: 10px;
  }

  .kpi {
    border-radius: 18px;
    padding: 12px 12px;
    background: rgba(0,0,0,0.03);
    border: 1px solid rgba(0,0,0,0.06);
  }
  @media (prefers-color-scheme: dark) {
    .kpi {
      background: rgba(255,255,255,0.06);
      border: 1px solid rgba(255,255,255,0.10);
    }
  }

  .kpiLabel { font-size: 12px; opacity: 0.78; font-weight: 800; }
  .kpiValue { font-size: 20px; font-weight: 950; margin-top: 2px; }

  div[data-baseweb="input"] > div { border-radius: 14px !important; }
  button { border-radius: 16px !important; }

  /* Slightly reduce top padding for mobile feel */
  section.main > div { padding-top: 18px; }

</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# =========================================================
# Helpers
# =========================================================
def is_missing(v) -> bool:
    try:
        return v is None or pd.isna(v)
    except Exception:
        return v is None


def safe_str(v) -> str:
    return "" if is_missing(v) else str(v)


def safe_lower(v) -> str:
    return safe_str(v).strip().lower()


def now_floor() -> datetime:
    n = datetime.now()
    return n.replace(second=0, microsecond=0)


def make_dt(d: date, t: time) -> datetime:
    return datetime.combine(d, t).replace(second=0, microsecond=0)


def new_row_id() -> str:
    return f"r{int(datetime.now().timestamp() * 1000)}"


def time_ago(ts: pd.Timestamp) -> str:
    if is_missing(ts):
        return "—"
    mins = int((pd.Timestamp.now() - ts).total_seconds() // 60)
    if mins < 1:
        return "just now"
    if mins < 60:
        return f"{mins}m ago"
    hrs = mins // 60
    rem = mins % 60
    if hrs < 24:
        return f"{hrs}h {rem}m ago"
    days = hrs // 24
    hrs2 = hrs % 24
    return f"{days}d {hrs2}h ago"


def within_night(ts: pd.Timestamp, night_start: time, night_end: time) -> bool:
    if is_missing(ts):
        return False
    t = ts.to_pydatetime().time()
    if night_start < night_end:
        return night_start <= t < night_end
    return t >= night_start or t < night_end


def timeline_label(row: pd.Series) -> str:
    dt_txt = ""
    if "datetime" in row and not is_missing(row["datetime"]):
        try:
            dt_txt = pd.to_datetime(row["datetime"]).strftime("%H:%M")
        except Exception:
            dt_txt = ""

    et = safe_lower(row.get("event_type"))
    fm = safe_lower(row.get("feed_method"))

    if et == "feed":
        if fm == "bottle":
            ml = pd.to_numeric(row.get("volume_ml"), errors="coerce")
            ml = int(ml) if not is_missing(ml) else 0
            return f"{dt_txt} 🍼 Bottle {ml} ml"
        mins = pd.to_numeric(row.get("duration_min"), errors="coerce")
        mins = int(mins) if not is_missing(mins) else 0
        side = safe_lower(row.get("side"))
        side_txt = f" {side.title()}" if side in ["left", "right"] else ""
        return f"{dt_txt} 🤱 Breast {mins} min{side_txt}"

    if et == "diaper":
        dtp = safe_lower(row.get("diaper_type"))
        icon = "💩" if dtp in ["dirty", "mixed"] else "💧"
        label = dtp.title() if dtp else "Diaper"
        return f"{dt_txt} {icon} {label}"

    if et == "growth":
        w = pd.to_numeric(row.get("weight_kg"), errors="coerce")
        h = pd.to_numeric(row.get("height_cm"), errors="coerce")
        parts = []
        if not is_missing(w):
            parts.append(f"{float(w):.2f}kg")
        if not is_missing(h):
            parts.append(f"{float(h):.1f}cm")
        return f"{dt_txt} 📈 Growth " + " ".join(parts)

    if et == "immunisation":
        title = safe_str(row.get("title")).strip()
        return f"{dt_txt} 💉 {title}".strip()

    if et == "milestone":
        title = safe_str(row.get("title")).strip()
        return f"{dt_txt} ✨ {title}".strip()

    if et == "medication":
        title = safe_str(row.get("title")).strip()
        return f"{dt_txt} 💊 {title}".strip()

    if et == "journal":
        mood = safe_str(row.get("mood")).strip()
        title = safe_str(row.get("title")).strip()
        return f"{dt_txt} {mood if mood else '📝'} {title if title else 'Journal'}".strip()

    return f"{dt_txt} Event"


def kpi_block(label: str, value: str):
    st.markdown(
        f"<div class='kpi'><div class='kpiLabel'>{label}</div><div class='kpiValue'>{value}</div></div>",
        unsafe_allow_html=True,
    )


# =========================================================
# Google Sheets
# =========================================================
LOG_REQUIRED_COLS = [
    "row_id",
    "datetime",
    "date",
    "time",
    "event_type",
    "feed_method",
    "side",
    "volume_ml",
    "duration_min",
    "diaper_type",
    "notes",
    "weight_kg",
    "height_cm",
    "mood",
    "title",
    "data_json",
]

CONFIG_SHEET = "Config"  # key, value


@st.cache_resource
def get_spreadsheet_and_ws():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scopes)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(st.secrets["gsheets"]["spreadsheet_id"])
    log_ws = sh.worksheet(st.secrets["gsheets"]["worksheet_name"])
    return sh, log_ws


def ws_headers(ws) -> list[str]:
    h = ws.row_values(1)
    return [str(x).strip() for x in h if str(x).strip()]


def ensure_log_headers():
    _, ws = get_spreadsheet_and_ws()
    h = ws_headers(ws)
    if not h:
        ws.update("A1", [LOG_REQUIRED_COLS])
        return LOG_REQUIRED_COLS
    missing = [c for c in LOG_REQUIRED_COLS if c not in h]
    if missing:
        ws.update("A1", [h + missing])
        return h + missing
    return h


def get_or_create_config_ws():
    sh, _ = get_spreadsheet_and_ws()
    try:
        ws = sh.worksheet(CONFIG_SHEET)
    except Exception:
        ws = sh.add_worksheet(title=CONFIG_SHEET, rows=50, cols=2)
        ws.update("A1:B1", [["key", "value"]])
        ws.update("A2:B6", [
            ["baby_name", "Baby Girl"],
            ["baby_dob", ""],
            ["night_start", "22:00"],
            ["night_end", "06:00"],
        ])
    # ensure header
    hv = ws.row_values(1)
    if not hv or [x.strip().lower() for x in hv[:2]] != ["key", "value"]:
        ws.update("A1:B1", [["key", "value"]])
    return ws


@st.cache_data(ttl=20)
def load_config(_rk: int) -> dict:
    ws = get_or_create_config_ws()
    rows = ws.get_all_values()
    cfg = {}
    for r in rows[1:]:
        if len(r) < 2:
            continue
        k = str(r[0]).strip()
        v = str(r[1]).strip()
        if k:
            cfg[k] = v
    return cfg


def save_config_value(key: str, value: str) -> None:
    ws = get_or_create_config_ws()
    rows = ws.get_all_values()
    # find row
    target_row = None
    for i, r in enumerate(rows[1:], start=2):
        if len(r) >= 1 and str(r[0]).strip() == key:
            target_row = i
            break
    if target_row is None:
        ws.append_row([key, value], value_input_option="USER_ENTERED")
    else:
        ws.update(f"B{target_row}", [[value]])


@st.cache_data(ttl=20)
def ws_read_log(_rk: int) -> pd.DataFrame:
    _, ws = get_spreadsheet_and_ws()
    headers = ensure_log_headers()
    records = ws.get_all_records()
    df = pd.DataFrame(records)

    for c in headers:
        if c not in df.columns:
            df[c] = pd.NA

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).copy()

    df["date"] = df["datetime"].dt.date
    df["time"] = df["datetime"].dt.strftime("%H:%M")

    for c in ["volume_ml", "duration_min", "weight_kg", "height_cm"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["event_type", "feed_method", "side", "diaper_type"]:
        df[c] = df[c].astype("string").str.strip().str.lower()

    df["row_id"] = df["row_id"].astype("string").fillna("").astype(str)
    df["notes"] = df["notes"].astype("string").fillna("")
    df["title"] = df["title"].astype("string").fillna("")
    df["mood"] = df["mood"].astype("string").fillna("")

    return df.sort_values("datetime").reset_index(drop=True)


def ws_append_log(row: dict):
    _, ws = get_spreadsheet_and_ws()
    headers = ensure_log_headers()
    ws.append_row([row.get(c, "") for c in headers], value_input_option="USER_ENTERED")


def find_rownum_by_row_id(rid: str) -> int | None:
    _, ws = get_spreadsheet_and_ws()
    headers = ws_headers(ws)
    if not headers:
        return None
    try:
        col_idx = headers.index("row_id") + 1
    except ValueError:
        col_idx = 1
    col_values = ws.col_values(col_idx)
    for i, v in enumerate(col_values[1:], start=2):
        if str(v).strip() == str(rid).strip():
            return i
    return None


def ws_update_log(rid: str, row: dict) -> bool:
    _, ws = get_spreadsheet_and_ws()
    headers = ensure_log_headers()
    rnum = find_rownum_by_row_id(rid)
    if rnum is None:
        return False
    end_col = len(headers)
    rng = gspread.utils.rowcol_to_a1(rnum, 1) + ":" + gspread.utils.rowcol_to_a1(rnum, end_col)
    ws.update(rng, [[row.get(c, "") for c in headers]])
    return True


def ws_delete_log(rid: str) -> bool:
    _, ws = get_spreadsheet_and_ws()
    rnum = find_rownum_by_row_id(rid)
    if rnum is None:
        return False
    ws.delete_rows(rnum)
    return True


# =========================================================
# Row builder
# =========================================================
def base_row(dt: datetime) -> dict:
    return {
        "row_id": new_row_id(),
        "datetime": dt.strftime("%Y-%m-%d %H:%M"),
        "date": dt.strftime("%Y-%m-%d"),
        "time": dt.strftime("%H:%M"),
        "event_type": "",
        "feed_method": "",
        "side": "",
        "volume_ml": "",
        "duration_min": "",
        "diaper_type": "",
        "notes": "",
        "weight_kg": "",
        "height_cm": "",
        "mood": "",
        "title": "",
        "data_json": "",
    }


def refresh():
    st.session_state.refresh_key += 1
    st.cache_data.clear()
    st.rerun()


# =========================================================
# State
# =========================================================
if "refresh_key" not in st.session_state:
    st.session_state.refresh_key = 0

if "bf_running" not in st.session_state:
    st.session_state.bf_running = False
if "bf_segments" not in st.session_state:
    st.session_state.bf_segments = []


# =========================================================
# Load data
# =========================================================
cfg = load_config(st.session_state.refresh_key)
baby_name = cfg.get("baby_name", "Baby Girl").strip() or "Baby Girl"
baby_dob = cfg.get("baby_dob", "").strip()

# night defaults
def parse_hhmm(s: str, default_t: time) -> time:
    try:
        s = (s or "").strip()
        if not s:
            return default_t
        hh, mm = s.split(":")
        return time(int(hh), int(mm))
    except Exception:
        return default_t

night_start = parse_hhmm(cfg.get("night_start", "22:00"), time(22, 0))
night_end = parse_hhmm(cfg.get("night_end", "06:00"), time(6, 0))

df_all = ws_read_log(st.session_state.refresh_key)


# =========================================================
# Always available lock, top
# =========================================================
st.markdown("<div class='container'>", unsafe_allow_html=True)

topA, topB = st.columns([1, 1])
with topA:
    st.markdown(f"<div class='title'>👶🏻 {baby_name}</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtle'>Mobile first logging, syncs across iPhone and Mac</div>", unsafe_allow_html=True)
with topB:
    if st.button("🔒 Lock", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()

st.write("")

# =========================================================
# Navigation (keeps your UI vibe)
# =========================================================
nav = st.radio(
    "Navigation",
    ["Home", "Feed", "Diaper", "Growth", "Insights", "More", "History"],
    horizontal=True,
    label_visibility="collapsed",
)

st.write("")


# =========================================================
# HOME
# =========================================================
if nav == "Home":
    today = datetime.now().date()
    today_df = df_all[df_all["date"] == today].copy() if len(df_all) else df_all.copy()

    feeds = today_df[today_df["event_type"] == "feed"].copy()
    diapers = today_df[today_df["event_type"] == "diaper"].copy()
    bottle = feeds[feeds["feed_method"] == "bottle"].copy()
    breast = feeds[feeds["feed_method"] == "breast"].copy()

    last_feed = feeds["datetime"].max() if len(feeds) else pd.NaT
    last_diaper = diapers["datetime"].max() if len(diapers) else pd.NaT

    bottle_ml = int(bottle["volume_ml"].fillna(0).sum()) if len(bottle) else 0
    breast_min = int(breast["duration_min"].fillna(0).sum()) if len(breast) else 0

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='kpiGrid2'>", unsafe_allow_html=True)
    kpi_block("Since last feed", time_ago(last_feed))
    kpi_block("Since last diaper", time_ago(last_diaper))
    kpi_block("Feeds today", str(int(len(feeds))))
    kpi_block("Diapers today", str(int(len(diapers))))
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown("<div class='kpiGrid2'>", unsafe_allow_html=True)
    kpi_block("Bottle ml today", str(bottle_ml))
    kpi_block("Breast min today", str(breast_min))
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Quick actions")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🍼 Bottle 15", use_container_width=True):
            row = base_row(now_floor())
            row["event_type"] = "feed"
            row["feed_method"] = "bottle"
            row["volume_ml"] = 15
            ws_append_log(row)
            st.toast("Bottle 15 logged", icon="✅")
            refresh()
    with c2:
        if st.button("🍼 Bottle 30", use_container_width=True):
            row = base_row(now_floor())
            row["event_type"] = "feed"
            row["feed_method"] = "bottle"
            row["volume_ml"] = 30
            ws_append_log(row)
            st.toast("Bottle 30 logged", icon="✅")
            refresh()
    with c3:
        if st.button("💧 Wet diaper", use_container_width=True):
            row = base_row(now_floor())
            row["event_type"] = "diaper"
            row["diaper_type"] = "wet"
            ws_append_log(row)
            st.toast("Wet logged", icon="✅")
            refresh()
    st.caption("If you need to edit or delete, use History.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Recent")
    recent = df_all.sort_values("datetime", ascending=False).head(8).copy()
    if len(recent) == 0:
        st.caption("No events yet. Start with Feed or Diaper.")
    else:
        for _, r in recent.iterrows():
            st.markdown(f"**{timeline_label(r)}**")
            notes = safe_str(r.get("notes")).strip()
            if notes:
                st.caption(notes)
    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# FEED
# =========================================================
elif nav == "Feed":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🍼 Feeding")
    mode = st.radio("Mode", ["Bottle", "Breast timer"], horizontal=True, label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

    if mode == "Bottle":
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Quick bottle")
        g1 = st.columns(3)
        with g1[0]:
            if st.button("15 ml", use_container_width=True):
                row = base_row(now_floor())
                row["event_type"] = "feed"
                row["feed_method"] = "bottle"
                row["volume_ml"] = 15
                ws_append_log(row)
                st.toast("Bottle 15 logged", icon="✅")
                refresh()
        with g1[1]:
            if st.button("30 ml", use_container_width=True):
                row = base_row(now_floor())
                row["event_type"] = "feed"
                row["feed_method"] = "bottle"
                row["volume_ml"] = 30
                ws_append_log(row)
                st.toast("Bottle 30 logged", icon="✅")
                refresh()
        with g1[2]:
            ml = st.number_input("Custom ml", min_value=0, max_value=300, value=0, step=5)
            if st.button("Log custom", use_container_width=True, disabled=(ml <= 0)):
                row = base_row(now_floor())
                row["event_type"] = "feed"
                row["feed_method"] = "bottle"
                row["volume_ml"] = int(ml)
                ws_append_log(row)
                st.toast(f"Bottle {int(ml)} logged", icon="✅")
                refresh()
        st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Breast timer")
        st.caption("Start, switch sides, stop, then save. Discard saves nothing.")

        def end_open_segment():
            if st.session_state.bf_segments and st.session_state.bf_segments[-1]["end"] is None:
                st.session_state.bf_segments[-1]["end"] = now_floor()

        def start_segment(side_name: str):
            st.session_state.bf_segments.append({"side": side_name, "start": now_floor(), "end": None})

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("▶️ Start", use_container_width=True):
                if not st.session_state.bf_running:
                    st.session_state.bf_running = True
                    st.session_state.bf_segments = [{"side": "left", "start": now_floor(), "end": None}]
                    st.toast("Started on left", icon="⏱️")
                st.rerun()
        with c2:
            if st.button("⏹️ Stop", use_container_width=True, disabled=(not st.session_state.bf_running)):
                end_open_segment()
                st.session_state.bf_running = False
                st.rerun()
        with c3:
            if st.button("🧽 Discard", use_container_width=True):
                st.session_state.bf_running = False
                st.session_state.bf_segments = []
                st.toast("Discarded", icon="🧽")
                st.rerun()

        if st.session_state.bf_segments:
            segs = st.session_state.bf_segments
            start_dt = segs[0]["start"]
            current_side = segs[-1]["side"]
            elapsed_min = int((datetime.now() - start_dt).total_seconds() // 60)

            st.markdown(f"**Started:** {start_dt.strftime('%H:%M')}  \n**Side:** {current_side.title()}  \n**Elapsed:** {elapsed_min} min")

            s1, s2 = st.columns(2)
            with s1:
                if st.button("Switch left", use_container_width=True, disabled=(not st.session_state.bf_running)):
                    end_open_segment()
                    start_segment("left")
                    st.rerun()
            with s2:
                if st.button("Switch right", use_container_width=True, disabled=(not st.session_state.bf_running)):
                    end_open_segment()
                    start_segment("right")
                    st.rerun()

            rows = []
            total_min = 0
            for s in segs:
                stt = s["start"]
                endt = s["end"] if s["end"] is not None else now_floor()
                mins = max(int((endt - stt).total_seconds() // 60), 0)
                total_min += mins
                rows.append({"side": s["side"], "start": stt.strftime("%H:%M"), "end": endt.strftime("%H:%M"), "min": mins})

            st.markdown(f"**Total:** {total_min} min")

            if st.toggle("Show breakdown", value=False):
                st.dataframe(pd.DataFrame(rows), use_container_width=True, height=220)

            if st.button("✅ Save breast feed", use_container_width=True, disabled=(total_min <= 0)):
                timer_end = rows[-1]["end"]
                breakdown = ", ".join([f"{r['side'][0].upper()}:{r['min']}m {r['start']}-{r['end']}" for r in rows])
                notes = f"Start {start_dt.strftime('%H:%M')} End {timer_end} | {breakdown}"

                row = base_row(start_dt)
                row["event_type"] = "feed"
                row["feed_method"] = "breast"
                row["duration_min"] = total_min
                row["side"] = current_side
                row["notes"] = notes

                ws_append_log(row)
                st.toast("Breast feed saved", icon="✅")
                st.session_state.bf_running = False
                st.session_state.bf_segments = []
                refresh()

        st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# DIAPER
# =========================================================
elif nav == "Diaper":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("💧 Diaper")
    st.caption("One tap logging")

    g = st.columns(3)
    with g[0]:
        if st.button("💧 Wet", use_container_width=True):
            row = base_row(now_floor())
            row["event_type"] = "diaper"
            row["diaper_type"] = "wet"
            ws_append_log(row)
            st.toast("Wet logged", icon="✅")
            refresh()
    with g[1]:
        if st.button("💩 Dirty", use_container_width=True):
            row = base_row(now_floor())
            row["event_type"] = "diaper"
            row["diaper_type"] = "dirty"
            ws_append_log(row)
            st.toast("Dirty logged", icon="✅")
            refresh()
    with g[2]:
        if st.button("💩 Mixed", use_container_width=True):
            row = base_row(now_floor())
            row["event_type"] = "diaper"
            row["diaper_type"] = "mixed"
            ws_append_log(row)
            st.toast("Mixed logged", icon="✅")
            refresh()

    st.markdown("</div>", unsafe_allow_html=True)

    today = datetime.now().date()
    today_df = df_all[(df_all["date"] == today) & (df_all["event_type"] == "diaper")].copy()
    wet_count = int((today_df["diaper_type"].isin(["wet", "mixed"])).sum()) if len(today_df) else 0
    dirty_count = int((today_df["diaper_type"].isin(["dirty", "mixed"])).sum()) if len(today_df) else 0

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Today")
    st.markdown("<div class='kpiGrid2'>", unsafe_allow_html=True)
    kpi_block("Wet count", str(wet_count))
    kpi_block("Dirty count", str(dirty_count))
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# GROWTH
# =========================================================
elif nav == "Growth":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📈 Growth")
    st.caption("Log weight and height, see trend")

    c1, c2 = st.columns(2)
    with c1:
        w = st.number_input("Weight kg", min_value=0.0, max_value=25.0, value=0.0, step=0.01, format="%.2f")
    with c2:
        h = st.number_input("Height cm", min_value=0.0, max_value=120.0, value=0.0, step=0.1, format="%.1f")

    if st.button("✅ Log growth", use_container_width=True, disabled=(w <= 0 and h <= 0)):
        row = base_row(now_floor())
        row["event_type"] = "growth"
        if w > 0:
            row["weight_kg"] = float(w)
        if h > 0:
            row["height_cm"] = float(h)
        ws_append_log(row)
        st.toast("Growth logged", icon="✅")
        refresh()

    st.markdown("</div>", unsafe_allow_html=True)

    gdf = df_all[df_all["event_type"] == "growth"].copy().sort_values("datetime")
    if len(gdf) == 0:
        st.markdown("<div class='card'>No growth entries yet.</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Charts")
        chart = gdf[["datetime", "weight_kg", "height_cm"]].copy()
        chart["date"] = chart["datetime"].dt.date
        daily = chart.groupby("date")[["weight_kg", "height_cm"]].mean().reset_index()
        daily = daily.rename(columns={"date": "index"}).set_index("index")
        daily.index = pd.to_datetime(daily.index)

        if daily["weight_kg"].notna().any():
            st.caption("Weight")
            st.line_chart(daily["weight_kg"])
        if daily["height_cm"].notna().any():
            st.caption("Height")
            st.line_chart(daily["height_cm"])

        if st.toggle("Show table", value=False):
            st.dataframe(
                gdf.sort_values("datetime", ascending=False)[["datetime", "weight_kg", "height_cm", "notes", "row_id"]],
                use_container_width=True,
                height=320,
            )
        st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# INSIGHTS
# =========================================================
elif nav == "Insights":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Insights")
    st.caption("Trends and patterns, layered and calm. Expand only what you need.")
    st.markdown("</div>", unsafe_allow_html=True)

    if len(df_all) == 0:
        st.markdown("<div class='card'>No data yet. Start logging feeds and diapers.</div>", unsafe_allow_html=True)
    else:
        now_ts = pd.Timestamp.now()
        today = now_ts.date()
        last7_cut = today - timedelta(days=6)
        last14_cut = today - timedelta(days=13)

        feeds_all = df_all[df_all["event_type"] == "feed"].copy().sort_values("datetime")
        diapers_all = df_all[df_all["event_type"] == "diaper"].copy().sort_values("datetime")
        bottle_all = feeds_all[feeds_all["feed_method"] == "bottle"].copy()
        breast_all = feeds_all[feeds_all["feed_method"] == "breast"].copy()

        # Summary card
        today_df = df_all[df_all["date"] == today].copy()
        feeds_today = today_df[today_df["event_type"] == "feed"].copy()
        diapers_today = today_df[today_df["event_type"] == "diaper"].copy()
        bottle_today = feeds_today[feeds_today["feed_method"] == "bottle"].copy()
        breast_today = feeds_today[feeds_today["feed_method"] == "breast"].copy()

        last_feed = feeds_today["datetime"].max() if len(feeds_today) else pd.NaT
        avg_gap_min = None
        if len(feeds_all) >= 2:
            gaps = feeds_all["datetime"].diff().dt.total_seconds() / 60
            gaps = gaps.dropna()
            gaps = gaps[(gaps >= 10) & (gaps <= 360)]
            if len(gaps):
                avg_gap_min = float(gaps.tail(30).mean())

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='kpiGrid2'>", unsafe_allow_html=True)
        kpi_block("Since last feed", time_ago(last_feed))
        kpi_block("Avg feed gap", f"{avg_gap_min:.0f} min" if avg_gap_min is not None else "—")
        kpi_block("Feeds today", str(int(len(feeds_today))))
        kpi_block("Diapers today", str(int(len(diapers_today))))
        st.markdown("</div>", unsafe_allow_html=True)

        st.write("")
        st.markdown("<div class='kpiGrid2'>", unsafe_allow_html=True)
        kpi_block("Bottle ml today", str(int(bottle_today["volume_ml"].fillna(0).sum()) if len(bottle_today) else 0))
        kpi_block("Breast min today", str(int(breast_today["duration_min"].fillna(0).sum()) if len(breast_today) else 0))
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Expanders
        with st.expander("Peak feed times and cluster awareness", expanded=True):
            last7 = df_all[df_all["date"] >= last7_cut].copy()
            feeds7 = last7[last7["event_type"] == "feed"].copy()
            if len(feeds7) == 0:
                st.caption("Not enough data yet.")
            else:
                feeds7["hour"] = feeds7["datetime"].dt.hour
                hist = feeds7.groupby("hour").size().reindex(range(24), fill_value=0).reset_index()
                hist.columns = ["hour", "count"]

                top = hist.sort_values("count", ascending=False).head(3)
                peaks = [f"{int(r.hour):02d}:00–{(int(r.hour)+1)%24:02d}:00" for _, r in top.iterrows() if int(r["count"]) > 0]
                st.markdown("**Most common feed windows (last 7 days)**")
                st.caption(", ".join(peaks) if peaks else "Not enough data yet to identify peaks.")
                st.caption("Feeds by hour (last 7 days)")
                st.bar_chart(hist.rename(columns={"hour": "index"}).set_index("index")["count"])

                # Cluster check in last 2h
                window_start = now_ts - pd.Timedelta(hours=2)
                recent2h = feeds_all[feeds_all["datetime"] >= window_start].copy()
                if len(recent2h) >= 3:
                    recent2h["gap_min"] = recent2h["datetime"].diff().dt.total_seconds() / 60
                    g = recent2h["gap_min"].dropna()
                    avg2h = float(g.mean()) if len(g) else None
                    if avg2h is not None and avg2h <= 35:
                        st.warning(f"Cluster feeding likely: {len(recent2h)} feeds in last 2h (avg gap {avg2h:.0f} min).", icon="🟣")
                    elif avg2h is not None:
                        st.info(f"Recent feeding: {len(recent2h)} feeds in last 2h (avg gap {avg2h:.0f} min).", icon="ℹ️")
                else:
                    st.caption("Cluster awareness uses last 2 hours. Log more feeds to activate it.")

        with st.expander("Daily trends (last 14 days)", expanded=False):
            last14 = df_all[df_all["date"] >= last14_cut].copy()
            if len(last14) == 0:
                st.caption("Not enough data yet.")
            else:
                feeds14 = last14[last14["event_type"] == "feed"].copy()
                diapers14 = last14[last14["event_type"] == "diaper"].copy()
                bottle14 = feeds14[feeds14["feed_method"] == "bottle"].copy()

                feeds_by_day = feeds14.groupby("date").size().rename("feeds")
                diapers_by_day = diapers14.groupby("date").size().rename("diapers")
                bottle_ml_by_day = bottle14.groupby("date")["volume_ml"].sum().rename("bottle_ml")

                idx = sorted(last14["date"].unique())
                daily = pd.DataFrame(index=idx)
                daily = daily.join(feeds_by_day, how="left").join(diapers_by_day, how="left").join(bottle_ml_by_day, how="left").fillna(0)
                daily.index = pd.to_datetime(daily.index)

                st.caption("Feeds and diapers per day")
                st.line_chart(daily[["feeds", "diapers"]])

                st.caption("Bottle ml per day")
                st.line_chart(daily[["bottle_ml"]])

        with st.expander("Night vs day split", expanded=False):
            if len(bottle_all) == 0:
                st.caption("No bottle feeds logged yet.")
            else:
                b = bottle_all.copy()
                b["is_night"] = b["datetime"].apply(lambda x: within_night(x, night_start, night_end))
                night_total = int(b[b["is_night"]]["volume_ml"].fillna(0).sum())
                day_total = int(b[~b["is_night"]]["volume_ml"].fillna(0).sum())

                st.markdown("<div class='kpiGrid2'>", unsafe_allow_html=True)
                kpi_block("Night bottle ml", str(night_total))
                kpi_block("Day bottle ml", str(day_total))
                st.markdown("</div>", unsafe_allow_html=True)

                last7 = df_all[df_all["date"] >= last7_cut].copy()
                b7 = last7[(last7["event_type"] == "feed") & (last7["feed_method"] == "bottle")].copy()
                if len(b7):
                    b7["is_night"] = b7["datetime"].apply(lambda x: within_night(x, night_start, night_end))
                    by_day = b7.groupby(["date", "is_night"])["volume_ml"].sum().reset_index()
                    pivot = by_day.pivot(index="date", columns="is_night", values="volume_ml").fillna(0)
                    pivot.columns = ["Day" if c is False else "Night" for c in pivot.columns]
                    pivot.index = pd.to_datetime(pivot.index)
                    st.caption("Bottle ml trend (last 7 days)")
                    st.line_chart(pivot)

        with st.expander("Feed gaps", expanded=False):
            if len(feeds_all) < 2:
                st.caption("Not enough feeds yet.")
            else:
                feeds_view = feeds_all[["datetime", "feed_method", "volume_ml", "duration_min", "side", "notes", "row_id"]].copy()
                feeds_view["gap_min"] = feeds_view["datetime"].diff().dt.total_seconds() / 60
                st.caption("Most recent feed gaps")
                st.dataframe(feeds_view.sort_values("datetime", ascending=False).head(30), use_container_width=True, height=360)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Insights settings")
        st.caption("These save to Google Sheets and sync across devices.")
        c1, c2 = st.columns(2)
        with c1:
            ns = st.time_input("Night starts", value=night_start)
        with c2:
            ne = st.time_input("Night ends", value=night_end)
        if st.button("Save insight settings", use_container_width=True):
            save_config_value("night_start", ns.strftime("%H:%M"))
            save_config_value("night_end", ne.strftime("%H:%M"))
            st.toast("Saved", icon="✅")
            refresh()
        st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# MORE
# =========================================================
elif nav == "More":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("More")
    st.caption("Baby profile, immunisation, milestones, medication, journal")
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Baby profile (syncs across iPhone and Mac)", expanded=True):
        new_name = st.text_input("Baby name", value=baby_name)
        new_dob = st.text_input("DOB (YYYY-MM-DD)", value=baby_dob, placeholder="e.g. 2026-01-10")
        if st.button("Save profile", use_container_width=True):
            save_config_value("baby_name", new_name.strip() or "Baby Girl")
            save_config_value("baby_dob", new_dob.strip())
            st.toast("Saved", icon="✅")
            refresh()

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    choice = st.radio("Section", ["Immunisation", "Milestones", "Medication", "Journal"], horizontal=True, label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

    if choice == "Immunisation":
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("💉 Immunisation")
        vname = st.text_input("Vaccine name", placeholder="e.g. 8 week vaccines")
        given = st.date_input("Given date", value=datetime.now().date())
        notes = st.text_input("Notes (optional)", placeholder="Batch, site, reaction, etc")
        if st.button("✅ Save vaccine", use_container_width=True, disabled=(not vname.strip())):
            row = base_row(make_dt(given, now_floor().time()))
            row["event_type"] = "immunisation"
            row["title"] = vname.strip()
            row["notes"] = notes
            ws_append_log(row)
            st.toast("Immunisation saved", icon="✅")
            refresh()

        vdf = df_all[df_all["event_type"] == "immunisation"].sort_values("datetime", ascending=False).head(25)
        if len(vdf):
            st.caption("Recent immunisations")
            for _, r in vdf.iterrows():
                st.markdown(f"**{timeline_label(r)}**")
                n = safe_str(r.get("notes")).strip()
                if n:
                    st.caption(n)
        st.markdown("</div>", unsafe_allow_html=True)

    if choice == "Milestones":
        DEFAULT = [
            "First smile",
            "Holds head up",
            "Rolls over",
            "First laugh",
            "Grasps objects",
            "Sits without support",
            "Crawls",
            "Stands with support",
            "First steps",
        ]
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("✨ Milestones")
        st.caption("Tap to log with timestamp")
        cols = st.columns(2)
        for i, m in enumerate(DEFAULT):
            with cols[i % 2]:
                if st.button(m, use_container_width=True):
                    row = base_row(now_floor())
                    row["event_type"] = "milestone"
                    row["title"] = m
                    ws_append_log(row)
                    st.toast("Milestone saved", icon="✅")
                    refresh()

        st.write("")
        custom = st.text_input("Custom milestone", placeholder="Type a milestone you want to log")
        if st.button("✅ Save custom milestone", use_container_width=True, disabled=(not custom.strip())):
            row = base_row(now_floor())
            row["event_type"] = "milestone"
            row["title"] = custom.strip()
            ws_append_log(row)
            st.toast("Milestone saved", icon="✅")
            refresh()

        mdf = df_all[df_all["event_type"] == "milestone"].sort_values("datetime", ascending=False).head(25)
        if len(mdf):
            st.caption("Recent milestones")
            for _, r in mdf.iterrows():
                st.markdown(f"**{timeline_label(r)}**")
        st.markdown("</div>", unsafe_allow_html=True)

    if choice == "Medication":
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("💊 Medication")
        med = st.text_input("Medication name", placeholder="e.g. Vitamin D drops")
        dose = st.text_input("Dose (optional)", placeholder="e.g. 0.3ml")
        notes = st.text_input("Notes (optional)", placeholder="e.g. after feed")
        if st.button("✅ Save medication", use_container_width=True, disabled=(not med.strip())):
            row = base_row(now_floor())
            row["event_type"] = "medication"
            row["title"] = med.strip()
            row["notes"] = (dose.strip() + " " + notes.strip()).strip()
            ws_append_log(row)
            st.toast("Medication saved", icon="✅")
            refresh()

        mdf = df_all[df_all["event_type"] == "medication"].sort_values("datetime", ascending=False).head(25)
        if len(mdf):
            st.caption("Recent medication")
            for _, r in mdf.iterrows():
                st.markdown(f"**{timeline_label(r)}**")
                n = safe_str(r.get("notes")).strip()
                if n:
                    st.caption(n)
        st.markdown("</div>", unsafe_allow_html=True)

    if choice == "Journal":
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📝 Journal")
        mood = st.radio("Mood", ["😊", "😴", "😢", "🤒", "😍", "🥳"], horizontal=True)
        title = st.text_input("Title", placeholder="e.g. First long sleep")
        text = st.text_area("Entry", height=140, placeholder="Write a quick note…")
        if st.button("✅ Save entry", use_container_width=True, disabled=(not title.strip() and not text.strip())):
            row = base_row(now_floor())
            row["event_type"] = "journal"
            row["mood"] = mood
            row["title"] = title.strip()
            row["notes"] = text.strip()
            ws_append_log(row)
            st.toast("Journal saved", icon="✅")
            refresh()

        jdf = df_all[df_all["event_type"] == "journal"].sort_values("datetime", ascending=False).head(20)
        if len(jdf):
            st.caption("Recent entries")
            for _, r in jdf.iterrows():
                st.markdown(f"**{timeline_label(r)}**")
                st.caption(safe_str(r.get("notes")).strip()[:160])
        st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# HISTORY
# =========================================================
elif nav == "History":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("History")
    st.caption("Filter, edit, or delete any entry")
    st.markdown("</div>", unsafe_allow_html=True)

    if len(df_all) == 0:
        st.markdown("<div class='card'>No entries yet.</div>", unsafe_allow_html=True)
    else:
        day = st.date_input("Day", value=datetime.now().date())
        types = st.multiselect(
            "Types",
            ["feed", "diaper", "growth", "immunisation", "milestone", "medication", "journal"],
            default=["feed", "diaper", "growth", "immunisation", "milestone", "medication", "journal"],
        )
        view = df_all[df_all["date"] == day].copy()
        if types:
            view = view[view["event_type"].isin(types)]
        view = view.sort_values("datetime", ascending=False)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if len(view) == 0:
            st.caption("No entries for this day.")
        else:
            for _, r in view.head(30).iterrows():
                st.markdown(f"**{timeline_label(r)}**")
                n = safe_str(r.get("notes")).strip()
                if n:
                    st.caption(n)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Edit or delete")
        recent = df_all.sort_values("datetime", ascending=False).head(500).copy()
        recent["pick"] = recent.apply(lambda x: f"{x['datetime'].strftime('%Y-%m-%d %H:%M')} | {timeline_label(x)}", axis=1)
        pick = st.selectbox("Pick an entry", recent["pick"].tolist())
        row0 = recent[recent["pick"] == pick].iloc[0]
        rid = safe_str(row0.get("row_id")).strip()

        dt0 = row0["datetime"].to_pydatetime()
        d0 = dt0.date()
        t0 = dt0.time().replace(second=0, microsecond=0)
        et0 = safe_lower(row0.get("event_type"))

        with st.form("edit_form"):
            c1, c2 = st.columns(2)
            with c1:
                nd = st.date_input("Date", value=d0)
            with c2:
                nt = st.time_input("Time", value=t0)

            et = st.selectbox("Event type", ["feed", "diaper", "growth", "immunisation", "milestone", "medication", "journal"],
                              index=["feed", "diaper", "growth", "immunisation", "milestone", "medication", "journal"].index(et0) if et0 in ["feed", "diaper", "growth", "immunisation", "milestone", "medication", "journal"] else 0)

            title = st.text_input("Title", value=safe_str(row0.get("title")))
            mood = st.text_input("Mood", value=safe_str(row0.get("mood")))
            notes = st.text_input("Notes", value=safe_str(row0.get("notes")))

            feed_method = safe_lower(row0.get("feed_method"))
            side = safe_lower(row0.get("side"))
            diaper_type = safe_lower(row0.get("diaper_type"))
            volume = float(row0.get("volume_ml")) if not is_missing(row0.get("volume_ml")) else 0.0
            duration = float(row0.get("duration_min")) if not is_missing(row0.get("duration_min")) else 0.0
            weight = float(row0.get("weight_kg")) if not is_missing(row0.get("weight_kg")) else 0.0
            height = float(row0.get("height_cm")) if not is_missing(row0.get("height_cm")) else 0.0

            if et == "feed":
                feed_method = st.selectbox("Feed method", ["bottle", "breast"], index=0 if feed_method != "breast" else 1)
                if feed_method == "bottle":
                    volume = st.number_input("Bottle ml", 0.0, 300.0, float(volume), step=5.0)
                else:
                    side = st.selectbox("Side", ["left", "right"], index=0 if side != "right" else 1)
                    duration = st.number_input("Duration min", 0.0, 240.0, float(duration), step=1.0)

            if et == "diaper":
                diaper_type = st.selectbox("Diaper type", ["wet", "dirty", "mixed"], index=["wet", "dirty", "mixed"].index(diaper_type) if diaper_type in ["wet", "dirty", "mixed"] else 0)

            if et == "growth":
                weight = st.number_input("Weight kg", 0.0, 25.0, float(weight), step=0.01, format="%.2f")
                height = st.number_input("Height cm", 0.0, 120.0, float(height), step=0.1, format="%.1f")

            save = st.form_submit_button("Save changes", use_container_width=True)

        if save:
            new_dt = make_dt(nd, nt)
            upd = base_row(new_dt)
            upd["row_id"] = rid
            upd["event_type"] = et
            upd["title"] = title
            upd["mood"] = mood
            upd["notes"] = notes

            if et == "feed":
                upd["feed_method"] = feed_method
                if feed_method == "bottle":
                    upd["volume_ml"] = int(volume)
                else:
                    upd["side"] = side
                    upd["duration_min"] = int(duration)

            if et == "diaper":
                upd["diaper_type"] = diaper_type

            if et == "growth":
                if weight > 0:
                    upd["weight_kg"] = float(weight)
                if height > 0:
                    upd["height_cm"] = float(height)

            ok = ws_update_log(rid, upd)
            if ok:
                st.toast("Updated", icon="✅")
                refresh()
            else:
                st.error("Could not update row. row_id not found.")

        if st.button("🗑️ Delete entry", use_container_width=True, type="secondary"):
            ok = ws_delete_log(rid)
            if ok:
                st.toast("Deleted", icon="🗑️")
                refresh()
            else:
                st.error("Could not delete row. row_id not found.")

        st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# Bottom controls
# =========================================================
st.write("")
b1, b2 = st.columns(2)
with b1:
    if st.button("🔄 Refresh", use_container_width=True):
        refresh()
with b2:
    if st.button("🔒 Lock", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()

st.markdown("</div>", unsafe_allow_html=True)
