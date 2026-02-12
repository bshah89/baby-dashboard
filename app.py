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
# PIN lock gate
# Secrets required
# [app_security]
# pin = "1234"
# =========================================================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False


def pin_gate():
    st.markdown("<div class='hub'>", unsafe_allow_html=True)
    st.markdown("<div class='appTitle'>🔒 Baby Hub</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtle'>Enter 4 digit PIN to continue</div>", unsafe_allow_html=True)

    pin_input = st.text_input("PIN", type="password", max_chars=4, placeholder="••••", key="pin_input")
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
# Theme
# =========================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&family=Quicksand:wght@400;500;600;700&display=swap');

:root{
  --bg: hsl(30 50% 98%);
  --fg: hsl(280 30% 20%);
  --card: hsl(0 0% 100%);
  --border: hsl(280 15% 88%);
  --muted: hsl(30 30% 94%);

  --pink: hsl(340 70% 65%);
  --blue: hsl(200 70% 75%);
  --yellow: hsl(45 90% 70%);
  --mint: hsl(160 50% 70%);
  --lav: hsl(270 50% 78%);
  --peach: hsl(20 80% 78%);
}

html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg) !important;
  color: var(--fg) !important;
  font-family: Nunito, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { display:none; }

.hub{
  max-width: 460px;
  margin: 0 auto;
  padding-bottom: 140px;
}

.appTitle{
  font-family: Quicksand, Nunito, sans-serif;
  font-size: 30px;
  font-weight: 800;
  letter-spacing: -0.02em;
  margin: 6px 0 2px 0;
}

.subtle{
  color: rgba(40, 20, 55, 0.62);
  font-size: 13px;
  line-height: 1.35;
}

.card{
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 14px 14px;
  background: var(--card);
  box-shadow: 0 10px 24px rgba(30, 10, 40, 0.06);
  margin-bottom: 12px;
}

.badgeRow{
  display:flex;
  gap:8px;
  flex-wrap: wrap;
  margin-top: 6px;
}

.badge{
  display:inline-flex;
  align-items:center;
  gap:6px;
  padding: 6px 10px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 800;
  border: 1px solid var(--border);
  background: var(--muted);
}

.badge.pink{ background: hsl(340 70% 95%); border-color: hsl(340 70% 85%); }
.badge.blue{ background: hsl(200 70% 94%); border-color: hsl(200 70% 84%); }
.badge.yellow{ background: hsl(45 90% 94%); border-color: hsl(45 90% 82%); }
.badge.mint{ background: hsl(160 50% 92%); border-color: hsl(160 50% 80%); }
.badge.lav{ background: hsl(270 50% 94%); border-color: hsl(270 50% 84%); }
.badge.peach{ background: hsl(20 80% 94%); border-color: hsl(20 80% 84%); }

.kpiGrid2{
  display:grid;
  grid-template-columns: repeat(2, minmax(0,1fr));
  gap:10px;
}

.kpi{
  border-radius: 18px;
  padding: 12px 12px;
  background: var(--muted);
  border: 1px solid var(--border);
}

.kpiLabel{ font-size: 12px; opacity: 0.78; font-weight: 800; }
.kpiValue{ font-size: 20px; font-weight: 950; margin-top: 2px; }

div[data-baseweb="input"] > div, div[data-baseweb="select"] > div { border-radius: 14px !important; }

button{
  border-radius: 16px !important;
  font-weight: 800 !important;
}

button[kind="primary"]{
  background: var(--pink) !important;
  border: 1px solid hsl(340 70% 55%) !important;
}
button[kind="primary"]:hover{ filter: brightness(0.97); }

/* bottom nav */
.bottomNav {
  position: fixed;
  left: 0;
  right: 0;
  bottom: 0;
  padding: 10px 12px 14px 12px;
  background: rgba(255, 248, 242, 0.92);
  border-top: 1px solid var(--border);
  backdrop-filter: blur(10px);
  z-index: 9999;
}

.bottomNavInner{
  max-width: 460px;
  margin: 0 auto;
  display: grid;
  grid-template-columns: repeat(5, minmax(0,1fr));
  gap: 8px;
}

.navItem{
  text-decoration: none !important;
  border: 1px solid var(--border);
  background: white;
  border-radius: 16px;
  padding: 10px 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  font-weight: 900;
  color: var(--fg) !important;
  font-size: 12px;
}

.navItemActive{
  background: hsl(340 70% 95%);
  border-color: hsl(340 70% 80%);
}
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
# Query param routing
# =========================================================
NAV_TABS = ["Home", "Feed", "Diaper", "Insights", "More"]
DEFAULT_TAB = "Home"


def qp_get(name: str, default: str = "") -> str:
    # Supports new and old Streamlit APIs
    try:
        qp = st.query_params
        val = qp.get(name, default)
        if isinstance(val, list):
            val = val[0] if val else default
        return str(val) if val is not None else default
    except Exception:
        qp = st.experimental_get_query_params()
        val = qp.get(name, [default])
        return str(val[0]) if val else default


def qp_set(**kwargs):
    try:
        for k, v in kwargs.items():
            st.query_params[k] = v
    except Exception:
        st.experimental_set_query_params(**kwargs)


nav = qp_get("tab", DEFAULT_TAB)
if nav not in NAV_TABS:
    nav = DEFAULT_TAB

more_section = qp_get("section", "Main")
if more_section not in ["Main", "Growth", "History"]:
    more_section = "Main"


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
CONFIG_SHEET = "Config"


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
        ws.update(
            "A2:B6",
            [
                ["baby_name", "Baby Girl"],
                ["baby_dob", ""],
                ["night_start", "22:00"],
                ["night_end", "06:00"],
            ],
        )
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

    for c in ["row_id", "notes", "title", "mood"]:
        df[c] = df[c].astype("string").fillna("").astype(str)

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
    col_idx = headers.index("row_id") + 1 if "row_id" in headers else 1
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


if "refresh_key" not in st.session_state:
    st.session_state.refresh_key = 0
if "bf_running" not in st.session_state:
    st.session_state.bf_running = False
if "bf_segments" not in st.session_state:
    st.session_state.bf_segments = []


def refresh():
    st.session_state.refresh_key += 1
    st.cache_data.clear()
    st.rerun()


cfg = load_config(st.session_state.refresh_key)
baby_name = cfg.get("baby_name", "Baby Girl").strip() or "Baby Girl"
baby_dob = cfg.get("baby_dob", "").strip()


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
# Header + always lock
# =========================================================
st.markdown("<div class='hub'>", unsafe_allow_html=True)

h1, h2 = st.columns([3, 1])
with h1:
    st.markdown(f"<div class='appTitle'>👶🏻 {baby_name}</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtle'>Tap to log. Syncs across iPhone and Mac.</div>", unsafe_allow_html=True)
with h2:
    if st.button("🔒 Lock", use_container_width=True, key="lock_top_btn"):
        st.session_state.authenticated = False
        st.rerun()

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
    st.markdown("<div class='badgeRow'>", unsafe_allow_html=True)
    st.markdown("<span class='badge pink'>👧 Baby girl</span>", unsafe_allow_html=True)
    st.markdown("<span class='badge blue'>📱 iPhone first</span>", unsafe_allow_html=True)
    st.markdown("<span class='badge lav'>☁️ Cloud synced</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.write("")

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
        if st.button("🍼 15", use_container_width=True, key="home_b15"):
            row = base_row(now_floor())
            row["event_type"] = "feed"
            row["feed_method"] = "bottle"
            row["volume_ml"] = 15
            ws_append_log(row)
            st.toast("Bottle 15 logged", icon="✅")
            refresh()
    with c2:
        if st.button("🍼 30", use_container_width=True, key="home_b30"):
            row = base_row(now_floor())
            row["event_type"] = "feed"
            row["feed_method"] = "bottle"
            row["volume_ml"] = 30
            ws_append_log(row)
            st.toast("Bottle 30 logged", icon="✅")
            refresh()
    with c3:
        if st.button("💧 Wet", use_container_width=True, key="home_wet"):
            row = base_row(now_floor())
            row["event_type"] = "diaper"
            row["diaper_type"] = "wet"
            ws_append_log(row)
            st.toast("Wet logged", icon="✅")
            refresh()
    st.caption("Edit or delete any mistake in Tools > History.")
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
    mode = st.radio("Mode", ["Bottle", "Breast timer"], horizontal=True, label_visibility="collapsed", key="feed_mode")
    st.markdown("</div>", unsafe_allow_html=True)

    if mode == "Bottle":
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Quick bottle")
        g1 = st.columns(3)
        with g1[0]:
            if st.button("15 ml", use_container_width=True, key="feed_b15"):
                row = base_row(now_floor())
                row["event_type"] = "feed"
                row["feed_method"] = "bottle"
                row["volume_ml"] = 15
                ws_append_log(row)
                st.toast("Bottle 15 logged", icon="✅")
                refresh()
        with g1[1]:
            if st.button("30 ml", use_container_width=True, key="feed_b30"):
                row = base_row(now_floor())
                row["event_type"] = "feed"
                row["feed_method"] = "bottle"
                row["volume_ml"] = 30
                ws_append_log(row)
                st.toast("Bottle 30 logged", icon="✅")
                refresh()
        with g1[2]:
            ml = st.number_input("Custom ml", min_value=0, max_value=300, value=0, step=5, key="feed_custom_ml")
            if st.button("Log", use_container_width=True, disabled=(ml <= 0), key="feed_log_custom"):
                row = base_row(now_floor())
                row["event_type"] = "feed"
                row["feed_method"] = "bottle"
                row["volume_ml"] = int(ml)
                ws_append_log(row)
                st.toast(f"Bottle {int(ml)} logged", icon="✅")
                refresh()
        st.caption("One tap logs instantly. Use Tools > History to edit or delete.")
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
            if st.button("▶️ Start", use_container_width=True, key="bf_start"):
                if not st.session_state.bf_running:
                    st.session_state.bf_running = True
                    st.session_state.bf_segments = [{"side": "left", "start": now_floor(), "end": None}]
                    st.toast("Started on left", icon="⏱️")
                st.rerun()
        with c2:
            if st.button("⏹️ Stop", use_container_width=True, disabled=(not st.session_state.bf_running), key="bf_stop"):
                end_open_segment()
                st.session_state.bf_running = False
                st.rerun()
        with c3:
            if st.button("🧽 Discard", use_container_width=True, key="bf_discard"):
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
                if st.button("Switch left", use_container_width=True, disabled=(not st.session_state.bf_running), key="bf_left"):
                    end_open_segment()
                    start_segment("left")
                    st.rerun()
            with s2:
                if st.button("Switch right", use_container_width=True, disabled=(not st.session_state.bf_running), key="bf_right"):
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

            if st.toggle("Show breakdown", value=False, key="bf_breakdown"):
                st.dataframe(pd.DataFrame(rows), use_container_width=True, height=220)

            if st.button("✅ Save breast feed", use_container_width=True, disabled=(total_min <= 0), key="bf_save"):
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
        if st.button("💧 Wet", use_container_width=True, key="d_wet"):
            row = base_row(now_floor())
            row["event_type"] = "diaper"
            row["diaper_type"] = "wet"
            ws_append_log(row)
            st.toast("Wet logged", icon="✅")
            refresh()
    with g[1]:
        if st.button("💩 Dirty", use_container_width=True, key="d_dirty"):
            row = base_row(now_floor())
            row["event_type"] = "diaper"
            row["diaper_type"] = "dirty"
            ws_append_log(row)
            st.toast("Dirty logged", icon="✅")
            refresh()
    with g[2]:
        if st.button("💩 Mixed", use_container_width=True, key="d_mixed"):
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
# INSIGHTS
# =========================================================
elif nav == "Insights":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Insights")
    st.markdown("<div class='subtle'>Trends and patterns. Expand only what you need.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if len(df_all) == 0:
        st.markdown("<div class='card'>No data yet. Start logging feeds and diapers.</div>", unsafe_allow_html=True)
    else:
        now_ts = pd.Timestamp.now()
        today = now_ts.date()
        last7_cut = today - timedelta(days=6)
        last14_cut = today - timedelta(days=13)

        feeds_all = df_all[df_all["event_type"] == "feed"].copy().sort_values("datetime")
        bottle_all = feeds_all[feeds_all["feed_method"] == "bottle"].copy()

        today_df = df_all[df_all["date"] == today].copy()
        feeds_today = today_df[today_df["event_type"] == "feed"].copy()
        diapers_today = today_df[today_df["event_type"] == "diaper"].copy()

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
        st.markdown("</div>", unsafe_allow_html=True)

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
                peaks = [
                    f"{int(r.hour):02d}:00–{(int(r.hour)+1)%24:02d}:00"
                    for _, r in top.iterrows()
                    if int(r["count"]) > 0
                ]
                st.markdown("**Most common feed windows (last 7 days)**")
                st.caption(", ".join(peaks) if peaks else "Not enough data yet to identify peaks.")
                st.caption("Feeds by hour (last 7 days)")
                st.bar_chart(hist.rename(columns={"hour": "index"}).set_index("index")["count"])

                window_start = now_ts - pd.Timedelta(hours=2)
                recent2h = feeds_all[feeds_all["datetime"] >= window_start].copy()
                if len(recent2h) >= 3:
                    recent2h["gap_min"] = recent2h["datetime"].diff().dt.total_seconds() / 60
                    g = recent2h["gap_min"].dropna()
                    avg2h = float(g.mean()) if len(g) else None
                    if avg2h is not None and avg2h <= 35:
                        st.warning(f"Cluster feeding likely: {len(recent2h)} feeds in last 2h (avg gap {avg2h:.0f} min).", icon="🟣")

        with st.expander("Daily trends (last 14 days)", expanded=False):
            last14 = df_all[df_all["date"] >= last14_cut].copy()
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

        with st.expander("Insights settings (syncs)", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                ns = st.time_input("Night starts", value=night_start, key="ins_night_start")
            with c2:
                ne = st.time_input("Night ends", value=night_end, key="ins_night_end")
            if st.button("Save", use_container_width=True, key="ins_save"):
                save_config_value("night_start", ns.strftime("%H:%M"))
                save_config_value("night_end", ne.strftime("%H:%M"))
                st.toast("Saved", icon="✅")
                refresh()


# =========================================================
# MORE (Main / Growth / History)
# =========================================================
elif nav == "More":
    # Tools bar
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Tools")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<a class='navItem' href='?tab=More&section=Growth'>📈 Growth</a>", unsafe_allow_html=True)
    with c2:
        st.markdown("<a class='navItem' href='?tab=More&section=History'>🧾 History</a>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if more_section == "Growth":
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📈 Growth")
        st.caption("Log weight and height, see trend")

        c1, c2 = st.columns(2)
        with c1:
            w = st.number_input("Weight kg", min_value=0.0, max_value=25.0, value=0.0, step=0.01, format="%.2f", key="g_w")
        with c2:
            h = st.number_input("Height cm", min_value=0.0, max_value=120.0, value=0.0, step=0.1, format="%.1f", key="g_h")

        if st.button("✅ Log growth", use_container_width=True, disabled=(w <= 0 and h <= 0), key="g_save"):
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

            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<a class='navItem' href='?tab=More&section=Main'>⬅ Back to More</a>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    elif more_section == "History":
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🧾 History")
        st.caption("Filter, edit, or delete any entry")
        st.markdown("</div>", unsafe_allow_html=True)

        if len(df_all) == 0:
            st.markdown("<div class='card'>No entries yet.</div>", unsafe_allow_html=True)
        else:
            day = st.date_input("Day", value=datetime.now().date(), key="hist_day")
            types = st.multiselect(
                "Types",
                ["feed", "diaper", "growth", "immunisation", "milestone", "medication", "journal"],
                default=["feed", "diaper", "growth", "immunisation", "milestone", "medication", "journal"],
                key="hist_types",
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
            pick = st.selectbox("Pick an entry", recent["pick"].tolist(), key="hist_pick")
            row0 = recent[recent["pick"] == pick].iloc[0]
            rid = safe_str(row0.get("row_id")).strip()

            dt0 = row0["datetime"].to_pydatetime()
            d0 = dt0.date()
            t0 = dt0.time().replace(second=0, microsecond=0)
            et0 = safe_lower(row0.get("event_type"))

            with st.form("edit_form"):
                c1, c2 = st.columns(2)
                with c1:
                    nd = st.date_input("Date", value=d0, key="edit_date")
                with c2:
                    nt = st.time_input("Time", value=t0, key="edit_time")

                et = st.selectbox(
                    "Event type",
                    ["feed", "diaper", "growth", "immunisation", "milestone", "medication", "journal"],
                    index=["feed", "diaper", "growth", "immunisation", "milestone", "medication", "journal"].index(et0)
                    if et0 in ["feed", "diaper", "growth", "immunisation", "milestone", "medication", "journal"]
                    else 0,
                    key="edit_event_type",
                )

                title = st.text_input("Title", value=safe_str(row0.get("title")), key="edit_title")
                mood = st.text_input("Mood", value=safe_str(row0.get("mood")), key="edit_mood")
                notes = st.text_input("Notes", value=safe_str(row0.get("notes")), key="edit_notes")

                feed_method = safe_lower(row0.get("feed_method"))
                side = safe_lower(row0.get("side"))
                diaper_type = safe_lower(row0.get("diaper_type"))
                volume = float(row0.get("volume_ml")) if not is_missing(row0.get("volume_ml")) else 0.0
                duration = float(row0.get("duration_min")) if not is_missing(row0.get("duration_min")) else 0.0
                weight = float(row0.get("weight_kg")) if not is_missing(row0.get("weight_kg")) else 0.0
                height = float(row0.get("height_cm")) if not is_missing(row0.get("height_cm")) else 0.0

                if et == "feed":
                    feed_method = st.selectbox("Feed method", ["bottle", "breast"], index=0 if feed_method != "breast" else 1, key="edit_feed_method")
                    if feed_method == "bottle":
                        volume = st.number_input("Bottle ml", 0.0, 300.0, float(volume), step=5.0, key="edit_volume")
                    else:
                        side = st.selectbox("Side", ["left", "right"], index=0 if side != "right" else 1, key="edit_side")
                        duration = st.number_input("Duration min", 0.0, 240.0, float(duration), step=1.0, key="edit_duration")

                if et == "diaper":
                    diaper_type = st.selectbox("Diaper type", ["wet", "dirty", "mixed"], index=["wet", "dirty", "mixed"].index(diaper_type) if diaper_type in ["wet", "dirty", "mixed"] else 0, key="edit_diaper")

                if et == "growth":
                    weight = st.number_input("Weight kg", 0.0, 25.0, float(weight), step=0.01, format="%.2f", key="edit_weight")
                    height = st.number_input("Height cm", 0.0, 120.0, float(height), step=0.1, format="%.1f", key="edit_height")

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

            if st.button("🗑️ Delete entry", use_container_width=True, type="secondary", key="hist_delete"):
                ok = ws_delete_log(rid)
                if ok:
                    st.toast("Deleted", icon="🗑️")
                    refresh()
                else:
                    st.error("Could not delete row. row_id not found.")

            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<a class='navItem' href='?tab=More&section=Main'>⬅ Back to More</a>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    else:
        # Main More
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("More")
        st.markdown("<div class='subtle'>Profile, immunisation, milestones, medication, journal</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("Baby profile (syncs)", expanded=True):
            new_name = st.text_input("Baby name", value=baby_name, key="profile_name")
            new_dob = st.text_input("DOB (YYYY-MM-DD)", value=baby_dob, placeholder="e.g. 2026-01-10", key="profile_dob")
            if st.button("Save profile", use_container_width=True, key="profile_save"):
                save_config_value("baby_name", new_name.strip() or "Baby Girl")
                save_config_value("baby_dob", new_dob.strip())
                st.toast("Saved", icon="✅")
                refresh()

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        choice = st.radio(
            "Section",
            ["Immunisation", "Milestones", "Medication", "Journal"],
            horizontal=True,
            label_visibility="collapsed",
            key="more_section_main",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        if choice == "Immunisation":
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("💉 Immunisation")
            vname = st.text_input("Vaccine name", placeholder="e.g. 8 week vaccines", key="imm_name")
            given = st.date_input("Given date", value=datetime.now().date(), key="imm_date")
            notes = st.text_input("Notes (optional)", placeholder="Batch, site, reaction, etc", key="imm_notes")
            if st.button("✅ Save vaccine", use_container_width=True, disabled=(not vname.strip()), key="imm_save"):
                row = base_row(make_dt(given, now_floor().time()))
                row["event_type"] = "immunisation"
                row["title"] = vname.strip()
                row["notes"] = notes
                ws_append_log(row)
                st.toast("Immunisation saved", icon="✅")
                refresh()
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
                    if st.button(m, use_container_width=True, key=f"ms_{i}"):
                        row = base_row(now_floor())
                        row["event_type"] = "milestone"
                        row["title"] = m
                        ws_append_log(row)
                        st.toast("Milestone saved", icon="✅")
                        refresh()
            st.markdown("</div>", unsafe_allow_html=True)

        if choice == "Medication":
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("💊 Medication")
            med = st.text_input("Medication name", placeholder="e.g. Vitamin D drops", key="med_name")
            dose = st.text_input("Dose (optional)", placeholder="e.g. 0.3ml", key="med_dose")
            notes = st.text_input("Notes (optional)", placeholder="e.g. after feed", key="med_notes")
            if st.button("✅ Save medication", use_container_width=True, disabled=(not med.strip()), key="med_save"):
                row = base_row(now_floor())
                row["event_type"] = "medication"
                row["title"] = med.strip()
                row["notes"] = (dose.strip() + " " + notes.strip()).strip()
                ws_append_log(row)
                st.toast("Medication saved", icon="✅")
                refresh()
            st.markdown("</div>", unsafe_allow_html=True)

        if choice == "Journal":
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("📝 Journal")
            mood = st.radio("Mood", ["😊", "😴", "😢", "🤒", "😍", "🥳"], horizontal=True, key="j_mood")
            title = st.text_input("Title", placeholder="e.g. First long sleep", key="j_title")
            text = st.text_area("Entry", height=140, placeholder="Write a quick note…", key="j_text")
            if st.button("✅ Save entry", use_container_width=True, disabled=(not title.strip() and not text.strip()), key="j_save"):
                row = base_row(now_floor())
                row["event_type"] = "journal"
                row["mood"] = mood
                row["title"] = title.strip()
                row["notes"] = text.strip()
                ws_append_log(row)
                st.toast("Journal saved", icon="✅")
                refresh()
            st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# Bottom actions
# =========================================================
st.write("")
b1, b2 = st.columns(2)
with b1:
    if st.button("🔄 Refresh", use_container_width=True, key="refresh_bottom_btn"):
        refresh()
with b2:
    if st.button("🔒 Lock", use_container_width=True, key="lock_bottom_btn"):
        st.session_state.authenticated = False
        st.rerun()


# =========================================================
# Fixed bottom nav (HTML)
# =========================================================
def bottom_nav(active_tab: str):
    items = [
        ("Home", "🏠"),
        ("Feed", "🍼"),
        ("Diaper", "💧"),
        ("Insights", "✨"),
        ("More", "🧸"),
    ]
    links = []
    for name, icon in items:
        cls = "navItem navItemActive" if name == active_tab else "navItem"
        links.append(f"<a class='{cls}' href='?tab={name}'>{icon} {name}</a>")
    st.markdown(
        "<div class='bottomNav'><div class='bottomNavInner'>"
        + "".join(links)
        + "</div></div>",
        unsafe_allow_html=True,
    )


bottom_nav(nav)

st.markdown("</div>", unsafe_allow_html=True)
