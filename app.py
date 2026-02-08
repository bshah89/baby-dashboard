import pandas as pd
import streamlit as st
from datetime import datetime, time, timedelta
import gspread
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="Baby Girl Care", page_icon="👶🏻", layout="wide")

# ===============================
# Simple 4-digit PIN protection
# ===============================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

def pin_gate():
    st.title("🔒 Baby Dashboard")
    st.caption("Enter 4-digit PIN to continue")

    pin_input = st.text_input(
        "PIN",
        type="password",
        max_chars=4,
        placeholder="••••",
    )

    if pin_input:
        if pin_input == st.secrets["app_security"]["pin"]:
            st.session_state.authenticated = True
            st.success("Unlocked")
            st.rerun()
        else:
            st.error("Incorrect PIN")

    st.stop()

if not st.session_state.authenticated:
    pin_gate()


# =========================================================
# Constants and schema
# =========================================================
EXPECTED_COLS = [
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
]

FEED_METHODS = ["bottle", "breast"]
SIDES = ["left", "right"]
DIAPER_TYPES = ["wet", "dirty", "mixed"]
EVENT_TYPES = ["feed", "diaper", "weight"]

DUPLICATE_GUARD_SECONDS = 15


# =========================================================
# Safe helpers
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


def safe_int(v, default=0) -> int:
    try:
        if is_missing(v):
            return default
        x = float(v)
        if pd.isna(x):
            return default
        return int(x)
    except Exception:
        return default


def safe_float(v, default=None):
    try:
        if is_missing(v):
            return default
        x = float(v)
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def now_floor_minute() -> datetime:
    n = datetime.now()
    return n.replace(second=0, microsecond=0)


def make_dt(d, t) -> datetime:
    return datetime.combine(d, t)


def new_row_id() -> str:
    return f"r{int(datetime.now().timestamp() * 1000)}"


def time_ago_str(ts: pd.Timestamp) -> str:
    if is_missing(ts):
        return "Not yet"
    mins = int((pd.Timestamp.now() - ts).total_seconds() // 60)
    if mins < 1:
        return "just now"
    if mins < 60:
        return f"{mins} min ago"
    hrs = mins // 60
    rem = mins % 60
    if hrs < 24:
        return f"{hrs}h {rem}m ago"
    days = hrs // 24
    hrs2 = hrs % 24
    return f"{days}d {hrs2}h ago"


def part_of_day(ts: pd.Timestamp) -> str:
    if is_missing(ts):
        return "Unknown"
    h = ts.hour
    if 5 <= h < 12:
        return "Morning"
    if 12 <= h < 17:
        return "Afternoon"
    if 17 <= h < 21:
        return "Evening"
    return "Night"


def is_night(ts: pd.Timestamp, night_start: time, night_end: time) -> bool:
    if is_missing(ts):
        return False
    tt = ts.time()
    if night_start < night_end:
        return night_start <= tt < night_end
    return tt >= night_start or tt < night_end


def last_night_window(now_dt: datetime, night_start_t: time, night_end_t: time):
    today = now_dt.date()

    if night_start_t < night_end_t:
        end_dt = datetime.combine(today, night_end_t)
        start_dt = datetime.combine(today, night_start_t)
        if now_dt < end_dt:
            end_dt = datetime.combine(today - timedelta(days=1), night_end_t)
            start_dt = datetime.combine(today - timedelta(days=1), night_start_t)
        return pd.Timestamp(start_dt), pd.Timestamp(end_dt)

    end_dt = datetime.combine(today, night_end_t)
    start_dt = datetime.combine(today - timedelta(days=1), night_start_t)
    if now_dt < end_dt:
        end_dt = datetime.combine(today - timedelta(days=1), night_end_t)
        start_dt = datetime.combine(today - timedelta(days=2), night_start_t)
    return pd.Timestamp(start_dt), pd.Timestamp(end_dt)


def normalize_row(
    dt: datetime,
    event_type: str,
    feed_method="",
    side="",
    volume_ml=None,
    duration_min=None,
    diaper_type="",
    notes="",
    weight_kg=None,
    row_id=None,
):
    rid = row_id or new_row_id()
    d = dt.date()

    et = safe_lower(event_type)
    fm = safe_lower(feed_method)
    sd = safe_lower(side)
    dp = safe_lower(diaper_type)

    # Clean based on event type
    if et != "feed":
        fm = ""
        sd = ""
        volume_ml = None
        duration_min = None
    if et != "diaper":
        dp = ""
    if et != "weight":
        weight_kg = None

    w = "" if weight_kg in [None, ""] else float(weight_kg)

    return {
        "row_id": rid,
        "datetime": dt.strftime("%Y-%m-%d %H:%M"),
        "date": d.strftime("%Y-%m-%d"),
        "time": dt.strftime("%H:%M"),
        "event_type": et,
        "feed_method": fm,
        "side": sd,
        "volume_ml": "" if volume_ml in [None, ""] else safe_int(volume_ml, 0),
        "duration_min": "" if duration_min in [None, ""] else safe_int(duration_min, 0),
        "diaper_type": dp,
        "notes": safe_str(notes),
        "weight_kg": w,
    }


def validate_row(row: dict) -> list[str]:
    errs = []
    et = safe_lower(row.get("event_type"))

    if et not in EVENT_TYPES:
        errs.append("Invalid event type.")

    if et == "feed":
        fm = safe_lower(row.get("feed_method"))
        if fm not in FEED_METHODS:
            errs.append("Choose bottle or breast.")
        if fm == "bottle":
            ml = safe_float(row.get("volume_ml"), 0)
            if ml is None or ml <= 0:
                errs.append("Bottle ml must be greater than 0.")
        if fm == "breast":
            mins = safe_float(row.get("duration_min"), 0)
            if mins is None or mins <= 0:
                errs.append("Breast duration must be greater than 0.")
    if et == "diaper":
        dp = safe_lower(row.get("diaper_type"))
        if dp not in DIAPER_TYPES:
            errs.append("Choose wet, dirty, or mixed.")
    if et == "weight":
        w = safe_float(row.get("weight_kg"))
        if w is None or w <= 0:
            errs.append("Weight must be greater than 0.")
        if w > 30:
            errs.append("Weight looks too high. Please double check kg value.")
    return errs


def timeline_label(row: pd.Series) -> str:
    dt_str = row["datetime"].strftime("%H:%M") if "datetime" in row and not is_missing(row["datetime"]) else ""
    et = safe_lower(row.get("event_type"))
    fm = safe_lower(row.get("feed_method"))

    if et == "weight":
        w = safe_float(row.get("weight_kg"))
        if w is None:
            return f"{dt_str} ⚖️ Weight"
        return f"{dt_str} ⚖️ {w:.3f} kg"

    if et == "feed":
        if fm == "bottle":
            return f"{dt_str} 🍼 Bottle {safe_int(row.get('volume_ml'), 0)} ml"
        if fm == "breast":
            mins = safe_int(row.get("duration_min"), 0)
            side = safe_lower(row.get("side"))
            side_txt = f" {side.title()}" if side in SIDES else ""
            return f"{dt_str} 🤱 Breast {mins} min{side_txt}"
        return f"{dt_str} 🍼 Feed"

    if et == "diaper":
        dtp = safe_lower(row.get("diaper_type"))
        icon = "💩" if dtp in ["dirty", "mixed"] else "💧"
        return f"{dt_str} {icon} {(dtp.title() if dtp else 'Diaper')}"

    return f"{dt_str} Event"


def gentle_insights(today_df: pd.DataFrame) -> list[str]:
    msgs = []
    feeds = today_df[today_df["event_type"] == "feed"].copy()
    diapers = today_df[today_df["event_type"] == "diaper"].copy()
    bottle = feeds[feeds["feed_method"] == "bottle"].copy()

    if len(feeds) == 0:
        return ["No feeds logged yet today."]

    last_feed = feeds["datetime"].max()
    mins = int((pd.Timestamp.now() - last_feed).total_seconds() // 60)

    if mins >= 180:
        msgs.append(f"It has been {mins // 60} hours since the last feed.")
    elif mins >= 90:
        msgs.append(f"It has been {mins} minutes since the last feed.")

    if len(diapers) == 0:
        msgs.append("No diapers logged yet today.")

    if len(bottle) > 0:
        msgs.append(f"Total bottle intake today: {int(bottle['volume_ml'].fillna(0).sum())} ml.")

    return msgs[:6]


# =========================================================
# Insights helpers (peak, cluster, last 24h, suggested next feed)
# =========================================================
def compute_feed_hour_hist(df_all: pd.DataFrame, days: int = 7) -> pd.DataFrame:
    if len(df_all) == 0:
        return pd.DataFrame(columns=["hour", "count"])
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
    feeds = df_all[(df_all["event_type"] == "feed") & (df_all["datetime"] >= cutoff)].copy()
    if len(feeds) == 0:
        return pd.DataFrame(columns=["hour", "count"])
    feeds["hour"] = feeds["datetime"].dt.hour
    hist = feeds.groupby("hour").size().reset_index(name="count")
    all_hours = pd.DataFrame({"hour": list(range(24))})
    hist = all_hours.merge(hist, on="hour", how="left").fillna(0)
    hist["count"] = hist["count"].astype(int)
    return hist


def top_peak_windows(hist: pd.DataFrame, top_n: int = 3) -> list[str]:
    if hist is None or len(hist) == 0:
        return []
    h = hist.sort_values("count", ascending=False).head(top_n)
    peaks = []
    for _, r in h.iterrows():
        hr = int(r["hour"])
        if int(r["count"]) > 0:
            peaks.append(f"{hr:02d}:00–{(hr + 1) % 24:02d}:00")
    return peaks


def cluster_feeding_status(df_all: pd.DataFrame):
    feeds = df_all[df_all["event_type"] == "feed"].sort_values("datetime").copy()
    if len(feeds) < 3:
        return {"is_cluster": False, "msg": "Not enough data yet for cluster feeding patterns."}

    now_ts = pd.Timestamp.now()
    window_start = now_ts - pd.Timedelta(hours=2)
    recent = feeds[feeds["datetime"] >= window_start].copy()

    if len(recent) < 3:
        return {"is_cluster": False, "msg": "No cluster feeding pattern in the last 2 hours."}

    recent["gap_min"] = recent["datetime"].diff().dt.total_seconds() / 60
    avg_gap = float(recent["gap_min"].dropna().mean()) if recent["gap_min"].notna().any() else None

    if avg_gap is not None and avg_gap <= 35:
        return {
            "is_cluster": True,
            "msg": f"Cluster feeding likely: {len(recent)} feeds in last 2h (avg gap {avg_gap:.0f} min).",
        }

    if avg_gap is not None:
        return {"is_cluster": False, "msg": f"Recent feeding: {len(recent)} feeds in last 2h (avg gap {avg_gap:.0f} min)."}
    return {"is_cluster": False, "msg": "Recent feeding detected."}


def hourly_activity_last_24h(df_all: pd.DataFrame) -> pd.DataFrame:
    now_ts = pd.Timestamp.now().floor("min")
    start = now_ts - pd.Timedelta(hours=24)

    if len(df_all) == 0:
        return pd.DataFrame()

    r = df_all[(df_all["datetime"] >= start) & (df_all["datetime"] <= now_ts)].copy()
    if len(r) == 0:
        return pd.DataFrame()

    r["hour_bucket"] = r["datetime"].dt.floor("H")
    feeds = r[r["event_type"] == "feed"].groupby("hour_bucket").size().rename("feeds")
    diapers = r[r["event_type"] == "diaper"].groupby("hour_bucket").size().rename("diapers")

    idx = pd.date_range(start=start.floor("H") + pd.Timedelta(hours=1), end=now_ts.floor("H"), freq="H")
    out = pd.DataFrame(index=idx)
    out = out.join(feeds, how="left").join(diapers, how="left").fillna(0)
    out.index = out.index.strftime("%m-%d %H:%M")
    return out.astype(int)


def longest_feed_gap_last_24h(df_all: pd.DataFrame) -> int:
    now_ts = pd.Timestamp.now()
    window_start = now_ts - pd.Timedelta(hours=24)
    feeds = df_all[df_all["event_type"] == "feed"].dropna(subset=["datetime"]).sort_values("datetime")
    feeds = feeds[feeds["datetime"] >= window_start]

    if len(feeds) == 0:
        return 24 * 60

    times = pd.concat(
        [pd.Series([window_start]), feeds["datetime"], pd.Series([now_ts])],
        ignore_index=True,
    ).sort_values().reset_index(drop=True)

    gaps = times.diff().dt.total_seconds().fillna(0) / 60
    return int(gaps.max())


def avg_feed_gap_minutes(df_all: pd.DataFrame, days: int = 7) -> float | None:
    if len(df_all) == 0:
        return None
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
    feeds = df_all[(df_all["event_type"] == "feed") & (df_all["datetime"] >= cutoff)].sort_values("datetime").copy()
    if len(feeds) < 2:
        return None
    gaps = feeds["datetime"].diff().dt.total_seconds() / 60
    gaps = gaps.dropna()
    if len(gaps) == 0:
        return None
    # trim extreme outliers for calmer recommendation
    gaps = gaps[(gaps >= 10) & (gaps <= 360)]
    if len(gaps) == 0:
        return None
    return float(gaps.mean())


def suggested_next_feed_window(df_all: pd.DataFrame) -> tuple[str, str]:
    avg_gap = avg_feed_gap_minutes(df_all, days=7)
    feeds = df_all[df_all["event_type"] == "feed"].sort_values("datetime")
    if len(feeds) == 0:
        return ("No suggestion yet", "Log a feed to start recommendations.")
    last_feed = feeds.iloc[-1]["datetime"]
    if is_missing(last_feed) or avg_gap is None:
        return ("No suggestion yet", "Need a bit more history to estimate timing.")
    start = last_feed + pd.Timedelta(minutes=avg_gap * 0.8)
    end = last_feed + pd.Timedelta(minutes=avg_gap * 1.2)
    return (f"{start.strftime('%H:%M')} to {end.strftime('%H:%M')}", f"Based on ~{avg_gap:.0f} min average gap (last 7 days).")


# =========================================================
# Google Sheets
# =========================================================
def col_letter(n: int) -> str:
    s = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


@st.cache_resource
def get_worksheet():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scopes)
    gc = gspread.authorize(creds)

    spreadsheet_id = st.secrets["gsheets"]["spreadsheet_id"]
    ws_name = st.secrets["gsheets"]["worksheet_name"]

    sh = gc.open_by_key(spreadsheet_id)
    ws = sh.worksheet(ws_name)
    return ws


@st.cache_data(ttl=20)
def ws_read_all_cached(_refresh_key: int) -> pd.DataFrame:
    ws = get_worksheet()
    records = ws.get_all_records()
    df = pd.DataFrame(records)

    for c in EXPECTED_COLS:
        if c not in df.columns:
            df[c] = pd.NA

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).copy()

    df["date"] = df["datetime"].dt.date
    df["time"] = df["datetime"].dt.strftime("%H:%M")

    df["volume_ml"] = pd.to_numeric(df["volume_ml"], errors="coerce")
    df["duration_min"] = pd.to_numeric(df["duration_min"], errors="coerce")
    df["weight_kg"] = pd.to_numeric(df["weight_kg"], errors="coerce")

    for c in ["event_type", "feed_method", "side", "diaper_type"]:
        df[c] = df[c].astype("string").str.strip().str.lower()

    df["notes"] = df["notes"].astype("string").fillna("")
    df["row_id"] = df["row_id"].astype("string").fillna("").astype(str)

    return df.sort_values("datetime").reset_index(drop=True)


def ws_append_row(row: dict) -> None:
    ws = get_worksheet()
    ws.append_row([row.get(c, "") for c in EXPECTED_COLS], value_input_option="USER_ENTERED")


def ws_find_rownum_by_row_id(row_id: str) -> int | None:
    ws = get_worksheet()
    col_values = ws.col_values(1)
    for idx, v in enumerate(col_values[1:], start=2):
        if str(v).strip() == str(row_id).strip():
            return idx
    return None


def ws_update_row(row_id: str, row: dict) -> bool:
    ws = get_worksheet()
    rnum = ws_find_rownum_by_row_id(row_id)
    if rnum is None:
        return False
    end_col = col_letter(len(EXPECTED_COLS))
    ws.update(f"A{rnum}:{end_col}{rnum}", [[row.get(c, "") for c in EXPECTED_COLS]])
    return True


def ws_delete_row(row_id: str) -> bool:
    ws = get_worksheet()
    rnum = ws_find_rownum_by_row_id(row_id)
    if rnum is None:
        return False
    ws.delete_rows(rnum)
    return True


# =========================================================
# Session state
# =========================================================
if "refresh_key" not in st.session_state:
    st.session_state.refresh_key = 0

if "bf_timer_active" not in st.session_state:
    st.session_state.bf_timer_active = False
if "bf_segments" not in st.session_state:
    st.session_state.bf_segments = []  # [{"side": "left/right", "start": dt, "end": dt|None}]
if "last_saved_row_id" not in st.session_state:
    st.session_state.last_saved_row_id = ""
if "last_saved_label" not in st.session_state:
    st.session_state.last_saved_label = ""
if "last_action_key" not in st.session_state:
    st.session_state.last_action_key = ""
if "last_action_at" not in st.session_state:
    st.session_state.last_action_at = None
if "pending_duplicate" not in st.session_state:
    st.session_state.pending_duplicate = None
if "show_edit_last" not in st.session_state:
    st.session_state.show_edit_last = False


def refresh_data():
    st.session_state.refresh_key += 1
    st.cache_data.clear()
    st.rerun()


# =========================================================
# Sidebar settings + filters + refresh
# =========================================================
with st.sidebar:
    if st.button("🔒 Lock app"):
        st.session_state.authenticated = False
        st.rerun()

    st.header("Settings")
    baby_name = st.text_input("Baby girl name", value="Baby Girl")

    night_mode = st.toggle("Night mode", value=False)
    if night_mode:
        st.markdown(
            """
            ...
            """,
            unsafe_allow_html=True,
        )
<style>
  .stApp { background: #0e1117; color: #e6e6e6; }
  .stMarkdown, .stTextInput label, .stSelectbox label, .stRadio label, .stNumberInput label, .stDateInput label, .stTimeInput label { color: #e6e6e6 !important; }
  div[data-testid="stMetricValue"] { color: #e6e6e6; }
  div[data-testid="stMetricLabel"] { color: #bdbdbd; }
</style>
            """,
            unsafe_allow_html=True,
        )

    st.divider()
    st.subheader("Sync")
    if st.button("🔄 Refresh", use_container_width=True):
        refresh_data()
    st.caption("Auto refresh every ~20 seconds. Use Refresh if you logged from another phone.")

# Read data (cached)
df_all = ws_read_all_cached(st.session_state.refresh_key)

with st.sidebar:
    st.divider()
    st.subheader("Filters")

    if len(df_all) > 0:
        min_date = df_all["date"].min()
        max_date = df_all["date"].max()
    else:
        today = datetime.now().date()
        min_date = today - timedelta(days=7)
        max_date = today

    date_range = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = end_date = date_range

    show_feeds = st.checkbox("Show feeds", value=True)
    show_diapers = st.checkbox("Show diapers", value=True)
    show_weights = st.checkbox("Show weights", value=True)

    only_bottle = st.checkbox("Bottle only", value=False)
    ml_min, ml_max = st.slider("Bottle ml range", 0, 300, (0, 150))

    night_start = st.time_input("Night starts", value=time(22, 0))
    night_end = st.time_input("Night ends", value=time(6, 0))

    st.caption("Backend: Google Sheet")

# Filtered view for History charts
df = df_all.copy()
if len(df) > 0:
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()

    allowed = []
    if show_feeds:
        allowed.append("feed")
    if show_diapers:
        allowed.append("diaper")
    if show_weights:
        allowed.append("weight")
    if allowed:
        df = df[df["event_type"].isin(allowed)].copy()

    if only_bottle:
        df = df[(df["event_type"] == "feed") & (df["feed_method"] == "bottle")].copy()
        v = df["volume_ml"].fillna(0)
        df = df[(v >= ml_min) & (v <= ml_max)].copy()

st.title(f"👶🏻 {baby_name} care board")
st.caption("Mobile first logging. Live sync across iPhone and Mac. Saved to Google Sheets.")

tabs = st.tabs(["Today", "Log", "Insights", "Weight", "History", "Export", "Data"])


# =========================================================
# Write helpers with duplicate guard + friendly errors
# =========================================================
def duplicate_guard(action_key: str) -> bool:
    """Returns True if allowed to proceed. False if blocked by duplicate guard UI."""
    now = datetime.now()
    last_key = safe_str(st.session_state.last_action_key)
    last_at = st.session_state.last_action_at

    if last_key == action_key and last_at is not None:
        elapsed = (now - last_at).total_seconds()
        if elapsed < DUPLICATE_GUARD_SECONDS:
            st.session_state.pending_duplicate = {
                "action_key": action_key,
                "elapsed": elapsed,
            }
            return False

    st.session_state.last_action_key = action_key
    st.session_state.last_action_at = now
    return True


def save_row(row: dict, success_msg: str):
    errs = validate_row(row)
    if errs:
        st.error("Fix this before saving:\n\n" + "\n".join([f"• {e}" for e in errs]))
        return

    try:
        ws_append_row(row)
        st.session_state.last_saved_row_id = row["row_id"]
        st.session_state.last_saved_label = success_msg
        st.toast(success_msg, icon="✅")
        refresh_data()
    except Exception as e:
        st.error("Could not save. Check internet and try again.")
        st.caption(str(e))


def undo_last():
    rid = safe_str(st.session_state.last_saved_row_id).strip()
    if not rid:
        st.info("Nothing to undo yet.")
        return
    try:
        ok = ws_delete_row(rid)
        st.session_state.last_saved_row_id = ""
        st.session_state.last_saved_label = ""
        if ok:
            st.toast("Undone", icon="↩️")
            refresh_data()
        else:
            st.error("Could not undo. Entry not found.")
    except Exception as e:
        st.error("Could not undo. Check internet and try again.")
        st.caption(str(e))


# =========================================================
# TODAY TAB (more actionable)
# =========================================================
with tabs[0]:
    today_date = datetime.now().date()
    today_df = df_all[df_all["date"] == today_date].copy() if len(df_all) else df_all.copy()

    feeds_today = today_df[today_df["event_type"] == "feed"].copy()
    diapers_today = today_df[today_df["event_type"] == "diaper"].copy()
    weights_today = today_df[today_df["event_type"] == "weight"].copy()

    last_feed_dt = feeds_today["datetime"].max() if len(feeds_today) else pd.NaT
    last_diaper_dt = diapers_today["datetime"].max() if len(diapers_today) else pd.NaT

    # Primary cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Time since last feed", time_ago_str(last_feed_dt))
    with c2:
        st.metric("Time since last diaper", time_ago_str(last_diaper_dt))
    with c3:
        st.metric("Feeds today", int(len(feeds_today)))
    with c4:
        st.metric("Diapers today", int(len(diapers_today)))

    st.divider()

    # Next action hints
    hint1, hint2, hint3 = st.columns(3)
    with hint1:
        window, note = suggested_next_feed_window(df_all)
        st.subheader("Suggested next feed")
        st.write(window)
        st.caption(note)

    with hint2:
        st.subheader("Cluster check")
        status = cluster_feeding_status(df_all)
        if status["is_cluster"]:
            st.warning(status["msg"], icon="🟣")
        else:
            st.info(status["msg"], icon="ℹ️")

    with hint3:
        st.subheader("Quick notes")
        for m in gentle_insights(today_df):
            st.write("• " + m)
        st.caption("You are doing great 💗")

    st.divider()

    # Last 6 events, big and readable
    st.subheader("Last events")
    last_events = df_all.sort_values("datetime", ascending=False).head(6).copy()
    if len(last_events) == 0:
        st.caption("No events yet. Use Log.")
    else:
        for _, r in last_events.iterrows():
            label = timeline_label(r)
            notes = safe_str(r.get("notes")).strip()
            st.markdown(f"**{label}**" + (f"  \n{notes}" if notes else ""))

    # Night summary (collapsible)
    st.divider()
    with st.expander("Night summary", expanded=False):
        start_night, end_night = last_night_window(datetime.now(), night_start, night_end)
        night_df2 = df_all[(df_all["datetime"] >= start_night) & (df_all["datetime"] < end_night)].copy()

        night_feeds = night_df2[night_df2["event_type"] == "feed"].copy()
        night_bottle = night_feeds[night_feeds["feed_method"] == "bottle"].copy()
        night_breast = night_feeds[night_feeds["feed_method"] == "breast"].copy()

        a, b, c = st.columns(3)
        a.metric("Night feeds", int(len(night_feeds)))
        b.metric("Night bottle ml", int(night_bottle["volume_ml"].fillna(0).sum()))
        c.metric("Night breast min", int(night_breast["duration_min"].fillna(0).sum()))
        st.caption(f"Window: {start_night.strftime('%a %H:%M')} to {end_night.strftime('%a %H:%M')}")


# =========================================================
# LOG TAB (one screen mobile app feel)
# =========================================================
with tabs[1]:
    st.subheader("Log now")
    st.caption("One tap logging. Saved instantly.")

    # Duplicate guard prompt
    if st.session_state.pending_duplicate:
        pdg = st.session_state.pending_duplicate
        st.warning(
            f"Looks like you just tapped this {int(pdg['elapsed'])} seconds ago. Log again?",
            icon="⚠️",
        )
        cA, cB = st.columns(2)
        with cA:
            if st.button("Yes, log again", use_container_width=True):
                st.session_state.pending_duplicate = None
                # Allow next save by resetting last_action_at backward
                st.session_state.last_action_at = datetime.now() - timedelta(seconds=DUPLICATE_GUARD_SECONDS + 1)
                st.rerun()
        with cB:
            if st.button("No, cancel", use_container_width=True):
                st.session_state.pending_duplicate = None
                st.rerun()

    # Pull last values
    feeds_all = df_all[df_all["event_type"] == "feed"].sort_values("datetime")
    diapers_all = df_all[df_all["event_type"] == "diaper"].sort_values("datetime")

    last_bottle = feeds_all[feeds_all["feed_method"] == "bottle"].tail(1)
    last_diaper = diapers_all.tail(1)

    last_bottle_ml = None
    last_bottle_label = "None yet"
    if len(last_bottle):
        r = last_bottle.iloc[0]
        last_bottle_ml = safe_int(r.get("volume_ml"), 0)
        last_bottle_label = f"{r['datetime'].strftime('%H:%M')} {last_bottle_ml} ml"

    last_diaper_type = None
    last_diaper_label = "None yet"
    if len(last_diaper):
        r = last_diaper.iloc[0]
        last_diaper_type = safe_lower(r.get("diaper_type"))
        last_diaper_label = f"{r['datetime'].strftime('%H:%M')} {(last_diaper_type.title() if last_diaper_type else 'Diaper')}"

    # Quick Log Card
    with st.container(border=True):
        st.markdown("### Quick Log")
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Last bottle")
            st.write(last_bottle_label)
        with c2:
            st.caption("Last diaper")
            st.write(last_diaper_label)

        st.write("")

        # Buttons rows
        r1a, r1b, r1c = st.columns(3)
        with r1a:
            if st.button("🍼 15 ml", use_container_width=True):
                if duplicate_guard("bottle_15"):
                    row = normalize_row(now_floor_minute(), "feed", feed_method="bottle", volume_ml=15)
                    save_row(row, "Bottle 15 ml saved")
        with r1b:
            if st.button("🍼 30 ml", use_container_width=True):
                if duplicate_guard("bottle_30"):
                    row = normalize_row(now_floor_minute(), "feed", feed_method="bottle", volume_ml=30)
                    save_row(row, "Bottle 30 ml saved")
        with r1c:
            if st.button("🍼 Repeat last", use_container_width=True, disabled=(last_bottle_ml is None or last_bottle_ml <= 0)):
                if duplicate_guard("bottle_repeat"):
                    row = normalize_row(now_floor_minute(), "feed", feed_method="bottle", volume_ml=last_bottle_ml)
                    save_row(row, f"Bottle {last_bottle_ml} ml saved")

        r2a, r2b, r2c = st.columns(3)
        with r2a:
            if st.button("💧 Wet", use_container_width=True):
                if duplicate_guard("diaper_wet"):
                    row = normalize_row(now_floor_minute(), "diaper", diaper_type="wet")
                    save_row(row, "Wet diaper saved")
        with r2b:
            if st.button("💩 Dirty", use_container_width=True):
                if duplicate_guard("diaper_dirty"):
                    row = normalize_row(now_floor_minute(), "diaper", diaper_type="dirty")
                    save_row(row, "Dirty diaper saved")
        with r2c:
            if st.button("💩 Mixed", use_container_width=True):
                if duplicate_guard("diaper_mixed"):
                    row = normalize_row(now_floor_minute(), "diaper", diaper_type="mixed")
                    save_row(row, "Mixed diaper saved")

        r3a, r3b, r3c = st.columns(3)
        with r3a:
            if st.button("🧷 Repeat last diaper", use_container_width=True, disabled=(not last_diaper_type)):
                if duplicate_guard("diaper_repeat"):
                    row = normalize_row(now_floor_minute(), "diaper", diaper_type=last_diaper_type)
                    save_row(row, f"{last_diaper_type.title()} diaper saved")
        with r3b:
            if st.button("↩️ Undo last save", use_container_width=True):
                undo_last()
        with r3c:
            if st.button("✏️ Edit last save", use_container_width=True, disabled=(not safe_str(st.session_state.last_saved_row_id).strip())):
                st.session_state.show_edit_last = not st.session_state.show_edit_last
                st.rerun()

        if safe_str(st.session_state.last_saved_label).strip():
            st.caption(f"Last saved: {st.session_state.last_saved_label}")

    # Edit last saved entry (simple, safe)
    if st.session_state.show_edit_last:
        rid = safe_str(st.session_state.last_saved_row_id).strip()
        match = df_all[df_all["row_id"] == rid].copy()
        with st.container(border=True):
            st.markdown("### Edit last saved entry")
            if len(match) == 0:
                st.info("Could not find that entry anymore. Try Refresh.")
            else:
                row0 = match.iloc[0]
                dt0 = row0["datetime"].to_pydatetime()
                et0 = safe_lower(row0.get("event_type")) or "feed"

                with st.form("edit_last_form"):
                    c1, c2 = st.columns(2)
                    with c1:
                        ed = st.date_input("Date", value=dt0.date(), key="edit_last_d")
                    with c2:
                        et = st.time_input("Time", value=dt0.time().replace(second=0, microsecond=0), key="edit_last_t")

                    e_notes = st.text_input("Notes", value=safe_str(row0.get("notes")))

                    if et0 == "feed":
                        fm0 = safe_lower(row0.get("feed_method")) or "bottle"
                        fm = st.radio("Feed method", FEED_METHODS, horizontal=True, index=0 if fm0 != "breast" else 1)
                        if fm == "bottle":
                            ml = st.number_input("Bottle ml", 0, 300, safe_int(row0.get("volume_ml"), 0), step=5)
                            updated = normalize_row(make_dt(ed, et), "feed", feed_method="bottle", volume_ml=ml, notes=e_notes, row_id=rid)
                        else:
                            side0 = safe_lower(row0.get("side")) or "left"
                            sd = st.radio("Side", SIDES, horizontal=True, index=0 if side0 != "right" else 1)
                            mins = st.number_input("Duration min", 0, 240, safe_int(row0.get("duration_min"), 0), step=1)
                            updated = normalize_row(make_dt(ed, et), "feed", feed_method="breast", side=sd, duration_min=mins, notes=e_notes, row_id=rid)

                    elif et0 == "diaper":
                        dp0 = safe_lower(row0.get("diaper_type")) or "wet"
                        if dp0 not in DIAPER_TYPES:
                            dp0 = "wet"
                        dp = st.radio("Diaper type", DIAPER_TYPES, horizontal=True, index=DIAPER_TYPES.index(dp0))
                        updated = normalize_row(make_dt(ed, et), "diaper", diaper_type=dp, notes=e_notes, row_id=rid)

                    else:
                        w0 = safe_float(row0.get("weight_kg"), default=0.0) or 0.0
                        w = st.number_input("Weight kg", 0.0, 25.0, float(w0), step=0.01, format="%.2f")
                        updated = normalize_row(make_dt(ed, et), "weight", weight_kg=w, notes=e_notes, row_id=rid)

                    save = st.form_submit_button("Save changes", use_container_width=True)

                if save:
                    errs = validate_row(updated)
                    if errs:
                        st.error("Fix this before saving:\n\n" + "\n".join([f"• {e}" for e in errs]))
                    else:
                        try:
                            ok = ws_update_row(rid, updated)
                            if ok:
                                st.toast("Updated", icon="✅")
                                st.session_state.show_edit_last = False
                                refresh_data()
                            else:
                                st.error("Could not update. Row id not found.")
                        except Exception as e:
                            st.error("Could not update. Check internet and try again.")
                            st.caption(str(e))

                if st.button("Delete this entry", use_container_width=True, type="secondary"):
                    try:
                        ok = ws_delete_row(rid)
                        if ok:
                            st.toast("Deleted", icon="🗑️")
                            st.session_state.show_edit_last = False
                            st.session_state.last_saved_row_id = ""
                            st.session_state.last_saved_label = ""
                            refresh_data()
                        else:
                            st.error("Could not delete. Row id not found.")
                    except Exception as e:
                        st.error("Could not delete. Check internet and try again.")
                        st.caption(str(e))

    # Breast timer (kept, but calmer)
    st.divider()
    with st.container(border=True):
        st.markdown("### Breast timer")
        st.caption("Start, switch sides, stop, then save. Discard saves nothing.")

        def end_current_segment():
            if st.session_state.bf_segments and st.session_state.bf_segments[-1]["end"] is None:
                st.session_state.bf_segments[-1]["end"] = now_floor_minute()

        def start_new_segment(side_name: str):
            st.session_state.bf_segments.append({"side": side_name, "start": now_floor_minute(), "end": None})

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("🤱 Start", use_container_width=True):
                if not st.session_state.bf_timer_active:
                    start = now_floor_minute()
                    st.session_state.bf_timer_active = True
                    st.session_state.bf_segments = [{"side": "left", "start": start, "end": None}]
                    st.toast("Timer started on left", icon="⏱️")
                st.rerun()
        with c2:
            if st.button("⏹️ Stop", use_container_width=True, disabled=(not st.session_state.bf_timer_active)):
                end_current_segment()
                st.session_state.bf_timer_active = False
                st.rerun()
        with c3:
            if st.button("🧽 Discard", use_container_width=True):
                st.session_state.bf_segments = []
                st.session_state.bf_timer_active = False
                st.toast("Discarded", icon="🧽")
                st.rerun()

        if st.session_state.bf_segments:
            segs = st.session_state.bf_segments
            start_dt = segs[0]["start"]
            current_side = safe_lower(segs[-1]["side"]) or "left"
            elapsed = int((datetime.now() - start_dt).total_seconds() // 60)

            m1, m2, m3 = st.columns(3)
            m1.metric("Started", start_dt.strftime("%H:%M"))
            m2.metric("Side", current_side.title())
            m3.metric("Elapsed", f"{elapsed} min")

            s1, s2 = st.columns(2)
            with s1:
                if st.button("Switch left", use_container_width=True, disabled=(not st.session_state.bf_timer_active)):
                    end_current_segment()
                    start_new_segment("left")
                    st.rerun()
            with s2:
                if st.button("Switch right", use_container_width=True, disabled=(not st.session_state.bf_timer_active)):
                    end_current_segment()
                    start_new_segment("right")
                    st.rerun()

            # Compute segment table
            rows = []
            total_min = 0
            for s in segs:
                stt = s["start"]
                end = s["end"] if s["end"] is not None else now_floor_minute()
                mins = int((end - stt).total_seconds() // 60)
                mins = max(mins, 0)
                total_min += mins
                rows.append({"side": s["side"], "start": stt.strftime("%H:%M"), "end": end.strftime("%H:%M"), "min": mins})

            st.metric("Total breast minutes", total_min)

            show_segments = st.toggle("Show segment breakdown", value=False)
            if show_segments:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, height=200)

            timer_start_dt = segs[0]["start"]
            timer_end_dt = segs[-1]["end"] if segs[-1]["end"] is not None else now_floor_minute()

            if st.button("Save breast feed ✅", use_container_width=True, disabled=(total_min <= 0)):
                if duplicate_guard("breast_timer_save"):
                    breakdown = ", ".join([f"{r['side'][0].upper()}:{r['min']}m {r['start']}-{r['end']}" for r in rows])
                    notes = f"Timer {timer_start_dt.strftime('%H:%M')}–{timer_end_dt.strftime('%H:%M')} | {breakdown}"
                    row = normalize_row(timer_start_dt, "feed", feed_method="breast", duration_min=total_min, notes=notes)
                    save_row(row, "Breast feed saved")
                    st.session_state.bf_segments = []
                    st.session_state.bf_timer_active = False

    # Backdate + detailed logging (hidden)
    st.divider()
    with st.expander("Backdate or add details", expanded=False):
        st.caption("Use this only when you need a custom time, notes, or a non standard amount.")
        with st.form("full_log_form", clear_on_submit=True):
            n = now_floor_minute()
            c1, c2 = st.columns(2)
            with c1:
                d = st.date_input("Date", value=n.date())
            with c2:
                tt = st.time_input("Time", value=n.time())

            event_type = st.radio("Event", ["feed", "diaper", "weight"], horizontal=True)

            notes = st.text_input("Notes (optional)")

            feed_method = ""
            side = ""
            volume_ml = None
            duration_min = None
            diaper_type = ""
            weight_kg = None

            if event_type == "feed":
                feed_method = st.radio("Feed method", FEED_METHODS, horizontal=True, index=0)
                if feed_method == "bottle":
                    volume_ml = st.number_input("Bottle amount ml", 0, 300, 30, step=5)
                else:
                    side = st.radio("Side", SIDES, horizontal=True)
                    duration_min = st.number_input("Duration min", 0, 240, 15, step=1)
            elif event_type == "diaper":
                diaper_type = st.radio("Diaper type", DIAPER_TYPES, horizontal=True)
            else:
                weight_kg = st.number_input("Weight (kg)", 0.0, 25.0, 0.0, step=0.01, format="%.2f")

            submitted = st.form_submit_button("Save entry", use_container_width=True)

        if submitted:
            dt = make_dt(d, tt)
            row = normalize_row(
                dt,
                event_type,
                feed_method=feed_method,
                side=side,
                volume_ml=volume_ml,
                duration_min=duration_min,
                diaper_type=diaper_type,
                notes=notes,
                weight_kg=weight_kg,
            )
            save_row(row, "Saved")


# =========================================================
# INSIGHTS TAB (calm layered collapsible)
# =========================================================
with tabs[2]:
    st.subheader("Insights")
    st.caption("A calm summary first. Details are tucked away.")

    with st.expander("Quick summary (Today + last 24h)", expanded=True):
        today_date = datetime.now().date()
        today_df = df_all[df_all["date"] == today_date].copy() if len(df_all) else df_all.copy()
        feeds_today = today_df[today_df["event_type"] == "feed"].copy()
        diapers_today = today_df[today_df["event_type"] == "diaper"].copy()
        bottle_today = feeds_today[feeds_today["feed_method"] == "bottle"].copy()
        bottle_today_ml = int(bottle_today["volume_ml"].fillna(0).sum()) if len(bottle_today) else 0

        now_ts = pd.Timestamp.now()
        last24_start = now_ts - pd.Timedelta(hours=24)
        last24 = df_all[(df_all["datetime"] >= last24_start) & (df_all["datetime"] <= now_ts)].copy()
        feeds_24 = last24[last24["event_type"] == "feed"].copy()
        diapers_24 = last24[last24["event_type"] == "diaper"].copy()
        bottle_24 = feeds_24[feeds_24["feed_method"] == "bottle"].copy()
        bottle_24_ml = int(bottle_24["volume_ml"].fillna(0).sum()) if len(bottle_24) else 0

        longest_gap_24 = longest_feed_gap_last_24h(df_all)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Feeds today", int(len(feeds_today)))
        k2.metric("Diapers today", int(len(diapers_today)))
        k3.metric("Bottle ml today", bottle_today_ml)
        k4.metric("Longest stretch (24h)", f"{longest_gap_24} min")

        k5, k6, k7 = st.columns(3)
        k5.metric("Feeds last 24h", int(len(feeds_24)))
        k6.metric("Diapers last 24h", int(len(diapers_24)))
        k7.metric("Bottle ml last 24h", bottle_24_ml)

        st.write("")
        st.caption("Hourly activity (last 24h)")
        activity = hourly_activity_last_24h(df_all)
        if len(activity):
            st.line_chart(activity)
        else:
            st.caption("No events logged in the last 24 hours.")

        show_table = st.toggle("Show last 24h table", value=False)
        if show_table and len(last24):
            view = last24.sort_values("datetime", ascending=False)[["datetime", "event_type", "feed_method", "volume_ml", "duration_min", "diaper_type", "weight_kg", "notes"]]
            st.dataframe(view, use_container_width=True, height=320)

    with st.expander("Weekly patterns (last 7 days)", expanded=False):
        hist = compute_feed_hour_hist(df_all, days=7)
        peaks = top_peak_windows(hist, top_n=3)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Peak feed hours**")
            st.caption(", ".join(peaks) if peaks else "Not enough data yet.")
        with c2:
            status = cluster_feeding_status(df_all)
            st.markdown("**Cluster feeding**")
            st.caption(status["msg"])

        if len(hist):
            st.caption("Feeds by hour (last 7 days)")
            st.bar_chart(hist.rename(columns={"hour": "index"}).set_index("index")["count"])

        # Daily totals simplified
        cutoff = datetime.now().date() - timedelta(days=6)
        last7 = df_all[df_all["date"] >= cutoff].copy()
        if len(last7):
            feeds7 = last7[last7["event_type"] == "feed"].groupby("date").size().rename("feeds")
            diapers7 = last7[last7["event_type"] == "diaper"].groupby("date").size().rename("diapers")
            bottle7 = (
                last7[(last7["event_type"] == "feed") & (last7["feed_method"] == "bottle")]
                .groupby("date")["volume_ml"]
                .sum()
                .rename("bottle_ml")
            )
            daily = pd.DataFrame(index=sorted(last7["date"].unique()))
            daily = daily.join(feeds7, how="left").join(diapers7, how="left").join(bottle7, how="left").fillna(0)
            daily.index = [str(d) for d in daily.index]

            st.caption("Daily totals (feeds and diapers)")
            st.line_chart(daily[["feeds", "diapers"]])
            st.caption("Daily bottle ml")
            st.line_chart(daily[["bottle_ml"]])
        else:
            st.caption("Not enough data yet for weekly totals.")

    with st.expander("Nights (last 7 nights)", expanded=False):
        st.caption("Night window is set in the sidebar.")
        nights = []
        base_now = datetime.now()
        for i in range(7):
            ref = base_now - timedelta(days=i)
            ns, ne = last_night_window(ref, night_start, night_end)
            nd = df_all[(df_all["datetime"] >= ns) & (df_all["datetime"] < ne)].copy()
            nf = nd[nd["event_type"] == "feed"].copy()
            nb = nf[nf["feed_method"] == "bottle"].copy()
            nbr = nf[nf["feed_method"] == "breast"].copy()

            nights.append(
                {
                    "night_start_date": ns.date(),
                    "night_feeds": int(len(nf)),
                    "night_bottle_ml": int(nb["volume_ml"].fillna(0).sum()) if len(nb) else 0,
                    "night_breast_min": int(nbr["duration_min"].fillna(0).sum()) if len(nbr) else 0,
                }
            )

        night_df7 = pd.DataFrame(nights).sort_values("night_start_date")
        night_df7 = night_df7.rename(columns={"night_start_date": "index"}).set_index("index")

        c1, c2 = st.columns(2)
        with c1:
            st.caption("Night feeds")
            st.line_chart(night_df7["night_feeds"])
        with c2:
            st.caption("Night bottle ml")
            st.line_chart(night_df7["night_bottle_ml"])

        show_night_table = st.toggle("Show night table", value=False)
        if show_night_table:
            st.dataframe(night_df7.reset_index(), use_container_width=True, height=280)


# =========================================================
# WEIGHT TAB (growth tracker)
# =========================================================
with tabs[3]:
    st.subheader("Weight")
    st.caption("Log weights and see growth trend.")

    weights = df_all[df_all["event_type"] == "weight"].dropna(subset=["datetime"]).copy()
    weights = weights.dropna(subset=["weight_kg"]).copy()
    weights = weights.sort_values("datetime")

    # iPhone friendly quick add
    with st.container(border=True):
        st.markdown("### Add weight")
        c1, c2 = st.columns(2)
        with c1:
            w = st.number_input("Weight (kg)", min_value=0.0, max_value=25.0, value=0.0, step=0.01, format="%.2f")
        with c2:
            use_now = st.checkbox("Use current time", value=True)

        if use_now:
            dt = now_floor_minute()
            d = dt.date()
            tt = dt.time()
        else:
            d = st.date_input("Date measured", value=datetime.now().date(), key="w_date")
            tt = st.time_input("Time measured", value=now_floor_minute().time(), key="w_time")

        cA, cB = st.columns(2)
        with cA:
            if st.button("Save weight", use_container_width=True, disabled=(w <= 0)):
                if duplicate_guard("weight_save"):
                    row = normalize_row(make_dt(d, tt), "weight", weight_kg=w)
                    save_row(row, "Weight saved")
        with cB:
            # Convenience birth weight logger (persists, no extra settings needed)
            with st.popover("Add birth weight"):
                st.caption("This logs a weight entry tagged as birth weight.")
                bd = st.date_input("Birth date", value=datetime.now().date(), key="bw_d")
                bt = st.time_input("Birth time", value=time(12, 0), key="bw_t")
                bw = st.number_input("Birth weight (kg)", 0.0, 25.0, 0.0, step=0.01, format="%.2f", key="bw_val")
                if st.button("Save birth weight", use_container_width=True, disabled=(bw <= 0)):
                    if duplicate_guard("birth_weight_save"):
                        row = normalize_row(make_dt(bd, bt), "weight", weight_kg=bw, notes="birth weight")
                        save_row(row, "Birth weight saved")

    st.divider()

    if len(weights) == 0:
        st.info("No weight entries yet.")
    else:
        latest = weights.iloc[-1]
        latest_w = float(latest["weight_kg"])
        latest_dt = latest["datetime"]

        cutoff = pd.Timestamp.now() - pd.Timedelta(days=14)
        last_14 = weights[weights["datetime"] >= cutoff]
        delta_14 = None
        if len(last_14) >= 2:
            delta_14 = float(last_14.iloc[-1]["weight_kg"]) - float(last_14.iloc[0]["weight_kg"])

        # Birth weight detection
        birth = weights[weights["notes"].astype("string").str.lower().str.contains("birth weight", na=False)].copy()
        birth_delta = None
        if len(birth):
            bw = float(birth.sort_values("datetime").iloc[0]["weight_kg"])
            birth_delta = latest_w - bw

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Latest weight", f"{latest_w:.3f} kg")
        k2.metric("Measured", latest_dt.strftime("%Y-%m-%d %H:%M"))
        k3.metric("Change (14 days)", f"{delta_14:.3f} kg" if delta_14 is not None else "Not enough data")
        k4.metric("Change since birth", f"{birth_delta:.3f} kg" if birth_delta is not None else "Add birth weight")

        # Chart with gentle smoothing (weekly average)
        chart = weights[["datetime", "weight_kg"]].copy()
        chart["date"] = chart["datetime"].dt.date
        daily = chart.groupby("date")["weight_kg"].mean().reset_index()
        daily = daily.rename(columns={"date": "index"}).set_index("index")
        daily.index = pd.to_datetime(daily.index)
        daily = daily.sort_index()

        # Smooth: 7 day rolling average if enough points
        smooth = daily.copy()
        smooth["rolling_7d"] = smooth["weight_kg"].rolling(window=7, min_periods=1).mean()

        st.caption("Growth chart (daily)")
        st.line_chart(smooth[["weight_kg", "rolling_7d"]])

        show_table = st.toggle("Show weight table", value=False)
        if show_table:
            show = weights.sort_values("datetime", ascending=False)[["datetime", "weight_kg", "notes", "row_id"]]
            st.dataframe(show, use_container_width=True, height=320)


# =========================================================
# HISTORY TAB (filters + day summary + grouped timeline)
# =========================================================
with tabs[4]:
    st.subheader("History")
    st.caption("Review, filter, and edit entries safely.")

    if len(df_all) == 0:
        st.caption("No entries yet.")
    else:
        # Local filters for History (in addition to sidebar)
        hf1, hf2, hf3 = st.columns(3)
        with hf1:
            type_filter = st.multiselect("Event types", EVENT_TYPES, default=EVENT_TYPES)
        with hf2:
            history_day = st.date_input("Day", value=datetime.now().date())
        with hf3:
            show_grouped = st.checkbox("Group by morning/afternoon/evening/night", value=True)

        day_df = df_all[df_all["date"] == history_day].copy()
        if type_filter:
            day_df = day_df[day_df["event_type"].isin(type_filter)].copy()
        day_df = day_df.sort_values("datetime")

        # Day summary row
        feeds = day_df[day_df["event_type"] == "feed"].copy()
        diapers = day_df[day_df["event_type"] == "diaper"].copy()
        weights = day_df[day_df["event_type"] == "weight"].copy()
        bottle = feeds[feeds["feed_method"] == "bottle"].copy()
        breast = feeds[feeds["feed_method"] == "breast"].copy()

        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Feeds", int(len(feeds)))
        s2.metric("Diapers", int(len(diapers)))
        s3.metric("Bottle ml", int(bottle["volume_ml"].fillna(0).sum()) if len(bottle) else 0)
        s4.metric("Breast min", int(breast["duration_min"].fillna(0).sum()) if len(breast) else 0)
        s5.metric("Weights", int(len(weights)))

        st.divider()

        # Timeline
        if len(day_df) == 0:
            st.caption("No events for this day.")
        else:
            if show_grouped:
                day_df["pod"] = day_df["datetime"].apply(part_of_day)
                for pod in ["Morning", "Afternoon", "Evening", "Night"]:
                    block = day_df[day_df["pod"] == pod]
                    if len(block) == 0:
                        continue
                    st.markdown(f"### {pod}")
                    for _, r in block.iterrows():
                        label = timeline_label(r)
                        notes = safe_str(r.get("notes")).strip()
                        st.markdown(f"**{label}**" + (f"  \n{notes}" if notes else ""))
            else:
                for _, r in day_df.iterrows():
                    label = timeline_label(r)
                    notes = safe_str(r.get("notes")).strip()
                    st.markdown(f"**{label}**" + (f"  \n{notes}" if notes else ""))

        st.divider()

        # Edit or delete
        st.markdown("### Edit or delete an entry")
        st.caption("Delete cannot be undone.")

        recent = df_all.sort_values("datetime", ascending=False).head(500).copy()
        recent["pick_label"] = recent.apply(
            lambda x: f"{x['datetime'].strftime('%Y-%m-%d %H:%M')} | {timeline_label(x)}",
            axis=1,
        )
        selected = st.selectbox("Pick an entry", recent["pick_label"].tolist())
        row = recent[recent["pick_label"] == selected].iloc[0]
        rid = safe_str(row.get("row_id")).strip()
        dt0 = row["datetime"].to_pydatetime()

        et0 = safe_lower(row.get("event_type")) or "feed"
        if et0 not in EVENT_TYPES:
            et0 = "feed"

        with st.form("edit_any_form"):
            c1, c2 = st.columns(2)
            with c1:
                ed = st.date_input("Date", value=dt0.date(), key="edit_any_d")
            with c2:
                et = st.time_input("Time", value=dt0.time().replace(second=0, microsecond=0), key="edit_any_t")

            e_event = st.radio("Event", EVENT_TYPES, horizontal=True, index=EVENT_TYPES.index(et0))
            e_notes = st.text_input("Notes", value=safe_str(row.get("notes")))

            # Fields per type
            e_feed_method = ""
            e_side = ""
            e_volume = None
            e_duration = None
            e_diaper_type = ""
            e_weight = None

            if e_event == "feed":
                fm_val = safe_lower(row.get("feed_method")) or "bottle"
                e_feed_method = st.radio("Feed method", FEED_METHODS, horizontal=True, index=0 if fm_val != "breast" else 1)
                if e_feed_method == "bottle":
                    e_volume = st.number_input("Bottle amount ml", 0, 300, safe_int(row.get("volume_ml"), 0), step=5)
                else:
                    s0 = safe_lower(row.get("side")) or "left"
                    e_side = st.radio("Side", SIDES, horizontal=True, index=0 if s0 != "right" else 1)
                    e_duration = st.number_input("Duration min", 0, 240, safe_int(row.get("duration_min"), 0), step=1)

            elif e_event == "diaper":
                dtp0 = safe_lower(row.get("diaper_type")) or "wet"
                if dtp0 not in DIAPER_TYPES:
                    dtp0 = "wet"
                e_diaper_type = st.radio("Diaper type", DIAPER_TYPES, horizontal=True, index=DIAPER_TYPES.index(dtp0))

            else:
                cur_w = safe_float(row.get("weight_kg"), default=0.0) or 0.0
                e_weight = st.number_input("Weight (kg)", 0.0, 25.0, float(cur_w), step=0.01, format="%.2f")

            save_edit = st.form_submit_button("Save changes", use_container_width=True)

        if save_edit:
            new_dt = make_dt(ed, et)
            updated = normalize_row(
                new_dt,
                e_event,
                feed_method=e_feed_method,
                side=e_side,
                volume_ml=e_volume,
                duration_min=e_duration,
                diaper_type=e_diaper_type,
                notes=e_notes,
                weight_kg=e_weight,
                row_id=rid,
            )
            errs = validate_row(updated)
            if errs:
                st.error("Fix this before saving:\n\n" + "\n".join([f"• {e}" for e in errs]))
            else:
                try:
                    ok = ws_update_row(rid, updated)
                    if ok:
                        st.toast("Updated", icon="✅")
                        refresh_data()
                    else:
                        st.error("Could not update. Row id not found.")
                except Exception as e:
                    st.error("Could not update. Check internet and try again.")
                    st.caption(str(e))

        if st.button("Delete this entry", use_container_width=True, type="secondary"):
            try:
                ok = ws_delete_row(rid)
                if ok:
                    st.toast("Deleted", icon="🗑️")
                    refresh_data()
                else:
                    st.error("Could not delete. Row id not found.")
            except Exception as e:
                st.error("Could not delete. Check internet and try again.")
                st.caption(str(e))


# =========================================================
# EXPORT TAB
# =========================================================
with tabs[5]:
    st.subheader("Export")
    st.caption("Download CSV for sharing or backup.")

    if len(df_all) == 0:
        st.caption("No data yet.")
    else:
        min_d = df_all["date"].min()
        max_d = df_all["date"].max()
        export_range = st.date_input("Export range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
        if isinstance(export_range, tuple) and len(export_range) == 2:
            ex_start, ex_end = export_range
        else:
            ex_start = ex_end = export_range

        df_export = df_all[(df_all["date"] >= ex_start) & (df_all["date"] <= ex_end)].copy()
        st.metric("Entries in export", int(len(df_export)))

        csv_df = df_export.copy()
        csv_df["datetime"] = csv_df["datetime"].dt.strftime("%Y-%m-%d %H:%M")
        csv_bytes = csv_df[EXPECTED_COLS].to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download CSV",
            data=csv_bytes,
            file_name="baby_log_export.csv",
            mime="text/csv",
            use_container_width=True,
        )


# =========================================================
# DATA TAB
# =========================================================
with tabs[6]:
    st.subheader("Raw data and backup")
    st.caption("This is your source of truth. You can always download a full backup.")

    st.dataframe(df_all.sort_values("datetime", ascending=False), use_container_width=True, height=520)

    csv_bytes = df_all.copy()
    csv_bytes["datetime"] = csv_bytes["datetime"].dt.strftime("%Y-%m-%d %H:%M")
    out = csv_bytes[EXPECTED_COLS].to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download full CSV backup",
        data=out,
        file_name="baby_log_full.csv",
        mime="text/csv",
        use_container_width=True,
    )
