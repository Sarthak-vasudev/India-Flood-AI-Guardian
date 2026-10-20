
import os, sys, sqlite3
import streamlit as st
import pandas as pd
import joblib
import folium
from streamlit_folium import st_folium
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import alert_engine, weather_fetcher, db_manager

BASE        = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(BASE, "models")
MODEL_PATH  = os.path.join(MODELS_DIR, "knn_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
FEAT_PATH   = os.path.join(MODELS_DIR, "features.pkl")

# CSV lookup — supports data/ subfolder or root
for _f in ["data/indian_district_master.csv", "districts.csv", "indian_district_master.csv"]:
    _p = os.path.join(BASE, _f)
    if os.path.exists(_p):
        DATA_PATH = _p
        break
else:
    st.error("❌ districts.csv not found.")
    st.stop()

INT2LABEL = {0: "Low", 1: "Moderate", 2: "High", 3: "Extreme"}
STYLES = {
    "Extreme" : {"color": "#C62828", "fill": "#EF5350", "radius": 12},
    "High"    : {"color": "#E65100", "fill": "#FF9800", "radius": 10},
    "Moderate": {"color": "#F9A825", "fill": "#FDD835", "radius":  8},
    "Low"     : {"color": "#2E7D32", "fill": "#66BB6A", "radius":  6},
}
RISK_EMOJI = {"Extreme": "🔴", "High": "🟠", "Moderate": "🟡", "Low": "🟢"}

# ── Init DB ────────────────────────────────────────────────────────────────────
db_manager.init_db()

st.set_page_config(page_title="India Flood AI Guardian", page_icon="🌊", layout="wide")
st.markdown("<style>[data-testid='stSidebar']{display:none}</style>", unsafe_allow_html=True)

# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    for p in [MODEL_PATH, SCALER_PATH, FEAT_PATH]:
        if not os.path.exists(p):
            return None, None, None
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH), joblib.load(FEAT_PATH)

knn, scaler, FEATURES = load_model()
if knn is None:
    st.error("❌ Model not found. Run: python train_model.py")
    st.stop()

# ── Load static district data ──────────────────────────────────────────────────
@st.cache_data
def load_static():
    df = pd.read_csv(DATA_PATH)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    static = ["elevation_m", "slope_deg", "soil_drainage",
              "historical_floods", "river_proximity_km"]
    df = df.dropna(subset=["lat", "lon"] + static)
    df = df[df["district"] != "Bhubaneswar2"]
    return df

df_static = load_static()

# ── 30-min window key ──────────────────────────────────────────────────────────
now     = datetime.now()
now_key = now.strftime("%Y%m%d_%H") + ("00" if now.minute < 30 else "30")

# ── Fetch live weather (cached 30 min) ─────────────────────────────────────────
@st.cache_data(ttl=1800)
def get_live(ts_key):
    return weather_fetcher.fetch_live_weather(df_static)

with st.spinner("🌐 Fetching live data..."):
    result = get_live(now_key)

df, failed_districts = result if isinstance(result, tuple) else (result, [])

# ── KNN prediction ─────────────────────────────────────────────────────────────
X      = scaler.transform(df[FEATURES].values)
idxs   = knn.predict(X)
probas = knn.predict_proba(X)
df     = df.copy()
df["knn_risk"] = [INT2LABEL[i] for i in idxs]
df["knn_conf"] = [round(float(p.max()) * 100, 1) for p in probas]

# ── Auto SMS alerts — once per 30-min window ──────────────────────────────────
if st.session_state.get("last_alert_key") != now_key:
    st.session_state.last_alert_key   = now_key
    st.session_state.alerts_sent      = []
    st.session_state.alerts_fired     = False
    st.session_state.clicked_district = None
    st.session_state.clicked_risk     = None

if not st.session_state.get("alerts_fired"):
    sent = alert_engine.process_all_alerts(df[["district", "knn_risk"]], now_key)
    st.session_state.alerts_sent  = sent
    st.session_state.alerts_fired = True

# ── Toast notifications — district-level only ──────────────────────────────────
if st.session_state.get("alerts_sent"):
    st.markdown(
        "<style>@keyframes fadeout{from{opacity:1}to{opacity:0;visibility:hidden}}</style>",
        unsafe_allow_html=True,
    )
    by_district: dict = {}
    for a in st.session_state.alerts_sent:
        d = a["district"]
        if d not in by_district:
            by_district[d] = {"risk": a["risk"], "sent": 0, "failed": 0}
        if a["sent"]:
            by_district[d]["sent"] += 1
        else:
            by_district[d]["failed"] += 1

    toasts = ""
    for i, (district, info) in enumerate(by_district.items()):
        top = 90 + i * 110
        if info["sent"] > 0:
            msg    = (f"🚨 {district} — {info['risk']} Risk!<br>"
                      f"✅ {info['sent']} alert(s) dispatched.<br>"
                      f"⏱ Reminder in {alert_engine.REMINDER_DELAY_SECONDS}s.")
            border = "#ef5350"
        else:
            msg    = f"⚠️ {district} — SMS failed ({info['failed']} user(s))."
            border = "#ffa726"
        toasts += (
            f'<div style="position:fixed;top:{top}px;right:24px;z-index:99999;'
            f'background:#1a1a2e;border-left:5px solid {border};border-radius:10px;'
            f'padding:16px 20px;max-width:320px;box-shadow:0 4px 20px rgba(0,0,0,0.6);'
            f'font-family:sans-serif;font-size:14px;color:#ffffff;line-height:1.8;'
            f'animation:fadeout 1s ease 6s forwards;">{msg}</div>'
        )
    st.markdown(toasts, unsafe_allow_html=True)

# ── Failed fetch warning ───────────────────────────────────────────────────────
if failed_districts:
    st.warning(
        f"⚠️ Live data fetch failed for {len(failed_districts)} district(s): "
        f"{', '.join(failed_districts[:5])}{'...' if len(failed_districts) > 5 else ''}. "
        f"Risk may be underestimated for those districts."
    )

# ── Map builder ────────────────────────────────────────────────────────────────
def build_map(df: pd.DataFrame) -> folium.Map:
    monitored = {u["district"].strip().lower() for u in db_manager.get_all_users()}

    # OpenStreetMap tiles — free, keyless. (Was "CartoDB positron", which now
    # requires a CARTO API key as of Aug 2026 or shows a watermark instead.)
    m = folium.Map(location=[20.5937, 78.9629], zoom_start=5, tiles="OpenStreetMap")

    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:9999;
         background:white;padding:12px 18px;border-radius:10px;
         box-shadow:0 2px 8px rgba(0,0,0,.3);font-size:13px;line-height:2">
      <b>🌊 Flood Risk — Live KNN</b><br>
      <span style="color:#C62828">⬤</span> Extreme &nbsp;
      <span style="color:#E65100">⬤</span> High<br>
      <span style="color:#F9A825">⬤</span> Moderate &nbsp;
      <span style="color:#2E7D32">⬤</span> Low<br>
      <span style="color:black">⬤</span> Monitored district<br>
      <span style="font-size:11px;color:#555">💡 Click any district to trigger manual alert</span>
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))

    for _, row in df.iterrows():
        risk   = row["knn_risk"]
        s      = STYLES[risk]
        is_mon = row["district"].strip().lower() in monitored
        popup_html = (
            f"<b>{row['district']}</b><br>"
            f"<span style='color:{s['color']};font-weight:bold'>{risk} ({row['knn_conf']}%)</span>"
            f"<hr style='margin:4px 0'>"
            f"🌧 Rain: {row['live_precip_mm']:.1f} mm &nbsp; "
            f"💨 Wind: {row['live_wind_kmh']:.1f} km/h<br>"
            f"🌱 Soil: {row['live_soil_moisture']:.3f} &nbsp; "
            f"❄️ Snow: {row['live_snowmelt_mm']:.1f} mm<br>"
            f"🌊 Discharge: {row['live_discharge']:.0f} m³/s<br>"
            f"⛰ Elev: {row['elevation_m']} m &nbsp; "
            f"🚣 River: {row['river_proximity_km']} km<br>"
            f"📊 Hist floods: {int(row['historical_floods'])}"
            + ("<br><b>📍 Monitored</b>" if is_mon else "")
            + "<br><span style='color:#aaa;font-size:11px'>👇 Scroll down to send manual alert</span>"
        )
        folium.CircleMarker(
            location     = [row["lat"], row["lon"]],
            radius       = s["radius"] + (5 if is_mon else 0),
            color        = "black" if is_mon else s["color"],
            weight       = 3 if is_mon else 1,
            fill         = True,
            fill_color   = s["fill"],
            fill_opacity = 0.85,
            popup        = folium.Popup(popup_html, max_width=260),
            tooltip      = f"{'📍 ' if is_mon else ''}{row['district']} — {risk} ({row['knn_conf']}%)",
        ).add_to(m)
    return m

# ── Page header ────────────────────────────────────────────────────────────────
st.title("🇮🇳 India Flood AI Guardian")
st.caption(
    f"🟢 Live: OpenWeatherMap + Open-Meteo · "
    f"{now.strftime('%d %b %Y %H:%M')} · Refreshes every 30 min"
)

# Metrics row
counts = df["knn_risk"].value_counts()
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("📍 Districts",   len(df))
c2.metric("🔴 Extreme",     counts.get("Extreme",  0))
c3.metric("🟠 High",        counts.get("High",     0))
c4.metric("🟡 Moderate",    counts.get("Moderate", 0))
c5.metric("🟢 Low",         counts.get("Low",      0))

# ── Map — capture clicked marker via tooltip ───────────────────────────────────
map_data = st_folium(
    build_map(df),
    width="100%",
    height=650,
    key=f"flood_map_{now_key}",
    returned_objects=["last_object_clicked_tooltip"],
)

# ── Parse clicked district from tooltip ───────────────────────────────────────
# Tooltip format: "📍 Visakhapatnam — Extreme (98.2%)" or "Chennai — Low (76.0%)"
clicked_raw = (map_data or {}).get("last_object_clicked_tooltip") or ""
if clicked_raw:
    label = clicked_raw.replace("📍 ", "").strip()
    if " — " in label:
        parts            = label.split(" — ")
        new_district     = parts[0].strip()
        risk_word        = parts[1].strip().split(" ")[0].strip() if len(parts) > 1 else ""
        new_risk         = risk_word if risk_word in STYLES else "Low"
        # Only update state when a genuinely new district is clicked
        if new_district != st.session_state.get("clicked_district"):
            st.session_state.clicked_district = new_district
            st.session_state.clicked_risk     = new_risk

# ── Manual Alert Panel ────────────────────────────────────────────────────────
st.divider()
st.subheader("📲 Manual Alert")

selected = st.session_state.get("clicked_district")
sel_risk = st.session_state.get("clicked_risk", "Low")

if not selected:
    st.markdown(
        "<p style='color:#888;font-size:14px;text-align:center;padding:12px'>"
        "💡 Click any district marker on the map above — "
        "an alert panel will appear here.</p>",
        unsafe_allow_html=True,
    )
else:
    # Count registered users in selected district from DB
    _c = sqlite3.connect(db_manager.DB_PATH, check_same_thread=False)
    _c.row_factory = sqlite3.Row
    reg_users = _c.execute(
        "SELECT id, name, phone, district FROM users "
        "WHERE active=1 AND LOWER(district)=?",
        (selected.strip().lower(),)
    ).fetchall()
    _c.close()
    user_count = len(reg_users)

    style_col = STYLES.get(sel_risk, STYLES["Low"])
    risk_em   = RISK_EMOJI.get(sel_risk, "🟢")

    # Info card
    st.markdown(
        f"""
        <div style="background:#1a1a2e;border-left:6px solid {style_col['color']};
             border-radius:10px;padding:18px 24px;margin-bottom:16px;color:#fff;">
          <h3 style="margin:0 0 6px 0">{risk_em} {selected}</h3>
          <p style="margin:0;font-size:15px;opacity:0.9">
            Current Risk: <b style="color:{style_col['fill']}">{sel_risk}</b>
            &nbsp;|&nbsp;
            Registered users in this district: <b>{user_count}</b>
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_send, col_clear = st.columns([3, 1])

    with col_send:
        if sel_risk in alert_engine.ALERT_LEVELS:
            if user_count == 0:
                st.info(
                    f"ℹ️ No registered users found in **{selected}**. "
                    f"Add users in db_manager._SEED_USERS to enable alerts."
                )
            else:
                if st.button(
                    f"📲 Send {sel_risk} Alert to {user_count} user(s) in {selected}",
                    type="primary",
                ):
                    single_row = df[df["district"].str.strip().str.lower() == selected.strip().lower()]
                    if not single_row.empty:
                        # Use a unique manual key → bypasses 30-min window dedup
                        manual_key = f"MANUAL_{now.strftime('%Y%m%d_%H%M%S')}"
                        with st.spinner(f"Sending SMS to {user_count} user(s) in {selected}..."):
                            sent_results = alert_engine.process_all_alerts(
                                single_row[["district", "knn_risk"]], manual_key
                            )
                        ok   = sum(1 for r in sent_results if r.get("sent"))
                        fail = len(sent_results) - ok
                        if ok > 0:
                            st.success(
                                f"✅ SMS alert sent to **{ok}** user(s) in **{selected}**. "
                                f"A reminder will follow in {alert_engine.REMINDER_DELAY_SECONDS}s."
                            )
                        if fail > 0:
                            errs = [r.get("error", "") for r in sent_results if not r.get("sent")]
                            st.error(f"❌ {fail} SMS(s) failed. Errors: {'; '.join(errs[:3])}")
                        if not sent_results:
                            st.info("ℹ️ Alert engine returned no results for this district.")
                    else:
                        st.warning(f"⚠️ {selected} not found in live prediction data.")
        else:
            st.info(
                f"ℹ️ **{selected}** is currently **{sel_risk}** risk. "
                f"Manual alerts are only available for Moderate / High / Extreme districts."
            )

    with col_clear:
        if st.button("✖ Clear", type="secondary"):
            st.session_state.clicked_district = None
            st.session_state.clicked_risk     = None
            st.rerun()

# ── District data table ────────────────────────────────────────────────────────
st.divider()
st.subheader("📋 Live District Data")
st.caption(
    "🟢 Rain, Wind, Soil Moisture, Snowmelt, Discharge — live (30 min cache) · "
    "⚪ Elevation, River km, Hist Floods — static from CSV"
)
show = df[[
    "district", "knn_risk",
    "live_precip_mm", "live_wind_kmh", "live_soil_moisture",
    "live_snowmelt_mm", "live_discharge",
    "elevation_m", "river_proximity_km", "historical_floods"
]].copy()
show.columns = [
    "District", "Risk",
    "Rain mm", "Wind km/h", "Soil Moist",
    "Snow mm", "Discharge m³/s",
    "Elev m", "River km", "Hist Floods"
]
st.dataframe(show.sort_values("Risk"), width="stretch")

# ── Model Performance — reads whatever train_model.py last generated ────────
st.divider()
st.subheader("📊 Model Performance")
st.caption(
    "Read from models/ — generated the last time train_model.py was run. "
    "Not live-computed; re-run train_model.py to refresh these."
)

HP_PATH   = os.path.join(MODELS_DIR, "hyperparameter_report.txt")
CM_PATH   = os.path.join(MODELS_DIR, "confusion_matrix.png")
ALGO_PATH = os.path.join(MODELS_DIR, "algorithm_comparison.csv")
IMP_PATH  = os.path.join(MODELS_DIR, "feature_importance.csv")

if not any(os.path.exists(p) for p in [HP_PATH, CM_PATH, ALGO_PATH, IMP_PATH]):
    st.info(
        "ℹ️ No evaluation metrics found yet. Run `python train_model.py` "
        "first — it saves hyperparameter_report.txt, confusion_matrix.png, "
        "algorithm_comparison.csv, and feature_importance.csv into models/."
    )
else:
    # ── Parse hyperparameter_report.txt into a simple key:value dict ───────
    hp = {}
    if os.path.exists(HP_PATH):
        with open(HP_PATH) as f:
            for line in f:
                if ":" in line:
                    k, v = line.split(":", 1)
                    hp[k.strip()] = v.strip()

    if hp:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Test Accuracy",          hp.get("Test accuracy", hp.get("accuracy", "—")))
        m2.metric("Balanced Accuracy",      hp.get("Test balanced accuracy", hp.get("balanced_accuracy", "—")))
        m3.metric("F1 (macro)",             hp.get("Test F1 (macro)", hp.get("f1_macro", "—")))
        m4.metric("MCC",                    hp.get("Test MCC", hp.get("mcc", "—")))

        m5, m6, m7 = st.columns(3)
        m5.metric("Best k",                 hp.get("Best k", "—"))
        m6.metric("Best weights",           hp.get("Best weights", "—"))
        m7.metric("Best CV f1_macro",       hp.get("Best CV f1_macro", "—"))

        with st.expander("Full hyperparameter report (raw)"):
            st.text(open(HP_PATH).read())

    col_cm, col_algo = st.columns(2)

    with col_cm:
        st.markdown("**Confusion Matrix**")
        if os.path.exists(CM_PATH):
            st.image(CM_PATH, width="stretch")
        else:
            st.caption("confusion_matrix.png not found.")

    with col_algo:
        st.markdown("**Algorithm Comparison** (same held-out test set)")
        if os.path.exists(ALGO_PATH):
            algo_df = pd.read_csv(ALGO_PATH)
            st.dataframe(algo_df, hide_index=True, width="stretch")
        else:
            st.caption("algorithm_comparison.csv not found.")

    if os.path.exists(IMP_PATH):
        st.markdown("**Permutation Feature Importance**")
        st.caption("KNN has no native feature importance — this measures the "
                    "drop in performance when each feature is shuffled.")
        imp_df = pd.read_csv(IMP_PATH)
        st.dataframe(imp_df, hide_index=True, width="stretch")

    st.warning(
        "⚠️ These metrics reflect whatever data trained the current model. "
        "If that data still contains synthetic or label-derived features "
        "(rather than real historical weather matched against real, "
        "independently-sourced flood events), these numbers do not "
        "represent validated real-world flood-prediction performance."
    )