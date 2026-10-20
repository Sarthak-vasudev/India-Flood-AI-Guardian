
import os
import time
import requests
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()  # reads .env in the project root

# ── API config ────────────────────────────────────────────────────────
OWM_KEY      = os.environ["OWM_API_KEY"]   # was hardcoded — now from .env
OWM_URL      = "https://api.openweathermap.org/data/2.5/weather"
WEATHER_URL  = "https://api.open-meteo.com/v1/forecast"
FLOOD_URL    = "https://flood-api.open-meteo.com/v1/flood"
BATCH_SIZE   = 20
TIMEOUT      = 45          # was 20 — too tight for multi-coordinate batches
RETRY_COUNT  = 2           # 1 initial attempt + 1 retry
RETRY_DELAY  = 3           # seconds between attempts


# ══════════════════════════════════════════════════════════════════════
# OpenWeatherMap — one call per district, run in parallel
# ══════════════════════════════════════════════════════════════════════
def _fetch_owm_one(row: dict) -> tuple:
    """Returns (district, precip_1h_mm, wind_kmh, ok). Falls back to 0 on error."""
    district = row["district"]
    try:
        r = requests.get(OWM_URL, params={
            "lat":   row["lat"],
            "lon":   row["lon"],
            "appid": OWM_KEY,
            "units": "metric",   # wind in m/s, temp in C
        }, timeout=15)
        if r.status_code == 200:
            data     = r.json()
            precip   = float(data.get("rain", {}).get("1h", 0.0))
            wind_kmh = round(float(data.get("wind", {}).get("speed", 0.0)) * 3.6, 2)
            return district, precip, wind_kmh, True
        elif r.status_code == 401:
            print(f"[OWM] 401 Unauthorized for {district} — check OWM_API_KEY in .env "
                  f"(new keys can take up to 2 hours to activate)")
        elif r.status_code == 429:
            print(f"[OWM] 429 Rate limit — too many calls per minute")
    except Exception as e:
        print(f"[OWM] {district}: {e}")
    return district, 0.0, 0.0, False


def _fetch_owm_all(rows: list) -> dict:
    """Parallel OWM fetch. Returns {district: (precip_mm, wind_kmh, ok)}."""
    results = {}
    with ThreadPoolExecutor(max_workers=20) as ex:
        futures = {ex.submit(_fetch_owm_one, r): r["district"] for r in rows}
        for fut in as_completed(futures):
            try:
                district, p, w, ok = fut.result()
                results[district] = (p, w, ok)
            except Exception:
                pass
    return results


# ══════════════════════════════════════════════════════════════════════
# Open-Meteo — batch fetch for soil, snow, discharge
# ══════════════════════════════════════════════════════════════════════
def _parse_weather_batch(data, districts: list) -> dict:
    results = {}
    if not isinstance(data, list):
        data = [data]
    for i, d in enumerate(data):
        if i >= len(districts):
            break
        district = districts[i]
        try:
            daily  = d.get("daily",  {})
            hourly = d.get("hourly", {})

            snow_vals = daily.get("snowfall_sum") or [0]
            snow = float(sum(v for v in snow_vals if v is not None))

            sm = hourly.get("soil_moisture_0_to_1cm") or []
            valid_sm = [v for v in sm[:8] if v is not None]
            soil = float(np.mean(valid_sm)) if valid_sm else 0.0

            results[district] = (soil, snow, True)
        except Exception:
            results[district] = (0.0, 0.0, False)
    return results


def _parse_flood_batch(data, districts: list) -> dict:
    results = {}
    if not isinstance(data, list):
        data = [data]
    for i, d in enumerate(data):
        if i >= len(districts):
            break
        district = districts[i]
        try:
            vals = d.get("daily", {}).get("river_discharge") or [0]
            results[district] = (float(vals[0] or 0), True)
        except Exception:
            results[district] = (0.0, False)
    return results


def _fetch_batch_weather(batch_rows: list) -> dict:
    """
    Fetch soil moisture + snowfall for a batch of districts.
    Retries once on failure before falling back to zeros (flagged ok=False).
    """
    lats      = ",".join(str(r["lat"]) for r in batch_rows)
    lons      = ",".join(str(r["lon"]) for r in batch_rows)
    districts = [r["district"] for r in batch_rows]

    for attempt in range(RETRY_COUNT):
        try:
            r = requests.get(WEATHER_URL, params={
                "latitude":      lats,
                "longitude":     lons,
                "daily":         "snowfall_sum",
                "hourly":        "soil_moisture_0_to_1cm",
                "timezone":      "Asia/Kolkata",
                "past_days":     7,
                "forecast_days": 1,
            }, timeout=TIMEOUT)
            if r.status_code == 200:
                return _parse_weather_batch(r.json(), districts)
            else:
                print(f"[Open-Meteo] batch HTTP {r.status_code} "
                      f"(attempt {attempt+1}/{RETRY_COUNT})")
        except Exception as e:
            print(f"[Open-Meteo] batch error (attempt {attempt+1}/{RETRY_COUNT}): {e}")

        if attempt < RETRY_COUNT - 1:
            time.sleep(RETRY_DELAY)

    print(f"[Open-Meteo] batch FAILED after {RETRY_COUNT} attempts for "
          f"{len(districts)} districts — falling back to zeros: {districts}")
    return {d: (0.0, 0.0, False) for d in districts}


def _fetch_batch_flood(batch_rows: list) -> dict:
    """
    Fetch river discharge for a batch of districts.
    Retries once on failure before falling back to zeros (flagged ok=False).
    """
    lats      = ",".join(str(r["lat"]) for r in batch_rows)
    lons      = ",".join(str(r["lon"]) for r in batch_rows)
    districts = [r["district"] for r in batch_rows]

    for attempt in range(RETRY_COUNT):
        try:
            r = requests.get(FLOOD_URL, params={
                "latitude":      lats,
                "longitude":     lons,
                "daily":         "river_discharge",
                "forecast_days": 1,
            }, timeout=TIMEOUT)
            if r.status_code == 200:
                return _parse_flood_batch(r.json(), districts)
            else:
                print(f"[Open-Meteo flood] batch HTTP {r.status_code} "
                      f"(attempt {attempt+1}/{RETRY_COUNT})")
        except Exception as e:
            print(f"[Open-Meteo flood] batch error (attempt {attempt+1}/{RETRY_COUNT}): {e}")

        if attempt < RETRY_COUNT - 1:
            time.sleep(RETRY_DELAY)

    print(f"[Open-Meteo flood] batch FAILED after {RETRY_COUNT} attempts for "
          f"{len(districts)} districts — falling back to zeros: {districts}")
    return {d: (0.0, False) for d in districts}


# ══════════════════════════════════════════════════════════════════════
# Main entry point — called by app.py
# ══════════════════════════════════════════════════════════════════════
def fetch_live_weather(df: pd.DataFrame) -> tuple:
    """
    Returns (df_enriched, failed_districts_list).

    Columns added to df:
      live_precip_mm     <- OWM  current 1h rain (mm)
      live_wind_kmh      <- OWM  current wind speed (km/h)
      live_soil_moisture <- Open-Meteo 8h mean soil moisture (m3/m3)
      live_snowmelt_mm   <- Open-Meteo 7-day snowfall sum (mm)
      live_discharge     <- Open-Meteo river discharge (m3/s)
      live_data_reliable <- bool, False if any source failed for that district

    failed_districts_list includes a district if ANY of OWM, Open-Meteo
    weather, or Open-Meteo flood failed for it. Treat this list as "do not
    fully trust this district's prediction this cycle" — values are present
    (as zeros/estimates) so the app doesn't crash, but they are not real
    readings.
    """
    rows    = df[["district", "lat", "lon"]].to_dict("records")
    batches = [rows[i: i + BATCH_SIZE] for i in range(0, len(rows), BATCH_SIZE)]

    # Open-Meteo batch (soil + snow)
    meteo_results: dict = {}
    for batch in batches:
        meteo_results.update(_fetch_batch_weather(batch))

    # Open-Meteo flood batch (discharge)
    flood_results: dict = {}
    for batch in batches:
        flood_results.update(_fetch_batch_flood(batch))

    # Discharge fallback: if flood API returned 0 AND failed, estimate from
    # soil moisture rather than leaving a hard zero (still flagged not-ok).
    for district, (discharge, ok) in list(flood_results.items()):
        if discharge == 0.0 and not ok:
            soil = meteo_results.get(district, (0.0, 0.0, False))[0]
            if soil > 0.05:
                flood_results[district] = (round(soil * 5000, 1), ok)

    # OWM parallel (rain + wind) — runs after meteo to avoid rate-limit stacking
    owm_results = _fetch_owm_all(rows)

    # A district is "failed" if ANY of the three sources didn't come back ok
    failed = sorted({
        d for d in df["district"]
        if not meteo_results.get(d, (0, 0, False))[2]
        or not flood_results.get(d, (0, False))[1]
        or not owm_results.get(d, (0, 0, False))[2]
    })

    if failed:
        print(f"⚠️  {len(failed)} district(s) have unreliable live data this "
              f"cycle (using fallback zeros/estimates): {failed}")

    df = df.copy()
    df["live_precip_mm"]     = df["district"].map(
        lambda d: owm_results.get(d, (0.0, 0.0, False))[0])
    df["live_wind_kmh"]      = df["district"].map(
        lambda d: owm_results.get(d, (0.0, 0.0, False))[1])
    df["live_soil_moisture"] = df["district"].map(
        lambda d: meteo_results.get(d, (0.0, 0.0, False))[0])
    df["live_snowmelt_mm"]   = df["district"].map(
        lambda d: meteo_results.get(d, (0.0, 0.0, False))[1])
    df["live_discharge"]     = df["district"].map(
        lambda d: flood_results.get(d, (0.0, False))[0])
    df["live_data_reliable"] = df["district"].map(lambda d: d not in failed)

    return df, failed