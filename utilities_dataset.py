
""" 
In this module there is a part( line 187-214) which is commented out because we already downloaded the data and cached it.
However, if you need to re-download or update the cache, you can uncomment that section.
Download was splitted into multiple parts due to API limitations.

"""

# 1) The first part of this file will aim to  create a dataset with the circuit specifications , for all circuits that hoested a grand prix from 2023 to 2025.

# 2) Then from this dataset we will construct a subset with the unique circuits plus drs dervied features( 1017- 1096).

# 3) From the cached data we will build the second dataset containg lap times weather gap to position ahed etc.( 1420 )

# 4) From OpenF1 API we will get the  Race control Data fot the same years, and merge them with the lap times dataset( line 2120).



# importing the necessary libraries

import fastf1
import matplotlib.pyplot as plt
import os
import pandas as pd
import yaml
from collections import OrderedDict
from collections import Counter
import unicodedata
import scipy as sp
import scipy.spatial
from scipy.spatial import cKDTree
import time
from datetime import datetime
import numpy as np
import openpyxl
import pickle
import warnings
import json 
from urllib.request import urlopen
import gc 
import pandas as pd


def load_circuit_specs(circuits_folder, output_path=None):
    """
    Load circuit specifications from YAML files in the circuits folder.
    
    Parameters:
    -----------
    circuits_folder : str
        Path to the folder containing circuit YAML files
    output_path : str, optional
        If provided, saves the circuit specs to a CSV file at this path
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing circuit specifications with columns:
        circuitRef, name, countryId, length_km, turns, drs_zones, width_m, type
    """
    data = []
    
    for file_name in os.listdir(circuits_folder):
        if file_name.endswith(".yml"):
            with open(os.path.join(circuits_folder, file_name), 'r') as f:
                circuit = yaml.safe_load(f)
            
            # Extract only the fields you need, handle missing safely
            data.append({
                "circuitRef": circuit.get("id"),
                "name": circuit.get("name"),
                "countryId": circuit.get("countryId"),
                "length_km": circuit.get("length"),
                "turns": circuit.get("turns"),
                "drs_zones": circuit.get("drsZones"),
                "width_m": circuit.get("width"),
                "type": circuit.get("type")
            })
    
    circuit_specs = pd.DataFrame(data)
    
    # Save to CSV if output_path is provided
    if output_path:
        circuit_specs.to_csv(output_path, index=False)
    
    return circuit_specs


def load_and_process_circuit_specs(csv_path):
    """
    Load your existing circuit specs file and add DRS zones.
    
    Parameters:
    -----------
    csv_path : str
        Path to the circuit specs CSV file
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with num_drs_zones added and drs_zones, width_m columns removed
    """
    # --- DRS zone counts (fill these manually as you check each circuit) ---
    drs_zones = {
        "adelaide": 2,
        "aida": 0,
        "ain-diab": 0,
        "aintree": 0,
        "andertorp": 0,
        "austin": 2,
        "avus": 0,
        "bahrain": 3,
        "baku": 2, 
        "brands-hatch": 0,
        "bremgarten": 0,
        "buddh":0, 
        "buenos-aires": 0,
        "bugatti": 0,
        "caesars-palace": 0,
        "catalunya": 2,
        "clermont-ferrand": 0,
        "dallas": 0,
        "detroit": 0,
        "dijon": 0,
        "donington": 0,
        "east-london": 0,
        "estoril": 0,
        "fuji": 0,
        "hockenheimring": 2,
        "hungaroring": 2,
        "imola": 1,
        "indianapolis": 0,
        "interlagos": 2,
        "istanbul": 2,
        "jacarepagua": 0,
        "jerema":0,
        "jerez": 0,
        "jeddah": 3,
        "kyalami": 0,
        "las-vegas": 2,
        "long-beach": 0,
        "lusail": 1,
        "magny-cours": 0,
        "marina-bay": 4,
        "melbourne": 4,
        "mexico-city": 3,
        "monaco": 1,
        "miami": 3,
        "monsanto": 0,
        "mont-tremblant": 0,
        "montjuic": 0,
        "montereal": 3,
        "monza": 2,
        "mosport": 0,
        "mugello": 1,
        "nivelles": 0,
        "nurburgring": 1,
        "paul-ricard": 2,
        "pedralbes": 0,
        "pescara": 0,
        "phoenix": 0,
        "portimao": 2,
        "porto":2, 
        "reims": 0,
        "rouen": 0,
        "sepang": 2,
        "shanghai": 2,
        "silverstone": 2,
        "sochi": 2,
        "spa-francorchamps": 2,
        "spielberg": 3,
        "suzuka": 1,
        "valencia": 1,
        "watkins-glen": 0,
        "yas-marina": 2,
        "yeongam": 2,
        "zandvoort": 2,
        "zeltweg": 0,
        "zolder":  0,
        "sebring":  0, 
        "jarama":  0,
        "montreal":  3, 
        "anderstorp":  0,
        "riverside":  0}
    
    # Load your existing circuit specs file
    circuit_specs = pd.read_csv(csv_path)
    
    # Add the new column 
    circuit_specs["num_drs_zones"] = circuit_specs["circuitRef"].map(drs_zones)
    
    # Drop unnecessary columns
    circuit_specs.drop(columns=["drs_zones", "width_m"], inplace=True)
    
    return circuit_specs


# THIS SECTION HAS ALREADY BEEN EXECUTED - DATA IS IN ./cache FOLDER
# The download was split into multiple parts due to API limitations
# All data for 2023-2025 is now cached and ready to use
# UNCOMMENT BELOW ONLY IF YOU NEED TO RE-DOWNLOAD OR UPDATE CACHE

""" 
# Enable cache
fastf1.Cache.enable_cache("./cache")

# Download data for 2018-2024
YEARS = range(2022, 2023)
start = datetime.now()

for year in YEARS:
    print(f"\n{'='*60}\nYear: {year}\n{'='*60}")
    schedule = fastf1.get_event_schedule(year)
    races = schedule[schedule['RoundNumber'] >= 1]
    
    for idx, race in races.iterrows():
        try:
            print(f"[{race['RoundNumber']:2d}] {race['EventName']:40s}", end=" ")
            
            # Download Race + Qualifying
            fastf1.get_session(year, race['EventName'], 'R').load()
            fastf1.get_session(year, race['EventName'], 'Q').load()
            
            print("✓")
        except Exception as e:
            print(f"✗ {str(e)[:40]}")

print(f"\n Done! Time: {datetime.now() - start}")

"""

def get_race_schedule_from_cache(year: int, cache_dir: str = "./cache") -> pd.DataFrame:
    """
    Build race schedule by reading the cache folder structure directly.
    This avoids needing to download schedule data from the API.
    
    Parameters:
    -----------
    year : int
        Year to read from cache
    cache_dir : str
        Path to the cache directory (default: "./cache")
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: event_name, round, location, year, folder_name
    """
    year_path = os.path.join(cache_dir, str(year))
    
    if not os.path.exists(year_path):
        print(f"  ⚠️  No cache folder found for {year}")
        return pd.DataFrame()
    
    schedules = []
    for folder_name in sorted(os.listdir(year_path)):
        if not os.path.isdir(os.path.join(year_path, folder_name)):
            continue
            
        # Folder format: "YYYY-MM-DD_Event_Name"
        parts = folder_name.split("_", 1)
        if len(parts) < 2:
            continue
            
        date_str, event_name = parts
        
        # Extract round number and clean event name
        # Event name might be like "03_Australian_Grand_Prix" or just "Australian_Grand_Prix"
        event_parts = event_name.split("_")
        if event_parts[0].isdigit():
            round_num = int(event_parts[0])
            event_clean = " ".join(event_parts[1:])
        else:
            # Try to infer round number from folder order
            round_num = len(schedules) + 1
            event_clean = " ".join(event_parts)
        
        # Try to extract location from event name
        # Most events end with "Grand Prix", location is usually before that
        if "Grand Prix" in event_clean or "Grand_Prix" in event_name:
            location = event_clean.replace("Grand Prix", "").strip()
        else:
            location = event_clean
        
        schedules.append({
            "event_name": event_clean,
            "round": round_num,
            "location": location,
            "year": year,
            "folder_name": folder_name
        })
    
    if not schedules:
        print(f"No race folders found in {year_path}")
        return pd.DataFrame()
    
    df = pd.DataFrame(schedules)
    # Ensure proper round numbering based on date order
    df = df.sort_values("folder_name").reset_index(drop=True)
    df["round"] = range(1, len(df) + 1)
    
    return df



def _norm(s: str) -> str:
    """
    Normalize string: remove diacritics and convert to lowercase.
    
    Parameters:
    -----------
    s : str
        String to normalize
    
    Returns:
    --------
    str
        Normalized string
    """
    if pd.isna(s):
        return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.strip().lower()


# Unified mapping: Event names, locations, and variations → circuitRef
NAME_TO_CIRCUIT = { _norm(k): v for k, v in {
    # === GRAND PRIX EVENT NAMES ===
    "australian grand prix": "melbourne",
    "bahrain grand prix": "bahrain",
    "chinese grand prix": "shanghai",
    "azerbaijan grand prix": "baku",
    "azerbaijani grand prix": "baku",
    "spanish grand prix": "catalunya",
    "monaco grand prix": "monaco",
    "canadian grand prix": "montreal",
    "french grand prix": "paul-ricard",
    "austrian grand prix": "spielberg",
    "british grand prix": "silverstone",
    "german grand prix": "hockenheimring",
    "hungarian grand prix": "hungaroring",
    "belgian grand prix": "spa-francorchamps",
    "italian grand prix": "monza",
    "singapore grand prix": "marina-bay",
    "russian grand prix": "sochi",
    "japanese grand prix": "suzuka",
    "united states grand prix": "austin",
    "mexican grand prix": "mexico-city",
    "mexico city grand prix": "mexico-city",
    "brazilian grand prix": "interlagos",
    "abu dhabi grand prix": "yas-marina",
    "saudi arabian grand prix": "jeddah",
    "miami grand prix": "miami",
    "dutch grand prix": "zandvoort",
    "qatar grand prix": "lusail",
    "las vegas grand prix": "las-vegas",
    
    "70th anniversary grand prix": "silverstone",
    "styrian grand prix": "spielberg",
    "eifel grand prix": "nurburgring",
    "tuscan grand prix": "mugello",
    "emilia romagna grand prix": "imola",
    "portuguese grand prix": "portimao",
    "turkish grand prix": "istanbul",
    "sakhir grand prix": "bahrain",  
    # === LOCATION/CITY NAMES ===
    "melbourne": "melbourne",
    "sakhir": "bahrain",
    "bahrain": "bahrain",
    "shanghai": "shanghai",
    "baku": "baku",
    "barcelona": "catalunya",
    "monaco": "monaco",
    "montreal": "montreal",
    "le castellet": "paul-ricard",
    "spielberg": "spielberg",
    "silverstone": "silverstone",
    "hockenheim": "hockenheimring",
    "budapest": "hungaroring",
    "spa-francorchamps": "spa-francorchamps",
    "zandvoort": "zandvoort",
    "monza": "monza",
    "marina bay": "marina-bay",
    "sochi": "sochi",
    "suzuka": "suzuka",
    "austin": "austin",
    "mexico city": "mexico-city",
    "sao paulo": "interlagos",
    "las vegas": "las-vegas",
    "jeddah": "jeddah",
    "miami": "miami",
    "imola": "imola",
    "portimao": "portimao",
    "istanbul": "istanbul",
    "mugello": "mugello",
    "nurburg": "nurburgring",
    "nuerburg": "nurburgring",
    "lusail": "lusail",
    "yas island": "yas-marina",
    "australian": "melbourne",
    "chinese": "shanghai",
    "hanoi": "hanoi",
    "sepang": "sepang",
    "yeongam": "yeongam",
    "buddh": "buddh",
}.items() }


def build_race_schedule(schedule_years, circuit_specs, cache_dir: str = "./cache", output_dir: str = "csv_output"):
    """
    Build race schedule with circuitRef by reading cache folder structure for multiple years.
    Maps event names and locations to circuits using NAME_TO_CIRCUIT mapping.
    Merges with circuit_specs and saves to CSV.
    
    Parameters:
    -----------
    schedule_years : range or list
        Years to process (e.g., range(2023, 2026))
    circuit_specs : pd.DataFrame
        DataFrame with circuit specifications (must have 'circuitRef' column)
    cache_dir : str
        Path to the cache directory (default: "./cache")
    output_dir : str
        Directory to save output CSV (default: "csv_output")
    
    Returns:
    --------
    pd.DataFrame
        Merged dataframe with circuit specs, year, and round information
    """
    print(f"\nConfigured to process years: {list(schedule_years)}")
    print("To process more years, check your cache folder and update schedule_years")
    
    all_schedules = []
    
    for year in schedule_years:
        print(f"\nReading cache for {year}...")
        schedule = get_race_schedule_from_cache(year, cache_dir=cache_dir)
        
        if schedule.empty:
            print(f"No races found for {year}, skipping")
            continue
        
        print(f"  ✓ Found {len(schedule)} race folders")
        
        # Map using event names first, then try location as fallback
        schedule["event_norm"] = schedule["event_name"].map(_norm)
        schedule["circuitRef"] = schedule["event_norm"].map(NAME_TO_CIRCUIT)
        
        # For unmapped, try location
        schedule["loc_norm"] = schedule["location"].map(_norm)
        schedule["circuitRef"] = schedule["circuitRef"].fillna(
            schedule["loc_norm"].map(NAME_TO_CIRCUIT)
        )
        
        # Count how many circuits were successfully mapped
        mapped_count = schedule["circuitRef"].notna().sum()
        print(f"  ✓ Mapped {mapped_count}/{len(schedule)} races to circuits")
        
        # Show unmapped races for debugging
        unmapped = schedule[schedule["circuitRef"].isna()]
        if not unmapped.empty:
            print(f"Could not map {len(unmapped)} races:")
            for _, row in unmapped.head(3).iterrows():
                print(f"      Event: '{row['event_name']}' → normalized: '{_norm(row['event_name'])}'")
                print(f"      Location: '{row['location']}' → normalized: '{_norm(row['location'])}'")
                print(f"      (Add one of these to NAME_TO_CIRCUIT mapping)")
        
        all_schedules.append(schedule)
    
    # Concatenate all schedules
    if not all_schedules:
        print("\nERROR: No race schedules found in cache!")
        print("Please check:")
        print("  1. Cache folder exists:", cache_dir)
        print("  2. Year folders exist (e.g., " + cache_dir + "/2018/)")
        print("  3. Race folders inside year folders")
        raise ValueError("No race data found in cache. Cannot proceed.")
    
    schedules_all_years = pd.concat(all_schedules, ignore_index=True)
    
    # One row per (year, round)
    race_keys_all_years = (
        schedules_all_years[["circuitRef", "year", "round"]]
          .dropna(subset=["circuitRef"])
          .drop_duplicates(subset=["year", "round"], keep="first")
          .reset_index(drop=True)
    )
    
    # Drop duplicates to ensure unique circuitRef in specs
    circuit_specs_clean = circuit_specs.drop_duplicates(subset=["circuitRef"], keep="first").reset_index(drop=True)
    
    # Merge with circuit_specs
    merged_all_years = pd.merge(circuit_specs_clean, race_keys_all_years, on="circuitRef", how="inner")
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    
    # Reorder & sort by year and round
    ordered_cols = ["circuitRef", "year", "round"] + [c for c in merged_all_years.columns if c not in ["circuitRef", "year", "round"]]
    merged_all_years = merged_all_years[ordered_cols].sort_values(["year", "round"]).reset_index(drop=True)
    
    # Create filename based on actual years in data
    min_year = int(merged_all_years["year"].min())
    max_year = int(merged_all_years["year"].max())
    schedule_filename = f"{output_dir}/circuit_specs_with_rounds_{min_year}_{max_year}.csv"
    merged_all_years.to_csv(schedule_filename, index=False)
    
    print(f"\n✓ Schedule file saved: {schedule_filename}")
    print(f"  Total race entries: {len(merged_all_years)}")
    print(f"  Years: {min_year}-{max_year}")
    print(f"  Unique circuits: {merged_all_years['circuitRef'].nunique()}")
    
    return merged_all_years

#------------------------------------------------------------------------------
# Working to obtain the geometry dataset 
#------------------------------------------------------------------------------

# Fast F1 cache setup
# Use ONLY cached data - no downloads during execution

os.makedirs("./cache", exist_ok=True)
fastf1.Cache.enable_cache("./cache")

# Set offline mode to prevent any network requests. FastF1 will only use data already in the cache

warnings.filterwarnings('ignore')  # Suppress warnings about missing data


# Helper to pick clean fastest lap 

def _pick_session_fastest_lap(session):
    """
    Return a clean fastest lap (any driver).
    - Drops in-/out-laps
    - Prefers fully green ('1') laps if TrackStatus exists
    """
    laps = session.laps
    if laps is None or laps.empty:
        return None

    # remove out-/in-laps
    laps = laps[(laps["PitOutTime"].isna()) & (laps["PitInTime"].isna())]

    # prefer fully green laps if available
    if "TrackStatus" in laps.columns:
        green = laps["TrackStatus"].fillna("").astype(str).str.fullmatch("1")
        if green.any():
            laps = laps[green]

    try:
        lap = laps.pick_fastest()
        if lap is not None and ("LapTime" in lap) and pd.notna(lap["LapTime"]):
            return lap
    except Exception:
        pass
    return None

def extract_apex_features(tel, corners):
    """
    Compute per-corner apex and classification features.
    Returns a DataFrame (apex_geom) with one row per corner.
    
    Corner classification uses quantile-based thresholds for era-robustness:
    - slow:   bottom third of apex speeds
    - medium: middle third of apex speeds
    - fast:   top third of apex speeds
    - flat:   special case for nearly-flat corners (minimal braking, low lateral g)
    """
    CIRCLE_WIN = 35
    ENTRY_WIN  = (-180, -15)
    EXIT_WIN   = (15, 180)
    MIN_WIN    = (-120, 120)
    
    # Flat corner detection thresholds (relative to max speed)
    FLAT_RELATIVE_SPEED = 0.92  # 92% of max lap speed
    FLAT_DROP  = 10.0           # Max speed drop (km/h)
    FLAT_GMAX  = 2.0            # Max lateral g
    
    # Quantile boundaries for slow/medium/fast classification
    SLOW_QUANTILE = 0.33
    FAST_QUANTILE = 0.67

    def _scale_xy_to_meters(df):
        x, y = df["X"].to_numpy(), df["Y"].to_numpy()
        L_xy = np.sum(np.hypot(np.diff(x), np.diff(y)))
        L_m  = float(df["Distance"].iloc[-1] - df["Distance"].iloc[0])
        scale = L_m / max(L_xy, 1e-9)
        df["X_m"] = df["X"] * scale
        df["Y_m"] = df["Y"] * scale
        return df

    def _curvature(df):
        s = df["Distance"].to_numpy()
        xm = df["X_m"].to_numpy(); ym = df["Y_m"].to_numpy()
        dx = np.gradient(xm, s); dy = np.gradient(ym, s)
        ddx = np.gradient(dx, s); ddy = np.gradient(dy, s)
        kap = (dx*ddy - dy*ddx) / (np.power(dx*dx + dy*dy, 1.5) + 1e-12)
        df["kap_smooth"] = pd.Series(kap).rolling(5, center=True, min_periods=1).median()
        return df

    def _nearest_indices_by_xy(tel, corners):
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(np.c_[tel["X"], tel["Y"]])
            idx0 = [int(tree.query([float(r.X), float(r.Y)], k=1)[1]) for r in corners.itertuples()]
        except Exception:
            XY = np.c_[tel["X"].to_numpy(), tel["Y"].to_numpy()]
            idx0 = [int(np.argmin(np.hypot(XY[:,0]-float(r.X), XY[:,1]-float(r.Y)))) for r in corners.itertuples()]
        return pd.DataFrame({"Number": corners["Number"].astype(int).to_numpy(), "idx0": np.array(idx0)})

    def _segment_bounds(idx0, n, pad=150):
        b = np.zeros(len(idx0)+1, dtype=int)
        b[1:-1] = np.round((idx0[:-1] + idx0[1:]) / 2).astype(int)
        b[0] = max(0, idx0[0] - pad)
        b[-1] = min(n-1, idx0[-1] + pad)
        return b

    def _circle_radius(xm, ym):
        xm = np.asarray(xm); ym = np.asarray(ym)
        x0, y0 = xm.mean(), ym.mean()
        X, Y = xm - x0, ym - y0
        Z = X*X + Y*Y
        A = np.c_[2*X, 2*Y, np.ones_like(X)]
        cx, cy, c0 = np.linalg.lstsq(A, Z, rcond=None)[0]
        return float(np.sqrt(cx*cx + cy*cy + c0))

    def _slice(tel, s0, a, b):
        m = (tel["Distance"] >= s0 + a) & (tel["Distance"] <= s0 + b)
        return tel.loc[m, ["Speed", "X_m", "Y_m"]]

    # Preprocess telemetry
    tel = (tel[["Distance", "Speed", "X", "Y"]]
           .dropna()
           .assign(Speed=lambda d: d["Speed"].rolling(3, center=True, min_periods=1).median()))
    tel = _scale_xy_to_meters(tel)
    tel = _curvature(tel)
    
    max_lap_speed = float(tel["Speed"].max())

    # Corner segmentation
    corner_idx = _nearest_indices_by_xy(tel, corners)
    idxs = corner_idx["idx0"].to_numpy()
    bounds = _segment_bounds(idxs, len(tel))

    # First pass: collect apex data
    rows = []
    for k in range(len(idxs)):
        lo, hi = bounds[k], bounds[k+1]
        seg = tel.iloc[lo:hi]
        if seg.empty:
            continue

        idx = seg["kap_smooth"].abs().idxmax()
        s_apex = float(tel.at[idx, "Distance"])
        v_kmh  = float(tel.at[idx, "Speed"])

        span = _slice(tel, s_apex, -CIRCLE_WIN, CIRCLE_WIN)
        if len(span) >= 6:
            R = _circle_radius(span["X_m"].to_numpy(), span["Y_m"].to_numpy())
            lat_g = ((v_kmh/3.6)**2) / (R * 9.80665)
        else:
            lat_g = np.nan

        ent = _slice(tel, s_apex, *ENTRY_WIN)
        exi = _slice(tel, s_apex, *EXIT_WIN)
        minw = _slice(tel, s_apex, *MIN_WIN)

        entry_speed = float(np.nanpercentile(ent["Speed"], 95)) if not ent.empty else np.nan
        exit_speed  = float(np.nanpercentile(exi["Speed"], 95)) if not exi.empty else np.nan
        min_speed   = float(minw["Speed"].min()) if not minw.empty else np.nan

        rows.append({
            "Number": int(corner_idx.iloc[k]["Number"]),
            "Distance_apex": s_apex,
            "apex_speed_kmh": v_kmh,
            "apex_lateral_g": lat_g,
            "entry_speed_kmh": entry_speed,
            "min_speed_kmh": min_speed,
            "exit_speed_kmh": exit_speed,
        })
    
    if not rows:
        return pd.DataFrame(columns=[
            "Number", "Distance_apex", "apex_speed_kmh", "apex_lateral_g",
            "entry_speed_kmh", "min_speed_kmh", "exit_speed_kmh", "corner_type_speed"
        ])
    
    df = pd.DataFrame(rows)
    
    # Second pass: compute quantile-based classification
    apex_speeds = df["apex_speed_kmh"].to_numpy()
    q_slow = np.nanpercentile(apex_speeds, SLOW_QUANTILE * 100)
    q_fast = np.nanpercentile(apex_speeds, FAST_QUANTILE * 100)
    flat_threshold = max_lap_speed * FLAT_RELATIVE_SPEED
    
    def _classify_corner(row):
        v = row["apex_speed_kmh"]
        entry = row["entry_speed_kmh"]
        min_spd = row["min_speed_kmh"]
        lat_g = row["apex_lateral_g"]
        
        # Check for flat corner (taken nearly flat-out with minimal braking)
        if (v >= flat_threshold and 
            np.isfinite(min_spd) and np.isfinite(entry) and
            (entry - min_spd) <= FLAT_DROP and 
            lat_g < FLAT_GMAX):
            return "flat"
        
        # Quantile-based classification
        if v <= q_slow:
            return "slow"
        elif v >= q_fast:
            return "fast"
        else:
            return "medium"
    
    df["corner_type_speed"] = df.apply(_classify_corner, axis=1)
    
    return df.sort_values("Number").reset_index(drop=True)


#Compute circuit summary features

def summarize_circuit(year, circuit_name, official_length_km=None, curvature_threshold=0.0010):
    """
    Uses race fastest lap (any driver); if none usable, falls back to qualifying fastest.
    - straights via curvature-based detection (geometrically correct)
    - wrap-around spacing (last -> start/finish)
    - robust lap distance via Distance diffs
    
    NOTE: This function expects data to be pre-cached. It will use ONLY cached data.
    
    Args:
        curvature_threshold: Sections with |curvature| < threshold are considered straights.
                           Default 0.0010 m^-1 corresponds to radius > 1000m
    """
    
    def _compute_curvature(tel):
        """Compute curvature from X,Y coordinates scaled to meters."""
        # Scale X,Y to meters
        x, y = tel["X"].to_numpy(), tel["Y"].to_numpy()
        L_xy = np.sum(np.hypot(np.diff(x), np.diff(y)))
        L_m = float(tel["Distance"].iloc[-1] - tel["Distance"].iloc[0])
        scale = L_m / max(L_xy, 1e-9)
        
        xm = x * scale
        ym = y * scale
        s = tel["Distance"].to_numpy()
        
        # Compute curvature: κ = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
        dx = np.gradient(xm, s)
        dy = np.gradient(ym, s)
        ddx = np.gradient(dx, s)
        ddy = np.gradient(dy, s)
        
        kap = (dx * ddy - dy * ddx) / (np.power(dx*dx + dy*dy, 1.5) + 1e-12)
        
        # Smooth to reduce noise
        kap_smooth = pd.Series(np.abs(kap)).rolling(7, center=True, min_periods=1).median()
        
        return kap_smooth.to_numpy()
    
    # Race session - load from cache only
    try:
        session = fastf1.get_session(year, circuit_name, "R")
        session.load(telemetry=True, weather=False, messages=False)
        fastest_lap = _pick_session_fastest_lap(session)
    except Exception as e:
        print(f"  Warning: Could not load race session from cache for {circuit_name} {year}: {e}")
        fastest_lap = None

    # Fallback to qualifying if needed
    if fastest_lap is None:
        try:
            session_q = fastf1.get_session(year, circuit_name, "Q")
            session_q.load(telemetry=True, weather=False, messages=False)
            fastest_lap = _pick_session_fastest_lap(session_q)
        except Exception as e:
            print(f"  Warning: Could not load qualifying session from cache for {circuit_name} {year}: {e}")

    if fastest_lap is None:
        raise RuntimeError(f"No usable lap found in cache for {circuit_name} {year}. Make sure data is pre-cached.")

    # Telemetry + robust distance
    try:
        tel = fastest_lap.get_telemetry()
        if tel is None or tel.empty:
            raise RuntimeError(f"Telemetry data is empty for {circuit_name} {year}")
        tel = tel.add_distance()
    except AttributeError as e:
        raise RuntimeError(f"Failed to get telemetry for {circuit_name} {year}: fastest_lap is None or invalid")
    tel = tel[["Distance", "Speed", "X", "Y"]].dropna().reset_index(drop=True)
    
    d_step = tel["Distance"].diff().clip(lower=0).fillna(0.0)
    lap_distance_m = float(d_step.sum())

    telemetry_length_km = lap_distance_m / 1000.0
    length_km = float(official_length_km) if official_length_km is not None else telemetry_length_km

    # Straights by curvature (geometrically correct)
    curvature = _compute_curvature(tel)
    is_straight = curvature < curvature_threshold
    straight_distance_m = float(d_step.where(is_straight).sum())
    straight_ratio = straight_distance_m / lap_distance_m if lap_distance_m > 0 else 0.0
    
    # Identify contiguous straight segments
    MIN_STRAIGHT_LEN = 70.0  # meters - drop noise segments shorter than this
    MAJOR_STRAIGHT_THRESHOLD = 700.0  # meters
    
    s = tel["Distance"].to_numpy()
    mask = is_straight.to_numpy() if isinstance(is_straight, pd.Series) else is_straight
    
    # Find transitions in the mask
    changes = np.where(np.diff(mask.astype(int)) != 0)[0] + 1
    starts = np.concatenate([[0], changes])
    ends = np.concatenate([changes, [len(s)]])
    
    # Keep only straight segments (where mask is True at segment start)
    straight_segments = []
    for i in range(len(starts)):
        if mask[starts[i]]:
            seg_start_idx = starts[i]
            seg_end_idx = ends[i] - 1  # inclusive end
            seg_len = s[seg_end_idx] - s[seg_start_idx]
            if seg_len >= MIN_STRAIGHT_LEN:  # filter out noise
                straight_segments.append({
                    'start_idx': seg_start_idx,
                    'end_idx': seg_end_idx,
                    'start_dist': s[seg_start_idx],
                    'end_dist': s[seg_end_idx],
                    'length': seg_len
                })
    
    # Handle wrap-around: if lap starts and ends on a straight, merge them
    if len(straight_segments) >= 2 and mask[0] and mask[-1]:
        first_seg = straight_segments[0]
        last_seg = straight_segments[-1]
        
        # Merged length spans end of lap to start
        wrap_length = (s[-1] - last_seg['start_dist']) + (first_seg['end_dist'] - s[0])
        
        if wrap_length >= MIN_STRAIGHT_LEN:
            # Replace first and last with merged segment
            merged_seg = {
                'start_idx': last_seg['start_idx'],
                'end_idx': first_seg['end_idx'],
                'start_dist': last_seg['start_dist'],
                'end_dist': first_seg['end_dist'],
                'length': wrap_length,
                'is_wraparound': True
            }
            straight_segments = [merged_seg] + straight_segments[1:-1]
        else:
            # If merged segment too short, drop both
            straight_segments = straight_segments[1:-1]
    
    # Compute metrics
    if straight_segments:
        lengths = np.array([seg['length'] for seg in straight_segments])
        straight_len_max_m = float(np.max(lengths))
        n_major_straights = int(np.sum(lengths >= MAJOR_STRAIGHT_THRESHOLD))
        
        # Find longest straight for potential future use (DRS, heavy braking)
        idx_longest = int(np.argmax(lengths))
        longest_straight = straight_segments[idx_longest]
    else:
        straight_len_max_m = 0.0
        n_major_straights = 0
        longest_straight = None

    # Corner info
    try:
        circuit_info = session.get_circuit_info()
        corners = circuit_info.corners.copy()
        if not corners.empty:
            corners = corners.sort_values("Distance").reset_index(drop=True)
    except Exception as e:
        print(f"  Warning: Could not load circuit info for {circuit_name} {year}: {e}")
        corners = pd.DataFrame()

    if corners.empty:
        raise RuntimeError(f"No corner data available for {circuit_name} {year}")
    
    apex_geom = extract_apex_features(tel, corners)
    counts = apex_geom["corner_type_speed"].value_counts()
    slow = int(counts.get("slow", 0))
    medium = int(counts.get("medium", 0))
    fast = int(counts.get("fast", 0))
    flat = int(counts.get("flat", 0))
    
    # Slow corner cluster analysis (longest consecutive slow section)
    if not apex_geom.empty and slow > 0:
        # Sort by lap position
        apex_sorted = apex_geom.sort_values("Distance_apex").reset_index(drop=True)
        corner_types = apex_sorted["corner_type_speed"].to_numpy()
        
        # Find runs of consecutive "slow" corners
        is_slow = (corner_types == "slow")
        
        # Detect run boundaries
        changes = np.concatenate([[0], np.where(np.diff(is_slow.astype(int)) != 0)[0] + 1, [len(is_slow)]])
        
        slow_runs = []
        for i in range(len(changes) - 1):
            start_idx = changes[i]
            end_idx = changes[i + 1]
            if is_slow[start_idx]:  # This run is slow corners
                run_length = end_idx - start_idx
                slow_runs.append(run_length)
        
        # Handle wrap-around: if lap starts AND ends with slow, merge those runs
        if len(slow_runs) >= 2 and is_slow[0] and is_slow[-1]:
            # First and last runs are both slow and should be merged
            wrapped_run = slow_runs[0] + slow_runs[-1]
            slow_runs = [wrapped_run] + slow_runs[1:-1]
        
        slow_cluster_max = int(max(slow_runs)) if slow_runs else 0
    else:
        slow_cluster_max = 0

    # Heavy-braking analysis
    HEAVY_BRAKE_THRESHOLD = 110.0  # km/h
    BRAKE_ZONE_PROXIMITY = 80.0    # meters after straight end
    
    if not apex_geom.empty:
        # Compute delta-v for each corner
        apex_geom["delta_v"] = apex_geom["entry_speed_kmh"] - apex_geom["min_speed_kmh"]
        
        # Identify heavy braking corners
        hb_mask = apex_geom["delta_v"] >= HEAVY_BRAKE_THRESHOLD
        hb_corners = apex_geom[hb_mask].copy()
        
        heavy_braking_zones = int(hb_mask.sum())
        
        if heavy_braking_zones > 0:
            heavy_braking_mean_dv_kmh = float(hb_corners["delta_v"].mean())
            
            # Spacing between heavy-brake corners (with wrap-around)
            hb_distances = np.sort(hb_corners["Distance_apex"].to_numpy())
            if len(hb_distances) >= 2:
                hb_spacings = np.diff(np.r_[hb_distances, hb_distances[0] + lap_distance_m])
                hb_spacing_std_m = float(np.std(hb_spacings))
            else:
                hb_spacing_std_m = np.nan
            
            # Check if heavy-brake follows longest straight
            hb_at_end_of_max = False
            if longest_straight is not None:
                straight_end = longest_straight['end_dist']
                # Handle wrap-around case
                if longest_straight.get('is_wraparound', False):
                    # Wrapped straight: check both near end_dist and near start (distance 0)
                    for hb_dist in hb_distances:
                        # Check proximity to end of wrapped segment (which is near start)
                        dist_from_end = min(
                            abs(hb_dist - straight_end),
                            abs(hb_dist - straight_end + lap_distance_m),
                            abs(hb_dist - straight_end - lap_distance_m)
                        )
                        if dist_from_end <= BRAKE_ZONE_PROXIMITY:
                            hb_at_end_of_max = True
                            break
                else:
                    # Normal straight: simple check with wrap
                    for hb_dist in hb_distances:
                        dist_from_end = hb_dist - straight_end
                        # Handle wrap-around: if straight ends near finish line
                        if dist_from_end < 0:
                            dist_from_end += lap_distance_m
                        if 0 <= dist_from_end <= BRAKE_ZONE_PROXIMITY:
                            hb_at_end_of_max = True
                            break
        else:
            heavy_braking_mean_dv_kmh = np.nan
            hb_spacing_std_m = np.nan
            hb_at_end_of_max = False
    else:
        heavy_braking_zones = 0
        heavy_braking_mean_dv_kmh = np.nan
        hb_spacing_std_m = np.nan
        hb_at_end_of_max = False

    total_angle = float(corners["Angle"].sum()) if not corners.empty else np.nan
    avg_angle = float(corners["Angle"].mean()) if not corners.empty else np.nan

    # Corner density (turns per kilometer)
    corner_density_tpkm = len(corners) / length_km if (not corners.empty and length_km > 0) else np.nan
    
    # Wrap-around spacing
    if not corners.empty and lap_distance_m > 0:
        s = np.sort(corners["Distance"].to_numpy(dtype=float))
        spacings = np.diff(np.r_[s, s[0] + lap_distance_m])  # include last -> start/finish
        avg_corner_distance = float(np.mean(spacings))
    else:
        avg_corner_distance = np.nan

    return pd.DataFrame([{
        "circuitRef": circuit_name.lower(),
        "length_km": round(length_km, 3),
        "num_turns": int(len(corners)),
        "slow_corners": slow,
        "medium_corners": medium,
        "fast_corners": fast,
        "flat_corners": flat,
        "slow_cluster_max": slow_cluster_max,
        "straight_distance_m": round(straight_distance_m, 1),
        "straight_ratio": round(straight_ratio, 3),
        "straight_len_max_m": round(straight_len_max_m, 1),
        "n_major_straights": n_major_straights,
        "heavy_braking_zones": heavy_braking_zones,
        "heavy_braking_mean_dv_kmh": round(heavy_braking_mean_dv_kmh, 1) if np.isfinite(heavy_braking_mean_dv_kmh) else np.nan,
        "hb_spacing_std_m": round(hb_spacing_std_m, 1) if np.isfinite(hb_spacing_std_m) else np.nan,
        "hb_at_end_of_max": hb_at_end_of_max,
        "corner_density_tpkm": round(corner_density_tpkm, 2) if np.isfinite(corner_density_tpkm) else np.nan,
        "avg_corner_angle": round(avg_angle, 1) if np.isfinite(avg_angle) else np.nan,
        "total_corner_angle": round(total_angle, 1) if np.isfinite(total_angle) else np.nan,
        "avg_corner_distance": round(avg_corner_distance, 1) if np.isfinite(avg_corner_distance) else np.nan
    }])



# Geometry changes for circuits

GEOMETRY_CHANGES = {
    # 'yas-marina': {
    #     'change_year': 2021,
    #     'description': 'Chicane removed, reprofiled hairpin; 21→19 turns'
    # },
    # 'melbourne': {
    #     'change_year': 2022,
    #     'description': 'T9/10 chicane removed, corners widened, full resurface'
    # },
    'catalunya': {
        'change_year': 2023,
        'description': 'Final chicane removed; lap 4.657 km'
    },
    'marina-bay': {
        'change_year': 2023,
        'description': 'Float section removed; 23→19 corners; lap 4.928 km'
     },
    # 'zandvoort': {
    #     'change_year': 2021,
    #     'description': 'Banked T3 & T14; new 4.259 km GP layout'
    # },
}


    
def create_unique_circuits(input_file: str, output_dir: str = "csv_output"):
    """
    Create a unique circuits dataset from circuit specs CSV.
    - Keeps the most recent version of each circuit (typically 2025)
    - ADDS old layout (one entry before geometry change) for circuits that changed
    
    Parameters:
    -----------
    input_file : str
        Path to the circuit specs CSV file (e.g., "csv_output/Circuit_specs_2018_2025.csv")
    output_dir : str
        Directory to save output CSV (default: "csv_output")
    
    Returns:
    --------
    pd.DataFrame
        Unique circuits dataframe with current and old layouts where applicable
    """
    # Load the full dataset
    df = pd.read_csv(input_file)
    
    # Process the dataset
    unique_circuits = []
    
    for circuit_ref in df['circuitRef'].unique():
        circuit_data = df[df['circuitRef'] == circuit_ref].copy()
        
        # Always get the most recent version (current layout)
        most_recent = circuit_data.sort_values('year', ascending=False).iloc[0].copy()
        unique_circuits.append(most_recent)
        
        # If this circuit had geometry changes, also add the old layout
        if circuit_ref in GEOMETRY_CHANGES:
            change_year = GEOMETRY_CHANGES[circuit_ref]['change_year']
            
            # Get data before the change (old layout)
            old_layout_data = circuit_data[circuit_data['year'] < change_year]
            
            if not old_layout_data.empty:
                # Get the most recent data from the old layout era
                old_most_recent = old_layout_data.sort_values('year', ascending=False).iloc[0].copy()
                
                # Modify the circuit reference and name to indicate it's the old layout
                old_most_recent['circuitRef'] = f"{circuit_ref}_old"
                old_most_recent['name'] = f"{old_most_recent['name']} (pre-{change_year})"
                
                unique_circuits.append(old_most_recent)
    
    # Create the unique circuits dataframe
    df_unique = pd.DataFrame(unique_circuits)
    
    # Sort by year and round
    df_unique = df_unique.sort_values(['year', 'round']).reset_index(drop=True)
    
    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/Circuit_specs_unique.csv"
    df_unique.to_csv(output_file, index=False)
    
    print(f"Dataset saved to: {output_file} ({len(df_unique)} circuits)")
    
    return df_unique



# Adding the DRS specification the the unique circuits dataset


CIRCUIT_TO_EVENT = {
    "austin": "United States", "bahrain": "Bahrain", "baku": "Azerbaijan",
    "catalunya": "Spanish", "catalunya_old": "Spanish", "hockenheimring": "German",
    "hungaroring": "Hungarian", "imola": "Emilia Romagna", "interlagos": "Brazilian",
    "istanbul": "Turkish", "jeddah": "Saudi Arabian", "las-vegas": "Las Vegas",
    "lusail": "Qatar", "marina-bay": "Singapore", "marina-bay_old": "Singapore",
    "melbourne": "Australian", "melbourne_old": "Australian", "mexico-city": "Mexican",
    "miami": "Miami", "monaco": "Monaco", "montreal": "Canadian", "monza": "Italian",
    "nurburgring": "Eifel", "paul-ricard": "French", "portimao": "Portuguese",
    "shanghai": "Chinese", "silverstone": "British", "sochi": "Russian",
    "spa-francorchamps": "Belgian", "spielberg": "Austrian", "suzuka": "Japanese",
    "yas-marina": "Abu Dhabi", "yas-marina_old": "Abu Dhabi", "zandvoort": "Dutch",
}

POS_JOIN_TOL = pd.Timedelta("200ms")
LAP_JOIN_DIR = "backward"
MIN_SEG_LEN_M = 50.0
EPS_CLUSTER_M = 200.0
MIN_SUPPORT_FRAC = 0.12
MIN_SUPPORT_ABS = 4
MAX_ZONE_FRAC = 0.35


def _drs_active_mask(series):
    return series.fillna(0).astype("Int64") > 0


def _ensure_dtypes(df):
    if "Time" in df.columns:
        df["Time"] = pd.to_timedelta(df["Time"])
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], utc=True)
    return df


def _clean_fastest_lap(session):
    try:
        laps = session.laps
        if laps is None or laps.empty:
            return None
        laps = laps[(laps["PitOutTime"].isna()) & (laps["PitInTime"].isna())]
        if "TrackStatus" in laps.columns:
            green = laps["TrackStatus"].fillna("").astype(str).str.fullmatch("1")
            if green.any():
                laps = laps[green]
        return laps.pick_fastest()
    except:
        return None


def _kd_map_distance(ref_tel, xy):
    tree = cKDTree(ref_tel[["X", "Y"]].to_numpy())
    idx = tree.query(xy, k=1)[1]
    return ref_tel["Distance"].to_numpy()[idx]


def _compute_curvature(tel):
    x, y = tel["X"].to_numpy(), tel["Y"].to_numpy()
    dx, dy = np.gradient(x), np.gradient(y)
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    num = np.abs(dx * ddy - dy * ddx)
    denom = (dx**2 + dy**2)**1.5
    return np.where(denom > 1e-9, num / denom, 0.0)


def _find_longest_straight(tel):
    curvature = _compute_curvature(tel)
    is_straight = curvature < 0.001
    s = tel["Distance"].to_numpy()
    changes = np.where(np.diff(is_straight.astype(int)) != 0)[0] + 1
    starts = np.concatenate([[0], changes])
    ends = np.concatenate([changes, [len(is_straight)]])
    segments = []
    for i in range(len(starts)):
        if is_straight[starts[i]]:
            length = float(s[ends[i] - 1] - s[starts[i]])
            if length >= 70.0:
                segments.append({"start": float(s[starts[i]]), "end": float(s[ends[i] - 1]), "length": length})
    return max(segments, key=lambda x: x["length"]) if segments else None


def _segments_from_mask(df):
    segs = []
    for lap, g in df.groupby("LapNumber", sort=True):
        m = g["drs_on"].to_numpy(dtype=bool)
        if not m.any():
            continue
        dm = np.diff(m.astype(int))
        starts = np.where(dm == 1)[0] + 1
        ends = np.where(dm == -1)[0] + 1
        if m[0]:
            starts = np.r_[0, starts]
        if m[-1]:
            ends = np.r_[ends, len(m)]
        svals = g["Distance"].to_numpy()
        for a, b in zip(starts, ends):
            seg_len = float(svals[b-1] - svals[a])
            if seg_len >= MIN_SEG_LEN_M:
                segs.append({"lap": int(lap), "start": float(svals[a]), "end": float(svals[b-1]), "len": seg_len})
    return segs


def _cluster_segments(segs, lap_len):
    if not segs:
        return []
    aug = []
    for s in segs:
        aug.append((s["start"], s["end"], s["lap"]))
        if lap_len - s["start"] < EPS_CLUSTER_M:
            aug.append((s["start"] + lap_len, s["end"] + lap_len, s["lap"]))
    aug = np.array(aug)
    aug = aug[np.argsort(aug[:, 0])]
    clusters = []
    cur = [aug[0]]
    for row in aug[1:]:
        if abs(row[0] - cur[-1][0]) <= EPS_CLUSTER_M:
            cur.append(row)
        else:
            clusters.append(np.array(cur))
            cur = [row]
    clusters.append(np.array(cur))
    zones = []
    for C in clusters:
        st = C[:, 0] % lap_len
        en = C[:, 1] % lap_len
        start_med = float(np.median(st))
        end_med = float(np.median(en))
        if end_med < start_med:
            zone_len = (end_med + lap_len) - start_med
            wraps = True
        else:
            zone_len = end_med - start_med
            wraps = False
        zones.append({"start": start_med, "len": max(float(zone_len), 0.0), "end": (start_med + max(float(zone_len), 0.0)) % lap_len, "wraps": wraps, "support": int(len(np.unique(C[:, 2])))})
    return zones


def _filter_zones(zones, laps_used, lap_len):
    kept = []
    for z in zones:
        support_frac = z["support"] / max(laps_used, 1)
        frac_len = z["len"] / max(lap_len, 1.0)
        if support_frac < MIN_SUPPORT_FRAC and z["support"] < MIN_SUPPORT_ABS:
            continue
        if z["len"] <= 0 or frac_len > MAX_ZONE_FRAC:
            continue
        kept.append(z)
    if not kept:
        return []
    kept.sort(key=lambda r: (-r["support"], -r["len"]))
    final = []
    used = np.zeros(len(kept), dtype=bool)
    for i in range(len(kept)):
        if used[i]:
            continue
        final.append(kept[i])
        for j in range(i+1, len(kept)):
            if used[j]:
                continue
            same_start = min(abs(kept[i]["start"] - kept[j]["start"]), abs((kept[i]["start"] + lap_len) - kept[j]["start"]), abs(kept[i]["start"] - (kept[j]["start"] + lap_len))) <= EPS_CLUSTER_M
            same_end = min(abs(kept[i]["end"] - kept[j]["end"]), abs((kept[i]["end"] + lap_len) - kept[j]["end"]), abs(kept[i]["end"] - (kept[j]["end"] + lap_len))) <= EPS_CLUSTER_M
            if same_start and same_end:
                used[j] = True
    return final


def _check_overlap(zones, longest_straight, lap_len):
    if not longest_straight or not zones:
        return False
    ss, se = longest_straight["start"], longest_straight["end"]
    for z in zones:
        zs, ze = z["start"], z["end"]
        if z["wraps"] or ze < zs:
            zone_segs = [(zs, lap_len), (0.0, ze)]
        else:
            zone_segs = [(zs, ze)]
        overlap = 0.0
        for zstart, zend in zone_segs:
            overlap += max(0.0, min(zend, se) - max(zstart, ss))
        if overlap / z["len"] >= 0.5:
            return True
    return False


def extract_drs_for_circuit(circuit_ref, year, event_name, cache_dir="./cache"):
    try:
        sess = fastf1.get_session(year, event_name, "R")
        sess.load(telemetry=True, laps=True)
        if not hasattr(sess, 'laps') or sess.laps is None or sess.laps.empty:
            return (np.nan, np.nan, False)
        all_drivers = sess.laps['DriverNumber'].dropna().unique().astype(str).tolist()
        ref_lap = _clean_fastest_lap(sess)
        if ref_lap is None:
            return (np.nan, np.nan, False)
        ref_tel = ref_lap.get_telemetry().add_distance()
        ref_tel = ref_tel.dropna(subset=["Distance", "X", "Y"])
        if ref_tel.empty:
            return (np.nan, np.nan, False)
        longest_straight = _find_longest_straight(ref_tel)
    except:
        return (np.nan, np.nan, False)
    best_result = None
    best_score = -1
    for driver in all_drivers:
        try:
            if driver not in sess.car_data or driver not in sess.pos_data:
                continue
            car = sess.car_data[driver].copy().dropna(subset=["Time", "Date", "DRS"]).reset_index(drop=True)
            pos = sess.pos_data[driver].copy().dropna(subset=["Time", "Date", "X", "Y"]).reset_index(drop=True)
            car = _ensure_dtypes(car)
            pos = _ensure_dtypes(pos)
            laps_drv = sess.laps[sess.laps["DriverNumber"] == driver][["LapNumber", "LapStartTime"]].dropna()
            if laps_drv.empty:
                continue
            laps_drv = laps_drv.sort_values("LapStartTime").reset_index(drop=True)
            car = car.sort_values("Time")
            car = pd.merge_asof(car, laps_drv, left_on="Time", right_on="LapStartTime", direction=LAP_JOIN_DIR, allow_exact_matches=True)
            car = car.dropna(subset=["LapNumber"]).copy()
            car["LapNumber"] = car["LapNumber"].astype(int)
            car = car.sort_values("Date")
            pos = pos.sort_values("Date")
            car = pd.merge_asof(car, pos[["Date", "X", "Y"]], on="Date", direction="nearest", tolerance=POS_JOIN_TOL)
            car = car.dropna(subset=["X", "Y"]).reset_index(drop=True)
            xy = car[["X", "Y"]].to_numpy()
            car["Distance"] = _kd_map_distance(ref_tel, xy)
            lap_len = car.groupby("LapNumber")["Distance"].agg(lambda s: float(s.max() - s.min()))
            lap_len = lap_len[lap_len > 1000.0]
            if lap_len.empty:
                continue
            lap_len_median = float(np.median(lap_len.values))
            laps_used = int(car["LapNumber"].nunique())
            car["drs_on"] = _drs_active_mask(car["DRS"])
            segs = _segments_from_mask(car)
            if not segs:
                continue
            zones = _cluster_segments(segs, lap_len_median)
            zones = _filter_zones(zones, laps_used, lap_len_median)
            if not zones:
                continue
            total_len = float(np.sum([z["len"] for z in zones]))
            num_zones = len(zones)
            on_longest = _check_overlap(zones, longest_straight, lap_len_median)
            score = num_zones * 1000 + laps_used
            if score > best_score:
                best_score = score
                best_result = (num_zones, total_len, on_longest)
        except Exception:
            continue
    return best_result if best_result else (np.nan, np.nan, False)


def find_event_name(circuit_ref, year, cache_dir="./cache"):
    pattern = CIRCUIT_TO_EVENT.get(circuit_ref)
    if not pattern:
        return None
    year_cache_dir = os.path.join(cache_dir, str(year))
    if not os.path.exists(year_cache_dir):
        return None
    pattern_norm = pattern.lower()
    try:
        for folder_name in os.listdir(year_cache_dir):
            folder_path = os.path.join(year_cache_dir, folder_name)
            if os.path.isdir(folder_path):
                parts = folder_name.split('_', 1)
                if len(parts) == 2:
                    event_name = parts[1].replace('_', ' ')
                    if pattern_norm in event_name.lower():
                        return event_name
    except Exception:
        pass
    return None


def add_drs_to_circuits(input_path, output_path, cache_dir="./cache"):
    #backup_path = output_path.replace(".csv", "_backup.csv")
    print(f"Reading {input_path}...")
    df = pd.read_csv(input_path)
    # df.to_csv(backup_path, index=False)
    # print(f"Backup saved to {backup_path}")
    if "drs_zones_detected" not in df.columns:
        df["drs_zones_detected"] = np.nan
    if "drs_total_len_m" not in df.columns:
        df["drs_total_len_m"] = np.nan
    if "drs_on_max" not in df.columns:
        df["drs_on_max"] = False
    print(f"\nProcessing {len(df)} circuits...\n")
    for idx, row in df.iterrows():
        circuit_ref = row["circuitRef"]
        year = int(row["year"])
        if pd.notna(row.get("drs_zones_detected")):
            print(f"[{idx+1}/{len(df)}] {year} {circuit_ref} - SKIP (already done)")
            continue
        print(f"[{idx+1}/{len(df)}] {year} {circuit_ref}...", end=" ")
        event_name = find_event_name(circuit_ref, year, cache_dir)
        if not event_name:
            print(f"✗ No event name")
            df.at[idx, "drs_zones_detected"] = 0
            df.at[idx, "drs_total_len_m"] = 0.0
            df.at[idx, "drs_on_max"] = False
            continue
        num_zones, total_len, on_longest = extract_drs_for_circuit(circuit_ref, year, event_name, cache_dir)
        df.at[idx, "drs_zones_detected"] = num_zones
        df.at[idx, "drs_total_len_m"] = total_len
        df.at[idx, "drs_on_max"] = on_longest
        if pd.notna(num_zones) and num_zones > 0:
            on_str = "✓longest" if on_longest else ""
            print(f"✓ {int(num_zones)} zones, {total_len:.0f}m {on_str}")
        else:
            print(f"✗ No zones")
        df.to_csv(output_path, index=False)
    print(f"\n{'='*72}")
    print(f"✓ Complete! Saved to {output_path}")
    print(f"{'='*72}")
    print(f"\nSummary:")
    print(f"  Total: {len(df)}")
    print(f"  With DRS: {(df['drs_zones_detected'] > 0).sum()}")
    print(f"  On longest straight: {df['drs_on_max'].sum()}")
    return df


#---------------------------------------------------------------------------------------------------------
# From the cached data , build the second dataset ( code has ben runned for 203, 2024 and 2025 separately)
#---------------------------------------------------------------------------------------------------------

def process_round_specs(input_csv, output_csv):
    """Process circuit specs and extract rounds information."""
    Circuit_spec_with_round = pd.read_csv(input_csv)
    Circuit_spec_with_round.drop(columns=["length_km", "turns", "type", "num_drs_zones"], inplace=True)
    Circuit_spec_with_round.to_csv(output_csv, index=False)
    return Circuit_spec_with_round


def discover_races_in_cache(cache_dir='cache', years=None, session_type='Race'):
    """
    Discover all race sessions in the cache directory.
    
    Parameters:
    -----------
    cache_dir : str, default='cache'
        Base cache directory path
    years : list, optional
        Specific years to process (e.g., [2023, 2024]).
        If None, all years in cache are processed.
    session_type : str, default='Race'
        Session type to find ('Race' or 'Qualifying'). Default is 'Race'.
    
    Returns:
    --------
    list of dict
        Each dict contains:
        - 'year': int (e.g., 2023)
        - 'race_name': str (e.g., '2023-09-03_Italian_Grand_Prix')
        - 'race_dir': str (e.g., 'cache/2023/2023-09-03_Italian_Grand_Prix')
        - 'session_path': str (e.g., 'cache/2023/2023-09-03_Italian_Grand_Prix/2023-09-03_Race')
        - 'session_date': str (e.g., '2023-09-03')
        - 'session_type': str ('Race' or 'Qualifying')
    """
    discovered_races = []
    cache_path = cache_dir
    
    if not os.path.isdir(cache_path):
        print(f"ERROR: Cache directory not found: {cache_path}")
        return discovered_races
    
    all_years = sorted([d for d in os.listdir(cache_path) 
                       if os.path.isdir(os.path.join(cache_path, d)) and d.isdigit()])
    
    if years:
        years_to_process = [str(y) for y in years if str(y) in all_years]
    else:
        years_to_process = all_years
    
    print(f"Scanning cache for {session_type} sessions in years: {years_to_process}")
    
    for year_str in years_to_process:
        year_path = os.path.join(cache_path, year_str)
        year_int = int(year_str)
        
        race_folders = sorted([d for d in os.listdir(year_path) 
                              if os.path.isdir(os.path.join(year_path, d))])
        
        for race_folder in race_folders:
            race_dir = os.path.join(year_path, race_folder)
            
            session_folders = sorted([d for d in os.listdir(race_dir)
                                     if os.path.isdir(os.path.join(race_dir, d)) and session_type in d])
            
            for session_folder in session_folders:
                session_path = os.path.join(race_dir, session_folder)
                session_date = session_folder.replace(f"_{session_type}", "")
                
                discovered_races.append({
                    'year': year_int,
                    'race_name': race_folder,
                    'race_dir': race_dir,
                    'session_path': session_path,
                    'session_date': session_date,
                    'session_type': session_type
                })
    
    print(f"✓ Discovered {len(discovered_races)} {session_type} sessions\n")
    return discovered_races


def build_race_output_dir(race_info, base_output='csv_output', create_dir=False):
    """
    Build organized output directory structure for a race.
    
    Parameters:
    -----------
    race_info : dict
        Dictionary from discover_races_in_cache() containing race metadata
    base_output : str, default='csv_output'
        Base output directory
    create_dir : bool, default=True
        If True, creates the directory if it doesn't exist
    
    Returns:
    --------
    dict
        Contains:
        - 'base_dir': str (e.g., 'csv_output/2023/Italian_Grand_Prix')
        - 'year': int
        - 'race_name_clean': str (race name without date prefix)
        - 'full_path': str (absolute path to output directory)
    """
    race_name_parts = race_info['race_name'].split('_')
    race_name_clean = '_'.join(race_name_parts[1:])
    
    year_str = str(race_info['year'])
    output_dir = os.path.join(base_output, year_str, race_name_clean)
    
    if create_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    return {
        'base_dir': output_dir,
        'year': race_info['year'],
        'race_name_clean': race_name_clean,
        'full_path': os.path.abspath(output_dir)
    }


# Extract Driver Info for the whole Season

def extract_drivers_for_race(base_path):
    """
    Extract driver information for a single race from a given cache path.
    
    Parameters:
    -----------
    base_path : str
        Path to race cache directory (e.g., "cache/2023/2023-09-03_Italian_Grand_Prix/2023-09-03_Race")
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with driver info including columns:
        - RacingNumber, FullName, TeamName
        - year, round, name, countryId (if circuit data found)
    """
    # Load driver info
    driver_path = os.path.join(base_path, "driver_info.ff1pkl")
    with open(driver_path, 'rb') as f:
        drivers = pickle.load(f)
    
    # Extract drivers info from the 'data' key (it's a defaultdict)
    drivers_data = drivers['data']
    
    # Extract all drivers
    drivers_info = []
    for driver_num, driver_dict in drivers_data.items():
        try:
            drivers_info.append({
                'RacingNumber': driver_dict['RacingNumber'],
                'FullName': driver_dict['FullName'],
                'TeamName': driver_dict['TeamName']
            })
        except (KeyError, TypeError):
            continue
    
    drivers_df = pd.DataFrame(drivers_info)
    
    # Try to add circuit info  
    try:
        circuit_df = pd.read_csv('csv_output/Rounds_2023_2025.csv')
        # Extract year from path
        year = int(base_path.split('/')[-3])
        
        # Extract round number by counting chronologically
        # Get year folder and find this race's position
        year_folder = base_path.split('/')[-3]
        race_folder = base_path.split('/')[-2]
        races_in_year = sorted([d for d in os.listdir(f"cache/{year_folder}") if os.path.isdir(f"cache/{year_folder}/{d}")])
        
        try:
            round_num = races_in_year.index(race_folder) + 1
        except (ValueError, IndexError):
            round_num = None
        
        # Get circuit info for this year and round
        if round_num:
            circuit_info = circuit_df[(circuit_df['year'] == year) & (circuit_df['round'] == round_num)]
            if not circuit_info.empty:
                circuit_info = circuit_info.iloc[0]
                for col in ['year', 'round', 'name', 'countryId']:
                    if col in circuit_info.index:
                        drivers_df[col] = circuit_info[col]
    except Exception as e:
        pass
    
    return drivers_df


def process_lap_timing_data(base_path, drivers_info, output_path=None, tolerance_seconds=0.1):
    """
    Process extended timing data and merge with driver information.
    
    Parameters:
    -----------
    base_path : str
        Path to the directory containing the '_extended_timing_data.ff1pkl' file
    drivers_info : pd.DataFrame
        DataFrame containing driver information with columns:
        ['RacingNumber', 'FullName', 'TeamName', 'year', 'round', 'name', 'countryId']
    output_path : str, optional
        Path to save the processed data as Excel file. If None, file is not saved.
        (default: None)
    tolerance_seconds : float, optional
        Tolerance in seconds for merging lap and position data (default: 0.05)
    
    Returns:
    --------
    pd.DataFrame
        Processed dataframe with columns:
        ['year', 'round', 'name', 'countryId', 'Name', 'Team', 'RacingNumber', 
         'Time', 'NumberOfLaps', 'LapTime', 'Position', 'IntervalToPositionAhead', 'NumberOfPitStops']
    """
    
    # Loading Data 
    fpath = os.path.join(base_path, "_extended_timing_data.ff1pkl")
    with open(fpath, "rb") as f:
        d = pickle.load(f)
    
    data = d["data"]  # the real stuff
    extended_timing_df = data[0].copy()
    
    # Extract Lap Data
    lap_data = extended_timing_df[['Driver', 'Time', 'NumberOfLaps', 'LapTime', 'NumberOfPitStops']].copy()
    lap_data_sorted = lap_data.sort_values(by=['Driver', 'NumberOfLaps']).reset_index(drop=True)
    
    # Extract Position Data
    position_data = data[1][['Driver', 'Time', 'Position', 'GapToLeader', 'IntervalToPositionAhead']].copy()
    
    # Clean Data 
    lap_data_clean = lap_data_sorted[lap_data_sorted['Time'].notna()].copy()
    position_data_clean = position_data[position_data['Time'].notna()].copy()
    
    # Convert Time to total seconds for more reliable merging
    def timedelta_to_seconds(td):
        """Convert timedelta to total seconds"""
        if pd.isna(td):
            return None
        return td.total_seconds()
    
    lap_data_clean['Time_seconds'] = lap_data_clean['Time'].apply(timedelta_to_seconds)
    position_data_clean['Time_seconds'] = position_data_clean['Time'].apply(timedelta_to_seconds)
    
    # Normalize Driver to string and remove rows with NaN Time_seconds
    lap_data_clean['Driver'] = lap_data_clean['Driver'].astype(str)
    position_data_clean['Driver'] = position_data_clean['Driver'].astype(str)
    lap_data_clean = lap_data_clean[lap_data_clean['Time_seconds'].notna()].copy()
    position_data_clean = position_data_clean[position_data_clean['Time_seconds'].notna()].copy()
    
    # Merge Lap and Position Data
    merged_parts = []
    for driver in lap_data_clean['Driver'].unique():
        lap_driver = lap_data_clean[lap_data_clean['Driver'] == driver].sort_values('Time_seconds').reset_index(drop=True)
        pos_driver = position_data_clean[position_data_clean['Driver'] == driver].sort_values('Time_seconds').reset_index(drop=True)
        
        if len(pos_driver) == 0:
            merged_parts.append(lap_driver)
            continue
        
        merged_driver = pd.merge_asof(
            lap_driver,
            pos_driver[['Time_seconds', 'Position', 'GapToLeader', 'IntervalToPositionAhead']],
            on='Time_seconds',
            direction='nearest',
            tolerance=tolerance_seconds
        )
        merged_parts.append(merged_driver)
    
    lap_data_merged = pd.concat(merged_parts, ignore_index=True)
    lap_data_merged = lap_data_merged.drop(columns=['Time_seconds'])
    lap_data_sorted = lap_data_merged.sort_values(by=['Driver', 'NumberOfLaps']).reset_index(drop=True)
    
    # Shift only IntervalToPositionAhead column down by one row
    lap_data_sorted['IntervalToPositionAhead'] = lap_data_sorted['IntervalToPositionAhead'].shift(1)
    
    print(f"✓ Merged {len(lap_data_sorted)} records with position and interval data")
    
    # Merge with Driver Info
    lap_data_detailed = lap_data_sorted.merge(
        drivers_info[['RacingNumber', 'FullName', 'TeamName', 'year', 'round', 'name', 'countryId']], 
        left_on='Driver', 
        right_on='RacingNumber',
        how='left'
    )
    lap_data_detailed = lap_data_detailed.drop(columns=['Driver'])
    
    # Convert LapTime to Seconds 
    lap_data_detailed['LapTime'] = lap_data_detailed['LapTime'].astype(str).str.split().str[-1]
    
    def time_to_seconds(time_str):
        """Convert time string (HH:MM:SS.microseconds) to total seconds"""
        if pd.isna(time_str) or time_str == 'NaT':
            return None
        try:
            parts = time_str.split(':')
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            total_seconds = hours * 3600 + minutes * 60 + seconds
            return round(total_seconds, 3)
        except:
            return None
    
    lap_data_detailed['LapTime'] = lap_data_detailed['LapTime'].apply(time_to_seconds)
    
    # Format and Export
    lap_data_detailed = lap_data_detailed.rename(columns={
        'FullName': 'Name',
        'TeamName': 'Team'
    })
    
    column_order = ['year', 'round', 'name', 'countryId', 'Name', 'Team', 'RacingNumber', 
                    'Time', 'NumberOfLaps', 'LapTime', 'Position', 'IntervalToPositionAhead', 'NumberOfPitStops']
    lap_data_detailed = lap_data_detailed[column_order]
    lap_data_detailed['Time'] = lap_data_detailed['Time'].astype(str)
    
    if output_path:
        lap_data_detailed.to_excel(output_path, index=False, sheet_name='Lap Data')
        
    
    return lap_data_detailed


# Function to extract timing app data - Specific columns only

def extract_timing_app_data(
    timing_app_path,
    output_csv_path=None,
    required_columns=None,
    sort_columns=None
):
    """
    Extract and process timing app data from a pickle file.
    
    Parameters:
    -----------
    timing_app_path : str
        Path to the timing_app_data.ff1pkl file
    output_csv_path : str, optional
        Path where the processed CSV should be saved. If None, file is not saved.
        (default: None)
    required_columns : list, optional
        List of columns to extract. Default: ['Driver', 'LapNumber', 'Stint', 
        'TotalLaps', 'Compound', 'New', 'TyresNotChanged']
    sort_columns : list, optional
        Columns to sort by. Default: ['Driver', 'LapNumber']
    
    Returns:
    --------
    pd.DataFrame
        The processed timing dataframe
    
    Raises:
    -------
    ValueError
        If the timing data structure is invalid or required columns not found
    """
    
    # Set defaults
    if required_columns is None:
        required_columns = ['Driver', 'LapNumber', 'Stint', 'TotalLaps', 'Compound', 'New', 'TyresNotChanged']
    
    if sort_columns is None:
        sort_columns = ['Driver', 'LapNumber']
    
    # Load timing app data
    with open(timing_app_path, "rb") as f:
        timing_data_raw = pickle.load(f)
    
    # Handle dictionary structure - extract the actual dataframe
    if isinstance(timing_data_raw, dict):
        if 'data' in timing_data_raw:
            timing_data = timing_data_raw['data']
        else:
            # Get the first non-empty value that looks like a dataframe
            timing_data = None
            for key, value in timing_data_raw.items():
                if hasattr(value, 'shape') and hasattr(value, 'columns'):
                    timing_data = value
                    break
            if timing_data is None:
                raise ValueError(f"Could not find DataFrame in timing data. Available keys: {list(timing_data_raw.keys())}")
    else:
        timing_data = timing_data_raw
    
    # Check which columns exist in the data
    existing_columns = [col for col in required_columns if col in timing_data.columns]
    
    # Extract only existing columns
    timing_df_filtered = timing_data[existing_columns].copy()
    
    # Fill TyresNotChanged: where not 0, fill with 1
    # 0 = tires were changed, 1 = tires NOT changed
    if 'TyresNotChanged' in timing_df_filtered.columns:
        timing_df_filtered['TyresNotChanged'] = timing_df_filtered['TyresNotChanged'].fillna(1)
    
    # Sort by specified columns (only use columns that exist)
    sort_cols_existing = [col for col in sort_columns if col in timing_df_filtered.columns]
    if sort_cols_existing:
        timing_df_filtered = timing_df_filtered.sort_values(by=sort_cols_existing).reset_index(drop=True)
    
    # Save to CSV if path provided
    if output_csv_path:
        timing_df_filtered.to_csv(output_csv_path, index=False)
    
    return timing_df_filtered

# Function to align tire information and clean timing data

def align_and_clean_timing_data(
    timing_df,
    output_csv_path=None,
    verbose=True,
    drop_columns=None
):
    """
    Align tire information and clean timing data by driver and stint.
    
    This function:
    1. Creates LapInStint column (calculated from actual lap numbers: CurrentLap - FirstLapInStint + 1)
    2. Removes rows with missing critical data (Compound or TotalLaps)
    3. Forward fills Compound and TyresNotChanged within each Driver+Stint group
    4. Creates per-driver lap counters
    5. Marks tire changes (New flag) at stint starts
    6. Converts TyresNotChanged to numeric type
    7. Optionally removes redundant columns
    
    Parameters:
    -----------
    timing_df : pd.DataFrame
        Input timing dataframe with columns: Driver, Stint, Compound, TotalLaps, TyresNotChanged, New
    output_csv_path : str, optional
        Path to save the cleaned CSV. If None, no file is saved.
    verbose : bool, default=True
        If True, prints alignment and completeness summaries
    drop_columns : list, optional
        Columns to drop before returning. Default: ['TotalLaps']
    
    Returns:
    --------
    pd.DataFrame
        Cleaned and aligned timing dataframe
    """
    
    if drop_columns is None:
        drop_columns = ['TotalLaps']
    
    # Create a working copy to avoid modifying the original
    df = timing_df.copy()
    
    # Step 1: Calculate LapInStint based on actual lap numbers (no filling/filtering needed)
    # For each Driver+Stint group, subtract the first lap number and add 1
    # This way: if stint starts at lap 15, then lap 15→1, lap 16→2, lap 17→3, etc.
    df['LapInStint'] = df.groupby(['Driver', 'Stint'])['TotalLaps'].transform(
        lambda x: x - x.min() + 1
    )
    
    # Step 2: Remove rows with missing critical data
    rows_initial = len(df)
    
    # Drop rows with no useful data (no Compound AND no TotalLaps)
    df_clean = df[
        df['Compound'].notna() | df['TotalLaps'].notna()
    ].copy()
    
    rows_after_first_filter = len(df_clean)
    
    # Remove rows where TotalLaps is missing (essential for lap sequencing)
    rows_before = len(df_clean)
    df_clean = df_clean[
        df_clean['TotalLaps'].notna()
    ].copy()
    rows_after = len(df_clean)
    
    # Step 3: Forward fill tire information within each Driver+Stint group
    for (driver, stint), group_idx in df_clean.groupby(['Driver', 'Stint']).groups.items():
        # Forward fill Compound (identifies tire type - same throughout stint)
        df_clean.loc[group_idx, 'Compound'] = df_clean.loc[group_idx, 'Compound'].ffill()
        
        # Forward fill TyresNotChanged (BEFORE filtering) to preserve tire state information
        df_clean.loc[group_idx, 'TyresNotChanged'] = df_clean.loc[group_idx, 'TyresNotChanged'].ffill()
    
    # Step 5: Create per-driver lap counters  
    df_clean['LapNumber'] = 0
    for driver in df_clean['Driver'].unique():
        driver_mask = df_clean['Driver'] == driver
        driver_indices = df_clean[driver_mask].index
        df_clean.loc[driver_indices, 'LapNumber'] = range(1, len(driver_indices) + 1)
    
    
    # Step 6: Mark "New" tires only at stint start
    df_clean['New'] = False  # Initialize all as FALSE
    
    for (driver, stint), group_idx in df_clean.groupby(['Driver', 'Stint']).groups.items():
        group_indices = list(group_idx)
        if len(group_indices) > 0:
            # Set ONLY the first row of each stint to TRUE (new tires at start)
            first_idx = group_indices[0]
            df_clean.loc[first_idx, 'New'] = True
    

    # Step 7: Convert TyresNotChanged to numeric
    df_clean['TyresNotChanged'] = pd.to_numeric(
        df_clean['TyresNotChanged'], 
        errors='coerce'
    ).fillna(1).astype(int)
    

    # Step 8: Drop redundant columns
    df_final = df_clean.drop(columns=drop_columns)
    

    # Step 9: Save to CSV (if path provided)
    if output_csv_path:
        df_final.to_csv(output_csv_path, index=False)
    
    return df_final


# Function to merge lap times and timing data

def merge_lap_timing_data(
    lap_times_path,
    timing_data_path,
    output_path=None,
    lap_times_file_type='xlsx',
    lap_number_column_lap_times='NumberOfLaps',
    lap_number_column_timing='LapNumber',
    driver_race_number_column='RacingNumber',
    timing_driver_column='Driver',
    race_identifier_lap_times='name',
    race_identifier_timing='race_name'
):
    """
    Merge lap times data with timing/tires data by RACE, driver (RacingNumber), and lap number.
    
    Parameters:
    -----------
    lap_times_path : str
        Path to the lap times file
    timing_data_path : str
        Path to the timing data CSV file
    output_path : str, optional
        Path to save the merged data as Excel or CSV
    lap_times_file_type : str, default='xlsx'
        File type of lap times data ('xlsx' or 'csv')
    lap_number_column_lap_times : str, default='NumberOfLaps'
        Name of the lap number column in lap times data
    lap_number_column_timing : str, default='LapNumber'
        Name of the lap number column in timing data
    driver_race_number_column : str, default='RacingNumber'
        Name of the racing number column in lap times data
    timing_driver_column : str, default='Driver'
        Name of the driver column in timing data
    race_identifier_lap_times : str, default='name'
        Name of the race identifier column in lap times data
    race_identifier_timing : str, default='race_name'
        Name of the race identifier column in timing data
    
    Returns:
    --------
    pd.DataFrame
        Merged dataframe
    """
    
    # Load data
    if lap_times_file_type.lower() == 'xlsx':
        lap_times_df = pd.read_excel(lap_times_path)
    else:
        lap_times_df = pd.read_csv(lap_times_path)
    
    timing_df = pd.read_csv(timing_data_path)
    
    # Create merge keys (include race identifier!)
    lap_times_df['_merge_race'] = lap_times_df[race_identifier_lap_times].astype(str)
    lap_times_df['_merge_driver'] = pd.to_numeric(lap_times_df[driver_race_number_column], errors='coerce')
    lap_times_df['_merge_lap'] = pd.to_numeric(lap_times_df[lap_number_column_lap_times], errors='coerce')
    
    timing_df['_merge_race'] = timing_df[race_identifier_timing].astype(str)
    timing_df['_merge_driver'] = pd.to_numeric(timing_df[timing_driver_column], errors='coerce')
    timing_df['_merge_lap'] = pd.to_numeric(timing_df[lap_number_column_timing], errors='coerce')
    
    # Remove invalid rows and merge
    lap_times_df = lap_times_df.dropna(subset=['_merge_driver', '_merge_lap'])
    timing_df = timing_df.dropna(subset=['_merge_driver', '_merge_lap'])
    
    # Merge on RACE + DRIVER + LAP (not just driver + lap)
    merged_df = pd.merge(
        lap_times_df,
        timing_df,
        left_on=['_merge_race', '_merge_driver', '_merge_lap'],
        right_on=['_merge_race', '_merge_driver', '_merge_lap'],
        how='inner'
    )
    
    # Clean up temporary columns
    merged_df = merged_df.drop(columns=['_merge_race', '_merge_driver', '_merge_lap'])
    
    # Remove duplicate columns
    cols_to_remove = []
    if lap_number_column_lap_times in merged_df.columns and lap_number_column_timing in merged_df.columns:
        cols_to_remove.append(lap_number_column_timing)
    if driver_race_number_column in merged_df.columns and timing_driver_column in merged_df.columns:
        cols_to_remove.append(timing_driver_column)
    
    if cols_to_remove:
        merged_df = merged_df.drop(columns=[col for col in cols_to_remove if col in merged_df.columns])
    
    # Save if path provided
    if output_path:
        if output_path.endswith('.xlsx'):
            merged_df.to_excel(output_path, index=False)
        else:
            merged_df.to_csv(output_path, index=False)
    
    return merged_df



# Merge Lap Data with Weather Data

def merge_race_data_with_weather(merged_race_data, race_cache_path, tolerance_seconds=30):
    """
    Merge race lap data with weather data using nearest time matching.
    
    Parameters:
    -----------
    merged_race_data : pd.DataFrame
        DataFrame with lap timing data including 'Time' and 'RacingNumber' columns
    race_cache_path : str
        Path to race cache directory (e.g., "cache/2023/2023-09-03_Italian_Grand_Prix/2023-09-03_Race")
    tolerance_seconds : int
        Time tolerance in seconds for matching (default: 30)
    
    Returns:
    --------
    pd.DataFrame
        Merged dataset with weather columns added, sorted by RacingNumber and lap progression
    """
    
    # Load weather data from pickle file
    weather_pkl_path = os.path.join(race_cache_path, 'weather_data.ff1pkl')
    with open(weather_pkl_path, 'rb') as f:
        weather_data_dict = pickle.load(f)
    
    df_weather = pd.DataFrame(weather_data_dict['data'])
    df_weather['Time'] = df_weather['Time'].astype(str)
    
    # Create a copy for merging (preserve original)
    df_for_merge = merged_race_data.copy()
    original_columns = list(merged_race_data.columns)
    
    # Convert Time columns to timedelta for matching
    df_for_merge['Time_td'] = pd.to_timedelta(df_for_merge['Time'])
    df_weather['Time_td'] = pd.to_timedelta(df_weather['Time'])
    
    # Sort by time
    df_for_merge_sorted = df_for_merge.sort_values('Time_td').reset_index(drop=True)
    df_weather_sorted = df_weather.sort_values('Time_td').reset_index(drop=True)
    
    # Merge on nearest time
    result = pd.merge_asof(
        df_for_merge_sorted,
        df_weather_sorted[['Time_td', 'AirTemp', 'Humidity', 'Pressure', 'Rainfall', 'TrackTemp', 'WindDirection', 'WindSpeed']],
        left_on='Time_td',
        right_on='Time_td',
        direction='nearest',
        tolerance=pd.Timedelta(f'{tolerance_seconds}s')
    )
    
    # Restore original column order and add weather columns at end
    weather_columns = ['AirTemp', 'Humidity', 'Pressure', 'Rainfall', 'TrackTemp', 'WindDirection', 'WindSpeed']
    final_columns = original_columns + weather_columns
    result = result[final_columns]
    
    # Sort by RacingNumber and lap progression
    result = result.sort_values(
        by=['RacingNumber', 'NumberOfLaps'],
        ascending=[True, True]
    ).reset_index(drop=True)
    
    return result


#--------------------------------------------------------------------------------------------
# From OpenF1 AP1, extract the race control dataset, and merge it with the lap timing dataset
#--------------------------------------------------------------------------------------------


def get_session_keys(year):
    """
    Extract session keys from OpenF1 API for a given year.
    
    Parameters:
    -----------
    year : int
        The year to fetch session keys for (e.g., 2023)
    
    Returns:
    --------
    dict
        Dictionary mapping circuit short names to session keys
    """
    sessions_url = f"https://api.openf1.org/v1/sessions?year={year}&session_name=Race"
    with urlopen(sessions_url) as resp:
        sessions = json.loads(resp.read().decode("utf-8"))
    
    session_keys = {
        s["circuit_short_name"]: s["session_key"]
        for s in sessions
    }
    
    return session_keys


def fetch_race_control_data(session_keys_dict, max_retries=3, base_delay=2):
    """
    Fetch race control data from OpenF1 API for all sessions.
    
    Parameters:
    -----------
    session_keys_dict : dict
        Dictionary mapping Grand Prix names to session keys
    max_retries : int, optional
        Maximum number of retry attempts for failed requests (default: 3)
    base_delay : int, optional
        Delay in seconds between requests (default: 2)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing all race control events with grand_prix and race_session_key columns
    """
    BASE_URL = "https://api.openf1.org/v1/race_control"
    all_events = []
    
    for idx, (gp_name, session_key) in enumerate(session_keys_dict.items(), 1):
        url = f"{BASE_URL}?session_key={session_key}"
        
        # Retry logic with exponential backoff
        for attempt in range(max_retries):
            try:
                print(f"[{idx}/{len(session_keys_dict)}] Fetching {gp_name} (session_key={session_key})...", end=" ")
                with urlopen(url) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                
                # Tag each event with GP name and session_key
                for ev in data:
                    ev["grand_prix"] = gp_name
                    ev["race_session_key"] = session_key
                    all_events.append(ev)
                
                print(f"✓ ({len(data)} events)")
                
                # Add delay between successful requests to avoid rate limiting
                if idx < len(session_keys_dict):
                    time.sleep(base_delay)
                break
                
            except HTTPError as e:
                if e.code == 429:  # Too Many Requests
                    wait_time = base_delay * (2 ** attempt)  # Exponential backoff
                    print(f"Rate limited! Retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    print(f"HTTP Error {e.code}: {e.reason}")
                    break
            except Exception as e:
                print(f"Error: {e}")
                break
    
    df_rc = pd.DataFrame(all_events)
    return df_rc

def process_race_control_data(df_input):
    """
    Process race control data: detect flags, classify clean laps, and format output.
    
    Parameters:
    -----------
    df_input : pd.DataFrame
        Input race control data with columns: message, flag, scope, category, 
        driver_number, lap_number, grand_prix
    
    Returns:
    --------
    pd.DataFrame
        Processed race control dataframe with normalized flags and clean lap classification
    """
    rc = df_input.copy()
    
    # Normalize text columns
    rc["msg_upper"] = rc["message"].str.upper().fillna("")
    rc["flag_upper"] = rc["flag"].astype(str).str.upper().fillna("")

    # Remove rows with specific flags (BLUE, CHEQUERED, BLACK AND WHITE, NONE, NAN, empty, Green, CLEAR)
    rc = rc[~rc["flag_upper"].isin(["BLUE", "CHEQUERED", "BLACK AND WHITE", "NONE", "NAN", "", "CLEAR", "GREEN"])].copy()
    
    # Safety car events
    rc["sc_deploy"] = rc["msg_upper"].str.contains("SAFETY CAR DEPLOYED", case=False, na=False)
    rc["sc_end"] = (
        rc["msg_upper"].str.contains("SAFETY CAR IN THIS LAP", case=False, na=False) |
        rc["msg_upper"].str.contains("SAFETY CAR ENDING", case=False, na=False) |
        rc["msg_upper"].str.contains("SAFETY CAR ENTERS PIT LANE", case=False, na=False)
    )
    
    # Virtual safety car events
    rc["vsc_deploy"] = rc["msg_upper"].str.contains("VIRTUAL SAFETY CAR DEPLOYED", case=False, na=False)
    rc["vsc_end"] = rc["msg_upper"].str.contains("VIRTUAL SAFETY CAR END", case=False, na=False)
    
    # Track flags (Yellow / Double Yellow / Red)
    track_mask = rc["scope"].isin(["Track", "Sector", "SafetyCar"]) if "scope" in rc.columns else pd.Series([False] * len(rc))
    track_rc = rc[track_mask].copy()
    
    if len(track_rc) > 0:
        track_rc["is_yellow"] = (
            track_rc["flag_upper"].str.contains("YELLOW", case=False, na=False) |
            track_rc["msg_upper"].str.contains("YELLOW", case=False, na=False)
        )
        track_rc["is_double_yellow"] = track_rc["msg_upper"].str.contains("DOUBLE YELLOW", case=False, na=False)
        track_rc["is_red"] = track_rc["msg_upper"].str.contains("RED FLAG", case=False, na=False)
        
        rc.loc[track_rc.index, "is_yellow"] = track_rc["is_yellow"]
        rc.loc[track_rc.index, "is_double_yellow"] = track_rc["is_double_yellow"]
        rc.loc[track_rc.index, "is_red"] = track_rc["is_red"]
    else:
        rc["is_yellow"] = False
        rc["is_double_yellow"] = False
        rc["is_red"] = False
    
    rc["is_yellow"] = rc["is_yellow"].fillna(False).astype(bool)
    rc["is_double_yellow"] = rc["is_double_yellow"].fillna(False).astype(bool)
    rc["is_red"] = rc["is_red"].fillna(False).astype(bool)
    
    # # Blue flags (Driver specific)
    # drv_mask = rc["scope"] == "Driver" if "scope" in rc.columns else pd.Series([False] * len(rc))
    # drv_rc = rc[drv_mask].copy()
    
    # if len(drv_rc) > 0:
    #     drv_rc["is_blue"] = (
    #         (drv_rc["flag_upper"] == "BLUE") |
    #         drv_rc["msg_upper"].str.contains("BLUE FLAG", case=False, na=False)
    #     )
    #     rc.loc[drv_rc.index, "is_blue"] = drv_rc["is_blue"]
    # else:
    #     rc["is_blue"] = False
    
    # rc["is_blue"] = rc["is_blue"].fillna(False).astype(bool)
    
    # Clean lap classification
    #lap_clean = False when is_yellow OR vsc_deploy OR sc_deploy OR is_red is True, True otherwise
    rc["lap_clean"] = ~((rc["is_yellow"]) | (rc["vsc_deploy"]) | (rc["sc_deploy"]) | (rc["is_red"]))
  
    
    laps_with_flags = rc[(rc["is_yellow"]) | (rc["is_red"])].groupby(["driver_number", "lap_number"])["lap_number"].count().index   #(rc["is_blue"]) removed , in case uncomment and add it 
    for driver, lap in laps_with_flags:
        rc.loc[(rc["driver_number"] == driver) & (rc["lap_number"] == lap), "lap_clean"] = False
    
    # Select key columns for output
    key_columns = [
        'lap_number', 'driver_number', 'grand_prix',  # metadata
        'flag_upper',  # normalized
        'sc_deploy', 'sc_end', 'vsc_deploy', 'vsc_end',  # SC/VSC events
        'is_yellow', 'is_double_yellow', 'is_red',  # track flags
        #'is_blue',  # driver flags
        'lap_clean'  # clean lap classification
    ]
    
    # Keep only columns that exist
    rc_processed = rc[[col for col in key_columns if col in rc.columns]].copy()
    
    # Create a sort key for driver_number that handles NaN values
    rc_processed['_sort_driver'] = rc_processed['driver_number'].fillna(float('inf'))
    
    # Sort: For every Grand Prix, sort by driver first, then by lap number
    rc_processed = rc_processed.sort_values(
        by=['grand_prix', '_sort_driver', 'lap_number'], 
        na_position='last'
    ).reset_index(drop=True)
    
    # Remove the helper column
    rc_processed = rc_processed.drop(columns=['_sort_driver'])
    
    return rc_processed



# CIRCUIT TO GRAND PRIX NAME MAPPING

CIRCUIT_TO_RACE_NAME_MAP = {
    "Sakhir": "Bahrain_Grand_Prix","Saudi Arabia": "Saudi_Arabian_Grand_Prix","Melbourne": "Australian_Grand_Prix",
    "Suzuka": "Japanese_Grand_Prix","Shanghai": "Chinese_Grand_Prix","Miami": "Miami_Grand_Prix",
    "Imola": "Emilia_Romagna_Grand_Prix","Monaco": "Monaco_Grand_Prix","Barcelona": "Spanish_Grand_Prix",
    "Montreal": "Canadian_Grand_Prix","Spielberg": "Austrian_Grand_Prix", "Silverstone": "British_Grand_Prix",
    "Budapest": "Hungarian_Grand_Prix", "Spa": "Belgian_Grand_Prix","Zandvoort": "Dutch_Grand_Prix",
    "Monza": "Italian_Grand_Prix","Azerbaijan": "Azerbaijan_Grand_Prix","Marina Bay": "Singapore_Grand_Prix",
    "Austin": "United_States_Grand_Prix","Mexico City": "Mexico_City_Grand_Prix","Sao Paulo": "Sao_Paulo_Grand_Prix","Las Vegas": "Las_Vegas_Grand_Prix",
    "Abu Dhabi": "Abu_Dhabi_Grand_Prix","Qatar": "Qatar_Grand_Prix",
}


# EXTRACT TRACK LIMIT VIOLATIONS AND OFF-TRACK INCIDENTS

def extract_rc_violations(df: pd.DataFrame, gp_col: str = "grand_prix", use_race_names: bool = True):
    """
    Extract track limit violations and off-track incidents from race control data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input race control data with columns: message, flag, driver_number, lap_number, grand_prix
    gp_col : str, optional
        Name of the Grand Prix column (default: "grand_prix")
    use_race_names : bool, optional
        If True, map circuit names to standardized race_name format (default: True)
    
    Returns:
    --------
    pd.DataFrame
        Filtered dataframe containing only violations with columns:
        race_name, car_number, lap_number, track_limit, off_track, any_violation, message
    """
    rc = df.copy()

    # Keep only messages involving violations (flag is None for these)
    mask = (
        rc["flag"].isna() &
        rc["message"].str.contains("TRACK LIMIT|OFF TRACK", case=False, na=False)
    )
    rc = rc[mask]

    # Boolean flags
    rc["track_limit"] = rc["message"].str.contains("TRACK LIMIT", case=False)
    rc["off_track"]   = rc["message"].str.contains("OFF TRACK",   case=False)
    rc["any_violation"] = rc["track_limit"] | rc["off_track"]

    # Extract car number from message (fallback)
    car_fallback = (
        rc["message"]
        .str.extract(r"CAR\s+(?P<car_number>\d+)", expand=True)
        .astype("Int64")
    )

    # Prefer driver_number when valid, else fallback
    rc["driver_number"] = rc["driver_number"].astype("Int64")
    rc["car_number"] = rc["driver_number"].where(
        rc["driver_number"].notna() & (rc["driver_number"] != 0),
        car_fallback["car_number"]
    )
    
    # Map circuit names to standardized race_name format if requested
    if use_race_names and gp_col in rc.columns:
        rc["race_name"] = rc[gp_col].map(CIRCUIT_TO_RACE_NAME_MAP).fillna(rc[gp_col])
    else:
        rc["race_name"] = rc[gp_col]

    # Select output columns
    out = rc[[
        "race_name",
        "car_number",
        "lap_number",
        "track_limit",
        "off_track",
        "any_violation",
        "message"
    ]].reset_index(drop=True)

    return out

# MERGING 

# Violation first 

def merge_violations_with_timing(
    timing_dir: str = "csv_output",
    violations_dir: str = "csv_output",
    output_dir: str = "csv_output",
    years: list = [2023, 2024, 2025]
):
    """
    Merge track limit violations with lap timing and weather data (year-by-year).
    Saves year-specific Excel files for inspection.
    
    Parameters:
    -----------
    timing_dir : str, optional
        Directory containing year-specific timing Excel files (default: "csv_output")
    violations_dir : str, optional
        Directory containing violation CSV files (default: "csv_output")
    output_dir : str, optional
        Output directory for merged Excel files (default: "csv_output")
    years : list, optional
        Years to process (default: [2023, 2024, 2025])
    """
    print(f"\n{'='*60}")
    print("Merging violations with lap timing data...")
    print(f"{'='*60}\n")
    
    for year in years:
        print(f"Processing year {year}...")
        
        try:
            # Load timing data
            timing_file = f"{timing_dir}/{year}_All_Races_Merged_Lap_Timing_With_Weather.xlsx"
            df_timing = pd.read_excel(timing_file)
            
            # Load violations
            violations_file = f"{violations_dir}/race_control_{year}_violations.csv"
            df_violations = pd.read_csv(violations_file)
            
            # Merge on race_name, RacingNumber, and lap number
            df_merged = df_timing.merge(
                df_violations,
                left_on=["race_name", "RacingNumber", "NumberOfLaps"],
                right_on=["race_name", "car_number", "lap_number"],
                how="left"
            )
            
            # Drop redundant columns
            df_merged = df_merged.drop(columns=["car_number", "lap_number"])
            
            # Fill violation columns with False where NaN (no violation)
            df_merged["track_limit"] = df_merged["track_limit"].fillna(False).astype(bool)
            df_merged["off_track"] = df_merged["off_track"].fillna(False).astype(bool)
            df_merged["any_violation"] = df_merged["any_violation"].fillna(False).astype(bool)
            
            # Save Excel file
            excel_file = f"{output_dir}/{year}_Merged_Lap_Timing_With_Violations.xlsx"
            df_merged.to_excel(excel_file, index=False)
            
            print(f"  ✓ Saved: {excel_file}\n")
            
        except FileNotFoundError as e:
            print(f"  ✗ Error for {year}: {e}\n")
            continue
    


def merge_race_flags_to_violations(
    violations_dir: str = "csv_output",
    race_control_dir: str = "csv_output",
    output_dir: str = "csv_output",
    years: list = [2023, 2024, 2025]
):
    """
    Add race control flags to violations-merged lap timing data.
    
    For each (grand_prix, lap_number), merges race control flags to all drivers at that lap.
    Example: Yellow flag on lap 20 in Austin applies to all drivers on lap 20 in Austin.
    
    Input: {year}_Merged_Lap_Timing_With_Violations.xlsx (timing + violations data)
           race_control_{year}_processed.csv (processed flags)
    
    Output: {year}_Merged_Lap_Timing_Violations_RaceControl.xlsx
    
    Parameters:
    -----------
    violations_dir : str
        Directory containing {year}_Merged_Lap_Timing_With_Violations.xlsx files
    race_control_dir : str
        Directory containing race_control_{year}_processed.csv files
    output_dir : str
        Output directory for merged CSV files
    years : list
        Years to process
    """
    yearly_data = {}
    
    for year in years:
        violations_file = f"{violations_dir}/{year}_Merged_Lap_Timing_With_Violations.xlsx"
        rc_file = f"{race_control_dir}/race_control_{year}_processed.csv"
        output_file = f"{output_dir}/{year}_Merged_Lap_Timing_Violations_RaceControl.xlsx"
        
        try:
            # Load data
            df_timing = pd.read_excel(violations_file)
            df_rc = pd.read_csv(rc_file)
            
            # Standardize column names
            if 'race_name' in df_timing.columns and 'grand_prix' not in df_timing.columns:
                df_timing = df_timing.rename(columns={'race_name': 'grand_prix'})
            elif 'name' in df_timing.columns and 'grand_prix' not in df_timing.columns:
                df_timing = df_timing.rename(columns={'name': 'grand_prix'})
            
            if 'NumberOfLaps' in df_timing.columns:
                df_timing = df_timing.rename(columns={'NumberOfLaps': 'lap_number'})
            
            # Extract race-level flags (one row per grand_prix, lap_number)
            rc_race_level = df_rc[[
                'grand_prix', 'lap_number', 'flag_upper',
                'sc_deploy', 'sc_end', 'vsc_deploy', 'vsc_end',
                'is_yellow', 'is_double_yellow', 'is_red', 'lap_clean'
            ]].copy()
            
            rc_race_level = rc_race_level.groupby(['grand_prix', 'lap_number']).first().reset_index()
            
            # Merge: same flags apply to all drivers in same lap/grand_prix
            df_merged = df_timing.merge(
                rc_race_level,
                on=['grand_prix', 'lap_number'],
                how='left'
            )
            
            # Fill missing flags with False
            flag_columns = [
                'sc_deploy', 'sc_end', 'vsc_deploy', 'vsc_end',
                'is_yellow', 'is_double_yellow', 'is_red', 'lap_clean'
            ]
            for col in flag_columns:
                if col in df_merged.columns:
                    df_merged[col] = df_merged[col].fillna(False).astype(bool)
            
            # Save output
            df_merged.to_excel(output_file, index=False)
            yearly_data[year] = df_merged
            
            # Clean up
            del df_timing, df_rc, rc_race_level, df_merged
            gc.collect()
            
        except FileNotFoundError as e:
            print(f"Skipping {year}: File not found ({e})")
            continue
        except Exception as e:
            print(f"Error processing {year}: {e}")
            continue
    
    return yearly_data if yearly_data else None

# Mergin the violation with the race contol dataset

def map_race_control_to_race_name(grand_prix):
    """
    Map race control grand_prix names to race_name format.
    Example: "Austin" -> "United_States_Grand_Prix"
    """
    if pd.isna(grand_prix):
        return None
    
    grand_prix = str(grand_prix).strip()
    
    # Map from race control grand_prix to race_name format
    mapping = {
        'Sakhir': 'Bahrain_Grand_Prix','Jeddah': 'Saudi_Arabian_Grand_Prix','Melbourne': 'Australian_Grand_Prix','Suzuka': 'Japanese_Grand_Prix',
        'Shanghai': 'Chinese_Grand_Prix','Miami': 'Miami_Grand_Prix','Imola': 'Emilia_Romagna_Grand_Prix','Monte Carlo': 'Monaco_Grand_Prix',
        'Barcelona': 'Spanish_Grand_Prix','Montreal': 'Canadian_Grand_Prix','Spielberg': 'Austrian_Grand_Prix','Silverstone': 'British_Grand_Prix',
        'Hungaroring': 'Hungarian_Grand_Prix','Spa-Francorchamps': 'Belgian_Grand_Prix', 'Zandvoort': 'Dutch_Grand_Prix','Monza': 'Italian_Grand_Prix',
        'Baku': 'Azerbaijan_Grand_Prix','Singapore': 'Singapore_Grand_Prix','SINGAPORE': 'Singapore_Grand_Prix','Austin': 'United_States_Grand_Prix',
        'Mexico City': 'Mexico_City_Grand_Prix','Interlagos': 'Sao_Paulo_Grand_Prix','Las Vegas': 'Las_Vegas_Grand_Prix','Abu Dhabi': 'Abu_Dhabi_Grand_Prix',
        'Qatar': 'Qatar_Grand_Prix',
    }
    
    # Check for exact match first
    if grand_prix in mapping:
        return mapping[grand_prix]
    
    # Check for case-insensitive match
    for key, value in mapping.items():
        if key.lower() == grand_prix.lower():
            return value
    
    # If no match found, return None
    return None


def merge_race_flags_to_violations(
    violations_dir: str = "csv_output",
    race_control_dir: str = "csv_output",
    output_dir: str = "csv_output",
    years: list = [2023, 2024, 2025]
):
    """
    Add race control flags to violations-merged lap timing data.
    
    For each (grand_prix, lap_number), merges race control flags to all drivers at that lap.
    Example: Yellow flag on lap 20 in Austin applies to all drivers on lap 20 in Austin.
    
    Output: {year}_Merged_Lap_Timing_Violations_RaceControl.xlsx
    

    """
    yearly_data = {}
    
    for year in years:
        violations_file = f"{violations_dir}/{year}_Merged_Lap_Timing_With_Violations.xlsx"
        rc_file = f"{race_control_dir}/race_control_{year}_processed.csv"
        output_file = f"{output_dir}/{year}_Merged_Lap_Timing_Violations_RaceControl.xlsx"
        
        try:
            # Load data
            df_timing = pd.read_excel(violations_file)
            df_rc = pd.read_csv(rc_file)
            
            # Rename lap number column in timing data if needed
            if 'NumberOfLaps' in df_timing.columns:
                df_timing = df_timing.rename(columns={'NumberOfLaps': 'lap_number'})
            
            # Map race_control grand_prix to race_name format
            df_rc['race_name'] = df_rc['grand_prix'].apply(map_race_control_to_race_name)
            
            # Extract race-level flags (one row per race_name, lap_number)
            rc_race_level = df_rc[[
                'race_name', 'lap_number', 'flag_upper',
                'sc_deploy', 'sc_end', 'vsc_deploy', 'vsc_end',
                'is_yellow', 'is_double_yellow', 'is_red', 'lap_clean'
            ]].copy()
            
            # Remove rows with None race_name (unmapped grand_prix)
            rc_race_level = rc_race_level[rc_race_level['race_name'].notna()]
            
            rc_race_level = rc_race_level.groupby(['race_name', 'lap_number']).first().reset_index()
            
            # Merge: same flags apply to all drivers in same lap/race
            df_merged = df_timing.merge(
                rc_race_level,
                on=['race_name', 'lap_number'],
                how='left'
            )
            
            # Fill missing flags with False (except lap_clean = True for clean laps)
            flag_columns = [
                'sc_deploy', 'sc_end', 'vsc_deploy', 'vsc_end',
                'is_yellow', 'is_double_yellow', 'is_red', 'lap_clean'
            ]
            for col in flag_columns:
                if col in df_merged.columns:
                    if col == 'lap_clean':
                        # If lap not in processed file (no flags), lap_clean = True
                        df_merged[col] = df_merged[col].fillna(True)
                    else:
                        # Other flags default to False
                        df_merged[col] = df_merged[col].fillna(False)
                    # Convert to bool efficiently
                    df_merged[col] = df_merged[col].astype('bool')
            
            # Save output
            df_merged.to_excel(output_file, index=False)
            yearly_data[year] = df_merged
            
            # Clean up
            del df_timing, df_rc, rc_race_level, df_merged
            gc.collect()
            
        except FileNotFoundError as e:
            print(f"Skipping {year}: File not found ({e})")
            continue
        except Exception as e:
            print(f"Error processing {year}: {e}")
            continue
    
    return yearly_data if yearly_data else None


# Actually combining the three years into one dataset

def combine_yearly_results(
    output_dir: str = "csv_output",
    years: list = [2023, 2024, 2025],
    combined_output: str = "csv_output/Combined_Lap_Timing_Violations_RaceControl.xlsx"
):
    """
    Combine the yearly merged files into one combined dataset.
    
    Input: {year}_Merged_Lap_Timing_Violations_RaceControl.xlsx
    Output: Combined_Lap_Timing_Violations_RaceControl.xlsx
    """
    all_data = []
    
    for year in years:
        file = f"{output_dir}/{year}_Merged_Lap_Timing_Violations_RaceControl.xlsx"
        try:
            df = pd.read_excel(file)
            all_data.append(df)
        except FileNotFoundError:
            print(f"Skipping {year}: File not found")
            continue
    
    if all_data:
        df_combined = pd.concat(all_data, ignore_index=True)
        df_combined.to_excel(combined_output, index=False)
        print(f"Combined {len(all_data)} years into {combined_output}")
        print(f"Total rows: {len(df_combined):,}")
        return df_combined
    else:
        print("No files to combine!")
        return None

# Cleaning the combined dataset and adding PitIn/PitOut indicators

def clean_combined_data(
    df_combined,
    output_file: str = "csv_output/Combined_Lap_Timing_Clean.xlsx"
):
    """
    Clean the combined dataset by removing unnecessary columns and reordering.
    
    Input: Combined dataset with all columns
    Output: Cleaned dataset with only modeling-relevant columns
    """
    df_clean = df_combined.copy()
    
    # Rename race_name to Grand_Prix FIRST (before sorting/grouping)
    if 'race_name' in df_clean.columns:
        df_clean = df_clean.rename(columns={'race_name': 'Grand_Prix'})
    
    # Determine race column name (use whatever exists)
    race_col = 'Grand_Prix' if 'Grand_Prix' in df_clean.columns else 'race_name'
    
    # Sort by driver and lap number for consistency
    df_clean = df_clean.sort_values(['year', 'round', race_col, 'RacingNumber', 'lap_number']).reset_index(drop=True)
    
    # Remove unnecessary columns (keep LapInStint for fallback pit detection)
    columns_to_drop = [
        'countryId', 'Time', 'NumberOfPitStops', 'TyresNotChanged',
        'track_limit', 'off_track', 'message', 'flag_upper',
        'sc_deploy', 'sc_end', 'vsc_deploy', 'vsc_end', 'is_yellow', 'is_double_yellow', 'is_red'
    ]
    df_clean = df_clean.drop(columns=[col for col in columns_to_drop if col in df_clean.columns])
    
    # Reorder columns for better readability
    column_order = [
        'year', 'round', 'name', 'Grand_Prix', 'Name', 'RacingNumber', 'Team',
        'lap_number', 'LapTime', 'IntervalToPositionAhead', 'Position',
        'Stint', 'Compound', 'New', 'LapInStint',
        'AirTemp', 'Humidity', 'Pressure', 'TrackTemp', 'WindDirection', 'WindSpeed', 'Rainfall',
        'any_violation', 'lap_clean'
    ]
    df_clean = df_clean[[col for col in column_order if col in df_clean.columns]]
    
    # Convert Rainfall to boolean (rain indicator: 0 or 1 → False or True)
    if 'Rainfall' in df_clean.columns:
        df_clean['Rainfall'] = df_clean['Rainfall'].astype('bool')
    
    df_clean.to_excel(output_file, index=False)
    print(f"\nCleaned data saved to {output_file}")
    print(f"Total rows: {len(df_clean):,}")
    print(f"Final columns ({len(df_clean.columns)}): {list(df_clean.columns)}")
    
    return df_clean


# OPENF1 PIT STOP DATA - Fetch and Merge

def map_openf1_gp_to_race_name(gp_name):
    """
    Map OpenF1 circuit names to standard race_name format.
    
    Example: 'Sakhir' -> 'Bahrain_Grand_Prix', 'Catalunya' -> 'Spanish_Grand_Prix'
    """
    if pd.isna(gp_name):
        return None
    
    gp_name = str(gp_name).strip()
    
    # Map from OpenF1 circuit names to race_name format
    # These are the same as race control circuit names
    mapping = {
        'Sakhir': 'Bahrain_Grand_Prix',
        'Jeddah': 'Saudi_Arabian_Grand_Prix',
        'Melbourne': 'Australian_Grand_Prix',
        'Suzuka': 'Japanese_Grand_Prix',
        'Shanghai': 'Chinese_Grand_Prix',
        'Miami': 'Miami_Grand_Prix',
        'Imola': 'Emilia_Romagna_Grand_Prix',
        'Monte Carlo': 'Monaco_Grand_Prix',
        'Catalunya': 'Spanish_Grand_Prix',
        'Montreal': 'Canadian_Grand_Prix',
        'Spielberg': 'Austrian_Grand_Prix',
        'Silverstone': 'British_Grand_Prix',
        'Hungaroring': 'Hungarian_Grand_Prix',
        'Spa-Francorchamps': 'Belgian_Grand_Prix',
        'Zandvoort': 'Dutch_Grand_Prix',
        'Monza': 'Italian_Grand_Prix',
        'Baku': 'Azerbaijan_Grand_Prix',
        'Singapore': 'Singapore_Grand_Prix',
        'Austin': 'United_States_Grand_Prix',
        'Mexico City': 'Mexico_City_Grand_Prix',
        'Interlagos': 'São_Paulo_Grand_Prix',
        'Las Vegas': 'Las_Vegas_Grand_Prix',
        'Yas Marina Circuit': 'Abu_Dhabi_Grand_Prix',
        'Lusail': 'Qatar_Grand_Prix',
    }
    
    # Check for exact match
    if gp_name in mapping:
        return mapping[gp_name]
    
    # Check case-insensitive
    for key, value in mapping.items():
        if key.lower() == gp_name.lower():
            return value
    
    return None


def fetch_openf1_pit_for_years(
    years=(2023, 2024, 2025),               
    max_retries=3,
    base_delay=2,
    progress=True
):
    """
    Pulls OpenF1 pit events for all RACE sessions in the given years.
    No external helpers required.

    Returns
    -------
    pd.DataFrame
        One row per pit event with useful session/meeting metadata attached.
    """
    all_rows = []
    sessions_base = "https://api.openf1.org/v1/sessions"
    pit_base = "https://api.openf1.org/v1/pit"

    for y in years:
        # 1) discover RACE sessions for the year
        sess_url = f"{sessions_base}?year={y}&session_name=Race"
        if progress: 
            print(f"\nYear {y}: fetching race sessions…", end=" ")

        with urlopen(sess_url) as resp:
            sessions = json.loads(resp.read().decode("utf-8"))

        if progress: 
            print(f"{len(sessions)} sessions")

        # 2) for each session, fetch pits (with retries)
        for i, s in enumerate(sessions, 1):
            session_key = s.get("session_key")
            if not session_key:
                continue

            # prefer meeting_name; fall back to circuit short name
            gp_name = s.get("meeting_name") or s.get("circuit_short_name") or "UnknownGP"
            meeting_key = s.get("meeting_key")
            round_no = s.get("meeting_round") or s.get("meeting_official_name")

            url = f"{pit_base}?session_key={session_key}"

            for attempt in range(max_retries):
                try:
                    if progress:
                        print(f"  [{i}/{len(sessions)}] {gp_name} (session_key={session_key}) …", end=" ")

                    with urlopen(url) as resp:
                        data = json.loads(resp.read().decode("utf-8"))

                    # attach metadata
                    for row in data:
                        row["season_year"]     = y
                        row["grand_prix"]      = gp_name
                        row["race_session_key"]= session_key
                        row["meeting_key"]     = meeting_key
                        row["round"]           = round_no
                    all_rows.extend(data)

                    if progress: 
                        print(f"✓ {len(data)} rows")
                    time.sleep(base_delay)  # gentle on API
                    break

                except Exception as e:
                    if hasattr(e, 'code') and e.code == 429 and attempt < max_retries - 1:
                        wait = base_delay * (2 ** attempt)
                        if progress: 
                            print(f"429 rate limit → retry in {wait}s")
                        time.sleep(wait)
                        continue
                    else:
                        if progress: 
                            print(f"Error: {e}")
                        break

    df = pd.DataFrame(all_rows)
    if not df.empty:
        # parse timestamp and order useful columns if present
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        preferred = [
            "season_year","round","grand_prix","race_session_key","meeting_key",
            "date","driver_number","lap_number","pit_duration"
        ]
        df = df[[c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]]

    return df

def merge_pit_stops_to_clean_dataset(df_clean, df_pits=None, use_stint_fallback=True):
    """
    Merge OpenF1 pit stop data into the cleaned dataset.
    Falls back to stint-based detection for races without OpenF1 data.
    
    Parameters:
    -----------
    df_clean : pd.DataFrame
        The cleaned dataset (output from clean_combined_data)
    df_pits : pd.DataFrame, optional
        Pre-loaded pit data. If None, loads from csv_output/openf1_pit_2023_2025.csv
    use_stint_fallback : bool, default True
        Use LapInStint/Stint changes to detect pit stops for races without OpenF1 data
    
    Returns:
    --------
    pd.DataFrame
        Dataset with added 'PitStop' and 'Pit_Out' columns (boolean)
    """
    # Load pit data from CSV if not provided
    if df_pits is None:
        csv_path = "csv_output/openf1_pit_2023_2025.csv"
        print(f"Loading pit data from {csv_path}...")
        df_pits = pd.read_csv(csv_path)
    
    # Map grand_prix names to standard format
    df_pits['grand_prix_mapped'] = df_pits['grand_prix'].apply(map_openf1_gp_to_race_name)
    
    # Remove unmapped entries
    unmapped_count = df_pits['grand_prix_mapped'].isna().sum()
    if unmapped_count > 0:
        print(f"Warning: {unmapped_count} pit stops with unmapped grand prix names")
        df_pits = df_pits[df_pits['grand_prix_mapped'].notna()]
    
    # Rename columns to match cleaned dataset
    df_pits = df_pits.rename(columns={
        'season_year': 'year',
        'driver_number': 'RacingNumber',
        'grand_prix_mapped': 'Grand_Prix'
    })
    
    # Add PitStop indicator
    df_pits['PitStop'] = True
    
    # Keep only merge keys
    df_pits = df_pits[['year', 'Grand_Prix', 'RacingNumber', 'lap_number', 'PitStop']].drop_duplicates()
    
    # Merge with cleaned dataset
    df_merged = pd.merge(
        df_clean,
        df_pits,
        on=['year', 'Grand_Prix', 'RacingNumber', 'lap_number'],
        how='left'
    )
    
    # Store original order columns
    sort_columns = ['year', 'round', 'Grand_Prix', 'RacingNumber', 'lap_number']
    
    # Sort to ensure correct ordering
    df_merged = df_merged.sort_values(sort_columns).reset_index(drop=True)
    
    # Initialize PitStop and Pit_Out columns properly
    # Keep NaN from merge to identify races without OpenF1 data
    has_openf1_data = df_merged['PitStop'].notna()
    
    # Identify races with OpenF1 data vs those needing fallback
    if use_stint_fallback and 'LapInStint' in df_merged.columns and 'Stint' in df_merged.columns:
        # Check which races have ANY OpenF1 pit data
        races_with_data = df_merged[has_openf1_data].groupby(['year', 'round', 'Grand_Prix']).size()
        all_races = df_merged.groupby(['year', 'round', 'Grand_Prix']).size()
        races_without_pits = all_races.index.difference(races_with_data.index)
        
        if len(races_without_pits) > 0:
            print(f"\n {len(races_without_pits)} races without OpenF1 data - using stint-based fallback:")
            
            for year, rnd, gp in races_without_pits:
                race_mask = (df_merged['year'] == year) & \
                           (df_merged['round'] == rnd) & \
                           (df_merged['Grand_Prix'] == gp)
                
                # PitStop: Last lap of stint (when stint changes)
                for (_, _, _, driver), group_idx in df_merged[race_mask].groupby(['year', 'round', 'Grand_Prix', 'RacingNumber']).groups.items():
                    driver_data = df_merged.loc[group_idx]
                    next_stint = driver_data['Stint'].shift(-1)
                    pit_in_mask = (driver_data['Stint'] != next_stint) & (next_stint.notna())
                    
                    # Set PitStop to True for stint changes
                    pit_indices = driver_data.index[pit_in_mask]
                    df_merged.loc[pit_indices, 'PitStop'] = True
                
                # Pit_Out: First lap of new stint (LapInStint == 1, but not lap 1 of race)
                pit_out_indices = df_merged[race_mask & (df_merged['LapInStint'] == 1) & (df_merged['lap_number'] > 1)].index
                df_merged.loc[pit_out_indices, 'Pit_Out'] = True
                
                fallback_pits = int(df_merged.loc[df_merged[race_mask].index, 'PitStop'].fillna(False).sum())
                fallback_outs = len(pit_out_indices)
                print(f"   {year} Round {rnd} - {gp}: {fallback_pits} pit stops, {fallback_outs} pit-outs (stint-based)")
    
    # Now fill remaining NaN with False and convert to bool
    df_merged['PitStop'] = df_merged['PitStop'].fillna(False).astype(bool)
    
    # Initialize Pit_Out if not already set
    if 'Pit_Out' not in df_merged.columns:
        df_merged['Pit_Out'] = False
    else:
        df_merged['Pit_Out'] = df_merged['Pit_Out'].fillna(False)
    
    # For races WITH OpenF1 data: mark pit-out laps (lap after PitStop)
    for (year, rnd, gp, driver), group_idx in df_merged.groupby(['year', 'round', 'Grand_Prix', 'RacingNumber']).groups.items():
        driver_data = df_merged.loc[group_idx]
        
        # Find rows where PitStop = True
        pit_stop_rows = driver_data[driver_data['PitStop'] == True]
        
        for pit_idx in pit_stop_rows.index:
            # Find the next row (same driver, next lap)
            next_rows = df_merged[(df_merged.index > pit_idx) & 
                                   (df_merged['year'] == year) &
                                   (df_merged['round'] == rnd) &
                                   (df_merged['Grand_Prix'] == gp) &
                                   (df_merged['RacingNumber'] == driver)]
            
            if len(next_rows) > 0:
                next_idx = next_rows.index[0]
                df_merged.loc[next_idx, 'Pit_Out'] = True
    
    # Convert Pit_Out to boolean
    df_merged['Pit_Out'] = df_merged['Pit_Out'].astype(bool)
    
    # Maintain original sorting order
    df_merged = df_merged.sort_values(sort_columns).reset_index(drop=True)
    
    # Summary
    pit_stop_count = df_merged['PitStop'].sum()
    pit_out_count = df_merged['Pit_Out'].sum()
    openf1_pit_count = pit_stop_count  # Track original count before fallback
    
    print(f"\n✓ Total pit stops: {pit_stop_count:,}")
    print(f"✓ Total pit-out laps: {pit_out_count:,}")
    
    return df_merged