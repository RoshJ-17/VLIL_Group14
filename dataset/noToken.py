import requests
from bs4 import BeautifulSoup
import os
from datetime import datetime, timedelta
from pyhdf.SD import SD, SDC
import numpy as np
import time

# ---------------- CONFIG ----------------
SAVE_MOD06 = "data/MOD06"
SAVE_MOD21 = "data/MOD021KM"

os.makedirs(SAVE_MOD06, exist_ok=True)
os.makedirs(SAVE_MOD21, exist_ok=True)

# ---------------- LIST FILES ----------------
def list_files(product, year, doy):
    url = f"https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/{product}/{year}/{doy}/"
    r = requests.get(url, timeout=20)  # .netrc handles auth
    if r.status_code != 200:
        print("❌ Failed:", url)
        return []
    soup = BeautifulSoup(r.text, "html.parser")
    files = [url + link.get("href") for link in soup.find_all("a") if link.get("href", "").endswith(".hdf")]
    print(f"{product} {year}/{doy}: {len(files)}")
    return files

# ---------------- TIME MATCH ----------------
def extract_time(url):
    name = url.split("/")[-1]
    parts = name.split(".")
    return parts[1][1:] + "_" + parts[2]

def build_dict(urls):
    return {extract_time(u): u for u in urls}

def match_files(mod06, mod21):
    d06 = build_dict(mod06)
    d21 = build_dict(mod21)
    common = sorted(set(d06.keys()) & set(d21.keys()))
    print("✅ Matched pairs:", len(common))
    return [(d06[t], d21[t]) for t in common]

# ---------------- TIME FILTER ----------------
def is_india_pass(url):
    name = url.split("/")[-1]
    parts = name.split(".")
    hour = int(parts[2][:2])
    return (3 <= hour <= 8) or (15 <= hour <= 20)

# ---------------- INDIA COVERAGE CHECK ----------------
def covers_india(file_path):
    try:
        hdf = SD(file_path, SDC.READ)
        lat = hdf.select('Latitude').get()
        lon = hdf.select('Longitude').get()
        return (
            np.nanmax(lat) >= 5 and np.nanmin(lat) <= 35 and
            np.nanmax(lon) >= 65 and np.nanmin(lon) <= 100
        )
    except:
        return False

# ---------------- SMART DOWNLOAD ----------------
def smart_download(url, save_dir):
    name = url.split("/")[-1]
    final_path = os.path.join(save_dir, name)
    if os.path.exists(final_path):
        print("✔ Exists:", name)
        return True
    temp_path = final_path + ".tmp"
    try:
        r = requests.get(url, stream=True, timeout=60)  # .netrc handles auth
        if r.status_code != 200:
            print("❌ Failed:", name)
            return False
        with open(temp_path, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        if covers_india(temp_path):
            os.rename(temp_path, final_path)
            print("✅ Saved (India):", name)
            return True
        else:
            os.remove(temp_path)
            print("❌ Skipped (not India):", name)
            return False
    except Exception as e:
        print("❌ Error:", name, e)
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False

# ---------------- DATE LOOP ----------------
def daterange(start_date, end_date):
    for n in range((end_date - start_date).days + 1):
        yield start_date + timedelta(n)

# ---------------- MAIN ----------------
start = datetime(2023, 5, 27)
end   = datetime(2023, 6, 30)

for date in daterange(start, end):
    year = date.strftime("%Y")
    doy  = date.strftime("%j")
    print(f"\n📅 Processing {year}-{doy}")

    mod06 = list_files("MOD06_L2", year, doy)
    mod21 = list_files("MOD021KM", year, doy)

    pairs = match_files(mod06, mod21)

    # -------- TIME FILTER --------
    filtered_pairs = [(u06, u21) for u06, u21 in pairs if is_india_pass(u06)]
    print("After time filter:", len(filtered_pairs))

    # -------- LIMIT FOR TEST --------
    filtered_pairs = filtered_pairs[:10]

    # -------- DOWNLOAD --------
    for u06, u21 in filtered_pairs:
        ok1 = smart_download(u06, SAVE_MOD06)
        ok2 = smart_download(u21, SAVE_MOD21)
        if not (ok1 and ok2):
            print("⚠️ Pair incomplete")
        time.sleep(1)