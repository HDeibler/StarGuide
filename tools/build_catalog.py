#!/usr/bin/env python3
"""Build the bundled star / name / constellation data from public sources.

Run once to (re)generate `starguide/data/*.json`:

    python tools/build_catalog.py

Sources (downloaded to a cache dir, then parsed — see data/README.md for licenses):
  • HYG database v4 (astronexus/HYG-Database) — Hipparcos stars, RA/Dec/mag,
    proper names, constellation membership.
  • Stellarium "western" constellationship.fab — 88 constellation stick figures
    as Hipparcos star-id pairs.

Output (all HIP-keyed, matching the loaders in starguide/astro.py):
  hip_catalog.json        {hip: {ra, dec, mag}}          stars to MAG_LIMIT
  hip_names.json          {hip: proper_name}             ~450 named stars
  constellation_data.json {Constellation: [[hipA, hipB], ...]}   88 figures
"""

import csv
import json
import os
import urllib.request

MAG_LIMIT = 7.5          # naked-eye is ~6.5; go deeper so a rich overlay has stars

HYG_URL = ("https://raw.githubusercontent.com/astronexus/HYG-Database/main/"
           "hyg/CURRENT/hygdata_v41.csv")
CSHIP_URL = ("https://raw.githubusercontent.com/Stellarium/stellarium/v0.20.0/"
             "skycultures/western/constellationship.fab")

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, ".cache")
DATA = os.path.join(os.path.dirname(HERE), "starguide", "data")

# IAU 3-letter abbreviation -> constellation name (matched case-insensitively).
ABBR = {
    "and": "Andromeda", "ant": "Antlia", "aps": "Apus", "aqr": "Aquarius",
    "aql": "Aquila", "ara": "Ara", "ari": "Aries", "aur": "Auriga",
    "boo": "Boötes", "cae": "Caelum", "cam": "Camelopardalis", "cnc": "Cancer",
    "cvn": "Canes Venatici", "cma": "Canis Major", "cmi": "Canis Minor",
    "cap": "Capricornus", "car": "Carina", "cas": "Cassiopeia",
    "cen": "Centaurus", "cep": "Cepheus", "cet": "Cetus", "cha": "Chamaeleon",
    "cir": "Circinus", "col": "Columba", "com": "Coma Berenices",
    "cra": "Corona Australis", "crb": "Corona Borealis", "crv": "Corvus",
    "crt": "Crater", "cru": "Crux", "cyg": "Cygnus", "del": "Delphinus",
    "dor": "Dorado", "dra": "Draco", "equ": "Equuleus", "eri": "Eridanus",
    "for": "Fornax", "gem": "Gemini", "gru": "Grus", "her": "Hercules",
    "hor": "Horologium", "hya": "Hydra", "hyi": "Hydrus", "ind": "Indus",
    "lac": "Lacerta", "leo": "Leo", "lmi": "Leo Minor", "lep": "Lepus",
    "lib": "Libra", "lup": "Lupus", "lyn": "Lynx", "lyr": "Lyra", "men": "Mensa",
    "mic": "Microscopium", "mon": "Monoceros", "mus": "Musca", "nor": "Norma",
    "oct": "Octans", "oph": "Ophiuchus", "ori": "Orion", "pav": "Pavo",
    "peg": "Pegasus", "per": "Perseus", "phe": "Phoenix", "pic": "Pictor",
    "psc": "Pisces", "psa": "Piscis Austrinus", "pup": "Puppis", "pyx": "Pyxis",
    "ret": "Reticulum", "sge": "Sagitta", "sgr": "Sagittarius", "sco": "Scorpius",
    "scl": "Sculptor", "sct": "Scutum", "ser": "Serpens", "sex": "Sextans",
    "tau": "Taurus", "tel": "Telescopium", "tri": "Triangulum",
    "tra": "Triangulum Australe", "tuc": "Tucana", "uma": "Ursa Major",
    "umi": "Ursa Minor", "vel": "Vela", "vir": "Virgo", "vol": "Volans",
    "vul": "Vulpecula",
}


def _fetch(url, name):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, name)
    if not os.path.exists(path):
        print(f"  downloading {name} …")
        urllib.request.urlretrieve(url, path)
    return path


def _finite(*xs):
    try:
        return all(x == x and abs(float(x)) != float("inf") for x in xs)
    except (TypeError, ValueError):
        return False


def main():
    hyg = _fetch(HYG_URL, "hygdata.csv")
    cship = _fetch(CSHIP_URL, "constellationship.fab")

    # --- stars + names from HYG (RA is in HOURS -> *15 for degrees) -----------
    catalog, names, by_hip = {}, {}, {}
    with open(hyg, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r["hip"]:
                continue
            hip = int(r["hip"])
            try:
                ra, dec, mag = float(r["ra"]) * 15.0, float(r["dec"]), float(r["mag"])
            except ValueError:
                continue
            if not _finite(ra, dec, mag):
                continue
            # keep the brightest component per HIP
            if hip in by_hip and by_hip[hip] <= mag:
                pass
            else:
                by_hip[hip] = mag
                catalog[str(hip)] = {"ra": round(ra, 4), "dec": round(dec, 4),
                                     "mag": round(mag, 2)}
            if r["proper"].strip():
                names[str(hip)] = r["proper"].strip()

    # --- 88 constellation figures from Stellarium (HIP pairs) -----------------
    constellations = {}
    missing = set()
    with open(cship, encoding="utf-8") as f:
        for line in f:
            tok = line.split()
            if len(tok) < 3:
                continue
            name = ABBR.get(tok[0].lower())
            if not name:
                continue
            ids = [int(x) for x in tok[2:]]
            pairs = [[ids[i], ids[i + 1]] for i in range(0, len(ids) - 1, 2)]
            constellations[name] = pairs
            for h in ids:
                if str(h) not in catalog:
                    missing.add(h)

    # constellation stars fainter than MAG_LIMIT must still exist so lines draw;
    # pull them straight from HYG regardless of magnitude.
    if missing:
        with open(hyg, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["hip"] and int(r["hip"]) in missing:
                    try:
                        ra, dec = float(r["ra"]) * 15.0, float(r["dec"])
                        mag = float(r["mag"])
                    except ValueError:
                        continue
                    if _finite(ra, dec, mag):
                        catalog[r["hip"]] = {"ra": round(ra, 4),
                                             "dec": round(dec, 4),
                                             "mag": round(mag, 2)}

    keep = {h: c for h, c in catalog.items()
            if c["mag"] <= MAG_LIMIT or h in {str(i) for i in
                                              sum(([a, b] for p in
                                                   constellations.values()
                                                   for a, b in p), [])}}

    os.makedirs(DATA, exist_ok=True)
    for fname, obj in [("hip_catalog.json", keep),
                       ("hip_names.json", names),
                       ("constellation_data.json", constellations)]:
        with open(os.path.join(DATA, fname), "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
    print(f"stars: {len(keep)}  names: {len(names)}  "
          f"constellations: {len(constellations)}")


if __name__ == "__main__":
    main()
