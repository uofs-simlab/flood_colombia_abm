"""The prespecified 24-setting, common-seed flood-response study."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agents import colombia_calibration as CAL

MAPS = ROOT / "santa_lucia_map_data"
FIXED = {
    "N_persons": 972, "shelter_cap_limit": 1, "healthcare_cap_limit": 5,
    "shelter_funding": 50000, "healthcare_funding": 100000,
    "pre_flood_days": 7, "flood_days": 5, "post_flood_days": 5,
    **{f"{kind}_file": str(MAPS / f"{kind}.zip") for kind in
       ["houses", "businesses", "schools", "shelter", "healthcare", "government"]},
    **{f"flood_file_{i}": str(MAPS / f"flood{i}.zip") for i in [1, 2, 3]},
    "model_crs": "EPSG:32618",
}
SITES = ["nearby_crossing", "current_hypothetical", "intermediate", "barranquilla"]
SITE_FILES = {
    site: str(MAPS / "shelter_distance_candidates" / f"shelter_{site}.zip")
    for site in SITES if site != "current_hypothetical"
}
SITE_FILES["current_hypothetical"] = str(MAPS / "shelter.zip")
SCENARIOS = {}
for level in [0, .25, .5, .75, 1]:
    name = f"A_org_{level:g}"
    SCENARIOS[name] = {
        "family": "organization", "label": f"Organization {level:g}",
        "parameters": {"community_organization_level": level},
    }
for label, params in [
    ("depth_only", {"org_sct_channel_on": False}),
    ("social_only", {"org_depth_channel_on": False}),
    ("weak_depth", {"org_depth_attenuation_coef": .1}),
]:
    SCENARIOS[f"A_{label}"] = {
        "family": "ablation", "label": label,
        "parameters": {"community_organization_level": 1, **params},
    }
for capacity in [1, 10, 30]:
    for site in SITES:
        SCENARIOS[f"D_cap{capacity}_{site}"] = {
            "family": "distance", "label": f"{site}, {capacity}% capacity",
            "parameters": {
                "shelter_cap_limit": capacity, "shelter_site_name": site,
                "shelter_file": SITE_FILES[site],
                "shelter_route_distance_km": CAL.SHELTER_ROUTE_DISTANCE_KM[site],
                "shelter_route_is_water_crossing": CAL.SHELTER_ROUTE_IS_WATER_CROSSING[site],
                "rescue_boats_available": 2,
            },
        }
for boats in [1, 5]:
    for site in ["nearby_crossing", "barranquilla"]:
        SCENARIOS[f"D_fleet{boats}_{site}"] = {
            "family": "fleet", "label": f"{site}, {boats} boats",
            "parameters": {
                "shelter_cap_limit": 10, "shelter_site_name": site,
                "shelter_file": SITE_FILES[site],
                "shelter_route_distance_km": CAL.SHELTER_ROUTE_DISTANCE_KM[site],
                "shelter_route_is_water_crossing": CAL.SHELTER_ROUTE_IS_WATER_CROSSING[site],
                "rescue_boats_available": boats,
            },
        }


def scenario_parameters(name):
    return {**FIXED, **SCENARIOS[name]["parameters"]}


def seeds(replications, seed_base):
    if (isinstance(replications, bool) or not isinstance(replications, int)
            or replications < 2):
        raise ValueError("The study requires at least two seed replications.")
    if (isinstance(seed_base, bool) or not isinstance(seed_base, int)
            or seed_base < 0 or seed_base + replications > 2**32):
        raise ValueError("Study seeds must be integers in [0, 2**32).")
    return list(range(seed_base, seed_base + replications))
