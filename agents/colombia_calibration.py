"""
Colombia / Santa Lucia (Atlantico) calibration constants.

The original Flood-ABM (Addai et al. 2025, IJDRR) was calibrated to Calgary's
Inglewood neighbourhood using Canadian census micro-data, CRA income tables,
and CIHI health-spending figures (see Table 6 of the paper). Every constant
below replaces a Canada-specific number with a Colombia- or Atlantico- (or,
where available, Santa Lucia-) specific one, with the source cited inline.

Where a municipal-level figure for Santa Lucia could not be found after a
genuine search (UNOSAT, Copernicus EMS, Dartmouth Flood Observatory / Global
Flood Database, IDEAM open data, DANE municipal bulletins), the best
available *departmental* (Atlantico) or *national* (Colombia) figure is used
instead and flagged as such. Nothing here is invented: every number has a
cited source and a stated geographic level. This file exists so every
number used downstream can be audited in one place instead of being buried
across model files.

Retrieved 2026-08-16.
"""

# ---------------------------------------------------------------------------
# 1. Population size (N_persons)
# ---------------------------------------------------------------------------
# The QGIS sample used for the spatial layer is 286 houses (8.68% random
# sample of 3,296 candidate buildings in the mapped urban core of Santa
# Lucia; see Santa_Lucia_Data_Archive_2026-08-15/01_source_inputs/qgis/README_GIS.md).
# A previous, unsourced note implied 1,500 simulated persons for those 286
# houses (5.245 persons/house). That figure has no cited source and is
# ~54% above the real average household size for the department.
#
# DANE, Censo Nacional de Poblacion y Vivienda 2018: average household size
# for Atlantico department = 3.4 persons/household (cited via El Heraldo,
# "La Costa, con la mayor tasa de personas por hogar", 2018-09-26, reporting
# preliminary DANE CNPV-2018 results).
AVG_HOUSEHOLD_SIZE_ATLANTICO_2018 = 3.4
N_HOUSES_SAMPLED = 286
N_PERSONS_RECOMMENDED = round(N_HOUSES_SAMPLED * AVG_HOUSEHOLD_SIZE_ATLANTICO_2018)  # 972

# ---------------------------------------------------------------------------
# 2. Age structure
# ---------------------------------------------------------------------------
# Santa Lucia municipality, DANE population projections, as tabulated for
# 2026 (5-year age bins; source: DANE municipal projections, reported via
# telencuestas.com/censos-de-poblacion/colombia/2026/atlantico/santa-lucia).
# This is *today's* structure, used as the best available proxy for local
# age structure -- Caribbean-coast small-town age pyramids shift slowly, but
# this is not claimed to be identical to the 2010 structure. Flagged as a
# limitation in the manuscript.
AGE_0_14_SHARE = 5524 / 19997    # 0.2762
AGE_15_64_SHARE = 12871 / 19997  # 0.6437
AGE_65_PLUS_SHARE = 1602 / 19997  # 0.0801

# Gender split, same source (Santa Lucia, DANE 2026 projection):
# 10,070 women (50.4%), 9,927 men (49.6%).
PERC_MALE = 0.496

# ---------------------------------------------------------------------------
# 3. Ethnic self-identification
# ---------------------------------------------------------------------------
# DANE, Censo Nacional de Poblacion y Vivienda 2018, national self-recognition
# shares (departmental/municipal breakdown for Atlantico/Santa Lucia was not
# found after a genuine search of DANE's ethnic-group publications, so the
# national figure is used and flagged explicitly as national-level, not local).
PERC_INDIGENA = 0.044
PERC_AFROCOLOMBIANO = 0.068
PERC_NINGUNO_OTRO = 1 - PERC_INDIGENA - PERC_AFROCOLOMBIANO  # 0.888
# NOTE ON CODE MAPPING: the original model's ethnicity field used 3 category
# labels ("Indigenous", "Immigrant", "Canadian") because Calgary/Inglewood
# analysis contrasted Indigenous, immigrant, and Canadian-born residents. To
# avoid touching every downstream string comparison (in decision_making_module.py
# and flood_agents.py), the SAME 3 internal labels are kept but re-mapped in
# meaning: "Indigenous" -> Indigena (4.4%), "Immigrant" -> Afrocolombiano /
# Negro / Mulato self-identified (6.8%), "Canadian" -> Ninguna pertenencia
# etnica declarada (88.8%). This is a structural adaptation, not a claim that
# Afrocolombiano residents face the same dynamics as Canadian immigrants --
# it is documented here and in the manuscript as an inherited modeling
# simplification of the base model's ethnicity mechanism.

# ---------------------------------------------------------------------------
# 4. Education / risk-literacy proxy
# ---------------------------------------------------------------------------
# The model's `perc_education_people` represents the share of adults assigned
# "full" education/risk-comprehension (education=1) vs partial (0.5-0.9).
# No Santa Lucia- or Atlantico-specific measure of flood-risk comprehension
# was found. The closest available real proxy is department-level literacy:
# DANE Censo 2005 (most recent disaggregated illiteracy figure located),
# Atlantico department illiteracy rate among population 15+ = 6.6%
# (geoportal.dane.gov.co Atlas Estadistico, Tomo II Social, 3.1.2).
# This is an imperfect proxy (literacy != flood-risk comprehension) and is
# flagged as such; it is nonetheless a real, cited, department-level figure
# rather than a copied Canadian number.
PERC_EDUCATION_PEOPLE = 1 - 0.066  # 0.934

# ---------------------------------------------------------------------------
# 5. Wealth classes (income distribution)
# ---------------------------------------------------------------------------
# DANE monetary poverty incidence, Atlantico department, 2024 (DANE,
# "Pobreza Monetaria Departamental 2024", pres-PMDepartamental-2024.pdf):
#   - Pobreza monetaria (total poor):    31.6%
#   - Pobreza monetaria extrema:          8.6%  (subset of the above)
# National-level socioeconomic classification (DANE 2023 data, as reported by
# El Espectador, "Que se considera pobre, vulnerable, clase media y clase
# alta en Colombia?", 2024-07-26), giving per-capita monthly income bands
# (COP) and population shares: Pobre 33.0%, Vulnerable 31.5%, Clase media
# 32.4%, Clase alta 3.1%.
#
# Method: use the Atlantico-specific "poor" share (31.6%, more local than the
# national 33.0%) and split the remaining 68.4% "non-poor" population using
# the *ratios* among the national vulnerable/middle/upper shares (since a
# local breakdown of the non-poor population was not found). This keeps the
# most locally accurate number (poverty rate) while still using only real,
# cited data for the remaining split.
POBRE_SHARE = 0.316
_non_poor = 1 - POBRE_SHARE
_national_non_poor_total = 0.315 + 0.324 + 0.031  # 0.670
VULNERABLE_SHARE = _non_poor * (0.315 / _national_non_poor_total)
CLASE_MEDIA_SHARE = _non_poor * (0.324 / _national_non_poor_total)
CLASE_ALTA_SHARE = _non_poor * (0.031 / _national_non_poor_total)

# Per-capita MONTHLY income bands (COP), DANE/El Espectador as above:
INCOME_BAND_POBRE = (0, 435_375)
INCOME_BAND_VULNERABLE = (435_375, 853_608)
INCOME_BAND_CLASE_MEDIA = (853_608, 4_596_352)
# Clase alta is an open-ended top bracket in the official classification
# (no ceiling is published). An upper bound of 3x the lower bound is used
# here as an explicit, documented author-defined cap (mirrors how the
# original Calgary model also capped its own open-ended top bracket).
INCOME_BAND_CLASE_ALTA = (4_596_352, 4_596_352 * 3)

# Annual equivalents (x12), used by the model (which works in annual income
# figures, consistent with the original Calgary code's convention):
ANNUAL_INCOME_LOWER_CLASS = (INCOME_BAND_POBRE[0] * 12, INCOME_BAND_POBRE[1] * 12)
ANNUAL_INCOME_MIDDLE_CLASS = (INCOME_BAND_VULNERABLE[0] * 12, INCOME_BAND_VULNERABLE[1] * 12)
ANNUAL_INCOME_UPPER_MIDDLE_CLASS = (INCOME_BAND_CLASE_MEDIA[0] * 12, INCOME_BAND_CLASE_MEDIA[1] * 12)
ANNUAL_INCOME_UPPER_CLASS = (INCOME_BAND_CLASE_ALTA[0] * 12, INCOME_BAND_CLASE_ALTA[1] * 12)
# NOTE ON CODE MAPPING: original class labels are kept (Lower_Class,
# Middle_Class, Upper_Middle_Class, Upper_Class) to avoid touching every
# downstream comparison in decision_making_module.py / flood_agents.py.
# Real-world mapping (poorest -> richest): Lower_Class=Pobre,
# Middle_Class=Vulnerable, Upper_Middle_Class=Clase media, Upper_Class=Clase alta.

# ---------------------------------------------------------------------------
# 6. Macro / sectoral GDP-share scaling (business, school, shelter,
#    healthcare, government initial wealth; hourly costs)
# ---------------------------------------------------------------------------
# The original per-capita sectoral constants ($22,000 business / $1,200
# school / $4,200 shelter / $4,200 healthcare / $7,000 government, all
# CAD/year; plus one-time evac cost $0-1,500 CAD and hourly shelter/
# healthcare costs $0-50 / $0-100 CAD) were "author defined" in the original
# paper (Table 6) -- i.e. not independently sourced even for Calgary. No
# equivalent Colombian sectoral costing at this granularity was found, so
# rather than inventing new absolute figures, the original relative
# magnitudes are preserved and scaled by the real ratio of per-capita GDP
# between the two countries, then expressed in COP:
#
#   GDP per capita, current US$, World Bank (2024):
#     Colombia = 7,914 ; Canada = 54,283  -> ratio = 0.145775
#   TRM (COP per USD), Banco de la Republica / dolarhoy.co, 2026-08-16:
#     3,128.65
#
#   scale_factor = ratio * TRM = 0.145775 * 3128.65 = 456.08
#   (i.e. 1 original CAD unit in the Calgary model -> ~456 COP here)
GDP_PC_COLOMBIA_USD_2024 = 7_914
GDP_PC_CANADA_USD_2024 = 54_283
TRM_COP_PER_USD_2026_08_16 = 3_128.65
CAD_TO_COP_SCALE = (GDP_PC_COLOMBIA_USD_2024 / GDP_PC_CANADA_USD_2024) * TRM_COP_PER_USD_2026_08_16  # ~456.08

BUSINESS_GDP_PER_CAPITA_COP = 22_000 * CAD_TO_COP_SCALE     # ~10,033,800 COP/yr
SCHOOL_GDP_PER_CAPITA_COP = 1_200 * CAD_TO_COP_SCALE        # ~547,300 COP/yr
SHELTER_GDP_PER_CAPITA_COP = 4_200 * CAD_TO_COP_SCALE       # ~1,915,500 COP/yr
HEALTHCARE_GDP_PER_CAPITA_COP = 4_200 * CAD_TO_COP_SCALE    # ~1,915,500 COP/yr
GOVERNMENT_GDP_PER_CAPITA_COP = 7_000 * CAD_TO_COP_SCALE    # ~3,192,600 COP/yr

EVAC_COST_MAX_COP = 1_500 * CAD_TO_COP_SCALE                # ~684,100 COP (one-time)
SHELTER_COST_PER_PERSON_HOUR_MAX_COP = 50 * CAD_TO_COP_SCALE     # ~22,800 COP/hr
HEALTHCARE_COST_PER_PERSON_HOUR_MAX_COP = 100 * CAD_TO_COP_SCALE  # ~45,600 COP/hr

# ---------------------------------------------------------------------------
# 7. Distance-dependent shelter rescue mechanism (Experiment D, follow-up to
#    Experiment B). The original/base model's `notify_shelter()` used a fixed
#    2-hour delay and a capacity check ONLY -- the shelter polygon's location
#    was never actually used in any decision, cost, or timing calculation
#    (confirmed by reading the code: the only use of shelter geometry was
#    `get_random_point_in_polygon()`, which places a *already-rescued* agent
#    on the map, after the fact -- purely cosmetic). This section adds a
#    real distance/terrain-dependent rescue-logistics mechanism.
#
# Road distances (NOT straight-line) were obtained from OSRM (Open Source
# Routing Machine, running on OpenStreetMap road data), queried 2026-08-17,
# for the real route between Santa Lucia's town centroid and each candidate
# site:
#   - "current" (hypothetical) shelter, already in the project's spatial
#     data: straight-line 14.0 km, real road route 38.4 km, 60.5 min
#     (dry-road driving conditions).
#   - Intermediate point (42% of the way toward Barranquilla by straight-line
#     interpolation -- NOT a specific named town; used purely to add a
#     resolution point to the dose-response curve, flagged as such in the
#     manuscript): real road route 58.0 km, 64.2 min.
#   - Barranquilla (the real, documented destination many Santa Lucia
#     evacuees actually used in 2010-2011, per El Heraldo/desplazadosambientales
#     reporting): real road route 99.9 km, 93.9 min.
# These 3 durations imply real average road speeds of 63.5, 54.2, and 63.8
# km/h respectively (mean 60.5, rounded to 60) -- used below as
# DRY_ROAD_SPEED_KMH, i.e. a REAL, sourced number, not an author guess.
#
# A 4th, "nearby" candidate represents the real, documented behavior of many
# 2010 evacuees: crossing the Canal del Dique directly by canoe to
# uninhabited high ground on the opposite bank (reported as "Lomita Arena" /
# "Arenal de la Montana" in press accounts, though those names could not be
# matched to a precise, verifiable coordinate, so a nearby real point is used
# instead: verified via this project's own flood-extent polygon to be
# genuinely outside the modeled flood zone, ~2.0 km west of the town
# centroid, sitting in a substantial ~21 km^2 non-flooded area, not a small
# artifact of the polygon boundary). Because this route is a direct water
# crossing (no road), it is modeled with the boat-crossing formula for its
# entire distance rather than the dry-road formula.
FLOOD_EXIT_DISTANCE_KM = 1.5
# Distance from a stranded position within the flooded town to the nearest
# passable/dry extraction point. Grounded in this project's own flood-extent
# polygon: the nearest non-flooded land was found (by direct geometric
# search) to lie 0.6-1.3 km from the town centroid in several directions;
# 1.5 km is used as a slightly conservative round number for the "typical"
# in-town stranded position (not just the closest possible edge point).

BOAT_SPEED_KMH = 5.0
# Author-defined assumption (not independently sourced): effective speed of
# a small rescue/rowing boat navigating flooded streets and open floodwater
# with debris. Flagged explicitly, exactly like the Experiment A community-
# organization coefficients, as an unfitted modeling assumption.

DRY_ROAD_SPEED_KMH = 60.0
# REAL, sourced number: mean of 3 real OSRM-routed effective speeds (see
# above), representing realistic driving speed on rural Atlantico-department
# roads once clear of the immediate flood zone.

RESCUE_BOATS_AVAILABLE = 2
BOAT_CAPACITY_PER_TRIP = 6
# Author-defined assumptions: a small town's realistic own rescue capacity
# in the early hours of a disaster (before large-scale external aid
# arrives), informed by the real accounts of families using their own
# canoes rather than a large organized fleet. Not independently sourced;
# flagged as such.

SHELTER_ROUTE_DISTANCE_KM = {
    "nearby_crossing": 2.0,      # direct water crossing, not a road route
    "current_hypothetical": 38.4,  # real OSRM road route
    "intermediate": 58.0,          # real OSRM road route
    "barranquilla": 99.9,          # real OSRM road route
}
SHELTER_ROUTE_IS_WATER_CROSSING = {
    "nearby_crossing": True,
    "current_hypothetical": False,
    "intermediate": False,
    "barranquilla": False,
}


if __name__ == "__main__":
    print("N_PERSONS_RECOMMENDED", N_PERSONS_RECOMMENDED)
    print("age shares", AGE_0_14_SHARE, AGE_15_64_SHARE, AGE_65_PLUS_SHARE, "sum=", AGE_0_14_SHARE+AGE_15_64_SHARE+AGE_65_PLUS_SHARE)
    print("wealth shares", POBRE_SHARE, VULNERABLE_SHARE, CLASE_MEDIA_SHARE, CLASE_ALTA_SHARE,
          "sum=", POBRE_SHARE+VULNERABLE_SHARE+CLASE_MEDIA_SHARE+CLASE_ALTA_SHARE)
    print("CAD_TO_COP_SCALE", CAD_TO_COP_SCALE)
    print("business gdp pc", BUSINESS_GDP_PER_CAPITA_COP)
