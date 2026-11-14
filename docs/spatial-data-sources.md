# Spatial input sources and attribution

The study uses twelve prepared spatial archives. QGIS 3.44.11 was used for
spatial preparation and selection. The layers are projected in EPSG:32618.
The source indicators below come from the layer attributes and the
study's calibration record. The study authors approved public release of
these prepared layers with source-specific attribution.

| Input | Recorded source and processing |
|---|---|
| `houses.zip` | 286 residential building geometries selected by QGIS from 3,296 candidate buildings (8.68%). The layer attributes identify the building source as `geofabrik_buildings` and retain OpenStreetMap IDs. |
| `businesses.zip`, `schools.zip`, `healthcare.zip` | Facility geometries associated with Geofabrik/OpenStreetMap building features. The attributes record manual Google Maps checks for facility names or classifications. |
| `government.zip` | Municipal government facility identified during manual Google Maps review; the layer also records an OpenStreetMap feature ID. |
| `flood1.zip`, `flood2.zip`, `flood3.zip` | The study authors identify the common flood footprint as derived from the Global Flood Database (GFD). The three files contain the same polygon. The GFD event identifier is not recorded in the layer metadata. |
| `shelter.zip` | Hypothetical external shelter created for the model scenario. |
| `shelter_distance_candidates/shelter_nearby_crossing.zip` | Model candidate for a 2.0 km water crossing. |
| `shelter_distance_candidates/shelter_intermediate.zip` | Model-defined intermediate candidate. |
| `shelter_distance_candidates/shelter_barranquilla.zip` | Barranquilla candidate used in the long-distance scenario. |

The calibration record documents OSRM route queries over OpenStreetMap
road data on 17 August 2026. Recorded routes are 38.4 km / 60.5 min,
58.0 km / 64.2 min, and 99.9 km / 93.9 min for the hypothetical,
intermediate, and Barranquilla candidates. The original request-response
files and OSM extract date are not part of the spatial ZIPs. The reported
route distances and durations are retained in the model inputs.

The public release copies omit the `path` attribute, which records a
local workstation path. The simulation code does not read this metadata
field; geometries and model-used attributes are preserved.

## Attribution and terms

- Building and road data derived from OpenStreetMap are attributed to
  OpenStreetMap contributors and made available under the Open Database
  License (ODbL) 1.0:
  <https://www.openstreetmap.org/copyright>.
- Geofabrik distributes OpenStreetMap extracts:
  <https://www.geofabrik.de/data/download.html>.
- The GFD is cited to Tellman et al. (2021), *Nature* 596, 80--86,
  <https://doi.org/10.1038/s41586-021-03695-w>. The Earth Engine catalog
  lists the database under CC BY-NC 4.0:
  <https://developers.google.com/earth-engine/datasets/catalog/GLOBAL_FLOOD_DB_MODIS_EVENTS_V1>.
- Google Maps was used for manual facility checks in the layers identified
  above. Google Maps attribution and terms are at
  <https://www.google.com/help/terms_maps/>.
- QGIS is the open-source GIS software used to prepare the spatial layers:
  <https://qgis.org/>.
- OSRM is the routing software used to calculate the road-route values:
  <https://project-osrm.org/>.

The MIT license in the repository applies to study code. The source terms
above apply to spatial data and their attribution.
