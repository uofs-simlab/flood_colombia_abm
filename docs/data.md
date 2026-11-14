# Spatial inputs and study outputs

## Spatial inputs

The model expects these twelve ZIP archives in the repository's `santa_lucia_map_data/` directory:

```text
santa_lucia_map_data/businesses.zip
santa_lucia_map_data/flood1.zip
santa_lucia_map_data/flood2.zip
santa_lucia_map_data/flood3.zip
santa_lucia_map_data/government.zip
santa_lucia_map_data/healthcare.zip
santa_lucia_map_data/houses.zip
santa_lucia_map_data/schools.zip
santa_lucia_map_data/shelter.zip
santa_lucia_map_data/shelter_distance_candidates/shelter_barranquilla.zip
santa_lucia_map_data/shelter_distance_candidates/shelter_intermediate.zip
santa_lucia_map_data/shelter_distance_candidates/shelter_nearby_crossing.zip
```

These study-specific layers include building inventories, facility locations, candidate shelters, and flood footprints. The [spatial-data source note](spatial-data-sources.md) records the source and attribution associated with each layer. Download the input bundle from [GitHub Releases](https://github.com/uofs-simlab/flood_colombia_abm/releases) and extract it so that all twelve archives retain the paths listed above.

## Simulation and analysis outputs

Run outputs are written under `data_collection/` and are not tracked in Git. A complete replication creates its own run data and manifests; use a new output directory when changing model code, inputs, environment, or design. Do not mix partial runs with a completed cohort.

The tracked files under `analysis/paper_generated/` are derived publication assets. They are useful for inspecting the reported tables and figures, but do not contain all hourly per-run records needed to independently rerun the complete statistical analysis. Full reproduction requires the authorized spatial inputs and execution of the documented experiment design.