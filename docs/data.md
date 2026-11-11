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

These are study-specific spatial layers, including building inventories, candidate shelter locations, and flood footprints. The associated manuscript describes the data sources. The archives are not distributed here because the original extraction records, complete attribution details, and redistribution permissions have not been established. Obtain authorized copies from the project authors, retain their original filenames, and confirm the applicable source terms before use or redistribution.

The repository's ignore rules exclude ZIP archives. Keep local input copies out of commits unless their provenance and redistribution terms have been documented and approved for release.

## Simulation and analysis outputs

Run outputs are written under `data_collection/` and are not tracked in Git. A complete replication creates its own run data and manifests; use a new output directory when changing model code, inputs, environment, or design. Do not mix partial runs with a completed cohort.

The tracked files under `analysis/paper_generated/` are derived publication assets. They are useful for inspecting the reported tables and figures, but do not contain all hourly per-run records needed to independently rerun the complete statistical analysis. Full reproduction requires the authorized spatial inputs and execution of the documented experiment design.