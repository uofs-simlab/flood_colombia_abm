"""Organization sweep using the shared, seeded study cohort."""

from study_design import FIXED

if __name__ == "__main__":
    from run_study import main
    main(default_family="organization")
