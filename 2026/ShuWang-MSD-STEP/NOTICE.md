# Notice

- **STEP code.** All STEP code in this repository was written by the authors.
- **JT code.** JT and its variants are the authors' own implementation of a filter based on the published description of JointTracker (Taetz et al., 2025). The JointTracker source code is not publicly available, so no part of it is included or used. Results from JT and its variants are not results of the original JointTracker implementation.
- **Experimental data.** The data in `section3_experiment/data/` were recorded by the authors at the University of Wisconsin-Madison.
  - IMU records: two WitMotion WT9011DCL IMUs. Accelerometer and gyroscope channels only; magnetometer channels and hardware identifiers are not included.
  - Optical joint angles: tracked from ArUco markers at 240 frames per second. Camera images are not included.
- **Third-party packages.** The Python packages listed in `requirements.txt` are used under their own licenses and are not redistributed here.
- **File names.** Some scripts and result files use the internal names `JT-BHS` (N-body scripts) and `A' + OC` (`Aprime_oc`, three-link scripts) for the variant called `JT-BHSF` in the paper. They are the same filter.
