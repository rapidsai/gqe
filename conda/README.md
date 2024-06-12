This directory contains yml files to construct conda environments to satisfy GQE dependencies.

**environment.yml**

This file installs all dependencies for GQE from conda. This is convenient but the libcudf installed is not compiled with per-thread default stream, which could hurt overlapping efficiency.

**docker-(arch).yml**

This file contains dependencies for libcudf and GQE. It is used for the Docker container to set up the environment to compile libcudf from scratch.
