#!/usr/bin/env bash
# The 14 volcanoes missing from the 250-site run, as batches b11..b17 (2 sites each, one machine each,
# hosts in sites/batches/hosts.csv). Same pipeline as b01..b10: this only calls launch_all.sh with
# these batches, so the finished b01..b10 are not touched.
#
#   bash scripts/launch_missing.sh start     # start b11..b17
#   bash scripts/launch_missing.sh status
#   bash scripts/launch_missing.sh stop
exec bash "$(dirname "$0")/launch_all.sh" "${1:-status}" b11 b12 b13 b14 b15 b16 b17
