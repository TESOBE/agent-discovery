#!/bin/bash
# Show what cpal sees on this device.
# Run: bash test-pi-cpal.sh

cargo run -- list-devices 2>&1
