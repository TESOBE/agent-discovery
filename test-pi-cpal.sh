#!/bin/bash
# Show what cpal sees on this device, saved to a file.
# Run: bash test-pi-cpal.sh

cargo run -- list-devices > cpal-devices.txt 2>&1
echo "Saved to cpal-devices.txt"
head -20 cpal-devices.txt
