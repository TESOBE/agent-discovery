#!/bin/bash
# Test the USB microphone on Raspberry Pi.
# Run: bash test-pi-mic.sh

echo "Recording 5 seconds from USB mic (card 2)..."
echo "Speak loudly or tap the mic NOW!"
arecord -D plughw:2,0 -d 5 -f cd test.wav 2>&1

echo ""
echo "File size:"
ls -lh test.wav

echo ""
echo "Playing back on USB speaker (card 3)..."
aplay -D plughw:3,0 test.wav 2>&1

echo ""
echo "If you heard yourself, the mic works."
echo "If silence, try: amixer -c 2 set Mic 100%"
