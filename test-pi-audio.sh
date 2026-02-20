#!/bin/bash
# Test all audio cards on Raspberry Pi to find the USB speaker and mic.
# Run: bash test-pi-audio.sh

echo "=== Audio cards ==="
echo ""
echo "--- Playback devices ---"
aplay -l 2>&1
echo ""
echo "--- Capture devices ---"
arecord -l 2>&1
echo ""

echo "=== Testing each playback card ==="
for card in 0 1 2 3; do
    echo ""
    echo "--- Testing card $card (plughw:$card,0) ---"
    echo "You should hear 'Front Left, Front Right' if this is your speaker."
    speaker-test -c 2 -D "plughw:$card,0" -t wav -l 1 2>&1 || echo "Card $card: failed"
    echo ""
done

echo "=== Done ==="
echo "Note which card number played sound — that is your USB speaker."
echo "Then edit setup-pi-audio.sh to use the correct card numbers and run it."
