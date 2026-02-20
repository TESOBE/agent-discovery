#!/bin/bash
# Test all audio cards on Raspberry Pi to find the USB speaker and mic.
# Run: bash test-pi-audio.sh

echo ""
echo "--- Playback devices ---"
aplay -l 2>&1
echo ""
echo "--- Capture devices ---"
arecord -l 2>&1
echo ""
echo "Will test each card. Press Enter between each test."
echo ""

for card in 0 1 2 3; do
    echo "========================================"
    echo "  TESTING CARD $card NOW"
    echo "========================================"
    read -p "Press Enter to play on card $card..."
    speaker-test -c 2 -D "plughw:$card,0" -t wav -l 1 2>/dev/null
    echo ""
    echo "Did you hear sound from the USB speaker on card $card? (remember the number)"
    echo ""
done

echo "Done. Update setup-pi-audio.sh with the card number that played from the USB speaker."
