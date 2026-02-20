#!/bin/bash
# Configure ALSA on Raspberry Pi to use USB mic (card 2) and USB speaker (card 3).
# Run: bash setup-pi-audio.sh
# Then test: arecord -d 3 -f cd test.wav && aplay test.wav

cat > ~/.asoundrc << 'EOF'
pcm.!default {
    type asym
    playback.pcm "plughw:3,0"
    capture.pcm "plughw:2,0"
}
ctl.!default {
    type hw
    card 3
}
EOF

echo "Written ~/.asoundrc (mic=card2, speaker=card3)"
echo "Test with: arecord -d 3 -f cd test.wav && aplay test.wav"
