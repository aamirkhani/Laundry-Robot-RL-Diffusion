#!/bin/bash
# Loud "it's done" announcement: raise volume, speak 3x, restore volume.
CUR=$(wpctl get-volume @DEFAULT_AUDIO_SINK@ | awk '{print $2}')
wpctl set-mute @DEFAULT_AUDIO_SINK@ 0
wpctl set-volume @DEFAULT_AUDIO_SINK@ 1.0
for i in 1 2 3; do
  spd-say -w -i 100 -r -20 "It's done. It's done."
  sleep 1
done
wpctl set-volume @DEFAULT_AUDIO_SINK@ "${CUR:-0.3}"
