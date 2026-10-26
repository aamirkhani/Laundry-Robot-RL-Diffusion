#!/bin/bash
# Re-record the 6 pre-texture videos (towel x3, shirt x3) on the textured
# table, 2 at a time. Run AFTER the pants videos finish. Idempotent: skips
# any vid_*.mp4 that already has a .textured marker.
cd "$(dirname "$0")"
render() {  # render <out.mp4> <extra eval args...>
  local out="$1"; shift
  if [ -f "$out.textured" ]; then echo "skip $out (done)"; return; fi
  rm -f "$out"
  PYTHONPATH=. .venv/bin/python eval_laundry.py "$@" --episodes 1 \
    --video "$out" --out "eval_vid_re_${out%.mp4}.json" \
    > "ev_re_${out%.mp4}.log" 2>&1
  if [ ! -f "$out" ]; then
    n=$(ls -t eval_laundry_cam_0_*.mp4 2>/dev/null | head -1)
    [ -n "$n" ] && mv "$n" "$out"
  fi
  [ -f "$out" ] && touch "$out.textured" && echo "RE-RENDERED $out"
}
render vid_towel_rl.mp4 --actor runs_towel/actor_380.pt --garment towel4.obj --seed 1300 &
render vid_towel_diffusion.mp4 --mode diffusion --uncrumple uncrumple_towel4.pt --fold fold_towel4.pt --garment towel4.obj --seed 1300 &
wait
render vid_towel_random.mp4 --actor "" --garment towel4.obj --seed 1300 &
render vid_tshirt_rl.mp4 --actor runs_tshirt_v2/actor_400.pt --garment tshirt4.obj --seed 1500 &
wait
render vid_tshirt_diffusion.mp4 --mode diffusion --uncrumple uncrumple_tshirt4.pt --fold fold_tshirt4.pt --garment tshirt4.obj --seed 1501 &
render vid_tshirt_random.mp4 --actor "" --garment tshirt4.obj --seed 1500 &
wait
echo "ALL RE-RENDERS DONE"
