#!/bin/bash
# One-command crash recovery: inspects what finished and prints (or runs)
# exactly the jobs that still need to run. Usage:
#   ./resume_pipeline.sh          # report state
#   ./resume_pipeline.sh --run    # relaunch missing jobs
cd "$(dirname "$0")"
SP="${SCRATCH:-/tmp}"
need=()
chk() {  # chk <json> <label> <cmd...>
  local j="$1"; shift; local label="$1"; shift
  if ps aux | grep -v grep | grep -q -- "--out $j"; then
    echo "RUNNING  $label"
  elif [ -s "$j" ] && python3 -c "import json;d=json.load(open('$j'));exit(0 if len(d.get('raw',[]))>=6 else 1)" 2>/dev/null; then
    echo "DONE     $label"
  elif [ -s "$j.partial" ]; then
    echo "PARTIAL  $label ($(python3 -c "import json;print(json.load(open('$j.partial'))['episodes_done'])" 2>/dev/null) eps saved)"
    need+=("$*")
  else
    echo "MISSING  $label"
    need+=("$*")
  fi
}
chk eval_random_towel.json "towel random" \
  ".venv/bin/python eval_laundry.py --actor '' --garment towel4.obj --episodes 6 --seed 1200 --out eval_random_towel.json > ev_random_towelR.log 2>&1 &"
chk eval_diff_towel.json "towel diffusion" \
  ".venv/bin/python eval_laundry.py --mode diffusion --uncrumple uncrumple_towel4.pt --fold fold_towel4.pt --garment towel4.obj --episodes 6 --seed 1200 --out eval_diff_towel.json > ev_diff_towelR.log 2>&1 &"
chk eval_rl_towel.json "towel RL" \
  ".venv/bin/python eval_laundry.py --actor runs_towel/actor_380.pt --garment towel4.obj --episodes 6 --seed 1200 --out eval_rl_towel.json > ev_rl_towelR.log 2>&1 &"
[ -f runs_towel/actor_380.pt ] && echo "DONE     towel RL checkpoint" || echo "MISSING  towel RL checkpoint"
if [ "$1" = "--run" ] && [ ${#need[@]} -gt 0 ]; then
  echo "--- relaunching ${#need[@]} job(s)"
  for c in "${need[@]}"; do echo "$c"; eval "PYTHONPATH=. nohup $c"; done
fi
