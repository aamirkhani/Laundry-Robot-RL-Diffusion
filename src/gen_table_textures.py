"""Generate 5 creative table textures (top + 4 sides) via the OpenAI image
API and save them under textures/. ALBEDO ONLY — applied as diffuse_texture
on non-colliding veneer panels, so physics/observations/rewards are
untouched and no trained model is invalidated.

Requires OPENAI_API_KEY in the environment.
Usage: python gen_table_textures.py [--model gpt-image-1]
"""
import argparse
import base64
import json
import os
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--model", type=str, default="gpt-image-1")
ap.add_argument("--outdir", type=str, default="textures")
args = ap.parse_args()

KEY = os.environ.get("OPENAI_API_KEY", "")
if not KEY:
    raise SystemExit("OPENAI_API_KEY not set — export it and rerun")
os.makedirs(args.outdir, exist_ok=True)

# very-very-creative brief, one distinct concept per face
PROMPTS = {
    "table_top": (
        "Seamless top-down texture of a luxurious walnut wood laundry-atelier "
        "worktable surface, rich chocolate grain with honey highlights, "
        "inlaid pale-maple marquetry forming a subtle geometric border of "
        "folded-shirt and towel silhouettes, tiny brass measuring-rule inlay "
        "along one edge, soft satin finish, photorealistic, even diffuse "
        "lighting, no shadows, orthographic flat view"),
    "side_front": (
        "Ornate carved walnut wood panel texture, bas-relief frieze of robot "
        "arms elegantly folding laundry in the style of a Greek temple "
        "procession, warm wood tones with gilded accents in the carving "
        "grooves, photorealistic, flat frontal view, even lighting"),
    "side_back": (
        "Carved wood panel texture in Japanese kumiko lattice style, "
        "asanoha hexagonal patterns interrupted by a central medallion of "
        "an origami crane, hinoki cypress pale golden tones, photorealistic "
        "flat frontal view, even diffuse lighting"),
    "side_left": (
        "Wood panel texture painted in Portuguese azulejo style: indigo blue "
        "ceramic-look tiles on wood depicting swirling textile waves, "
        "clotheslines with dancing garments, and washerwomen at a river, "
        "craquelure glaze detail, flat frontal view, even lighting"),
    "side_right": (
        "Steampunk cabinet panel texture: burl wood with embedded brass "
        "gears, copper pipework, a small pressure gauge labelled STEAM, "
        "rivets along the edges, aged patina, warm workshop tones, "
        "photorealistic flat frontal view, even diffuse lighting"),
}
SIZES = {
    "table_top": "1536x1024",     # 2.2 x 1.6 m top (closest wide aspect)
    "side_front": "1536x1024",
    "side_back": "1536x1024",
    "side_left": "1024x1024",
    "side_right": "1024x1024",
}

for name, prompt in PROMPTS.items():
    out = os.path.join(args.outdir, f"{name}.png")
    if os.path.exists(out):
        print(f"{out} exists — skip")
        continue
    body = json.dumps({"model": args.model, "prompt": prompt,
                       "size": SIZES[name], "n": 1}).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/images/generations", data=body,
        headers={"Authorization": f"Bearer {KEY}",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        resp = json.load(r)
    d = resp["data"][0]
    if "b64_json" in d:
        img = base64.b64decode(d["b64_json"])
    else:
        with urllib.request.urlopen(d["url"], timeout=120) as r2:
            img = r2.read()
    with open(out, "wb") as f:
        f.write(img)
    print(f"wrote {out} ({len(img)//1024} KB)")
print("ALL TEXTURES DONE")
