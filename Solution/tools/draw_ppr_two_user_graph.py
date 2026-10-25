"""Draw results/analysis/ppr_two_user_graph.json as one SVG: heavy user's PPR subgraph on the left and the light
user's on the right, mirrored (programmes in the middle), 10 nodes each.
Usage: python3 tools/draw_ppr_two_user_graph.py results/analysis/ppr_two_user_graph.json results/analysis/ppr_two_user_graph.svg
"""
import json
import math
import sys
from html import escape

d = json.load(open(sys.argv[1], encoding="utf-8"))
W, H, PAD_TOP = 1240, 470, 64
COLORS = {"heavy": "#1d4ed8", "light": "#be185d"}
GREY, HIT = "#9ca3af", "#16a34a"


def short(t, n=32):
    t = t.strip().strip(".")
    return t if len(t) <= n else t[: n - 1].rstrip() + "…"


out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
       f'font-family="Arial, Liberation Sans, sans-serif">',
       f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>']

half = W / 2
for k, p in enumerate(d["panels"]):
    role, col = p["role"], COLORS[p["role"]]
    mirror = k == 1
    x0 = half * k
    # column x positions: users outside, programmes towards the centre of the figure
    xu = x0 + (half - 160 if mirror else 160)
    xi = x0 + (half - 330 if mirror else 330)
    rows = [PAD_TOP + 30 + r * 72 for r in range(5)]
    seed = p["users"][0]
    label = "A — heavy viewer" if role == "heavy" else "B — light viewer"
    out.append(f'<text x="{x0 + half / 2}" y="26" text-anchor="middle" font-size="16" font-weight="bold" fill="{col}">'
               f'{label} ({seed["fullViews"]} full views of {seed["degree"]} programmes), restart at {"A" if role == "heavy" else "B"}</text>')
    out.append(f'<text x="{xu}" y="{PAD_TOP - 4}" text-anchor="middle" font-size="13" fill="#4b5563">users</text>')
    out.append(f'<text x="{xi}" y="{PAD_TOP - 4}" text-anchor="middle" font-size="13" fill="#4b5563">PPR Top-5 programmes</text>')
    upos = {u["id"]: (xu, rows[r]) for r, u in enumerate(p["users"])}
    ipos = {it["id"]: (xi, rows[r]) for r, it in enumerate(p["items"])}
    # edges first (under the nodes); co-viewer edges grey, seed edges in the panel colour with the view count
    for e in sorted(p["edges"], key=lambda e: e["user"] == seed["id"]):
        (ux, uy), (ix, iy) = upos[e["user"]], ipos[e["item"]]
        is_seed = e["user"] == seed["id"]
        wpx = 1.0 + 1.9 * math.log1p(e["w"])
        out.append(f'<line x1="{ux}" y1="{uy}" x2="{ix}" y2="{iy}" stroke="{col if is_seed else GREY}" '
                   f'stroke-width="{wpx:.2f}" stroke-opacity="{0.9 if is_seed else 0.45}"/>')
        if is_seed:
            mx, my = ux + (ix - ux) * 0.62, uy + (iy - uy) * 0.62
            out.append(f'<text x="{mx:.1f}" y="{my - 5:.1f}" text-anchor="middle" font-size="12" font-weight="bold" '
                       f'fill="{col}" stroke="#ffffff" stroke-width="3" paint-order="stroke">{e["w"]}</text>')
    # user nodes
    for r, u in enumerate(p["users"]):
        x, y = upos[u["id"]]
        if u["seed"]:
            out.append(f'<circle cx="{x}" cy="{y}" r="17" fill="{col}"/>')
            out.append(f'<text x="{x}" y="{y + 5}" text-anchor="middle" font-size="15" font-weight="bold" fill="#ffffff">{"A" if role == "heavy" else "B"}</text>')
            txt = "seed"
        else:
            out.append(f'<circle cx="{x}" cy="{y}" r="11" fill="#e5e7eb" stroke="#6b7280" stroke-width="1.5"/>')
            txt = f'other user: {u["degree"]} progr.'
        anchor, dx = ("start", 22) if mirror else ("end", -22)
        out.append(f'<text x="{x + dx}" y="{y - 2}" text-anchor="{anchor}" font-size="12" fill="#111827">{escape(txt)}</text>')
        out.append(f'<text x="{x + dx}" y="{y + 13}" text-anchor="{anchor}" font-size="11" fill="#6b7280">PPR {u["ppr"]:.4f}</text>')
    # programme nodes: radius by PPR score, green ring = in the user's actual weeks 21-30 top-5, star = new to the user
    smax = max(it["ppr"] for it in p["items"])
    for it in p["items"]:
        x, y = ipos[it["id"]]
        rad = 7 + 13 * math.sqrt(it["ppr"] / smax)
        ring = f'stroke="{HIT}" stroke-width="4"' if it["hit"] else 'stroke="#374151" stroke-width="1.5"'
        out.append(f'<circle cx="{x}" cy="{y}" r="{rad:.1f}" fill="#fde68a" {ring}/>')
        anchor, dx = ("end", -(rad + 8)) if mirror else ("start", rad + 8)
        star = " ★" if it["seedViews"] == 0 else ""
        out.append(f'<text x="{x + dx:.1f}" y="{y - 2}" text-anchor="{anchor}" font-size="12.5" font-weight="bold" fill="#111827">'
                   f'#{it["rank"]} {escape(short(it["title"]))}{star}</text>')
        out.append(f'<text x="{x + dx:.1f}" y="{y + 13}" text-anchor="{anchor}" font-size="11" fill="#6b7280">'
                   f'{it["id"]} · {escape(it["category"])} · PPR {it["ppr"]:.4f}</text>')
out.append(f'<line x1="{half}" y1="44" x2="{half}" y2="{H - 70}" stroke="#e5e7eb" stroke-width="1.5"/>')
# legend
ly = H - 40
leg = [
    f'<line x1="20" y1="{ly}" x2="50" y2="{ly}" stroke="{COLORS["heavy"]}" stroke-width="4"/><line x1="54" y1="{ly}" x2="84" y2="{ly}" stroke="{COLORS["light"]}" stroke-width="4"/>'
    f'<text x="92" y="{ly + 4}" font-size="12" fill="#374151">seed user’s edges, label = full views in weeks 11–20</text>',
    f'<line x1="400" y1="{ly}" x2="430" y2="{ly}" stroke="{GREY}" stroke-width="3" stroke-opacity="0.6"/>'
    f'<text x="438" y="{ly + 4}" font-size="12" fill="#374151">other users’ edges (width = log of full views)</text>',
    f'<circle cx="760" cy="{ly}" r="9" fill="#fde68a" stroke="{HIT}" stroke-width="4"/>'
    f'<text x="776" y="{ly + 4}" font-size="12" fill="#374151">in the user’s actual weeks 21–30 top-5</text>',
    f'<text x="1000" y="{ly + 4}" font-size="12" fill="#374151">★ = new to the user</text>',
]
out.extend(leg)
out.append(f'<text x="20" y="{H - 14}" font-size="11.5" fill="#6b7280">Training graph weeks 11–20; PPR α = 0.7, edge weight = full-view count; '
           f'programme radius = PPR score; the 4 other users are those with the highest PPR probability from the seed (where the walk goes).</text>')
out.append("</svg>")
open(sys.argv[2], "w", encoding="utf-8").write("\n".join(out))
print("written", sys.argv[2])
