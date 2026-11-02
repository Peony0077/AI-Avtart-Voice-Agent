"""Remove flyaway hair strands from an avatar photo before build_avatar.py animates it.

Thin strands sticking out of the hair into the background (common in portraits) are animated by the
face rig with the head, and against a dark background they read as "pieces of hair standing up".
This makes a clean copy of the photo:

  1. BiSeNet face parsing (face-parse-bisent, already used by MuseTalk) gives the person: skin, hair,
     neck, clothes... (everything that is not background).
  2. A morphological opening removes the thin parts of that silhouette - the strands - while the head
     and the body keep their shape; the outline is then smoothed and feathered by a pixel or two.
  3. Background pixels that differ sharply from their neighbourhood (a thin line on a smooth, out of
     focus background: strands the parser already called background) are found by a median filter.
  4. The strands (both kinds) are filled in from the background around them (OpenCV inpainting), and
     the person is put back over it with the soft outline.

The original photo is never changed: the result is written next to it (default: photo_clean.png).

    avatar/musetalk/.venv/bin/python avatar/musetalk/clean_hair.py data/avatars/dev_girl/photo.png
    avatar/musetalk/.venv/bin/python avatar/musetalk/clean_hair.py photo.png --out clean.png --debug dbg/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor" / "MuseTalk"))

BACKGROUND, HAIR, HAT = 0, 17, 18


def parse(image_bgr: np.ndarray) -> np.ndarray:
    """BiSeNet classes (CelebAMask-HQ: 0 background ... 17 hair) at the image's size."""
    import torch
    from common import WEIGHTS   # MODEL_HOME/avatar/models/face-parse-bisent
    from musetalk.utils.face_parsing.model import BiSeNet

    folder = WEIGHTS / "models" / "face-parse-bisent"
    net = BiSeNet(n_classes=19, resnet_path=str(folder / "resnet18-5c106cde.pth"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net.load_state_dict(torch.load(folder / "79999_iter.pth", map_location=device))
    net.to(device).eval()
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(cv2.resize(image_bgr, (512, 512), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    x = (rgb.astype(np.float32) / 255.0 - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)
    x = torch.from_numpy(x.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    with torch.no_grad():
        out = net(x)[0][0].argmax(0).cpu().numpy().astype(np.uint8)
    return cv2.resize(out, (w, h), interpolation=cv2.INTER_NEAREST)


def largest_component(mask: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return mask
    keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == keep).astype(np.uint8)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    flood = mask.copy() * 255
    pad = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, pad, (0, 0), 255)
    return (mask.astype(bool) | (flood == 0)).astype(np.uint8)


def large_lights(image: np.ndarray, scale: int) -> np.ndarray:
    """Bright areas of at least ~0.1 % of the picture (a lamp), as a 0/1 mask."""
    bright = (image.max(axis=2) >= 232).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bright, 8)
    keep = np.zeros(n, np.uint8)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= max(50, int(0.001 * scale * scale))
    return keep[lab]


def disk(size: int) -> np.ndarray:
    size = max(3, int(round(size)) | 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def clean(image: np.ndarray, open_frac: float = 0.035, line_thresh: float = 9.0,
          feather_frac: float = 0.004, wisp_frac: float = 0.03, wisp_thresh: float = 24.0,
          spike_frac: float = 0.012, edge_frac: float = 0.035, smooth_bg: bool = True,
          debug: Path | None = None) -> np.ndarray:
    h, w = image.shape[:2]
    scale = max(h, w)
    labels = parse(image)
    person = (labels != BACKGROUND).astype(np.uint8)
    hair = ((labels == HAIR) | (labels == HAT)).astype(np.uint8)
    rest = person & (1 - hair)              # face, neck, clothes: their outline is kept as parsed

    # 2. the hair without its thin parts (strands, the tuft on the crown), its outline smoothed. Only
    # the hair: opening the whole person would also round off the shoulders and the chin.
    ellipse = disk(open_frac * scale)
    hair_body = cv2.morphologyEx(hair, cv2.MORPH_OPEN, ellipse)
    s = max(1.0, open_frac * scale / 3)
    hair_body = (cv2.GaussianBlur(hair_body.astype(np.float32), (0, 0), s) > 0.5).astype(np.uint8)
    # a thin spike on the outline (the tuft at the crown) is rounded off: a wider blur, cut a little above half
    hair_body = (cv2.GaussianBlur(hair_body.astype(np.float32), (0, 0), max(1.0, spike_frac * scale))
                 > 0.58).astype(np.uint8) if spike_frac > 0 else hair_body
    body = fill_holes(largest_component(hair_body | cv2.morphologyEx(rest, cv2.MORPH_OPEN, disk(5))))
    # the frame's bottom edge is clothes: what the parser called person there stays person
    body[int(h * 0.9):] |= person[int(h * 0.9):]

    # 2b. the hair's wispy fringe: the parser counts a fringe of loose strands over the background as
    # hair (a strand sticking out at the cheek stayed). In the hair's outer band, where the neighbourhood
    # looks like the background around it rather than like hair, it is background: the silhouette is
    # trimmed to the dense hair there, and left exactly as it was where the hair is dense.
    if wisp_frac > 0:
        bg_guess = cv2.inpaint(image, cv2.dilate(body, disk(0.02 * scale)) * 255, max(3, int(0.012 * scale)),
                               cv2.INPAINT_TELEA)
        med = cv2.medianBlur(image, max(5, int(round(0.025 * scale)) | 1)).astype(np.float32)
        hair_like = np.abs(med - bg_guess.astype(np.float32)).max(axis=2) >= wisp_thresh
        band = (body & (1 - cv2.erode(body, disk(wisp_frac * scale))) & cv2.dilate(hair_body, disk(3))).astype(bool)
        body = (body.astype(bool) & ~(band & ~hair_like)).astype(np.uint8)
        body = cv2.morphologyEx(body, cv2.MORPH_OPEN, disk(0.012 * scale))
        body = fill_holes(largest_component(body))
        body = (cv2.GaussianBlur(body.astype(np.float32), (0, 0), max(1.0, 0.004 * scale)) > 0.5).astype(np.uint8)
        body[int(h * 0.9):] |= person[int(h * 0.9):]

    # 3. strands the parser left in the background: thin lines that a median filter removes. Per
    # colour channel (a brown strand on a grey-blue background differs more in colour than in grey).
    m = max(5, int(round(0.03 * scale)) | 1)
    diff = np.abs(image.astype(np.float32) - cv2.medianBlur(image, m).astype(np.float32)).max(axis=2)
    lines = (diff > line_thresh).astype(np.uint8)
    near_hair = cv2.dilate(hair_body | hair, disk(0.26 * scale))
    lines &= near_hair            # only around the hair: leave the rest of the background alone
    # never the edge of a light in the background (a lamp's rim is a sharp edge too, not a strand). A light
    # is a large bright area: the bright tip of a strand is not one (it was kept, as a white dot).
    lights = large_lights(image, scale)
    lines &= 1 - cv2.dilate(lights, disk(0.03 * scale))
    lines &= 1 - cv2.dilate(body, disk(3))
    # a strand is a line, not a blob: drop specks of a few pixels (bokeh noise)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(lines, 8)
    keep = np.zeros(n, np.uint8)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= max(4, int(0.00002 * scale * scale))
    lines = keep[lab]

    # 4. fill strands from the surrounding background, then put the person back with a soft edge
    holes = ((person & (1 - body)) | lines).astype(np.uint8)
    holes = cv2.dilate(holes, disk(0.008 * scale))      # and the strand's soft glow
    bg_only = image.copy()
    fill = (holes | cv2.dilate(body, ellipse)).astype(np.uint8) * 255   # the person is not background
    bg_only = cv2.inpaint(bg_only, fill, max(3, int(0.012 * scale)), cv2.INPAINT_TELEA)
    alpha = cv2.GaussianBlur(body.astype(np.float32), (0, 0), max(0.6, feather_frac * scale))[..., None]
    out = image.astype(np.float32) * alpha + bg_only.astype(np.float32) * (1 - alpha)
    # away from the strands and the outline the photo is untouched
    changed = cv2.dilate((holes | (cv2.dilate(body, ellipse) - cv2.erode(body, ellipse))).astype(np.uint8),
                         ellipse)
    out = np.where(changed[..., None].astype(bool), out, image.astype(np.float32))
    out = np.clip(out + 0.5, 0, 255).astype(np.uint8)

    # 5. bright strands lying across the hair's edge (one sticks out at the cheek): thin bright lines in the
    # outer band of the hair (a white top-hat finds them) take the hair's own texture around them (median)
    if edge_frac > 0:
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, disk(0.014 * scale))
        band = cv2.dilate(body, disk(0.035 * scale)) & (1 - cv2.erode(body, disk(edge_frac * scale)))
        band &= cv2.dilate(hair_body, disk(0.01 * scale))
        strands = ((tophat > 28) & band.astype(bool)).astype(np.uint8)
        strands = cv2.dilate(strands, disk(3))
        smooth = cv2.medianBlur(out, max(5, int(round(0.018 * scale)) | 1))
        soft = cv2.GaussianBlur(strands.astype(np.float32), (0, 0), 1.0)[..., None]
        out = np.clip(out * (1 - soft) + smooth * soft + 0.5, 0, 255).astype(np.uint8)

    # 6. what is left in the plain background around the head - a faint trace, a dot where a strand was -
    # is smoothed away (the background there is out of focus: a median filter changes nothing else). Not
    # near a light, with a soft edge so no seam shows.
    if smooth_bg:
        region = (cv2.dilate(hair_body | hair, disk(0.2 * scale)) & (1 - cv2.dilate(body, disk(0.004 * scale)))
                  & (1 - cv2.dilate(lights, disk(0.03 * scale))))
        region[int(h * 0.9):] = 0
        weight = cv2.GaussianBlur(region.astype(np.float32), (0, 0), max(1.0, 0.006 * scale))[..., None]
        calm = cv2.medianBlur(out, max(5, int(round(0.03 * scale)) | 1)).astype(np.float32)
        out = np.clip(out * (1 - weight) + calm * weight + 0.5, 0, 255).astype(np.uint8)

    if debug is not None:
        debug.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug / "person.png"), person * 255)
        cv2.imwrite(str(debug / "body.png"), body * 255)
        cv2.imwrite(str(debug / "lines.png"), lines * 255)
        cv2.imwrite(str(debug / "holes.png"), holes * 255)
        cv2.imwrite(str(debug / "background.png"), bg_only)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photo", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="default: <photo>_clean.png next to the photo")
    ap.add_argument("--open", type=float, default=0.035, help="strand width removed, as a share of the image")
    ap.add_argument("--line-thresh", type=float, default=9.0, help="colour levels a strand stands out by")
    ap.add_argument("--debug", type=Path, default=None, help="write the masks to this folder")
    args = ap.parse_args()
    image = cv2.imread(str(args.photo), cv2.IMREAD_COLOR)
    if image is None:
        sys.exit(f"cannot read {args.photo}")
    out = clean(image, args.open, args.line_thresh, debug=args.debug)
    target = args.out or args.photo.with_name(args.photo.stem + "_clean.png")
    cv2.imwrite(str(target), out)
    print(f"written {target}")


if __name__ == "__main__":
    main()
