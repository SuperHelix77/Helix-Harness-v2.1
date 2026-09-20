#!/usr/bin/env python3
"""Regenerate legacy-named public branding assets from the Helix app icon.

The filenames are retained because installers and older packaged builds still look
for them. The Unsloth provider badge (public/rounded.png) is intentionally not
part of this generator; it identifies the upstream model owner in the Hub.
"""

from pathlib import Path

from PIL import Image


REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "studio/src-tauri/icons/icon.png"
PUBLIC = REPO / "studio/frontend/public"


def main() -> None:
    source = Image.open(SOURCE).convert("RGBA")
    icon_512 = source.resize((512, 512), Image.Resampling.LANCZOS)

    for filename in ("rounded-512.png", "sticker.png", "unsloth-gem.png"):
        icon_512.save(PUBLIC / filename)

    # Preserve the historical wide lockup dimensions without carrying the old
    # Unsloth wordmark. The canonical Helix mark is centered on transparency.
    lockup = Image.new("RGBA", (960, 294), (0, 0, 0, 0))
    mark = source.resize((294, 294), Image.Resampling.LANCZOS)
    lockup.alpha_composite(mark, ((lockup.width - mark.width) // 2, 0))
    lockup.save(PUBLIC / "logotext.png")

    source.save(
        PUBLIC / "unsloth.ico",
        format="ICO",
        sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )


if __name__ == "__main__":
    main()
