# Icons

Phosphor Icons (regular weight), MIT licensed — https://phosphoricons.com
Vendored as individual SVG files so the control panel can inline them and stay
fully offline (no CDN, no font).

Each file is a single `<path>` with `fill="currentColor"`, so inlining it into the
page and sizing it with CSS gives an icon that inherits text colour.

Icons are decorative when they sit next to a visible Thai label (`aria-hidden="true"`);
an icon that carries meaning on its own must have an accessible name.
