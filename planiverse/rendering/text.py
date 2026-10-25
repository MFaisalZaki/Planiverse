"""The characters a state's `picture()` is drawn in.

A picture is plain text, black on white once typeset: a board with its walls joined into lines,
a field or a map laid out on a grid of characters, a system drawn as a schematic. What they
share is here, so that every environment draws a wall, a gauge or a box the same way:

    boxed(rows, wall="#", glyphs={"@": "☻"})   # walls joined into lines, two columns a cell
    canvas = Canvas(40, 12)                    # a grid of characters to draw on
    canvas.hline(0, 39, 11); canvas.write(2, 0, "label"); str(canvas)
    bar(3.5, 10, 20)                           # '███████░░░░░░░░░░░░░'
    tank(0.6, 5, 6)                            # a vessel filled to 60%, five rows tall

Everything returns or holds `str`, and every character used is in DejaVu Sans Mono, the face
the renderer typesets in, so a picture prints in a terminal as it renders in a GIF.
"""

#: A wall cell's line by which of its four neighbours, (up, right, down, left), are walls.
JOINS = {
    (0, 0, 0, 0): "■", (1, 0, 1, 0): "│", (0, 1, 0, 1): "─", (0, 1, 1, 0): "┌",
    (0, 0, 1, 1): "┐", (1, 1, 0, 0): "└", (1, 0, 0, 1): "┘", (1, 1, 1, 0): "├",
    (1, 0, 1, 1): "┤", (0, 1, 1, 1): "┬", (1, 1, 0, 1): "┴", (1, 1, 1, 1): "┼",
    (1, 0, 0, 0): "╵", (0, 1, 0, 0): "╶", (0, 0, 1, 0): "╷", (0, 0, 0, 1): "╴",
}

#: The joins of two crossing or meeting lines, for `Canvas.hline` and `Canvas.vline`.
_UP, _RIGHT, _DOWN, _LEFT = 1, 2, 4, 8
_ARMS = {"─": _LEFT | _RIGHT, "│": _UP | _DOWN, "┌": _RIGHT | _DOWN, "┐": _LEFT | _DOWN,
         "└": _UP | _RIGHT, "┘": _UP | _LEFT, "├": _UP | _DOWN | _RIGHT,
         "┤": _UP | _DOWN | _LEFT, "┬": _LEFT | _RIGHT | _DOWN, "┴": _LEFT | _RIGHT | _UP,
         "┼": _UP | _DOWN | _LEFT | _RIGHT, "╴": _LEFT, "╶": _RIGHT, "╵": _UP, "╷": _DOWN}
_BY_ARMS = {arms: char for char, arms in _ARMS.items()}


def boxed(rows, wall="#", glyphs=None, blank=" "):
    """A board with its walls drawn as joined lines and every other cell through `glyphs`.

    Each cell takes two columns, the second carrying a wall's line on to its right-hand
    neighbour, so that the lines join and a cell is about as wide as it is tall. A cell
    `glyphs` does not name is drawn as itself.
    """
    glyphs = glyphs or {}
    rows = [str(row) for row in rows]

    def is_wall(r, c):
        return 0 <= r < len(rows) and 0 <= c < len(rows[r]) and rows[r][c] == wall

    lines = []
    for r, row in enumerate(rows):
        line = []
        for c, cell in enumerate(row):
            if cell == wall:
                joins = (is_wall(r - 1, c), is_wall(r, c + 1), is_wall(r + 1, c),
                         is_wall(r, c - 1))
                line.append(JOINS[tuple(int(j) for j in joins)] + ("─" if joins[1] else blank))
            else:
                line.append(glyphs.get(cell, cell) + blank)
        lines.append("".join(line).rstrip())
    return "\n".join(lines)


class Canvas:
    """A grid of characters, `width` columns by `height` rows, origin at the top left.

    Drawing outside the grid is clipped rather than refused, so a picture can place things
    by their coordinates without checking every one. Lines drawn with `hline`, `vline` and
    `rect` join where they meet.
    """

    def __init__(self, width, height, fill=" "):
        self.width, self.height = width, height
        self.cells = [[fill] * width for _ in range(height)]

    def put(self, x, y, char):
        x, y = int(round(x)), int(round(y))
        if 0 <= x < self.width and 0 <= y < self.height:
            self.cells[y][x] = char

    def get(self, x, y):
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.cells[y][x]
        return " "

    def write(self, x, y, text):
        """Text left to right from `(x, y)`."""
        for k, char in enumerate(str(text)):
            self.put(x + k, y, char)

    def _join(self, x, y, arms):
        current = _ARMS.get(self.get(x, y), 0)
        self.put(x, y, _BY_ARMS.get(current | arms, "┼"))

    def hline(self, x0, x1, y, char=None):
        """A horizontal line from `x0` to `x1`: box-drawing that joins, or `char` throughout."""
        x0, x1 = sorted((int(round(x0)), int(round(x1))))
        for x in range(x0, x1 + 1):
            if char:
                self.put(x, y, char)
            else:
                self._join(x, y, (_LEFT if x > x0 else 0) | (_RIGHT if x < x1 else 0))

    def vline(self, x, y0, y1, char=None):
        """A vertical line from `y0` to `y1`: box-drawing that joins, or `char` throughout."""
        y0, y1 = sorted((int(round(y0)), int(round(y1))))
        for y in range(y0, y1 + 1):
            if char:
                self.put(x, y, char)
            else:
                self._join(x, y, (_UP if y > y0 else 0) | (_DOWN if y < y1 else 0))

    def rect(self, x0, y0, x1, y1, label=None):
        """A box from corner to corner, its label, if any, written inside on its first row."""
        self.hline(x0, x1, y0)
        self.hline(x0, x1, y1)
        self.vline(x0, y0, y1)
        self.vline(x1, y0, y1)
        if label:
            self.write(x0 + 1, y0 + 1, str(label)[:max(0, x1 - x0 - 1)])

    def fill(self, x0, y0, x1, y1, char):
        """Every cell of a rectangle, corners included."""
        for y in range(int(round(min(y0, y1))), int(round(max(y0, y1))) + 1):
            for x in range(int(round(min(x0, x1))), int(round(max(x0, x1))) + 1):
                self.put(x, y, char)

    def __str__(self):
        return "\n".join("".join(row).rstrip() for row in self.cells).rstrip("\n")


def bar(value, high, width, full="█", empty="░"):
    """A horizontal gauge `width` characters long, filled to `value` out of `high`."""
    share = 0.0 if not high else max(0.0, min(1.0, value / high))
    filled = int(round(share * width))
    return full * filled + empty * (width - filled)


def tank(share, height, width, full="█", empty=" ", wall="│", floor="└", corner="┘"):
    """A vessel `height` rows tall and `width` wide inside, filled to `share` (0 to 1), as a
    list of lines from the top: walls either side and a floor under it."""
    share = max(0.0, min(1.0, share))
    level = int(round(share * height))
    lines = [wall + (full if height - row <= level else empty) * width + wall
             for row in range(height)]
    lines.append(floor + "─" * width + corner)
    return lines


def side_by_side(blocks, gap=3):
    """Blocks of text (each a `str` or a list of lines) set next to each other, tops aligned."""
    blocks = [block.split("\n") if isinstance(block, str) else list(block) for block in blocks]
    widths = [max((len(line) for line in block), default=0) for block in blocks]
    height = max(len(block) for block in blocks)
    lines = []
    for row in range(height):
        parts = [(block[row] if row < len(block) else "").ljust(width)
                 for block, width in zip(blocks, widths)]
        lines.append((" " * gap).join(parts).rstrip())
    return "\n".join(lines)


def missing_glyphs(text):
    """The characters of `text` the renderer's monospace face has no glyph for, which would
    fall back to another font and break the grid. Empty when the face cannot be found."""
    import os

    try:
        import matplotlib
        from fontTools.ttLib import TTFont
    except ImportError:
        return set()
    path = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf",
                        "DejaVuSansMono.ttf")
    if not os.path.isfile(path):
        return set()
    cmap = _cmap(path)
    return {char for char in set(text) - {"\n"} if ord(char) not in cmap}


_CMAPS = {}


def _cmap(path):
    if path not in _CMAPS:
        from fontTools.ttLib import TTFont
        _CMAPS[path] = set(TTFont(path).getBestCmap())
    return _CMAPS[path]
