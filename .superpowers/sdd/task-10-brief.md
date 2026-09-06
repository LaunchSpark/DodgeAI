### Task 10: Text overlay and button glyphs

PICO-8 glyphs are 3×5 px, which no system font renders legibly at native scale. Text is therefore queued during the frame and drawn onto the *upscaled* window, where there is room for a real font. Button glyphs inside strings (`"press ❎ to play again"`) become small drawn icons.

**Files:**
- Modify: `pico8/backends.py`
- Create: `tests/test_text_overlay.py`

**Interfaces:**
- Consumes: `PygameBackend.text_queue`, `pico8.preprocess.BUTTONS`
- Produces: `pico8.backends.split_glyphs(text: str) -> list[tuple[str, str]]` — `("text"|"glyph", chunk)` pairs

- [ ] **Step 1: Write the failing test**

```python
# tests/test_text_overlay.py
from pico8.backends import split_glyphs


def test_plain_text_is_one_chunk():
    assert split_glyphs("score:12") == [("text", "score:12")]


def test_glyph_is_split_out():
    assert split_glyphs("press ❎ to play") == [
        ("text", "press "), ("glyph", "❎"), ("text", " to play"),
    ]


def test_adjacent_glyphs():
    assert split_glyphs("⬅️➡️") == [
        ("glyph", "⬅️"), ("glyph", "➡️"),
    ]


def test_empty_string():
    assert split_glyphs("") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_text_overlay.py -v`
Expected: FAIL with `ImportError: cannot import name 'split_glyphs'`

- [ ] **Step 3: Implement**

Add to `pico8/backends.py`:

```python
import re

from .preprocess import BUTTONS

_GLYPH_SPLIT = re.compile("(" + "|".join(re.escape(g) for g in BUTTONS) + ")")

# What each button glyph is drawn as, since no system font has them.
GLYPH_LABEL = {
    "⬅️": "<", "➡️": ">", "⬆️": "^",
    "⬇️": "v", "\U0001f17e️": "O", "❎": "X",
}


def split_glyphs(text):
    """Split a PICO-8 string into ('text'|'glyph', chunk) pairs."""
    return [("glyph" if chunk in BUTTONS else "text", chunk)
            for chunk in _GLYPH_SPLIT.split(text) if chunk]
```

Replace the placeholder `_draw_text` on `PygameBackend` with the real one, and initialise the font in `__init__` after `pygame.init()`:

```python
        self.font = pygame.font.SysFont("consolas,dejavusansmono,monospace",
                                        max(8, 4 * scale))
```

```python
    def _draw_text(self):
        """Draw queued text onto the upscaled window.

        PICO-8 characters occupy a 4x6 cell, so a chunk of n characters is
        n*4 pixels wide in game space; that is the advance used here.
        """
        s = self.scale
        for text, x, y, colour in self.text_queue:
            rgb = PICO8[self.display_pal[colour]]
            cursor = x
            for kind, chunk in split_glyphs(text):
                if kind == "glyph":
                    label = GLYPH_LABEL[chunk]
                    box = pygame.Rect(cursor * s, y * s, 7 * s, 5 * s)
                    pygame.draw.rect(self.window, rgb, box, width=max(1, s // 2))
                    glyph = self.font.render(label, True, rgb)
                    self.window.blit(glyph, glyph.get_rect(center=box.center))
                    cursor += 8
                else:
                    self.window.blit(self.font.render(chunk, True, rgb),
                                     (cursor * s, y * s))
                    cursor += len(chunk) * 4
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/ -v`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add pico8/backends.py tests/test_text_overlay.py
git commit -m "feat: scaled text overlay with drawn button glyphs"
```

---

