"""A played game saved compactly, to be rendered later at any speed.

Dodge is deterministic given the seed and the actions, so a run file holds
only those plus what the player drew over each frame (its Overlay), and a
checksum of each frame's pixels so a replay can prove it reproduces the game.
Deciding is what makes a heuristic slow to watch; replaying actions is not.

The file is a gzip stream of pickled records, flushed as it goes, so a run
that is cut short is still readable up to the last flush:
    {"settings": ...}                  first, the engine settings
    ("game", seed)                     each new game
    ("frame", action, crc, overlay)    each frame; overlay is None or
                                       (field float16 | None, boxes, paths)
"""
import gzip
import pickle
import zlib

import numpy as np

from .overlay import Overlay

FLUSH_EVERY = 600       # frames: at most ten seconds of play lost if cut short


def pixels_crc(pixels):
    return zlib.crc32(np.ascontiguousarray(pixels, dtype=np.uint8).tobytes())


def pack_overlay(overlay):
    if overlay is None:
        return None
    field = None if overlay.field is None else np.asarray(overlay.field, dtype=np.float16)
    boxes = [tuple(float(v) for v in b[:4]) + (tuple(b[4]),) for b in overlay.boxes]
    paths = [(np.asarray(points, dtype=np.float32), tuple(colour))
             for points, colour in overlay.paths]
    return field, boxes, paths


def unpack_overlay(packed):
    if packed is None:
        return None
    field, boxes, paths = packed
    return Overlay(field=None if field is None else field.astype(np.float32),
                   boxes=list(boxes), paths=[(p.tolist(), c) for p, c in paths])


def open_video(path, fps, crf=32):
    """An MP4 writer taking (height, width, 3) uint8 frames.

    yuv420p H.264 plays everywhere a video gets sent: phones, chat apps,
    browsers. CRF 32 keeps a 90 s game with the overlay near 10 MB and still
    sharp; faststart lets it play before it has fully downloaded."""
    import imageio.v2 as imageio
    return imageio.get_writer(path, fps=fps, codec="libx264", quality=None,
                              pixelformat="yuv420p", macro_block_size=16,
                              output_params=["-crf", str(crf), "-preset", "slow",
                                             "-movflags", "+faststart"])


class RunWriter:
    def __init__(self, path, settings):
        self._file = gzip.open(path, "wb", compresslevel=6)
        self._frames = 0
        self._dump({"settings": settings})

    def _dump(self, record):
        pickle.dump(record, self._file, protocol=pickle.HIGHEST_PROTOCOL)

    def game(self, seed):
        self._dump(("game", int(seed)))

    def frame(self, action, pixels, overlay):
        self._dump(("frame", int(action), pixels_crc(pixels), pack_overlay(overlay)))
        self._frames += 1
        if self._frames % FLUSH_EVERY == 0:
            self._file.flush(zlib.Z_SYNC_FLUSH)

    def close(self):
        self._file.close()


def read_run(path):
    """(settings, records): records yields ("game", seed) and ("frame", action,
    crc, overlay) in order, stopping quietly where a cut-short file ends."""
    f = gzip.open(path, "rb")
    settings = pickle.load(f)["settings"]

    def records():
        try:
            while True:
                yield pickle.load(f)
        except (EOFError, gzip.BadGzipFile, pickle.UnpicklingError, OSError, zlib.error):
            pass
        finally:
            f.close()
    return settings, records()
