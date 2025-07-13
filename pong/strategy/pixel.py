import numpy as np
import cv2
from collections import deque
from pong.constants import *
from .multiball import MultiBallStepStrategy


class PixelStepStrategy(MultiBallStepStrategy):
    """
    • 4-frame stack of 84×84 grayscale images ∈ [0, 1] float32
    • Action-repeat (frame-skip) = 4
    • ±1 reward clipping
    """
    handles_pixels = True
    
    def __init__(self, cfg: dict | None = None):
        super().__init__(cfg)
        cfg = cfg or {}
        self.skip      = cfg.get("frame_skip", 4)
        self.stack_len = cfg.get("frame_stack", 4)
        self.obs_shape = (84, 84)
        self._frames: deque[np.ndarray] = deque(maxlen=self.stack_len)

    # ───────────────────────── helpers ──────────────────────────
    def _preprocess(self, rgb: np.ndarray) -> np.ndarray:
        gray     = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        resized  = cv2.resize(gray, self.obs_shape, interpolation=cv2.INTER_AREA)
        return (resized.astype(np.float32) / 255.0)          # (H,W) float32

    def _stack(self) -> np.ndarray:
        """Return (stack_len, 84, 84) float32, oldest-first."""
        while len(self._frames) < self.stack_len:
            self._frames.append(self._frames[-1].copy())
        return np.stack(self._frames, axis=0, dtype=np.float32)

    # ───────────────────────── gameplay ─────────────────────────
    def reset(self, env):
        super().reset(env)                       # sets game objects
        img = env._capture_frame()               # one freshly drawn RGB frame
        self._frames.clear()
        pre = self._preprocess(img)
        for _ in range(self.stack_len):
            self._frames.append(pre)
        return self._stack()

    def execute(self, env, action):
        R, term, trunc = 0.0, False, False
        for t in range(self.skip):
            _, r, term, trunc, info = super().execute(env, action)
            R += np.sign(r)                          # reward clip
            if t == self.skip - 1 or term or trunc:
                img = env._capture_frame()
                self._frames.append(self._preprocess(img))
            if term or trunc:
                break
        return self._stack(), R, term, trunc, info