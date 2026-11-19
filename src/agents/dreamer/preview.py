"""Videos of what the world model predicts, next to what really happened."""

from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image

# Border colors of frames that the model saw and of frames it predicted without seeing them
SEEN_COLOR = (40, 180, 40)
PREDICTED_COLOR = (200, 40, 40)
# One step holds the action for 4 of the game's 60 frames per second
FRAMES_PER_SECOND = 15


def render_preview(
    true: npt.NDArray[np.uint8] | None, predicted: npt.NDArray[np.uint8], context: int, scale: int = 2
) -> list[npt.NDArray[np.uint8]]:
    """
    Video frames of sequences side by side. Each column shows the true frames, the predicted ones and their
    difference (gray where they agree), with a green border while the model sees the frames and a red one after.

    :param true: True frames of shape (sequence, time, height, width, channel), None for imagined sequences only
    :param predicted: Predicted frames of the same shape
    :param context: Number of steps the model saw at the start of each sequence
    :return: RGB video frames
    """
    rows = [predicted] if true is None else [true, predicted, ((predicted.astype(int) - true + 255) // 2)]
    video = []
    for t in range(predicted.shape[1]):
        color = SEEN_COLOR if t < context else PREDICTED_COLOR
        columns = [np.concatenate([_framed(row[i, t], color, scale) for row in rows], 0) for i in range(len(predicted))]
        video.append(np.concatenate(columns, 1))
    return video


def _framed(frame: npt.NDArray, color: tuple[int, int, int], scale: int) -> npt.NDArray[np.uint8]:
    frame = np.asarray(frame, np.uint8)
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, -1)
    frame = frame.repeat(scale, 0).repeat(scale, 1)
    framed = np.empty((frame.shape[0] + 4, frame.shape[1] + 4, 3), np.uint8)
    framed[:] = color
    framed[2:-2, 2:-2] = frame
    return framed


def save_gif(video: list[npt.NDArray[np.uint8]], path: Path, fps: int = FRAMES_PER_SECOND) -> None:
    images = [Image.fromarray(frame) for frame in video]
    images[0].save(path, save_all=True, append_images=images[1:], duration=round(1000 / fps), loop=0)
