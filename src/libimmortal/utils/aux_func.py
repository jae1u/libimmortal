from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Sequence, Type, List, Union
import numpy as np
from .enums import GraphicObservationColorMap
import socket
from contextlib import closing
from collections import deque


@dataclass(frozen=True)
class ColorMapEncoder:
    palette: np.ndarray  # (K,3) uint8
    names: Tuple[str, ...]  # length K
    name2id: Dict[str, int]
    unknown_id: int

    @classmethod
    def from_enum(
        cls,
        enum_cls: Type[GraphicObservationColorMap] = GraphicObservationColorMap,
        *,
        unknown_to: str | int = "BLANK",
    ) -> "ColorMapEncoder":

        items = [(k, v) for k, v in enum_cls.__dict__.items() if k.isupper()]
        names = tuple(k for k, _ in items)
        palette = np.asarray([v for _, v in items], dtype=np.uint8)
        if palette.ndim != 2 or palette.shape[1] != 3:
            raise ValueError("Palette must be shape (K,3) of uint8 RGB.")
        name2id = {name: i for i, name in enumerate(names)}

        if isinstance(unknown_to, str):
            if unknown_to not in name2id:
                raise KeyError(f"unknown_to='{unknown_to}' not in {list(name2id)}")
            unknown_id = name2id[unknown_to]
        else:
            unknown_id = int(unknown_to)
            if not (0 <= unknown_id < len(names)):
                raise ValueError(f"unknown_to id out of range 0..{len(names)-1}")

        return cls(palette=palette, names=names, name2id=name2id, unknown_id=unknown_id)

    def encode_ids(self, img: np.ndarray) -> np.ndarray:
        img_hwc = self._to_hwc(img)
        matches = (img_hwc[..., None, :] == self.palette[None, None, :, :]).all(
            axis=-1
        )  # (H,W,K)
        matched_any = matches.any(axis=-1)
        id_map = matches.argmax(axis=-1).astype(np.uint8)
        if not matched_any.all():
            id_map[~matched_any] = np.uint8(self.unknown_id)
        return id_map

    def encode_onehot(self, img: np.ndarray) -> np.ndarray:
        img_hwc = self._to_hwc(img)
        matches = (img_hwc[..., None, :] == self.palette[None, None, :, :]).all(
            axis=-1
        )  # (H,W,K)
        matched_any = matches.any(axis=-1)
        onehot = matches.astype(np.uint8).transpose(2, 0, 1)  # (K,H,W)
        if not matched_any.all():
            onehot[:, ~matched_any] = 0
            onehot[self.unknown_id, ~matched_any] = 1
        return onehot

    def encode(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        img_hwc = self._to_hwc(img)
        matches = (img_hwc[..., None, :] == self.palette[None, None, :, :]).all(
            axis=-1
        )  # (H,W,K)
        matched_any = matches.any(axis=-1)

        id_map = matches.argmax(axis=-1).astype(np.uint8)
        if not matched_any.all():
            id_map[~matched_any] = np.uint8(self.unknown_id)

        onehot = matches.astype(np.uint8).transpose(2, 0, 1)  # (K,H,W)
        if not matched_any.all():
            onehot[:, ~matched_any] = 0
            onehot[self.unknown_id, ~matched_any] = 1

        return id_map, onehot

    @staticmethod
    def _to_hwc(img: np.ndarray) -> np.ndarray:
        if img.ndim != 3:
            raise AssertionError("expected 3D array")
        if img.shape[0] == 3 and img.shape[-1] != 3:
            # (3,H,W) -> (H,W,3)
            img = np.moveaxis(img, 0, -1)
        if img.dtype != np.uint8:
            img = img.astype(np.uint8, copy=False)
        return img


DEFAULT_ENCODER = ColorMapEncoder.from_enum()


def colormap_to_ids_and_onehot(img_chw: np.ndarray):
    return DEFAULT_ENCODER.encode(img_chw)


def find_free_tcp_port(host: str = "127.0.0.1") -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, 0))
        return s.getsockname()[1]


def find_n_free_tcp_ports(n: int, host: str = "127.0.0.1") -> List[int]:
    ports = []
    for _ in range(n):
        ports.append(find_free_tcp_port(host))
    if len(set(ports)) != len(ports):
        return find_n_free_tcp_ports(n, host)
    return ports


__all__ = [
    "ColorMapEncoder",
    "colormap_to_ids_and_onehot",
    "DEFAULT_ENCODER",
    "find_free_tcp_port",
    "find_n_free_tcp_ports",
    "calculate_distance_map",
    "get_grid_pos",
]


def calculate_distance_map(id_map: np.ndarray) -> np.ndarray:
    wall_id = DEFAULT_ENCODER.name2id["WALL"]
    goal_id = DEFAULT_ENCODER.name2id["GOAL"]

    id_map[45:53, 114:130] = wall_id
    id_map[75:83, 120:144] = wall_id

    rows, cols = id_map.shape
    dist_map = np.full((rows, cols), -1, dtype=np.int32)
    queue = deque()
    goal_coords = np.argwhere(id_map == goal_id)

    for r, c in goal_coords:
        dist_map[r, c] = 0
        queue.append((r, c))

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while queue:
        curr_r, curr_c = queue.popleft()
        current_dist = dist_map[curr_r, curr_c]

        for dr, dc in directions:
            next_r, next_c = curr_r + dr, curr_c + dc

            if 0 <= next_r < rows and 0 <= next_c < cols:
                if dist_map[next_r, next_c] == -1 and id_map[next_r, next_c] != wall_id:
                    dist_map[next_r, next_c] = current_dist + 1
                    queue.append((next_r, next_c))

    return dist_map


_grid_pos_data = None


def get_grid_pos(player_x, player_y):
    global _grid_pos_data

    if _grid_pos_data is None:
        # 프로젝트 루트에서 log.txt 찾기
        import os

        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
        log_path = os.path.join(project_root, "log.txt")

        data = np.loadtxt(log_path, delimiter=",")
        slope_x, intercept_x = np.polyfit(data[:, 0], data[:, 2], 1)
        slope_y, intercept_y = np.polyfit(data[:, 1], data[:, 3], 1)
        _grid_pos_data = (slope_x, intercept_x, slope_y, intercept_y)

    slope_x, intercept_x, slope_y, intercept_y = _grid_pos_data
    gx = int(slope_x * player_x + intercept_x)
    gy = int(slope_y * player_y + intercept_y)
    return gx, gy
