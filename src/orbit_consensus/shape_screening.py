from __future__ import annotations

from dataclasses import dataclass

from .schema import SupportPair

Shape = tuple[int, int]


@dataclass(frozen=True)
class ShapeScreen:
    allowed_shapes: frozenset[Shape]
    relations: tuple[str, ...]


def infer_shape_screen(support_pairs: tuple[SupportPair, ...], query_shape: Shape) -> ShapeScreen:
    """Infer only dimension relations shared by every support pair."""
    if not support_pairs:
        return ShapeScreen(frozenset(), ())

    allowed: set[Shape] = set()
    relations: list[str] = []
    input_shapes = [pair.input_grid.shape for pair in support_pairs]
    output_shapes = [pair.output_grid.shape for pair in support_pairs]

    if all(input_shape == output_shape for input_shape, output_shape in zip(input_shapes, output_shapes)):
        allowed.add(query_shape)
        relations.append("identity")

    if all(shape == output_shapes[0] for shape in output_shapes):
        allowed.add((int(output_shapes[0][0]), int(output_shapes[0][1])))
        relations.append("constant_output")

    scales: list[Shape] = []
    for input_shape, output_shape in zip(input_shapes, output_shapes):
        if output_shape[0] % input_shape[0] or output_shape[1] % input_shape[1]:
            scales = []
            break
        scales.append((output_shape[0] // input_shape[0], output_shape[1] // input_shape[1]))
    if scales and all(scale == scales[0] for scale in scales) and scales[0] != (1, 1):
        allowed.add((query_shape[0] * scales[0][0], query_shape[1] * scales[0][1]))
        relations.append(f"integer_scale_{scales[0][0]}x{scales[0][1]}")

    valid = frozenset(shape for shape in allowed if 1 <= shape[0] <= 30 and 1 <= shape[1] <= 30)
    return ShapeScreen(valid, tuple(relations))
