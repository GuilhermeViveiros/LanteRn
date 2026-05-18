"""Tests for the Tetris synthetic benchmark (CPU, no model required)."""
from synthetic.Tetris.analogy_simulator import AnalogySimulator
from synthetic.Tetris.pieces import TETROMINOES, _normalize, _rotate_90cw, get_unique_rotations
from synthetic.Tetris.renderer import render_piece


def test_normalize_zeroes_origin():
    cells = [(2, 3), (3, 3), (4, 3)]
    result = _normalize(cells)
    rows = [r for r, _ in result]
    cols = [c for _, c in result]
    assert min(rows) == 0
    assert min(cols) == 0


def test_rotate_90cw_changes_shape():
    cells = [(0, 0), (0, 1), (0, 2)]  # horizontal bar
    rotated = _rotate_90cw(cells)
    assert set(map(tuple, rotated)) != set(map(tuple, cells))


def test_unique_rotations_square_has_one():
    square = [(0, 0), (0, 1), (1, 0), (1, 1)]
    rotations = get_unique_rotations(square)
    assert len(rotations) == 1


def test_unique_rotations_line_has_two():
    line = [(0, 0), (0, 1), (0, 2), (0, 3)]
    rotations = get_unique_rotations(line)
    assert len(rotations) == 2


def test_all_tetrominoes_defined():
    assert len(TETROMINOES) == 7
    for name, cells in TETROMINOES.items():
        assert len(cells) == 4, f"{name} should have 4 cells"


def test_render_piece_returns_image():
    from PIL import Image
    cells = [(0, 0), (0, 1), (1, 0), (1, 1)]
    img = render_piece(cells, cell_size=20)
    assert isinstance(img, Image.Image)
    assert img.width > 0 and img.height > 0


def test_analogy_simulator_generates_sample():
    sim = AnalogySimulator()
    sample = sim.sample()
    assert "question_img" in sample
    assert "answer" in sample
    assert sample["answer"] in ("A", "B", "C", "D")
