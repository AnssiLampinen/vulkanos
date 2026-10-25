import numpy as np

from vulkanos.dataset import IGNORE, POSITIVE_ONLY, chip_labels, edge_band, split_of


def test_split_is_deterministic_and_roughly_sized():
    cells = [f"cell{i}" for i in range(2000)]
    splits = [split_of(c, 0.2) for c in cells]
    assert splits == [split_of(c, 0.2) for c in cells]
    assert 0.15 < splits.count("val") / len(cells) < 0.25


def test_edge_band_covers_both_sides_of_outline():
    b = np.zeros((20, 20), bool)
    b[5:15, 5:15] = True
    band = edge_band(b, 1)
    assert band[5, 10] and band[4, 10]  # inside and outside the edge
    assert not band[10, 10]  # building interior
    assert not band[0, 0]  # far background


def test_chip_labels_split_and_ignore():
    buildings = np.zeros((8, 8), bool)
    buildings[1:3, 1:3] = True
    valid = np.ones((8, 8), bool)
    valid[7, 7] = False
    pixel_split = np.full((8, 8), "", dtype=object)
    pixel_split[:, :4] = "train"
    pixel_split[:, 4:6] = "val"

    labels = chip_labels(buildings, valid, pixel_split, ignore_edge_px=0)
    train, val = labels["train"], labels["val"]
    assert (train[:, 4:] == IGNORE).all()  # val cells and unlabelled cells ignored
    assert (train[1:3, 1:3] == 1).all()
    assert train[5, 0] == 0
    assert (val[:, :4] == IGNORE).all() and (val[:, 6:] == IGNORE).all()
    assert (val[:, 4:6] == 0).all()
    assert train[7, 7] == IGNORE and val[7, 7] == IGNORE


def test_positive_only_cells_add_buildings_to_train_only():
    buildings = np.zeros((8, 8), bool)
    buildings[1:3, 1:3] = True  # in a complete train cell
    buildings[1:3, 5:7] = True  # in a not-complete cell
    valid = np.ones((8, 8), bool)
    pixel_split = np.full((8, 8), "", dtype=object)
    pixel_split[:, :4] = "train"
    pixel_split[:, 4:] = POSITIVE_ONLY

    labels = chip_labels(buildings, valid, pixel_split, ignore_edge_px=0)
    train, val = labels["train"], labels["val"]
    assert (train[1:3, 5:7] == 1).all()  # buildings from the other cell are positives
    assert train[5, 5] == IGNORE  # its background is not trusted
    assert train[5, 0] == 0  # complete-cell background unchanged
    assert (val == IGNORE).all()  # never in validation


def test_without_positive_only_cells_labels_are_unchanged():
    rng = np.random.default_rng(0)
    buildings = rng.random((16, 16)) > 0.7
    valid = rng.random((16, 16)) > 0.1
    pixel_split = rng.choice(np.array(["train", "val", ""], dtype=object), size=(16, 16))
    labels = chip_labels(buildings, valid, pixel_split, ignore_edge_px=1)
    unusable = ~valid | edge_band(buildings, 1)
    for split in ("train", "val"):
        expected = buildings.astype(np.uint8)
        expected[unusable | (pixel_split != split)] = IGNORE
        assert (labels[split] == expected).all()
