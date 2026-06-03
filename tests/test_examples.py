"""The committed example photos must blind-solve correctly — always-on coverage
of the still-image path against data that ships in the repo."""

import os

from starguide import solve_image

IMAGES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "examples", "image", "input")


def _named(sky):
    return {s.name for s, *_ in sky.matches if not s.name.startswith(("HIP", "HR "))}


def test_orion_example_identifies_the_belt():
    sky = solve_image(os.path.join(IMAGES, "orion.jpg"), verbose=False)
    assert sky.solution.n_inliers >= 30
    names = _named(sky)
    # Orion's Belt + a shoulder/foot — unmistakable if the solve is right.
    assert {"Alnilam", "Alnitak", "Mintaka"} <= names
    assert "Betelgeuse" in names or "Rigel" in names


def test_bootes_example_identifies_arcturus():
    sky = solve_image(os.path.join(IMAGES, "bootes.jpg"), verbose=False)
    assert sky.solution.n_inliers >= 20
    assert "Arcturus" in _named(sky)
