"""The crop planner frames the head as the rules ask, or says why no 35x45 box fits inside the photo."""
from edutap.image_api.core.crop import ASPECT, FACE_CENTRE_RANGE, HEAD_RANGE, MAX_OFFSET, plan_crop


def _face(centre_x, centre_y, size):
    return {'x': centre_x - size / 2, 'y': centre_y - size / 2, 'width': size, 'height': size}


def _framing(box, bbox, chin_y, crown_y):
    # What check_face_size and check_face_centered would measure on the cropped photo
    head_ratio = (chin_y - crown_y) / box.height
    centre_x = bbox['x'] + bbox['width'] / 2 - box.x
    centre_y = bbox['y'] + bbox['height'] / 2 - box.y
    return head_ratio, abs(centre_x / box.width - 0.5), centre_y / box.height


def test_square_photo_keeps_its_height_and_loses_the_sides():
    # Like tests/test-images/9.jpg: the head already fills 74% of the height
    bbox = _face(1015, 959, 1046)
    box, reason = plan_crop(2000, 2000, bbox, chin_y=1508, crown_y=36)
    assert reason is None
    assert (box.width, box.height) == (1556, 2000) and box.y == 0
    head, offset, centre = _framing(box, bbox, 1508, 36)
    assert abs(head - 0.736) < 0.001 and offset < 0.01 and FACE_CENTRE_RANGE[0] <= centre <= FACE_CENTRE_RANGE[1]


def test_small_head_is_cropped_in_to_the_allowed_size():
    bbox = _face(1000, 1000, 500)
    box, reason = plan_crop(2000, 2000, bbox, chin_y=1300, crown_y=700)
    assert reason is None
    assert abs(box.width / box.height - ASPECT) < 0.005
    head, offset, centre = _framing(box, bbox, 1300, 700)
    assert HEAD_RANGE[0] - 0.005 <= head <= HEAD_RANGE[1] + 0.005
    assert offset <= MAX_OFFSET + 0.005 and abs(centre - 0.60) < 0.01


def test_off_centre_face_is_centred():
    bbox = _face(1300, 1000, 400)
    box, _ = plan_crop(2000, 2000, bbox, chin_y=1250, crown_y=730)
    _, offset, _ = _framing(box, bbox, 1250, 730)
    assert offset < 0.005


def test_face_at_the_edge_cannot_be_cropped_without_adding_background():
    box, reason = plan_crop(2000, 2000, _face(150, 1000, 300), chin_y=1200, crown_y=800)
    assert box is None and 'left edge' in reason


def test_head_that_fills_the_photo_cannot_be_cropped():
    box, reason = plan_crop(1000, 800, _face(500, 400, 600), chin_y=790, crown_y=10)
    assert box is None and 'fills too much' in reason


def test_unmeasured_head_gives_a_reason():
    box, reason = plan_crop(1000, 1000, _face(500, 500, 300), chin_y=400, crown_y=400)
    assert box is None and reason
