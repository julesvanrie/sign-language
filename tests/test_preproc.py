from signlanguage.preproc import read_frame
from pyarrow.parquet import read_table
import numpy as np

def test_has_correct_shape():
    testfile = 'tests/test.parquet'
    pqtable = read_table(testfile)
    reshaped = read_frame(testfile)
    length = int(len(pqtable['frame']) / (1629/3))
    assert reshaped.shape == [length, 1629]


def get_example():
    testfile = 'tests/test.parquet'
    pqtable = read_table(testfile)
    x = pqtable['x']
    y = pqtable['y']
    z = pqtable['z']
    reshaped = read_frame(testfile)
    return x, y, z, reshaped


def test_first_frame_face_landmark_0():
    x, y, z, reshaped = get_example()
    orig = [float(reshaped[0,0].numpy()),
            float(reshaped[0,1].numpy()),
            float(reshaped[0,2].numpy()),]
    new = [x[0].as_py(),
           y[0].as_py(),
           z[0].as_py(),]
    assert np.allclose(orig, new, equal_nan=True)


def test_first_frame_left_hand_landmark_0():
    x, y, z, reshaped = get_example()

    # First frame, left hand, landmark 0
    orig = [float(reshaped[0,1404+0].numpy()),
            float(reshaped[0,1404+1].numpy()),
            float(reshaped[0,1404+2].numpy()),]
    new = [i if i is not None else float('nan') for i in
            [x[468].as_py(),
             y[468].as_py(),
             z[468].as_py(),]]
    assert np.allclose(orig, new, equal_nan=True)


def test_first_frame_pose_landmark_0():
    x, y, z, reshaped = get_example()
    # First frame, pose, landmark 0
    orig = [float(reshaped[0,1467+0].numpy()),
            float(reshaped[0,1467+1].numpy()),
            float(reshaped[0,1467+2].numpy()),]
    new = [i if i is not None else float('nan') for i in
            [x[489].as_py(),
             y[489].as_py(),
             z[489].as_py(),]]
    assert np.allclose(orig, new, equal_nan=True)


def test_first_frame_right_hand_landmark_0():
    x, y, z, reshaped = get_example()
    # First frame, right hand, landmark 0
    orig = [float(reshaped[0,1566+0].numpy()),
            float(reshaped[0,1566+1].numpy()),
            float(reshaped[0,1566+2].numpy()),]
    new = [i if i is not None else float('nan') for i in
            [x[522].as_py(),
             y[522].as_py(),
             z[522].as_py(),]]
    assert np.allclose(orig, new, equal_nan=True)


def test_second_frame_face_landmark_0_1_2_x():
    x, y, z, reshaped = get_example()
    orig = [float(reshaped[1,0].numpy()),
            float(reshaped[1,3].numpy()),
            float(reshaped[1,6].numpy()),]
    new = [i if i is not None else float('nan') for i in
            [x[543+0].as_py(),
             x[543+1].as_py(),
             x[543+2].as_py(),]]
    assert np.allclose(orig, new, equal_nan=True)
