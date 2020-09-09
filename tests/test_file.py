import os
import pytest
from pyult import file
from pathlib import Path

TEST_ROOT = os.path.dirname(__file__)

def test_mainpart ():
    teststrs = ['xxx_Track0', 'xxx_Track01', 'xxxUS', 'xxx_corrected']
    cleaned = [ file.mainpart(i) for i in teststrs ]
    assert len(set(cleaned))==1

def test_check_wdir ():
    path = os.path.join(TEST_ROOT, 'resources/sample_recording')
    assert file.check_wdir(path, verbose=False)

def test_unique_target_stems ():
    path = os.path.join(TEST_ROOT, 'resources/sample_recording')
    paths = file.unique_target_stems(path)
    cond1 = len(set(paths))==1
    cond2 = paths[0]=='sample'
    assert all([cond1, cond2])

def test_find_target_file ():
    wdir = os.path.join(TEST_ROOT, 'resources/sample_recording')
    stem = 'sample'
    extension = '\\.wav$'
    path = file.find_target_file(wdir, stem, extension)
    assert Path(path).exists()

