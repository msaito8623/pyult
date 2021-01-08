import os
import pytest
from pyult import file
from pathlib import Path

TEST_ROOT = Path(__file__).parent

def test_mainpart ():
    teststrs = ['xxx_Track0', 'xxx_Track01', 'xxxUS', 'xxx_corrected']
    cleaned = [ file.mainpart(i) for i in teststrs ]
    assert len(set(cleaned))==1

def test_check_wdir ():
    path = str(TEST_ROOT / 'resources/sample_recording')
    assert file.check_wdir(path, verbose=False)

def test_unique_target_stems ():
    path = str(TEST_ROOT / 'resources/sample_recording')
    paths = file.unique_target_stems(path)
    assert len(set(paths))==2
    assert all([ i==j for i,j in zip(paths, ['sample_01','sample_02']) ])

def test_find_target_file ():
    wdir = str(TEST_ROOT / 'resources/sample_recording')
    stem = 'sample_02'
    extension = '\\.wav$'
    path = file.find_target_file(wdir, stem, extension)
    assert Path(path).exists()

