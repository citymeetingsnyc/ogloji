import os
import time

import pytest

from ogloji import evict_from_local_storage


@pytest.fixture
def storage_with_files(tmp_path):
    # Create test files with different access times
    files = {
        "old.png": (1000000, time.time() - 3600),  # 1MB, accessed 1 hour ago
        "medium.png": (2000000, time.time() - 1800),  # 2MB, accessed 30 mins ago
        "new.png": (1500000, time.time()),  # 1.5MB, accessed now
    }

    # Create the files
    for name, (size, atime) in files.items():
        path = tmp_path / name
        with open(path, "wb") as f:
            f.write(b"0" * size)
        os.utime(path, (atime, atime))

    return tmp_path


def test_no_eviction_needed(storage_with_files):
    # Test eviction with 4.5MB capacity (should keep all files)
    evicted = evict_from_local_storage(4.5, str(storage_with_files))
    assert len(evicted) == 0
    assert len(list(storage_with_files.glob("*.png"))) == 3


def test_evict_oldest_file(storage_with_files):
    # Test eviction with 3MB capacity (should evict oldest file)
    evicted = evict_from_local_storage(3.5, str(storage_with_files))
    assert len(evicted) == 1
    assert os.path.basename(evicted[0]) == "old.png"
    assert len(list(storage_with_files.glob("*.png"))) == 2
    assert not (storage_with_files / "old.png").exists()
    assert (storage_with_files / "medium.png").exists()
    assert (storage_with_files / "new.png").exists()


def test_evict_multiple_files(storage_with_files):
    # Test eviction with 1MB capacity (should evict all but newest)
    evicted = evict_from_local_storage(1.5, str(storage_with_files))
    assert len(evicted) == 2
    assert set(os.path.basename(p) for p in evicted) == {"old.png", "medium.png"}
    assert len(list(storage_with_files.glob("*.png"))) == 1
    assert (storage_with_files / "new.png").exists()


def test_evict_from_local_storage_empty_dir(tmp_path):
    # Test with empty directory
    evicted = evict_from_local_storage(1.0, str(tmp_path))
    assert len(evicted) == 0


def test_evict_from_local_storage_updates_on_access(tmp_path):
    # Create two test files
    files = {
        "old.png": (1000000, time.time() - 3600),
        "new.png": (1000000, time.time() - 1800),
    }

    for name, (size, atime) in files.items():
        path = tmp_path / name
        with open(path, "wb") as f:
            f.write(b"0" * size)
        os.utime(path, (atime, atime))

    # Update atime of old file to now
    old_path = tmp_path / "old.png"
    current_time = time.time()
    os.utime(old_path, (current_time, current_time))

    # Now evict with capacity that only allows one file
    evicted = evict_from_local_storage(1.0, str(tmp_path))
    assert len(evicted) == 1
    assert os.path.basename(evicted[0]) == "new.png"
    assert (tmp_path / "old.png").exists()
