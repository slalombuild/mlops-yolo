from scripts.utilities import rename_files_in_dir,convert_mp4_to_jpg, count_files_in_dir, get_random_sample, update_yaml
import os
import cv2
from unittest.mock import MagicMock, patch
import pytest
import yaml
import tempfile


def test_rename_files_in_dir(tmp_path):
    # Create some files with a specific text in the name
    file1 = tmp_path / "file1_text_to_remove.txt"
    file1.write_text("content1")
    file2 = tmp_path / "file2_text_to_remove.txt"
    file2.write_text("content2")

    # Call the function to rename files in the directory
    txt_to_remove = "_text_to_remove"
    txt_to_insert = "_text_to_insert"
    rename_files_in_dir(tmp_path, txt_to_remove, txt_to_insert)

    # Check that the files have been renamed correctly
    assert (tmp_path / f"file1{txt_to_insert}.txt").is_file()
    assert (tmp_path / f"file1{txt_to_insert}.txt").read_text() == "content1"
    assert not (tmp_path / "file1_text_to_remove.txt").exists()
    assert (tmp_path / f"file2{txt_to_insert}.txt").is_file()
    assert (tmp_path / f"file2{txt_to_insert}.txt").read_text() == "content2"
    assert not (tmp_path / "file2_text_to_remove.txt").exists()


def test_count_files_in_dir(tmp_path):
    # Create some files with different extensions in a temporary directory
    file_paths = [
        tmp_path / "file1.jpg",
        tmp_path / "file2.JPG",
        tmp_path / "file3.png",
        tmp_path / "file4.mp4",
    ]
    for path in file_paths:
        path.touch()

    # Test the function with different arguments
    assert count_files_in_dir(tmp_path) == 4  # Count all files
    assert count_files_in_dir(tmp_path, [".jpg", ".png"]) == 3
    assert count_files_in_dir(tmp_path, [".jpeg"]) == 0


@pytest.fixture
def test_dirs(tmpdir):
    # create temporary read and write directories for testing
    read_dir = tmpdir.mkdir("read")
    write_dir = tmpdir.mkdir("write")
    # create some test files in the read directory
    for i in range(10):
        with open(os.path.join(read_dir, f"test_file_{i}.txt"), "w") as f:
            f.write("test content")
    return read_dir, write_dir

def test_get_random_sample(test_dirs):
    # unpack test directories from fixture
    read_dir, write_dir = test_dirs
    # call function to get random sample
    get_random_sample(read_path=str(read_dir), write_path=str(write_dir),sub_sample=0.5)
    # check that files were copied to write directory
    assert len(os.listdir(write_dir)) > 0

def test_update_yaml():
    # create temporary yaml file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_yaml_file:
        temp_yaml_path = temp_yaml_file.name
        temp_yaml_file.write('foo: bar\n')
        temp_yaml_file.write('baz: qux\n')
    # call function to update yaml
    update_yaml(temp_yaml_path, {'foo': 'baz', 'quux': 'corge'})
    # read updated yaml file and check if it is correct
    with open(temp_yaml_path, 'r') as updated_yaml_file:
        updated_yaml = yaml.safe_load(updated_yaml_file)
        assert updated_yaml == {'foo': 'baz', 'baz': 'qux', 'quux': 'corge'}
    # remove temporary yaml file
    os.remove(temp_yaml_path)