import numpy as np
import pytest

from permutect import constants
from permutect.data import memory_mapped_data as memory_mapped_data_module
from permutect.data.datum import FLOAT_DTYPE
from permutect.data.datum import Data
from permutect.data.memory_mapped_data import MemoryMappedData
from permutect.data.memory_mapped_data import StackedData
from permutect.test.data.test_memory_mapped_data import make_datum
from permutect.tools import edit_dataset
from permutect.tools.edit_dataset import EditType
from permutect.utils.enums import Label


def test_edit_type_must_be_one_of_the_choices(capsys):
    argv = ["--" + constants.TRAIN_TAR_NAME, "data.tar", "--" + constants.OUTPUT_NAME, "out.tar"]
    for edit_type in EditType:
        args = edit_dataset.parse_arguments(argv + ["--" + constants.DATASET_EDIT_TYPE_NAME, edit_type.value])
        assert getattr(args, constants.DATASET_EDIT_TYPE_NAME) == edit_type.value

    # a typo is an error when parsing arguments, not after loading every dataset
    with pytest.raises(SystemExit):
        edit_dataset.parse_arguments(argv + ["--" + constants.DATASET_EDIT_TYPE_NAME, "unlabel_artifact"])
    assert "invalid choice" in capsys.readouterr().err


def run_edit_dataset(tmp_path, tarfiles: list[str], edit_type: EditType, source: int = None) -> MemoryMappedData:
    output = str(tmp_path / "edited.tar")
    argv = ["--" + constants.TRAIN_TAR_NAME, *tarfiles, "--" + constants.OUTPUT_NAME, output]
    argv += ["--" + constants.DATASET_EDIT_TYPE_NAME, edit_type.value]
    argv += [] if source is None else ["--" + constants.SOURCE_NAME, str(source)]
    edit_dataset.main_without_parsing(edit_dataset.parse_arguments(argv))
    return MemoryMappedData.load_from_tarfile(output)


@pytest.mark.parametrize("source", [None, 3])
@pytest.mark.parametrize("edit_type", list(EditType))
def test_edits(tmp_path, monkeypatch, edit_type, source):
    # data of every label, some with no reads, in two datasets that are edited four data at a time
    monkeypatch.setattr(edit_dataset, "DATA_PER_EDIT", 4)
    labels = [Label.ARTIFACT, Label.VARIANT, Label.UNLABELED] * 4
    data = [make_datum(index, num_reads=index % 5, label=label) for index, label in enumerate(labels)]
    tarfiles = []
    for n, dataset in enumerate((data[:7], data[7:])):
        tarfiles.append(str(tmp_path / f"dataset{n}.tar"))
        num_reads = sum(len(datum.reads_re) for datum in dataset)
        MemoryMappedData.from_generator(dataset, len(dataset), num_reads).save_to_tarfile(tarfiles[-1])
    edited = run_edit_dataset(tmp_path, tarfiles, edit_type, source)

    # as each datum was edited before the edits were vectorized
    removed = {EditType.REMOVE_ARTIFACTS: Label.ARTIFACT, EditType.REMOVE_VARIANTS: Label.VARIANT}.get(edit_type)
    unlabeled = {EditType.UNLABEL_ARTIFACTS: Label.ARTIFACT, EditType.UNLABEL_VARIANTS: Label.VARIANT}.get(edit_type)
    expected = [
        (
            datum.get(Data.POSITION),
            Label.UNLABELED if edit_type == EditType.UNLABEL_EVERYTHING or label == unlabeled else label,
            datum.get(Data.SOURCE) if source is None else source,
            datum.reads_re.tobytes(),
        )
        for datum, label in zip(data, labels)
        if label != removed
    ]
    edited_data = [
        (datum.get(Data.POSITION), datum.get(Data.LABEL), datum.get(Data.SOURCE), datum.reads_re.tobytes())
        for datum in edited.generate()
    ]
    assert edited_data == expected


def test_float64_float_arrays_become_float16(tmp_path):
    # as a Datum made them float16 when data were edited one at a time, eg those that preprocess_dataset saved as float64
    data = [make_datum(index, num_reads=index % 3) for index in range(10)]
    stacked = StackedData.of(data)
    floats_ve = np.random.default_rng(0).normal(size=stacked.float_array_ve.shape)
    tarfile = str(tmp_path / "float64.tar")
    num_reads = len(stacked.reads_re)
    MemoryMappedData(stacked.int_array_ve, floats_ve, len(data), stacked.reads_re, num_reads).save_to_tarfile(tarfile)

    edited = run_edit_dataset(tmp_path, [tarfile], EditType.KEEP_EVERYTHING)
    assert edited.float_mmap.dtype == FLOAT_DTYPE
    assert edited.float_mmap.tobytes() == floats_ve.astype(FLOAT_DTYPE).tobytes()
    assert edited.int_mmap.tobytes() == stacked.int_array_ve.tobytes()
    assert edited.reads_mmap.tobytes() == stacked.reads_re.tobytes()


def test_edited_data_are_written_straight_into_the_tarfile(tmp_path, monkeypatch):
    # rather than into temporary memory maps first, which doubled the bytes written
    labels = [Label.ARTIFACT, Label.VARIANT, Label.UNLABELED] * 3
    data = [make_datum(index, num_reads=index % 4, label=label) for index, label in enumerate(labels)]
    tarfile, num_reads = str(tmp_path / "dataset.tar"), sum(len(datum.reads_re) for datum in data)
    MemoryMappedData.from_generator(data, len(data), num_reads).save_to_tarfile(tarfile)

    def no_temporary_files(*args, **kwargs):
        raise AssertionError("edit_dataset made a temporary file")

    monkeypatch.setattr(memory_mapped_data_module, "NamedTemporaryFile", no_temporary_files)
    edited = run_edit_dataset(tmp_path, [tarfile], EditType.REMOVE_ARTIFACTS)
    assert [datum.get(Data.POSITION) for datum in edited.generate()] == [1, 2, 4, 5, 7, 8]


def test_datasets_must_have_the_same_widths(tmp_path):
    tarfiles = []
    for n, num_info in enumerate((3, 4)):
        stacked = StackedData.of([make_datum(index, num_reads=2) for index in range(3)])
        floats_ve = np.zeros((3, Data.NUM_SCALAR_FLOAT_ELEMENTS + num_info), dtype=FLOAT_DTYPE)
        tarfiles.append(str(tmp_path / f"dataset{n}.tar"))
        MemoryMappedData(stacked.int_array_ve, floats_ve, 3, stacked.reads_re, 6).save_to_tarfile(tarfiles[-1])
    with pytest.raises(ValueError):
        run_edit_dataset(tmp_path, tarfiles, EditType.KEEP_EVERYTHING)
