from __future__ import annotations

from sleep_stager.data.raw import _subject_id_from_record
from sleep_stager.data.splits import (
    SplitConfig,
    subject_wise_loso_split,
    subject_wise_kfold_splits,
    subject_wise_split,
    unique_subject_ids,
)


def test_subject_wise_split_has_no_overlap():
    subject_ids = ["S1", "S1", "S2", "S2", "S3", "S3", "S4", "S4", "S5", "S5"]
    cfg = SplitConfig(train_ratio=0.6, val_ratio=0.2, seed=123)
    splits = subject_wise_split(subject_ids, cfg)
    train = set(splits["train"])
    val = set(splits["val"])
    test = set(splits["test"])
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert train | val | test == set(unique_subject_ids(subject_ids))


def test_subject_id_strips_night_suffix():
    assert _subject_id_from_record("ST0001J0") == "ST0001J"
    assert _subject_id_from_record("ST0001J1") == "ST0001J"


def test_loso_split_keeps_nights_together():
    record_ids = [
        "ST0001J0",
        "ST0001J1",
        "ST0002J0",
        "ST0002J1",
        "ST0003J0",
        "ST0003J1",
    ]
    subject_ids = [_subject_id_from_record(record_id) for record_id in record_ids]
    splits, held_out = subject_wise_loso_split(subject_ids, val_ratio=0.2, seed=7)
    assert held_out in splits["test"]
    assert held_out not in splits["train"]
    assert set(splits["train"]).isdisjoint(splits["test"])
    for subject in set(subject_ids):
        in_train = subject in splits["train"]
        in_test = subject in splits["test"]
        assert not (in_train and in_test)


def test_kfold_keeps_subject_nights_together():
    record_ids = [
        "ST0001J0",
        "ST0001J1",
        "ST0002J0",
        "ST0002J1",
        "ST0003J0",
        "ST0003J1",
        "ST0004J0",
        "ST0004J1",
        "ST0005J0",
        "ST0005J1",
    ]
    subject_ids = [record_id[:-1] for record_id in record_ids]
    splits = subject_wise_kfold_splits(subject_ids, k=5, seed=7, val_ratio=0.2)
    subject_to_fold = {}
    for fold_idx, split in enumerate(splits):
        train = set(split["train"])
        val = set(split["val"])
        test = set(split["test"])
        assert train.isdisjoint(val)
        assert train.isdisjoint(test)
        assert val.isdisjoint(test)
        for subject in test:
            assert subject not in subject_to_fold
            subject_to_fold[subject] = fold_idx
    for subject in set(subject_ids):
        folds = {
            subject_to_fold[subject]
            for record_id in record_ids
            if record_id[:-1] == subject
        }
        assert len(folds) == 1
