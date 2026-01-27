from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import mne
from mne.datasets import eegbci
from mne.io import read_raw_edf

def load_raw_subject_runs(subject_id: int, runs: List[int], l_freq: float, h_freq: float,
                          resample_sfreq: Optional[float] = None) -> mne.io.BaseRaw:
    files = eegbci.load_data(subject_id, runs)
    raws = [read_raw_edf(f, preload=True, stim_channel='auto', verbose=False) for f in files]
    raw = mne.concatenate_raws(raws, verbose=False)

    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)

    if resample_sfreq is not None:
        raw.resample(resample_sfreq, verbose=False)

    return raw

def raw_to_epochs_xy(raw: mne.io.BaseRaw, tmin: float, tmax: float, picks: str = "eeg"
                     ) -> Tuple[np.ndarray, np.ndarray, mne.Info]:
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    # EEGBCI motor imagery: older examples used 'T1'/'T2' for left/right, newer use 'left'/'right'.
    if "left" in event_id and "right" in event_id:
        left_key, right_key = "left", "right"
    elif "T1" in event_id and "T2" in event_id:
        left_key, right_key = "T1", "T2"
    else:
        raise ValueError(f"Missing left/right (or T1/T2) in event_id keys: {list(event_id.keys())}")

    epochs = mne.Epochs(
        raw,
        events,
        event_id={"left": event_id[left_key], "right": event_id[right_key]},
        tmin=tmin, tmax=tmax, baseline=None, preload=True,
        picks=picks, verbose=False
    )
    X = epochs.get_data().astype(np.float32)  # (trials, chans, times)
    y_event = epochs.events[:, -1]
    # Map left->0, right->1 using the resolved right_key
    y = (y_event == event_id[right_key]).astype(np.int64)
    return X, y, epochs.info

def load_subject_train_test(subject_id: int,
                            train_runs: List[int],
                            test_runs: List[int],
                            l_freq: float, h_freq: float,
                            tmin: float, tmax: float,
                            resample_sfreq: Optional[float] = None,
                            picks: str = "eeg") -> Dict[str, Any]:
    raw_train = load_raw_subject_runs(subject_id, train_runs, l_freq, h_freq, resample_sfreq)
    raw_test = load_raw_subject_runs(subject_id, test_runs, l_freq, h_freq, resample_sfreq)

    return {
        "raw_train": raw_train,
        "raw_test": raw_test,
        "subject": subject_id,
        "meta": {"train_runs": train_runs, "test_runs": test_runs, "l_freq": l_freq, "h_freq": h_freq}
    }

def load_subject_all_runs(subject_id: int,
                          runs: List[int],
                          l_freq: float, h_freq: float,
                          tmin: float, tmax: float,
                          resample_sfreq: Optional[float] = None,
                          picks: str = "eeg") -> Dict[str, Any]:
    raw = load_raw_subject_runs(subject_id, runs, l_freq, h_freq, resample_sfreq)
    X, y, info = raw_to_epochs_xy(raw, tmin, tmax, picks=picks)
    return {"X": X, "y": y, "info": info, "subject": subject_id}
