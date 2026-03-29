"""
Label utilities for PhysioNet CT-ICH dataset.
Loads slice-level CSV labels and derives patient-level binary hemorrhage labels.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_patient_labels(csv_path):
    """
    Load hemorrhage_diagnosis_raw_ct.csv and derive patient-level binary labels.

    For each patient: if ANY slice has No_Hemorrhage == 0, label = 1 (hemorrhage present).
    If ALL slices have No_Hemorrhage == 1, label = 0 (no hemorrhage).

    Returns:
        dict[int, int]: {patient_id: label}
    """
    df = pd.read_csv(csv_path, encoding='utf-8-sig')

    # The first column may have BOM artifacts -- normalize name
    cols = df.columns.tolist()
    cols[0] = 'PatientNumber'
    df.columns = cols

    labels = {}
    for pid, group in df.groupby('PatientNumber'):
        pid = int(pid)
        # If any slice has No_Hemorrhage == 0, the patient has hemorrhage
        has_hemorrhage = int((group['No_Hemorrhage'] == 0).any())
        labels[pid] = has_hemorrhage

    return labels


def get_available_patients(ct_dir):
    """
    List all .nii files in ct_scans/ and extract integer patient IDs.

    Returns:
        list[int]: Sorted list of patient IDs
    """
    ct_path = Path(ct_dir) / 'ct_scans'
    nii_files = sorted(ct_path.glob('*.nii'))
    patient_ids = []
    for f in nii_files:
        try:
            pid = int(f.stem)
            patient_ids.append(pid)
        except ValueError:
            continue
    return sorted(patient_ids)


def get_aligned_labels(csv_path, ct_dir):
    """
    Return aligned arrays of patient IDs and labels for patients that have
    both a .nii scan file AND a CSV label entry.

    Returns:
        tuple[np.ndarray, np.ndarray]: (patient_ids, labels)
    """
    all_labels = load_patient_labels(csv_path)
    available = get_available_patients(ct_dir)

    # Only keep patients present in both
    aligned_ids = [pid for pid in available if pid in all_labels]
    aligned_labels = [all_labels[pid] for pid in aligned_ids]

    ids_arr = np.array(aligned_ids, dtype=np.int64)
    labels_arr = np.array(aligned_labels, dtype=np.int64)

    n_pos = int(labels_arr.sum())
    n_neg = len(labels_arr) - n_pos
    print(f"Labels loaded: {len(labels_arr)} patients ({n_pos} hemorrhage, {n_neg} no hemorrhage)")

    return ids_arr, labels_arr


if __name__ == '__main__':
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else '/mnt/c/Users/mehme/Desktop/ct-scan-test'
    csv_path = str(Path(data_dir) / 'hemorrhage_diagnosis_raw_ct.csv')
    ids, labels = get_aligned_labels(csv_path, data_dir)
    print(f"Patient IDs: {ids}")
    print(f"Labels:      {labels}")
