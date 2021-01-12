import os, json
import nibabel as nib
import numpy as np

def convert_gsd_to_nifti_dataset(path_to_gsd_dataset: str, path_to_new_dataset_dir: str):
    params = np.load(path_to_gsd_dataset, allow_pickle=True)['params'].item()
    ids = np.load(path_to_gsd_dataset, allow_pickle=True)['ids']
    ct_inputs = np.load(path_to_gsd_dataset, allow_pickle=True)['ct_inputs']
    try:
        ct_lesion_GT = np.load(path_to_gsd_dataset, allow_pickle=True)['ct_lesion_GT']
    except:
        ct_lesion_GT = np.load(path_to_gsd_dataset, allow_pickle=True)['lesion_GT']

    brain_masks = np.load(path_to_gsd_dataset, allow_pickle=True)['brain_masks']

    print('Loading a total of', ct_inputs.shape[0], 'subjects.')
    print('Sequences used:', params)

    if not os.path.exists(path_to_new_dataset_dir):
        os.mkdir(path_to_new_dataset_dir)

    with open(os.path.join(path_to_new_dataset_dir, 'params.json'), 'w') as fp:
        json.dump(params, fp, indent=4)

    for subj_idx, id in enumerate(ids):
        subj_path = os.path.join(path_to_new_dataset_dir, id)
        if not os.path.exists(subj_path):
            os.mkdir(subj_path)

        subj_pct_img = nib.Nifti1Image(ct_inputs[subj_idx], np.eye(4))
        subj_lesion_img = nib.Nifti1Image(ct_lesion_GT[subj_idx], np.eye(4))
        subj_mask_img = nib.Nifti1Image(brain_masks[subj_idx].astype(int), np.eye(4))

        nib.save(subj_pct_img, os.path.join(subj_path, 'PCT.nii'))
        nib.save(subj_lesion_img, os.path.join(subj_path, 'LESION.nii'))
        nib.save(subj_mask_img, os.path.join(subj_path, 'MASK.nii'))