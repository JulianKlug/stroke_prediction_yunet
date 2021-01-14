import argparse
import h5py
import os, json
import numpy as np
from tqdm import tqdm


def convert_gsd_to_hdf5(path_to_gsd_dataset: str, path_to_new_dataset_dir: str):
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

    #  change axis order (as used in yunet)
    ct_inputs = ct_inputs.transpose(0, 3, 1, 2, 4)
    ct_lesion_GT = ct_lesion_GT.transpose(0, 3, 1, 2)
    brain_masks = brain_masks.transpose(0, 3, 1, 2)

    for subj_idx, id in enumerate(tqdm(ids)):
        subj_path = os.path.join(path_to_new_dataset_dir, id)
        os.mkdir(subj_path)

        hf = h5py.File(os.path.join(subj_path, 'inputs_aug0.hdf5'), 'w')
        hf.create_dataset('init', data=ct_inputs[subj_idx])
        hf.close()

        hf = h5py.File(os.path.join(subj_path, 'output_aug0.hdf5'), 'w')
        hf.create_dataset('init', data=ct_lesion_GT[subj_idx])
        hf.close()

        hf = h5py.File(os.path.join(subj_path, 'mask_aug0.hdf5'), 'w')
        hf.create_dataset('init', data=brain_masks[subj_idx])
        hf.close()

    with open(os.path.join(path_to_new_dataset_dir, 'params.json'), 'w') as fp:
        json.dump(params, fp, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Geneva Stroke Dataset into HDF5 format')
    parser.add_argument('dataset_path')
    parser.add_argument('-o', '--outdir',  help='Name of output directory', required=True, default=None)

    args = parser.parse_args()
    convert_gsd_to_hdf5(args.dataset_path, args.outdir)