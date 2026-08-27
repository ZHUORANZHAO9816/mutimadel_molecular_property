import numpy as np


def load_data_for_pretrain_1(pretrain_data_path='./data/pretrain_data/CHEMBL_wash_500_pretrain'):
    tokens_idx_list = []
    global_labels_list = []
    atom_labels_list = []
    atom_mask_list = []
    smile_list= []
    n_files = 0
    # Count available files first
    import glob
    pattern = pretrain_data_path + '_*.npy'
    n_files = len(glob.glob(pattern))
    if n_files == 0:
        raise FileNotFoundError(f"No pretrain data files found at: {pattern}")
    for i in range(n_files):
        pretrain_data = np.load(pretrain_data_path+'_{}.npy'.format(i+1), allow_pickle=True)
        tokens_idx_list = tokens_idx_list + [x for x in pretrain_data[0]]
        global_labels_list = global_labels_list + [x for x in pretrain_data[1]]
        atom_labels_list = atom_labels_list + [x for x in pretrain_data[2]]
        atom_mask_list = atom_mask_list + [x for x in pretrain_data[3]]
        smile_list = smile_list + [x for x in pretrain_data[4]]
        print(pretrain_data_path+'_{}.npy'.format(i+1) + ' is loaded')
    pretrain_data_final = []
    for i in range(len(tokens_idx_list)):
        a_pretrain_data = [tokens_idx_list[i], global_labels_list[i], atom_labels_list[i], atom_mask_list[i], smile_list[i]]
        pretrain_data_final.append(a_pretrain_data)
    return pretrain_data_final

def load_data_for_pretrain_2(pretrain_data_path='./data/pretrain_data/CHEMBL_wash_500_pretrain'):
    """
    Load the second half of pretraining data files.
    """
    tokens_idx_list = []
    tokens_idx_list = []
    global_labels_list = []
    atom_labels_list = []
    atom_mask_list = []
    smile_list= []
    import glob
    pattern = pretrain_data_path + '_*.npy'
    n_files = len(glob.glob(pattern))
    # load second half (offset by first half count)
    for i in range(n_files // 2, n_files):
        pretrain_data = np.load(pretrain_data_path+'_{}.npy'.format(i+1), allow_pickle=True)
        tokens_idx_list = tokens_idx_list + [x for x in pretrain_data[0]]
        global_labels_list = global_labels_list + [x for x in pretrain_data[1]]
        atom_labels_list = atom_labels_list + [x for x in pretrain_data[2]]
        atom_mask_list = atom_mask_list + [x for x in pretrain_data[3]]
        smile_list = smile_list + [x for x in pretrain_data[4]]
        print(pretrain_data_path+'_{}.npy'.format(i+1) + ' is loaded')
    pretrain_data_final = []
    for i in range(len(tokens_idx_list)):
        a_pretrain_data = [tokens_idx_list[i], global_labels_list[i], atom_labels_list[i], atom_mask_list[i], smile_list[i]]
        pretrain_data_final.append(a_pretrain_data)
    return pretrain_data_final