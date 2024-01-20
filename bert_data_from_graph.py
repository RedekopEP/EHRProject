import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Train BERT')
parser.add_argument('--path_to_graph_pt', type=str,
                    default='/opt/genomics/workingdir/zichenwang/share/graph_dataset_clean_quarter_prediction4_window20_simplified.pt')
parser.add_argument('--path_to_save_label_df', type=str, default=None)
parser.add_argument('--path_to_codes_vocab', type=str,
                    default='/opt/genomics/workingdir/zichenwang/share/depression_code_vocab/')
parser.add_argument('--path_to_save_train_df', type=str,
                    default='/opt/genomics/workingdir/eredekop/bert_depr_dataset/data_from_graph_12mo_train.pkl')

args = parser.parse_args()

if __name__ == '__main__':
    data = torch.load(args.path_to_graph_pt)

    df_diag = pd.read_csv(
        '{0}/Diagnosis_codes_short_min_pts_400_clean.csv'.format(args.path_to_codes_vocab))
    df_med = pd.read_csv(
        '{0}/Medication_codes_short_min_pts_400_clean.csv'.format(args.path_to_codes_vocab))
    df_proc = pd.read_csv(
        '{0}/Procedure_codes_min_pts_400_clean.csv'.format(args.path_to_codes_vocab))

    dx = data[('diagnosis', 'rev_dx', 'patients')]
    rx = data[('medication', 'rev_rx', 'patients')]
    pt = data[('procedure', 'rev_pt', 'patients')]

    dx_edge_index = dx['edge_index']  # first row is item node index, second row is patient node index
    dx_edge_time = dx['edge_time']  # number of quarters
    dx_edge_count = dx['edge_count']  # number of edge counts

    rx_edge_index = rx['edge_index']  # first row is item node index, second row is patient node index
    rx_edge_time = rx['edge_time']  # number of quarters
    rx_edge_count = rx['edge_count']  # number of edge counts

    pt_edge_index = pt['edge_index']  # first row is item node index, second row is patient node index
    pt_edge_time = pt['edge_time']  # number of quarters
    pt_edge_count = pt['edge_count']  # number of edge counts

    patients = data["patients"]
    genders = list(patients['sex'].numpy())  # ['idx'].unique()
    split_mask = list(patients['dataset_split'].numpy())
    labels = list(patients['depression_label'].numpy())

    label_df = pd.DataFrame({'ID': [0], 'label': [0]})  # no topic
    for patient_idx in tqdm(range(len(labels))):
        label = labels[patient_idx]
        pn_data = pd.DataFrame({'ID': [patient_idx], 'label': [label]})
        label_df = pd.concat([label_df, pn_data], axis=0)

    label_df.to_pickle(
        args.path_to_save_label_df)

    # extract individual patient history
    from tqdm import tqdm

    count = 0
    df_train = pd.DataFrame(
        {'ID': [0], 'code': [['a', 'b', 'SEP']], 'age': [[0, 0, 0]], 'gender': [[0, 0, 0]]})  # no topic
    df_val = pd.DataFrame(
        {'ID': [0], 'code': [['a', 'b', 'SEP']], 'age': [[0, 0, 0]], 'gender': [[0, 0, 0]]})  # no topic
    df_test = pd.DataFrame(
        {'ID': [0], 'code': [['a', 'b', 'SEP']], 'age': [[0, 0, 0]], 'gender': [[0, 0, 0]]})  # no topic
    for patient_idx in tqdm(range(len(labels))):
        patient_mask = dx_edge_index[1] == patient_idx
        patient_dx_idx = dx_edge_index[0][patient_mask].numpy()
        patient_dx_time = dx_edge_time[patient_mask].numpy()
        patient_dx_count = dx_edge_count[patient_mask].numpy()

        patient_mask = rx_edge_index[1] == patient_idx
        patient_rx_idx = rx_edge_index[0][patient_mask].numpy()
        patient_rx_time = rx_edge_time[patient_mask].numpy()
        patient_rx_count = rx_edge_count[patient_mask].numpy()

        patient_mask = pt_edge_index[1] == patient_idx
        patient_pt_idx = pt_edge_index[0][patient_mask].numpy()
        patient_pt_time = pt_edge_time[patient_mask].numpy()
        patient_pt_count = pt_edge_count[patient_mask].numpy()

        all_quarters = list(set(patient_dx_time).union(patient_rx_time, patient_pt_time))

        sequence = []
        age_code = []
        gender = genders[patient_idx]
        for quarter in np.unique(all_quarters):
            mask_patient_dx_time = patient_dx_time == quarter
            codes = patient_dx_idx[mask_patient_dx_time]
            for item in codes:
                sequence.append('d' + str(df_diag[df_diag['index'] == item]['code_short'].iloc[0]))
            mask_patient_rx_time = patient_rx_time == quarter
            codes = patient_rx_idx[mask_patient_rx_time]  # .astype('str')
            for item in codes:
                sequence.append('m' + str(df_med[df_med['index'] == item]['code_short'].iloc[0]))
            #             sequence.extend(patient_rx_idx[mask_patient_rx_time].astype('str'))
            mask_patient_pt_time = patient_pt_time == quarter
            #             sequence.extend(patient_pt_idx[mask_patient_pt_time].astype('str'))
            codes = patient_pt_idx[mask_patient_pt_time]  # .astype('str')
            for item in codes:
                sequence.append('p' + str(df_proc[df_proc['index'] == item]['code'].iloc[0]))
            sequence.append('SEP')
            age_code.extend([int(quarter)] * (len(patient_dx_idx[mask_patient_dx_time])))
            age_code.extend([int(quarter)] * (len(patient_rx_idx[mask_patient_rx_time])))
            age_code.extend([int(quarter)] * (len(patient_pt_idx[mask_patient_pt_time]) + 1))
        if len(sequence) > 0:
            pn_data = pd.DataFrame(
                {'ID': [patient_idx], 'code': [sequence], 'age': [age_code], 'gender': [[gender] * len(sequence)]})
            if split_mask[patient_idx] == 0:
                df_train = pd.concat([df_train, pn_data], axis=0)
            elif split_mask[patient_idx] == 1:
                df_val = pd.concat([df_val, pn_data], axis=0)
            else:
                df_test = pd.concat([df_test, pn_data], axis=0)

    df_train = df_train.reset_index(drop=True)
    df_train = df_train.drop(df_train.index[0])
    df_train.to_pickle(args.path_to_save_train_df)

    df_val = df_val.reset_index(drop=True)
    df_val = df_val.drop(df_val.index[0])
    df_val.to_pickle(args.path_to_save_train_df.replace('train', 'val'))

    df_test = df_test.reset_index(drop=True)
    df_test = df_test.drop(df_test.index[0])
    df_test.to_pickle(args.path_to_save_train_df.replace('train', 'test'))
