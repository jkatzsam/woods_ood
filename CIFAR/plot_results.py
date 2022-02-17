import pandas as pd
import os
import pickle
import numpy as np
from sklearn.metrics import det_curve, roc_auc_score

''''
This file plots results from a set of algorithm runs.

    Specify at the top of the file the two variables:
        results_root: Directory containing results from the run
        figs_root: Directory to save test results on outlier datasets
'''

#change these strings for plotting
results_root = 'results_ssnd_10'
figs_root = 'figs_ssnd_10'


def test_fnr_using_valid_threshold(valid_scores_in, valid_scores_out, test_scores_in, test_scores_out, fpr_cutoff=0.05):
    valid_labels_in = np.zeros(len(valid_scores_in))
    valid_labels_out = np.ones(len(valid_scores_out))
    y_true_valid = np.concatenate([valid_labels_in, valid_labels_out])
    y_score_valid = np.concatenate([valid_scores_in, valid_scores_out])

    fpr, fnr, thresholds = det_curve(y_true=y_true_valid, y_score=y_score_valid)

    idx = np.argmin(np.abs(fpr - fpr_cutoff))
    t = thresholds[idx]

    fpr_test = len(np.array(test_scores_in)[np.array(test_scores_in) >= t]) / len(test_scores_in)
    fnr_test = len(np.array(test_scores_out)[np.array(test_scores_out) < t]) / len(test_scores_out)

    return fnr_test


def compute_auroc(out_scores, in_scores):

    in_labels = np.zeros(len(in_scores))
    out_labels = np.ones(len(out_scores))
    y_true = np.concatenate([in_labels, out_labels])
    y_score = np.concatenate([in_scores, out_scores])
    auroc = roc_auc_score(y_true=y_true, y_score=y_score)

    return auroc


def load_results(results_root, figs_root):
    files = []
    valid_fnr = []
    test_fnr = []
    scores = []
    for _, dsets, _ in os.walk(results_root):
        for dset in dsets:
            dset_dir = os.path.join(results_root, dset)
            for _, out_dsets, _ in os.walk(dset_dir):
                for out_dset in out_dsets:
                    out_dset_dir = os.path.join(dset_dir, out_dset)
                    for _, scores, _ in os.walk(out_dset_dir):
                        for score in scores:
                            score_dir = os.path.join(out_dset_dir, score)
                            for f in os.listdir(score_dir):
                                f_path = os.path.join(score_dir, f)
                                files.append(f_path)

    results_all = []
    num_files = len(files)
    count = 0

    for file in files:
        if not file.endswith('pkl'):
            continue

        count += 1

        # make corresponding fig directory for OOD score plots
        d = os.path.normpath(file).split(os.path.sep)
        fig_dir = os.path.join(figs_root, *d[1:-1])
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)

        with open(file, 'rb') as f:
            print('[{}/{}] loading {}'.format(count, num_files, file))
            try:
                results = pickle.load(f)
            except EOFError:
                print('EOFError, skipping file')
                continue
            except pickle.UnpicklingError:
                print('UnpicklingError, skipping file')
                continue

            # results from best validation epoch
            e = results['best_epoch_valid']
            # results['fnr_train_best_epoch_valid'] = results['fnr_train'][e]
            results['min_valid_fnr'] = results['fnr_valid'][e]
            # results['fnr_test_best_epoch_valid'] = results['fnr_test'][e]
            results['train_acc_best_epoch_valid'] = results['train_accuracy'][e]
            results['valid_acc_best_epoch_valid'] = results['valid_accuracy'][e]
            results['test_acc_best_epoch_valid'] = results['test_accuracy'][e]

            results['fnr_test_best_epoch_valid'] = test_fnr_using_valid_threshold(results['OOD_scores_P0_valid'][e],
                                                                                  results['OOD_scores_PX_valid'][e],
                                                                                  results['OOD_scores_P0_test'][e],
                                                                                  results['OOD_scores_Ptest'][e])
            results['test_auroc_best_epoch_valid'] = compute_auroc(in_scores=results['OOD_scores_P0_test'][e],
                                                                   out_scores=results['OOD_scores_Ptest'][e])

            # results from best clean validation epoch
            results['min_valid_fnr_clean'] = min(results['fnr_valid_clean'])
            e = results['fnr_valid_clean'].index(results['min_valid_fnr_clean'])
            results['best_epoch_valid_clean'] = e
            results['train_acc_best_epoch_valid_clean'] = results['train_accuracy'][e]
            results['valid_acc_best_epoch_valid_clean'] = results['valid_accuracy'][e]
            results['test_acc_best_epoch_valid_clean'] = results['test_accuracy'][e]

            results['fnr_test_best_epoch_valid_clean'] = test_fnr_using_valid_threshold(results['OOD_scores_P0_valid_clean'][e],
                                                                                  results['OOD_scores_PX_valid_clean'][e],
                                                                                  results['OOD_scores_P0_test'][e],
                                                                                  results['OOD_scores_Ptest'][e])
            # results['fnr_test_best_epoch_valid_clean'] = results['fnr_test'][e]
            results['test_auroc_best_epoch_valid_clean'] = compute_auroc(in_scores=results['OOD_scores_P0_test'][e],
                                                                         out_scores=results['OOD_scores_Ptest'][e])

            # results from best test epoch
            e = results['best_epoch_test']
            results['fnr_valid_best_epoch_test'] = results['fnr_valid'][e]
            results['min_test_fnr'] = results['fnr_test'][e]
            results['train_acc_best_epoch_test'] = results['train_accuracy'][e]
            results['valid_acc_best_epoch_test'] = results['valid_accuracy'][e]
            results['test_acc_best_epoch_test'] = results['test_accuracy'][e]
            results['test_auroc_best_epoch_test'] = compute_auroc(in_scores=results['OOD_scores_P0_test'][e],
                                                                  out_scores=results['OOD_scores_Ptest'][e])

            # results from last epoch
            e = results['epoch']
            results['fnr_valid_last_epoch'] = results['fnr_valid'][e]
            results['fnr_test_last_epoch'] = results['fnr_test'][e]
            results['train_acc_last_epoch'] = results['train_accuracy'][e]
            results['valid_acc_last_epoch'] = results['valid_accuracy'][e]
            results['test_acc_last_epoch'] = results['test_accuracy'][e]
            results['test_auroc_last_epoch'] = compute_auroc(in_scores=results['OOD_scores_P0_test'][e],
                                                             out_scores=results['OOD_scores_Ptest'][e])

            # for plotting valid vs test FNR
            valid_fnr.extend(results['fnr_valid'])
            test_fnr.extend(results['fnr_test'])
            scores.extend([results['pi']] * len(results['fnr_test']))

            # save results
            results_all.append(results)

    return results_all, valid_fnr, test_fnr, scores


def save_all_results(results_all, figs_root):
    results_df = pd.DataFrame(results_all)
    results_save = results_df[['wandb_name',
                               'classification', 'pi', 'score', 'dataset', 'aux_out_dataset', 'test_out_dataset',
                               'epoch', 'epochs',
                               'false_alarm_cutoff', 'in_constraint_weight', 'out_constraint_weight', 'penalty_mult',
                               'lr_lam',
                               'oe_lambda',
                               'energy_vos_lambda',
                               'train_in_size', # 'train_out_size',
                               'valid_in_size', # 'valid_out_size',
                               'test_in_size', 'test_out_size',
                               'best_epoch_valid', 'min_valid_fnr', 'fnr_test_best_epoch_valid',
                               'train_acc_best_epoch_valid', 'valid_acc_best_epoch_valid', 'test_acc_best_epoch_valid',
                               'best_epoch_valid_clean', 'min_valid_fnr_clean', 'fnr_test_best_epoch_valid_clean',
                               'train_acc_best_epoch_valid_clean', 'valid_acc_best_epoch_valid_clean', 'test_acc_best_epoch_valid_clean',
                               'best_epoch_test', 'min_test_fnr', 'fnr_valid_best_epoch_test',
                               'train_acc_best_epoch_test', 'valid_acc_best_epoch_test', 'test_acc_best_epoch_test',
                               'fnr_valid_last_epoch', 'fnr_test_last_epoch', 'train_acc_last_epoch',
                               'valid_acc_last_epoch', 'test_acc_last_epoch',
                               'test_auroc_best_epoch_valid', 'test_auroc_best_epoch_valid_clean',
                               'test_auroc_best_epoch_test', 'test_auroc_last_epoch'
                               ]]
    results_save = results_save.sort_values(by=['classification', 'pi', 'score', 'dataset', 'aux_out_dataset',
                                                'test_out_dataset', 'false_alarm_cutoff', 'in_constraint_weight',
                                                'out_constraint_weight', 'lr_lam', 'energy_vos_lambda',
                                                'oe_lambda']).reset_index(drop=True)
    results_save.to_csv(os.path.join(figs_root, 'results_all.csv'), index=False)
    return results_save


def save_min_fnr_results(results_save, figs_root):

    idx_list = ['dataset', 'test_out_dataset', 'pi']
    idx_list_score = idx_list + ['score']
    scores = list(pd.unique(results_save['score']))

    # min test FNR
    test_min_fnr = pd.pivot_table(data=results_save,
                                  values='min_test_fnr',
                                  index=idx_list,
                                  columns=['score'],
                                  aggfunc='min').reset_index().sort_values(idx_list)
    test_min_fnr = test_min_fnr[idx_list + scores]
    test_min_fnr[scores] = test_min_fnr[scores] * 100
    test_min_fnr.to_csv(os.path.join(figs_root, 'test_min_fnr.csv'), index=False)

    # test auroc at min test FNR
    test_min_fnr_idx = results_save.groupby(by=idx_list_score)['min_test_fnr'].idxmin()
    results_test_min_fnr = results_save.iloc[test_min_fnr_idx]
    test_auroc_at_test_min_fnr = pd.pivot_table(data=results_test_min_fnr,
                                              values='test_auroc_best_epoch_test',
                                              index=idx_list,
                                              columns='score',
                                              aggfunc='mean').reset_index().sort_values(idx_list)
    test_auroc_at_test_min_fnr = test_auroc_at_test_min_fnr[idx_list + scores]
    test_auroc_at_test_min_fnr[scores] = test_auroc_at_test_min_fnr[scores] * 100
    test_auroc_at_test_min_fnr.to_csv(os.path.join(figs_root, 'test_auroc_at_test_min_fnr.csv'), index=False)

    # test accuracy at min test FNR
    test_acc_at_test_min_fnr = pd.pivot_table(data=results_test_min_fnr,
                                              values='test_acc_best_epoch_test',
                                              index=idx_list,
                                              columns='score',
                                              aggfunc='mean').reset_index().sort_values(idx_list)
    test_acc_at_test_min_fnr = test_acc_at_test_min_fnr[idx_list + scores]
    test_acc_at_test_min_fnr[scores] = test_acc_at_test_min_fnr[scores] * 100
    test_acc_at_test_min_fnr.to_csv(os.path.join(figs_root, 'test_acc_at_test_min_fnr.csv'), index=False)

    # test auroc at min test FNR
    test_auroc_at_test_min_fnr = pd.pivot_table(data=results_test_min_fnr,
                                              values='test_auroc_best_epoch_test',
                                              index=idx_list,
                                              columns='score',
                                              aggfunc='mean').reset_index().sort_values(idx_list)
    test_auroc_at_test_min_fnr = test_auroc_at_test_min_fnr[idx_list + scores]
    test_auroc_at_test_min_fnr[scores] = test_auroc_at_test_min_fnr[scores] * 100
    test_auroc_at_test_min_fnr.to_csv(os.path.join(figs_root, 'test_auroc_at_test_min_fnr.csv'), index=False)

    # test FNR at min valid FNR
    valid_min_fnr_idx = results_save.groupby(by=idx_list_score)['min_valid_fnr'].idxmin()
    results_valid_min_fnr = results_save.iloc[valid_min_fnr_idx]
    test_fnr_at_valid_min_fnr = pd.pivot_table(data=results_valid_min_fnr,
                                               values='fnr_test_best_epoch_valid',
                                               index=idx_list,
                                               columns='score',
                                               aggfunc='mean').reset_index().sort_values(idx_list)
    test_fnr_at_valid_min_fnr = test_fnr_at_valid_min_fnr[idx_list + scores]
    test_fnr_at_valid_min_fnr[scores] = test_fnr_at_valid_min_fnr[scores] * 100
    test_fnr_at_valid_min_fnr.to_csv(os.path.join(figs_root, 'test_fnr_at_valid_min_fnr.csv'), index=False)

    # test auroc at min valid FNR
    test_auroc_at_valid_min_fnr = pd.pivot_table(data=results_valid_min_fnr,
                                               values='test_auroc_best_epoch_valid',
                                               index=idx_list,
                                               columns='score',
                                               aggfunc='mean').reset_index().sort_values(idx_list)
    test_auroc_at_valid_min_fnr = test_auroc_at_valid_min_fnr[idx_list + scores]
    test_auroc_at_valid_min_fnr[scores] = test_auroc_at_valid_min_fnr[scores] * 100
    test_auroc_at_valid_min_fnr.to_csv(os.path.join(figs_root, 'test_auroc_at_valid_min_fnr.csv'), index=False)

    # test acc at min valid FNR
    test_acc_at_valid_min_fnr = pd.pivot_table(data=results_valid_min_fnr,
                                               values='test_acc_best_epoch_valid',
                                               index=idx_list,
                                               columns='score',
                                               aggfunc='mean').reset_index().sort_values(idx_list)
    test_acc_at_valid_min_fnr = test_acc_at_valid_min_fnr[idx_list + scores]
    test_acc_at_valid_min_fnr[scores] = test_acc_at_valid_min_fnr[scores] * 100
    test_acc_at_valid_min_fnr.to_csv(os.path.join(figs_root, 'test_acc_at_valid_min_fnr.csv'), index=False)

    return test_min_fnr, test_acc_at_test_min_fnr, test_fnr_at_valid_min_fnr, test_acc_at_valid_min_fnr


def save_min_fnr_results_valid_clean(results_save, figs_root):

    idx_list = ['dataset', 'test_out_dataset', 'pi']
    idx_list_score = idx_list + ['score']
    scores = list(pd.unique(results_save['score']))

    # test FNR at min valid FNR
    valid_min_fnr_idx = results_save.groupby(by=idx_list_score)['min_valid_fnr_clean'].idxmin()
    results_valid_min_fnr = results_save.iloc[valid_min_fnr_idx]
    test_fnr_at_valid_min_fnr = pd.pivot_table(data=results_valid_min_fnr,
                                               values='fnr_test_best_epoch_valid_clean',
                                               index=idx_list,
                                               columns='score',
                                               aggfunc='mean').reset_index().sort_values(idx_list)
    test_fnr_at_valid_min_fnr = test_fnr_at_valid_min_fnr[idx_list + scores]
    test_fnr_at_valid_min_fnr[scores] = test_fnr_at_valid_min_fnr[scores] * 100
    test_fnr_at_valid_min_fnr.to_csv(os.path.join(figs_root, 'test_fnr_at_valid_min_fnr_clean.csv'), index=False)

    # test auroc at min valid FNR
    test_auroc_at_valid_min_fnr = pd.pivot_table(data=results_valid_min_fnr,
                                               values='test_auroc_best_epoch_valid_clean',
                                               index=idx_list,
                                               columns='score',
                                               aggfunc='mean').reset_index().sort_values(idx_list)
    test_auroc_at_valid_min_fnr = test_auroc_at_valid_min_fnr[idx_list + scores]
    test_auroc_at_valid_min_fnr[scores] = test_auroc_at_valid_min_fnr[scores] * 100
    test_auroc_at_valid_min_fnr.to_csv(os.path.join(figs_root, 'test_auroc_at_valid_min_fnr_clean.csv'), index=False)

    # test acc at min valid FNR
    test_acc_at_valid_min_fnr = pd.pivot_table(data=results_valid_min_fnr,
                                               values='test_acc_best_epoch_valid_clean',
                                               index=idx_list,
                                               columns='score',
                                               aggfunc='mean').reset_index().sort_values(idx_list)
    test_acc_at_valid_min_fnr = test_acc_at_valid_min_fnr[idx_list + scores]
    test_acc_at_valid_min_fnr[scores] = test_acc_at_valid_min_fnr[scores] * 100
    test_acc_at_valid_min_fnr.to_csv(os.path.join(figs_root, 'test_acc_at_valid_min_fnr_clean.csv'), index=False)


def save_results_last_epoch(results_save, figs_root):

    idx_list = ['dataset', 'test_out_dataset', 'pi']
    idx_list_score = idx_list + ['score']
    scores = list(pd.unique(results_save['score']))

    # min test FNR
    test_min_fnr = pd.pivot_table(data=results_save,
                                  values='fnr_test_last_epoch',
                                  index=idx_list,
                                  columns=['score'],
                                  aggfunc='min').reset_index().sort_values(idx_list)
    test_min_fnr = test_min_fnr[idx_list + scores]
    test_min_fnr[scores] = test_min_fnr[scores] * 100
    test_min_fnr.to_csv(os.path.join(figs_root, 'test_min_fnr_last_epoch.csv'), index=False)

    # test accuracy at min test FNR
    test_min_fnr_idx = results_save.groupby(by=idx_list_score)['fnr_test_last_epoch'].idxmin()
    results_test_min_fnr = results_save.iloc[test_min_fnr_idx]
    test_acc_at_test_min_fnr = pd.pivot_table(data=results_test_min_fnr,
                                              values='test_acc_last_epoch',
                                              index=idx_list,
                                              columns='score',
                                              aggfunc='mean').reset_index().sort_values(idx_list)
    test_acc_at_test_min_fnr = test_acc_at_test_min_fnr[idx_list + scores]
    test_acc_at_test_min_fnr[scores] = test_acc_at_test_min_fnr[scores] * 100
    test_acc_at_test_min_fnr.to_csv(os.path.join(figs_root, 'test_acc_at_test_min_fnr_last_epoch.csv'), index=False)

    # test FNR at min valid FNR
    valid_min_fnr_idx = results_save.groupby(by=idx_list_score)['fnr_valid_last_epoch'].idxmin()
    results_valid_min_fnr = results_save.iloc[valid_min_fnr_idx]
    test_fnr_at_valid_min_fnr = pd.pivot_table(data=results_valid_min_fnr,
                                               values='fnr_test_last_epoch',
                                               index=idx_list,
                                               columns='score',
                                               aggfunc='mean').reset_index().sort_values(idx_list)
    test_fnr_at_valid_min_fnr = test_fnr_at_valid_min_fnr[idx_list + scores]
    test_fnr_at_valid_min_fnr[scores] = test_fnr_at_valid_min_fnr[scores] * 100
    test_fnr_at_valid_min_fnr.to_csv(os.path.join(figs_root, 'test_fnr_at_valid_min_fnr_last_epoch.csv'), index=False)

    # test acc at min valid FNR
    test_acc_at_valid_min_fnr = pd.pivot_table(data=results_valid_min_fnr,
                                               values='test_acc_last_epoch',
                                               index=idx_list,
                                               columns='score',
                                               aggfunc='mean').reset_index().sort_values(idx_list)
    test_acc_at_valid_min_fnr = test_acc_at_valid_min_fnr[idx_list + scores]
    test_acc_at_valid_min_fnr[scores] = test_acc_at_valid_min_fnr[scores] * 100
    test_acc_at_valid_min_fnr.to_csv(os.path.join(figs_root, 'test_acc_at_valid_min_fnr_last_epoch.csv'), index=False)

    return test_min_fnr, test_acc_at_test_min_fnr, test_fnr_at_valid_min_fnr, test_acc_at_valid_min_fnr


def plot_single_dir():

    # load results from .pkl files
    results_all, _, _, _ = load_results(results_root, figs_root)

    # save all results to df/csv
    results_save = save_all_results(results_all, figs_root)
    results_save = results_save.replace('dtd', 'textures')

    # save FNR and accuracy results
    save_min_fnr_results(results_save, figs_root)
    save_min_fnr_results_valid_clean(results_save, figs_root)


def main():
    plot_single_dir()


if __name__ == '__main__':
    main()