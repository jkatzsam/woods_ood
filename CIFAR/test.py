from sklearn.manifold import TSNE
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import pickle
import pandas as pd
from sklearn.metrics import det_curve, accuracy_score, roc_auc_score
from models.wrn import WideResNet
from models.wrn_ssnd import *
from make_datasets import *

''''
This file tests models on a set of outlier datasets. 

    Inputs: 
        --results_root: Directory containing results from the run
        --figs_root: Directory to save test results on outlier datasets (note: must exist prior to running)
        --use_test_score: Whether to use the best test score on the run to choose the best model from a particular run
        --use_last_epoch: Whether to use the model from the last epoch
        --use_clean_validation_score: Whether to use the validation score to choose the best model
    
    Outputs:
        test_min_fnr_last_epoch.csv: fnr results choosing by the best test fnr in the last epoch
        test_at_clean_valid_min_fnr_last_epoch.csv: fnr results choosing by the best clean validation fnr in the last epoch
        test_at_valid_min_fnr_last_epoch.csv: fnr results choosing by the best validation fnr in the last epoch
        
        test_min_fnr.csv: fnr results choosing by the best test fnr in the last epoch
        test_at_clean_valid_min_fnr.csv: fnr results choosing by the best clean validation fnr in the last epoch
        test_at_valid_min_fnr.csv: fnr results choosing by the best validation fnr in the last epoch    
        
        test_min_auroc_last_epoch.csv: auroc results choosing by the best test fnr in the last epoch
        test_at_clean_valid_min_auroc_last_epoch.csv: auroc results choosing by the best clean validation fnr in the last epoch
        test_at_valid_min_auroc_last_epoch.csv: auroc results choosing by the best validation fnr in the last epoch
        
        test_min_auroc.csv: auroc results choosing by the best test fnr in the last epoch
        test_at_clean_valid_min_auroc.csv: auroc results choosing by the best clean validation fnr in the last epoch
        test_at_valid_min_auroc.csv: auroc results choosing by the best validation fnr in the last epoch    
        
        test_acc_last_epoch.csv: acc results choosing by the best test fnr in the last epoch
        test_at_clean_valid_acc_last_epoch.csv: acc results choosing by the best clean validation fnr in the last epoch
        test_at_valid_acc_last_epoch.csv: acc results choosing by the best validation fnr in the last epoch
        
        test_acc.csv: acc results choosing by the best test fnr in the last epoch
        test_at_clean_valid_acc.csv: acc results choosing by the best clean validation fnr in the last epoch
        test_at_valid_acc.csv: acc results choosing by the best validation fnr in the last epoch    
'''

parser = argparse.ArgumentParser(description='Tunes a CIFAR Classifier with OE',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--use_test_score', '-b', type=int,
                    default=0, help='whether to use test score.')
parser.add_argument('--use_last_epoch', '-l', type=int,
                    default=0, help='whether to use scores from last epoch.')
parser.add_argument('--use_clean_validation_score', '-c', type=int,
                    default=0, help='whether to use clean validation score.')
parser.add_argument('--results_root', type=str,
                    default='results_ood',
                    help='Directory containing results.')
parser.add_argument('--figs_root', type=str,
                    default='figs_ood',
                    help='Directory to save compiled results.')
args = parser.parse_args()

#specify results root
results_root = args.results_root

#specify write root
figs_root = args.figs_root
if not os.path.exists(figs_root):
    os.makedirs(figs_root)

#if true, use the test score to judge model; otherwise, use the validation score
if args.use_test_score == 0:
    use_test_score = False
else:
    use_test_score = True

#if true, uses the clean validation score to judge model
if args.use_clean_validation_score == 0:
    use_clean_validation_score = False
else:
    use_clean_validation_score = True
    use_test_score = False #clean validation overrides test score

if args.use_last_epoch == 0:
    use_last_epoch = False
else:
    use_last_epoch = True

#datasets to test
ood_datasets = ['dtd', 'svhn', 'isun', 'lsun_r','lsun_c', 'places']

def to_np(x): return x.data.cpu().numpy()

def compute_fnr(out_scores, in_scores, fpr_cutoff=.05):
    '''
    compute fnr at 05
    '''

    in_labels = np.zeros(len(in_scores))
    out_labels = np.ones(len(out_scores))
    y_true = np.concatenate([in_labels, out_labels])
    y_score = np.concatenate([in_scores, out_scores])
    fpr, fnr, thresholds = det_curve(y_true=y_true, y_score=y_score)

    idx = np.argmin(np.abs(fpr - fpr_cutoff))

    fpr_at_fpr_cutoff = fpr[idx]
    fnr_at_fpr_cutoff = fnr[idx]

    if fpr_at_fpr_cutoff > 0.1:
        fnr_at_fpr_cutoff = 1.0

    return fnr_at_fpr_cutoff

def compute_auroc(out_scores, in_scores):

    in_labels = np.zeros(len(in_scores))
    out_labels = np.ones(len(out_scores))
    y_true = np.concatenate([in_labels, out_labels])
    y_score = np.concatenate([in_scores, out_scores])
    auroc = roc_auc_score(y_true=y_true, y_score=y_score)

    return auroc

def test(net, in_loader, out_loader, state):

    if state['dataset'] in ['cifar10']:
        num_classes = 10
    elif state['dataset'] in ['cifar100']:
        num_classes = 100

    net.eval()

    # in-distribution performance
    with torch.no_grad():

        accuracies = []
        OOD_scores_P0 = []

        for data, target in in_loader:

            if state['ngpu'] > 0:
                data, target = data.cuda(), target.cuda()

            # forward
            output = net(data)

            if state['score'] in ["woods_nn"]:

                # classification accuracy
                output_classification = output[:len(data), :num_classes]
                pred = output_classification.data.max(1)[1]
                accuracies.append(accuracy_score(list(to_np(pred)), list(to_np(target))))

                # OOD scores
                np_in = to_np(output[:, num_classes])
                np_in_list = list(np_in)
                OOD_scores_P0.extend(np_in_list)

            elif state['score'] in ['energy', 'energy_vos', 'woods']:

                # classification accuracy
                pred = output.data.max(1)[1]
                accuracies.append(accuracy_score(list(to_np(pred)), list(to_np(target))))

                # OOD scores
                OOD_scores_P0.extend(list(-to_np((state['T'] * torch.logsumexp(output / state['T'], dim=1)))))

            elif state['score'] == 'OE':

                # classification accuracy
                pred = output.data.max(1)[1]
                accuracies.append(accuracy_score(list(to_np(pred)), list(to_np(target))))

                # OOD scores
                smax = to_np(F.softmax(output, dim=1))
                OOD_scores_P0.extend(list(-np.max(smax, axis=1)))

    # OOD performance
    with torch.no_grad():

        OOD_scores_P_out = []

        for data, target in out_loader:

            if state['ngpu'] > 0:
                data, target = data.cuda(), target.cuda()

            output = net(data)

            if state['score'] in ["woods_nn"]:

                np_out = to_np(output[:, num_classes])
                np_out_list = list(np_out)
                OOD_scores_P_out.extend(np_out_list)

            elif state['score'] in ['energy', 'energy_vos', 'woods']:

                OOD_scores_P_out.extend(list(-to_np((state['T']*torch.logsumexp(output / state['T'], dim=1)))))

            elif state['score'] == 'OE':

                smax = to_np(F.softmax(output, dim=1))
                OOD_scores_P_out.extend(list(-np.max(smax, axis=1)))

    # compute FNR, AUROC, and accuracy
    fnr = compute_fnr(np.array(OOD_scores_P_out), np.array(OOD_scores_P0))
    auroc = compute_auroc(np.array(OOD_scores_P_out), np.array(OOD_scores_P0))
    acc = sum(accuracies) / len(accuracies)

    print('\tFNR = {}'.format(fnr))
    print('\tAUROC = {}'.format(auroc))
    print('\taccuracy = {}'.format(acc))

    return fnr, acc, auroc, OOD_scores_P0, OOD_scores_P_out

'''
collect best models
'''
#collect all of the files of the results
files = []
for _, dsets, _ in os.walk(results_root):
    print(dsets)
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

#collect best models by validation score
best_models = {}
for file in files:
    if not file.endswith('pkl'):
        continue

    print(file)

    with open(file, 'rb') as f:
        try:
            results = pickle.load(f)
        except EOFError:
            print('EOFError, skipping file')
            continue
        except pickle.UnpicklingError:
            print('UnpicklingError, skipping file')
            continue

    if results['score'] not in best_models:
        best_models[results['score']] = {}

    if results['dataset'] not in best_models[results['score']]:
        best_models[results['score']][results['dataset']] = {}

    if results['pi'] not in best_models[results['score']][results['dataset']]:
        best_models[results['score']][results['dataset']][results['pi']] = {}

        #score to judge model
        best_models[results['score']][results['dataset']][results['pi']]['best_score'] = 1

    #get score to judge model
    if use_last_epoch:
        if use_test_score:
            print("using test score from last epoch....")
            current_score = results['fnr_test'][-1]
        elif use_clean_validation_score:
            print("using clean validation score from last epoch....")
            current_score = results['fnr_valid_clean'][-1]
        else:
            print("using validation score from last epoch....")
            current_score = results['fnr_valid'][-1]

    else:
        if use_test_score:
            print("using min test score....")
            current_score = np.min(results['fnr_test'])
        elif use_clean_validation_score:
            print("using clean validation score....")
            current_score = np.min(results['fnr_valid_clean'])
        else:
            print("using min validation score....")
            current_score = np.min(results['fnr_valid'])

    if current_score < best_models[results['score']][results['dataset']][results['pi']]['best_score']:
        best_models[results['score']][results['dataset']][results['pi']]['best_score'] = current_score


        # best_models[results['score']][results['dataset']][results['pi']]['model_loc'] = results['model_loc']
        best_models[results['score']][results['dataset']][results['pi']]['results'] = results

results_best = []

with open(os.path.join(figs_root, 'results_test.csv'), 'w') as f:
    f.write("score, dataset, pi, ood_dataset, fnr, auroc, accuracy\n")

for score, score_dict in best_models.items():
        for dataset, dataset_dict in score_dict.items():

            for pi, pi_dict in dataset_dict.items():

                results = pi_dict['results']
                print("score {} best score {}".format(score, pi_dict['best_score']))
                # get score to judge model
                print("working on score {}...".format(score))
                if not use_last_epoch:
                    if use_test_score:
                        print("loading model from best test epoch...")
                        model_loc = results['model_loc_test']
                    elif use_clean_validation_score:
                        print("loading model from best clean validation epoch...")
                        model_loc = results['model_loc_valid_clean']
                    else:
                        print("loading model from best validation epoch...")
                        model_loc = results['model_loc_valid']

                else:
                    print('loading model from last epoch...')
                    model_loc = results['model_loc_last']

                #get ready to load in model
                if dataset in ['cifar10']:
                    num_classes = 10
                elif dataset in ['cifar100']:
                    num_classes = 100

                #load model
                if score in ['woods_nn']:
                    net = WideResNet(results['layers'],
                                     num_classes,
                                     results['widen_factor'],
                                     dropRate=results['droprate'])
                    net = WideResNet_SSND(net)


                elif score in ['OE', 'energy', 'energy_vos', 'woods']:
                    net = WideResNet(results['layers'],
                                     num_classes,
                                     results['widen_factor'],
                                     dropRate=results['droprate'])

                print('\tloading {}'.format(model_loc))

                net.load_state_dict(torch.load(model_loc))

                if results['ngpu'] > 0:
                    net.cuda()
                    torch.cuda.manual_seed(1)

                #test on odd_datasets
                for ood_dataset in ood_datasets:
                    print("testing {}".format(ood_dataset))

                    test_in_loader, test_out_loader = make_test_dataset(dataset, ood_dataset, results)

                    # load test data
                    test_in_loader, test_out_loader = make_test_dataset(results['dataset'], ood_dataset, results)

                    # test
                    print('\n\ttesting on {}...'.format(ood_dataset))
                    fnr, acc, auroc, OOD_scores_P0_test, OOD_scores_Ptest = test(net, test_in_loader, test_out_loader, results)


                    with open(os.path.join(figs_root, 'results_test.csv'), 'a') as f:
                        f.write("{}, {}, {}, {}, {}, {}, {}\n".format(score, dataset, pi, ood_dataset, fnr, auroc, acc))

                    results_dataset = results.copy()
                    results_dataset['ood_dataset'] = ood_dataset
                    results_dataset['fnr'] = fnr
                    results_dataset['auroc'] = auroc
                    results_dataset['acc'] = acc
                    results_best.append(results_dataset)

                #create pandas dictionary and save
                results_df = pd.DataFrame(results_best)
                results_save = results_df[['wandb_name', 'classification', 'pi', 'score', 'dataset', 'ood_dataset', 'aux_out_dataset', 'fnr', 'auroc', 'acc',
                                           'epoch', 'epochs',
                                           'false_alarm_cutoff', 'in_constraint_weight', 'out_constraint_weight',
                                           'lr_lam',
                                           'penalty_mult',
                                           'energy_vos_lambda',
                                           'oe_lambda',
                                           'train_in_size', 'valid_in_size',
                                           'test_in_size', 'test_out_size',
                                           'best_epoch_test', 'best_epoch_valid'
                                           ]]
                results_save = results_save.sort_values(by=['classification', 'pi', 'score', 'dataset', 'ood_dataset', 'false_alarm_cutoff', 'in_constraint_weight',
                                                            'out_constraint_weight', 'lr_lam', 'penalty_mult', 'energy_vos_lambda',
                                                            'oe_lambda']).reset_index(drop=True)
                results_save.to_csv(os.path.join(figs_root, 'results_all_test.csv'), index=False)

                #collapse into single row per ood dataset for fnr
                idx_list = ['dataset', 'ood_dataset', 'pi']
                idx_list_score = idx_list + ['score']
                scores = list(results_save['score'].unique())

                # min test FNR
                test_min_fnr = pd.pivot_table(data=results_save,
                                              values='fnr',
                                              index=idx_list,
                                              columns=['score'],
                                              aggfunc='min').reset_index().sort_values(idx_list)
                test_min_fnr = test_min_fnr[idx_list + scores]
                test_min_fnr[scores] = test_min_fnr[scores] * 100

                if use_last_epoch:
                    if use_test_score:
                        test_min_fnr.to_csv(os.path.join(figs_root, 'test_min_fnr_last_epoch.csv'), index=False)
                    elif use_clean_validation_score:
                        test_min_fnr.to_csv(os.path.join(figs_root, 'test_at_clean_valid_min_fnr_last_epoch.csv'), index=False)
                    else:
                        test_min_fnr.to_csv(os.path.join(figs_root, 'test_at_valid_min_fnr_last_epoch.csv'), index=False)
                else:
                    if use_test_score:
                        test_min_fnr.to_csv(os.path.join(figs_root, 'test_min_fnr.csv'), index=False)
                    elif use_clean_validation_score:
                        test_min_fnr.to_csv(os.path.join(figs_root, 'test_at_clean_valid_min_fnr.csv'), index=False)
                    else:
                        test_min_fnr.to_csv(os.path.join(figs_root, 'test_at_valid_min_fnr.csv'), index=False)


                #collapse into single row per ood dataset for auroc
                idx_list = ['pi', 'dataset', 'aux_out_dataset','ood_dataset']
                idx_list_score = idx_list + ['score']

                # auroc at min test FNR
                test_auroc = pd.pivot_table(data=results_save,
                                            values='auroc',
                                            index=idx_list,
                                            columns=['score']).reset_index().sort_values(idx_list)
                test_auroc = test_auroc[idx_list + scores]
                test_auroc[scores] = test_auroc[scores] * 100

                if use_last_epoch:
                    if use_test_score:
                        test_auroc.to_csv(os.path.join(figs_root, 'test_auroc_last_epoch.csv'), index=False)
                    elif use_clean_validation_score:
                        test_auroc.to_csv(
                            os.path.join(figs_root, 'test_at_clean_valid_min_auroc_last_epoch.csv'),
                            index=False)
                    else:
                        test_auroc.to_csv(os.path.join(figs_root, 'test_at_valid_min_auroc_last_epoch.csv'),
                                        index=False)
                else:
                    if use_test_score:
                        test_auroc.to_csv(os.path.join(figs_root, 'test_auroc.csv'), index=False)
                    elif use_clean_validation_score:
                        test_auroc.to_csv(os.path.join(figs_root, 'test_at_clean_valid_min_auroc.csv'),
                                        index=False)
                    else:
                        test_auroc.to_csv(os.path.join(figs_root, 'test_at_valid_min_auroc.csv'), index=False)

                # collapse into single row per ood dataset for acc
                idx_list = ['pi', 'dataset', 'aux_out_dataset', 'ood_dataset']
                idx_list_score = idx_list + ['score']

                # acc at min test FNR
                test_acc = pd.pivot_table(data=results_save,
                                              values='acc',
                                              index=idx_list,
                                              columns=['score']).reset_index().sort_values(idx_list)
                test_acc = test_acc[idx_list + scores]
                test_acc[scores] = test_acc[scores] * 100

                if use_last_epoch:
                    if use_test_score:
                        test_acc.to_csv(os.path.join(figs_root, 'test_acc_last_epoch.csv'), index=False)
                    elif use_clean_validation_score:
                        test_acc.to_csv(os.path.join(figs_root, 'test_at_clean_valid_min_acc_last_epoch.csv'), index=False)
                    else:
                        test_acc.to_csv(os.path.join(figs_root, 'test_at_valid_min_acc_last_epoch.csv'), index=False)
                else:
                    if use_test_score:
                        test_acc.to_csv(os.path.join(figs_root, 'test_acc.csv'), index=False)
                    elif use_clean_validation_score:
                        test_acc.to_csv(os.path.join(figs_root, 'test_at_clean_valid_min_acc.csv'), index=False)
                    else:
                        test_acc.to_csv(os.path.join(figs_root, 'test_at_valid_min_acc.csv'), index=False)


