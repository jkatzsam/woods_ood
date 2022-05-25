# -*- coding: utf-8 -*-
from sklearn.metrics import det_curve, accuracy_score, roc_auc_score
from make_datasets_new import *
from models.wrn_ssnd import *

import wandb

if __package__ is None:
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

'''
This code implements training and testing functions. 
'''


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser(description='Tunes a CIFAR Classifier with OE',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('dataset', type=str, choices=['cifar10', 'cifar100'],
                    default='cifar10',
                    help='Choose between CIFAR-10, CIFAR-100.')
parser.add_argument('--model', '-m', type=str, default='allconv',
                    choices=['allconv', 'wrn', 'densenet'], help='Choose architecture.')
# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=10,
                    help='Number of epochs to train.')
parser.add_argument('--learning_rate', '-lr', type=float,
                    default=0.001, help='The initial learning rate.')
parser.add_argument('--batch_size', '-b', type=int,
                    default=128, help='Batch size.')
parser.add_argument('--oe_batch_size', type=int,
                    default=256, help='Batch size.')
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float,
                    default=0.0005, help='Weight decay (L2 penalty).')
# WRN Architecture
parser.add_argument('--layers', default=40, type=int,
                    help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float,
                    help='dropout probability')
# Checkpoints
parser.add_argument('--results_dir', type=str,
                    default='results', help='Folder to save .pkl results.')
parser.add_argument('--checkpoints_dir', type=str,
                    default='checkpoints', help='Folder to save .pt checkpoints.')

parser.add_argument('--load_pretrained', type=str,
                    default='snapshots/pretrained', help='Load pretrained model to test or resume training.')
parser.add_argument('--test', '-t', action='store_true',
                    help='Test only flag.')

# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--gpu_id', type=int, default=1, help='Which GPU to run on.')
parser.add_argument('--prefetch', type=int, default=4,
                    help='Pre-fetching threads.')
# EG specific
parser.add_argument('--score', type=str, default='SSND', help='SSND|OE|energy|VOS')
parser.add_argument('--seed', type=int, default=1,
                    help='seed for np(tinyimages80M sampling); 1|2|8|100|107')
parser.add_argument('--classification', type=boolean_string, default=True)

# dataset related
parser.add_argument('--aux_out_dataset', type=str, default='svhn', choices=['svhn', 'lsun_c', 'lsun_r',
                    'isun', 'dtd', 'places', 'tinyimages_300k'],
                    help='Auxiliary out of distribution dataset')
parser.add_argument('--test_out_dataset', type=str,choices=['svhn', 'lsun_c', 'lsun_r',
                    'isun', 'dtd', 'places', 'tinyimages_300k'],
                    default='svhn', help='Test out of distribution dataset')
parser.add_argument('--pi', type=float, default=1,
                    help='pi in ssnd framework, proportion of ood data in auxiliary dataset')

###woods/woods_nn specific
parser.add_argument('--in_constraint_weight', type=float, default=1,
                    help='weight for in-distribution penalty in loss function')
parser.add_argument('--out_constraint_weight', type=float, default=1,
                    help='weight for out-of-distribution penalty in loss function')
parser.add_argument('--ce_constraint_weight', type=float, default=1,
                    help='weight for classification penalty in loss function')
parser.add_argument('--false_alarm_cutoff', type=float,
                    default=0.05, help='false alarm cutoff')

parser.add_argument('--lr_lam', type=float, default=1, help='learning rate for the updating lam (SSND_alm)')
parser.add_argument('--ce_tol', type=float,
                    default=2, help='tolerance for the loss constraint')

parser.add_argument('--penalty_mult', type=float,
                    default=1.5, help='multiplicative factor for penalty method')

parser.add_argument('--constraint_tol', type=float,
                    default=0, help='tolerance for considering constraint violated')

# Energy Method Specific
parser.add_argument('--m_in', type=float, default=-25.,
                    help='margin for in-distribution; above this value will be penalized')
parser.add_argument('--m_out', type=float, default=-5.,
                    help='margin for out-distribution; below this value will be penalized')
parser.add_argument('--T', default=1., type=float, help='temperature: energy|Odin')  # T = 1 suggested by energy paper

#energy vos method
parser.add_argument('--energy_vos_lambda', type=float, default=2, help='energy vos weight')

# OE specific
parser.add_argument('--oe_lambda', type=float, default=.5, help='OE weight')


# parse argument
args = parser.parse_args()

# method_data_name gives path to the model
if args.score in ['woods_nn']:
    method_data_name = "{}_{}_{}_{}_{}_{}_{}_{}".format(args.score,
                                               str(args.in_constraint_weight),
                                               str(args.out_constraint_weight),
                                               str(args.ce_constraint_weight),
                                               str(args.false_alarm_cutoff),
                                               str(args.lr_lam),
                                               str(args.penalty_mult),
                                               str(args.pi))
elif args.score == "energy":
    method_data_name = "{}_{}_{}_{}".format(args.score,
                                      str(args.m_in),
                                      str(args.m_out),
                                      args.pi)
elif args.score == "OE":
    method_data_name = "{}_{}_{}".format(args.score,
                                   str(args.oe_lambda),
                                   str(args.pi))
elif args.score == "energy_vos":
    method_data_name = "{}_{}_{}".format(args.score,
                                   str(args.energy_vos_lambda),
                                   str(args.pi))
elif args.score in ['woods']:
    method_data_name = "{}_{}_{}_{}_{}_{}_{}_{}".format(args.score,
                                            str(args.in_constraint_weight),
                                            str(args.out_constraint_weight),
                                            str(args.false_alarm_cutoff),
                                            str(args.ce_constraint_weight),
                                            str(args.lr_lam),
                                            str(args.penalty_mult),
                                            str(args.pi))


state = {k: v for k, v in args._get_kwargs()}
print(state)

#save wandb hyperparameters
# wandb.config = state
wandb.init(project="OOD", entity="ood_learning", config=state)
state['wandb_name'] = wandb.run.name

# store train, test, and valid FNR
state['fnr_train'] = []
state['fnr_valid'] = []
state['fnr_valid_clean'] = []
state['fnr_test'] = []

# in-distribution classification accuracy
state['train_accuracy'] = []
state['valid_accuracy'] = []
state['valid_accuracy_clean'] = []
state['test_accuracy'] = []

# store train, valid, and test OOD scores
state['OOD_scores_P0_train'] = []
state['OOD_scores_PX_train'] = []
state['OOD_scores_P0_valid'] = []
state['OOD_scores_PX_valid'] = []
state['OOD_scores_P0_valid_clean'] = []
state['OOD_scores_PX_valid_clean'] = []
state['OOD_scores_P0_test'] = []
state['OOD_scores_Ptest'] = []

# optimization constraints
state['in_dist_constraint'] = []
state['train_loss_constraint'] = []

def to_np(x): return x.data.cpu().numpy()

torch.manual_seed(args.seed)
rng = np.random.default_rng(args.seed)

#make the data_loaders
train_loader_in, train_loader_aux_in, train_loader_aux_out, test_loader, test_loader_ood, valid_loader_in, valid_loader_aux_in, valid_loader_aux_out = make_datasets(
    args.dataset, args.aux_out_dataset, args.test_out_dataset, args.pi, state)

print("\n len(train_loader_in.dataset) {} " \
      "len(train_loader_aux_in.dataset) {}, " \
      "len(train_loader_aux_out.dataset) {}, " \
      "len(test_loader.dataset) {}, " \
      "len(test_loader_ood.dataset) {}, " \
      "len(valid_loader_in.dataset) {}, " \
      "len(valid_loader_aux_in.dataset) {}" \
      "len(valid_loader_aux_out.dataset) {}".format(
    len(train_loader_in.dataset),
    len(train_loader_aux_in.dataset),
    len(train_loader_aux_out.dataset),
    len(test_loader.dataset),
    len(test_loader_ood.dataset),
    len(valid_loader_in.dataset),
    len(valid_loader_aux_in.dataset),
    len(valid_loader_aux_out.dataset)))

state['train_in_size'] = len(train_loader_in.dataset)
state['train_aux_in_size'] = len(train_loader_aux_in.dataset)
state['train_aux_out_size'] = len(train_loader_aux_out.dataset)
state['valid_in_size'] = len(valid_loader_in.dataset)
state['valid_aux_in_size'] = len(valid_loader_aux_in.dataset)
state['valid_aux_out_size'] = len(valid_loader_aux_out.dataset)
state['test_in_size'] = len(test_loader.dataset)
state['test_out_size'] = len(test_loader_ood.dataset)

if args.dataset in ['cifar10']:
    num_classes = 10
elif args.dataset in ['cifar100']:
    num_classes = 100

# WRN architecture with 10 output classes (extra NN is added later for SSND methods)
net = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate)

#create logistic regression layer for energy_vos and woods
if args.score in ['energy_vos', 'woods']:
    logistic_regression = nn.Linear(1, 1)
    logistic_regression.cuda()

# Restore model
model_found = False
print(args.load_pretrained)
if args.load_pretrained == 'snapshots/pretrained':
    print('Restoring trained model...')
    for i in range(100, -1, -1):

        model_name = os.path.join(args.load_pretrained, args.dataset + '_' + args.model +
                                  '_pretrained_epoch_' + str(i) + '.pt')
        if os.path.isfile(model_name):
            print('found pretrained model: {}'.format(model_name))
            net.load_state_dict(torch.load(model_name))
            print('Model restored! Epoch:', i)
            model_found = True
            break
    if not model_found:
        assert False, "could not find model to restore"

# add extra NN for OOD detection (for SSND methods)
if args.score in ['woods_nn']:
    net = WideResNet_SSND(wrn=net)

if args.ngpu > 1:
    print('Available CUDA devices:', torch.cuda.device_count())
    print('CUDA available:', torch.cuda.is_available())
    print('Running in parallel across', args.ngpu, 'GPUs')
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))
    net.cuda()
    torch.cuda.manual_seed(1)
elif args.ngpu > 0:
    print('CUDA available:', torch.cuda.is_available())
    print('Available CUDA devices:', torch.cuda.device_count())
    print('Sending model to device', torch.cuda.current_device(), ':', torch.cuda.get_device_name())
    net.cuda()
    torch.cuda.manual_seed(1)

# cudnn.benchmark = True  # fire on all cylinders
cudnn.benchmark = False  # control reproducibility/stochastic behavior

#energy_vos, woods also use logistic regression in optimization
if args.score in ['energy_vos', 'woods']:
    optimizer = torch.optim.SGD(
        list(net.parameters()) + list(logistic_regression.parameters()),
        state['learning_rate'], momentum=state['momentum'],
        weight_decay=state['decay'], nesterov=True)

else:
    optimizer = torch.optim.SGD(
        net.parameters(), state['learning_rate'], momentum=state['momentum'],
        weight_decay=state['decay'], nesterov=True)

#define scheduler for learning rate
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                 milestones=[int(args.epochs*.5), int(args.epochs*.75), int(args.epochs*.9)], gamma=0.5)

# /////////////// Training ///////////////

# Create extra variable needed for training

# make in_constraint a global variable
in_constraint_weight = args.in_constraint_weight

# make loss_ce_constraint a global variable
ce_constraint_weight = args.ce_constraint_weight

# create the lagrangian variable for lagrangian methods
if args.score in ['woods_nn', 'woods']:
    lam = torch.tensor(0).float()
    lam = lam.cuda()

    lam2 = torch.tensor(0).float()
    lam2 = lam.cuda()


def mix_batches(aux_in_set, aux_out_set):
    '''
    Args:
        aux_in_set: minibatch from in_distribution
        aux_out_set: minibatch from out distribution

    Returns:
        mixture of minibatches with mixture proportion pi of aux_out_set
    '''

    # create a mask to decide which sample is in the batch
    mask = rng.choice(a=[False, True], size=(args.batch_size,), p=[1 - args.pi, args.pi])

    aux_out_set_subsampled = aux_out_set[0][mask]
    aux_in_set_subsampled = aux_in_set[0][np.invert(mask)]

    # note: ordering of aux_out_set_subsampled, aux_in_set_subsampled does not matter because you always take the sum
    aux_set = torch.cat((aux_out_set_subsampled, aux_in_set_subsampled), 0)

    return aux_set


def train(epoch):
    '''
    Train the model using the specified score
    '''

    # make the variables global for optimization purposes
    global in_constraint_weight
    global ce_constraint_weight

    # declare lam global
    if args.score in ['woods_nn',  'woods']:
        global lam
        global lam2

    # print the learning rate
    for param_group in optimizer.param_groups:
        print("lr {}".format(param_group['lr']))

    net.train()  # enter train mode

    # track train classification accuracy
    train_accuracies = []

    # # start at a random point of the dataset for; this induces more randomness without obliterating locality
    train_loader_aux_in.dataset.offset = rng.integers(
        len(train_loader_aux_in.dataset))
    train_loader_aux_out.dataset.offset = rng.integers(
        len(train_loader_aux_out.dataset))
    batch_num = 1
    loaders = zip(train_loader_in, train_loader_aux_in, train_loader_aux_out)

    # for logging in weights & biases
    losses_ce = []
    in_losses = []
    out_losses = []
    out_losses_weighted = []
    losses = []

    for in_set, aux_in_set, aux_out_set in loaders:
        #create the mixed batch
        aux_set = mix_batches(aux_in_set, aux_out_set)

        batch_num += 1
        data = torch.cat((in_set[0], aux_set), 0)
        target = in_set[1]

        if args.ngpu > 0:
            data, target = data.cuda(), target.cuda()

        # forward
        x = net(data)

        # in-distribution classification accuracy
        if args.score in ['woods_nn']:
            x_classification = x[:len(in_set[0]), :num_classes]
        elif args.score in ['energy', 'OE', 'energy_vos', 'woods']:
            x_classification = x[:len(in_set[0])]
        pred = x_classification.data.max(1)[1]
        train_accuracies.append(accuracy_score(list(to_np(pred)), list(to_np(target))))

        optimizer.zero_grad()

        # cross-entropy loss
        if args.classification:
            loss_ce = F.cross_entropy(x_classification, target)
        else:
            loss_ce = torch.Tensor([0]).cuda()

        losses_ce.append(loss_ce.item())

        if args.score == 'woods_nn':
            '''
            This is the same as woods_nn but it now uses separate
            weight for in distribution scores and classification scores.

            it also updates the weights separately
            '''

            # penalty for the mixture/auxiliary dataset
            out_x_ood_task = x[len(in_set[0]):, num_classes]
            out_loss = torch.mean(F.relu(1 - out_x_ood_task))
            out_loss_weighted = args.out_constraint_weight * out_loss

            in_x_ood_task = x[:len(in_set[0]), num_classes]
            f_term = torch.mean(F.relu(1 + in_x_ood_task)) - args.false_alarm_cutoff
            if in_constraint_weight * f_term + lam >= 0:
                in_loss = f_term * lam + in_constraint_weight / 2 * torch.pow(f_term, 2)
            else:
                in_loss = - torch.pow(lam, 2) * 0.5 / in_constraint_weight

            loss_ce_constraint = loss_ce - args.ce_tol * full_train_loss
            if ce_constraint_weight * loss_ce_constraint + lam2 >= 0:
                loss_ce = loss_ce_constraint * lam2 + ce_constraint_weight / 2 * torch.pow(loss_ce_constraint, 2)
            else:
                loss_ce = - torch.pow(lam2, 2) * 0.5 / ce_constraint_weight

            # add the losses together
            loss = loss_ce + out_loss_weighted + in_loss

            in_losses.append(in_loss.item())
            out_losses.append(out_loss.item())
            out_losses_weighted.append(out_loss.item())
            losses.append(loss.item())

        elif args.score == 'energy':

            Ec_out = -torch.logsumexp(x[len(in_set[0]):], dim=1)
            Ec_in = -torch.logsumexp(x[:len(in_set[0])], dim=1)
            loss_energy = 0.1 * (torch.pow(F.relu(Ec_in - args.m_in), 2).mean() + torch.pow(F.relu(args.m_out - Ec_out),
                                                                                            2).mean())
            loss = loss_ce + loss_energy

            losses.append(loss.item())

        elif args.score == 'energy_vos':

            Ec_out = torch.logsumexp(x[len(in_set[0]):], dim=1)
            Ec_in = torch.logsumexp(x[:len(in_set[0])], dim=1)
            binary_labels = torch.ones(len(x)).cuda()
            binary_labels[len(in_set[0]):] = 0
            loss_energy = F.binary_cross_entropy_with_logits(logistic_regression(
                torch.cat([Ec_in, Ec_out], -1).unsqueeze(1)).squeeze(),
                                                                 binary_labels)

            loss = loss_ce + args.energy_vos_lambda * loss_energy

            losses.append(loss.item())

        elif args.score == 'woods':

            #compute energies
            Ec_out = torch.logsumexp(x[len(in_set[0]):], dim=1)
            Ec_in = torch.logsumexp(x[:len(in_set[0])], dim=1)

            #apply the sigmoid loss
            loss_energy_in =  torch.mean(torch.sigmoid(logistic_regression(
                Ec_in.unsqueeze(1)).squeeze()))
            loss_energy_out = torch.mean(torch.sigmoid(-logistic_regression(
                Ec_out.unsqueeze(1)).squeeze()))

            #alm function for the in distribution constraint
            in_constraint_term = loss_energy_in - args.false_alarm_cutoff
            if in_constraint_weight * in_constraint_term + lam >= 0:
                in_loss = in_constraint_term * lam + in_constraint_weight / 2 * torch.pow(in_constraint_term, 2)
            else:
                in_loss = - torch.pow(lam, 2) * 0.5 / in_constraint_weight

            #alm function for the cross entropy constraint
            loss_ce_constraint = loss_ce - args.ce_tol * full_train_loss
            if ce_constraint_weight * loss_ce_constraint + lam2 >= 0:
                loss_ce = loss_ce_constraint * lam2 + ce_constraint_weight / 2 * torch.pow(loss_ce_constraint, 2)
            else:
                loss_ce = - torch.pow(lam2, 2) * 0.5 / ce_constraint_weight

            loss = loss_ce + args.out_constraint_weight*loss_energy_out + in_loss

            #wandb
            in_losses.append(in_loss.item())
            out_losses.append(loss_energy_out.item())
            out_losses_weighted.append(args.out_constraint_weight * loss_energy_out.item())
            losses.append(loss.item())

        elif args.score == 'OE':

            loss_oe = args.oe_lambda * -(x[len(in_set[0]):].mean(1) - torch.logsumexp(x[len(in_set[0]):], dim=1)).mean()
            loss = loss_ce + loss_oe
            losses.append(loss.item())

        loss.backward()
        optimizer.step()

    loss_ce_avg = np.mean(losses_ce)
    in_loss_avg = np.mean(in_losses)
    out_loss_avg = np.mean(out_losses)
    out_loss_weighted_avg = np.mean(out_losses_weighted)
    loss_avg = np.mean(losses)
    train_acc_avg = np.mean(train_accuracies)

    wandb.log({
        'epoch':epoch,
        "learning rate": optimizer.param_groups[0]['lr'],
        'CE loss':loss_ce_avg,
        'in loss':in_loss_avg,
        'out loss':out_loss_avg,
        'out loss (weighted)':out_loss_weighted_avg,
        'loss':loss_avg,
        'train accuracy':train_acc_avg
    })

    # store train accuracy
    state['train_accuracy'].append(train_acc_avg)

    # updates for alm methods
    if args.score in ["woods_nn"]:
        print("making updates for SSND alm methods...")

        # compute terms for constraints
        in_term, ce_loss = compute_constraint_terms()

        # update lam for in-distribution term
        if args.score in ["woods_nn"]:
            print("updating lam...")

            in_term_constraint = in_term - args.false_alarm_cutoff
            print("in_distribution constraint value {}".format(in_term_constraint))
            state['in_dist_constraint'].append(in_term_constraint.item())

            # wandb
            wandb.log({"in_term_constraint": in_term_constraint.item(),
                       'in_constraint_weight':in_constraint_weight,
                       'epoch':epoch})

            # update lambda
            if in_term_constraint * in_constraint_weight + lam >= 0:
                lam += args.lr_lam * in_term_constraint
            else:
                lam += -args.lr_lam * lam / in_constraint_weight

        # update lam2
        if args.score in ["woods_nn"]:
            print("updating lam2...")

            ce_constraint = ce_loss - args.ce_tol * full_train_loss
            print("cross entropy constraint {}".format(ce_constraint))
            state['train_loss_constraint'].append(ce_constraint.item())

            # wandb
            wandb.log({"ce_term_constraint": ce_constraint.item(),
                       'ce_constraint_weight':ce_constraint_weight,
                       'epoch':epoch})

            # update lambda2
            if ce_constraint * ce_constraint_weight + lam2 >= 0:
                lam2 += args.lr_lam * ce_constraint
            else:
                lam2 += -args.lr_lam * lam2 / ce_constraint_weight

        # update weight for alm_full_2
        if args.score == 'woods_nn' and in_term_constraint > args.constraint_tol:
            print('increasing in_constraint_weight weight....\n')
            in_constraint_weight *= args.penalty_mult

        if args.score == 'woods_nn' and ce_constraint > args.constraint_tol:
            print('increasing ce_constraint_weight weight....\n')
            ce_constraint_weight *= args.penalty_mult

    #alm update for energy_vos alm methods
    if args.score in ['woods']:
        print("making updates for energy alm methods...")
        avg_sigmoid_energy_losses, _, avg_ce_loss = evaluate_energy_logistic_loss()

        in_term_constraint = avg_sigmoid_energy_losses -  args.false_alarm_cutoff
        print("in_distribution constraint value {}".format(in_term_constraint))
        state['in_dist_constraint'].append(in_term_constraint.item())

        # update lambda
        print("updating lam...")
        if in_term_constraint * in_constraint_weight + lam >= 0:
            lam += args.lr_lam * in_term_constraint
        else:
            lam += -args.lr_lam * lam / in_constraint_weight

        # wandb
        wandb.log({"in_term_constraint": in_term_constraint.item(),
                   'in_constraint_weight':in_constraint_weight,
                   "avg_sigmoid_energy_losses": avg_sigmoid_energy_losses.item(),
                   'lam': lam,
                   'epoch':epoch})

        # update lam2
        if args.score in ['woods']:
            print("updating lam2...")

            ce_constraint = avg_ce_loss - args.ce_tol * full_train_loss
            print("cross entropy constraint {}".format(ce_constraint))
            state['train_loss_constraint'].append(ce_constraint.item())

            # wandb
            wandb.log({"ce_term_constraint": ce_constraint.item(),
                       'ce_constraint_weight':ce_constraint_weight,
                       'epoch':epoch})

            # update lambda2
            if ce_constraint * ce_constraint_weight + lam2 >= 0:
                lam2 += args.lr_lam * ce_constraint
            else:
                lam2 += -args.lr_lam * lam2 / ce_constraint_weight

        # update in-distribution weight for alm
        if args.score in ['woods'] and in_term_constraint > args.constraint_tol:
            print("energy in distribution constraint violated, so updating in_constraint_weight...")
            in_constraint_weight *= args.penalty_mult

        # update ce_loss weight for alm
        if args.score in ['woods'] and ce_constraint > args.constraint_tol:
            print('increasing ce_constraint_weight weight....\n')
            ce_constraint_weight *= args.penalty_mult


def compute_constraint_terms():
    '''

    Compute the in-distribution term and the cross-entropy loss over the whole training set
    '''

    net.eval()

    # create list for the in-distribution term and the ce_loss
    in_terms = []
    ce_losses = []
    num_batches = 0
    for in_set in train_loader_in:
        num_batches += 1
        data = in_set[0]
        target = in_set[1]

        if args.ngpu > 0:
            data, target = data.cuda(), target.cuda()

        # forward
        net(data)
        z = net(data)

        # compute in-distribution term
        in_x_ood_task = z[:, num_classes]
        in_terms.extend(list(to_np(F.relu(1 + in_x_ood_task))))

        # compute cross entropy term
        z_classification = z[:, :num_classes]
        loss_ce = F.cross_entropy(z_classification, target, reduction='none')
        ce_losses.extend(list(to_np(loss_ce)))

    return np.mean(np.array(in_terms)), np.mean(np.array(ce_losses))


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

# test function
def test(test_dataset=True, clean_dataset=True):
    """
    tests current model

    test_dataset: if true, then uses test dataloaders, if false uses validation dataloaders
    """
    if test_dataset:
        print('testing...')
    else:
        print('validation...')

    # decide which dataloader to use for in-distribution
    if test_dataset:
        in_loader = test_loader
    else:
        in_loader = valid_loader_in

    net.eval()

    # in-distribution performance
    print("computing over in-distribution data...\n")
    with torch.no_grad():

        accuracies = []
        OOD_scores_P0 = []

        for data, target in in_loader:

            if args.ngpu > 0:
                data, target = data.cuda(), target.cuda()

            # forward
            output = net(data)

            if args.score in ["woods_nn"]:

                # classification accuracy
                output_classification = output[:len(data), :num_classes]
                pred = output_classification.data.max(1)[1]
                accuracies.append(accuracy_score(list(to_np(pred)), list(to_np(target))))

                # OOD scores
                np_in = to_np(output[:, num_classes])
                np_in_list = list(np_in)
                OOD_scores_P0.extend(np_in_list)

            elif args.score in ['energy', 'OE', 'energy_vos', 'woods']:

                # classification accuracy
                pred = output.data.max(1)[1]
                accuracies.append(accuracy_score(list(to_np(pred)), list(to_np(target))))

                if args.score in ['energy', 'energy_vos', 'woods']:

                    # OOD scores
                    OOD_scores_P0.extend(list(-to_np((args.T * torch.logsumexp(output / args.T, dim=1)))))

                elif args.score == 'OE':

                    # OOD scores
                    smax = to_np(F.softmax(output, dim=1))
                    OOD_scores_P0.extend(list(-np.max(smax, axis=1)))

    # OOD scores for either mixture or PX
    OOD_scores_P_out = []

    if test_dataset:
        print("computing over ood data for testing...\n")

        # OOD performance
        with torch.no_grad():

            for data, target in test_loader_ood:

                if args.ngpu > 0:
                    data, target = data.cuda(), target.cuda()

                output = net(data)

                if args.score in [ "woods_nn"]:

                    np_out = to_np(output[:, num_classes])
                    np_out_list = list(np_out)
                    OOD_scores_P_out.extend(np_out_list)

                elif args.score in ['energy', 'energy_vos', 'woods']:

                    OOD_scores_P_out.extend(list(-to_np((args.T * torch.logsumexp(output / args.T, dim=1)))))

                elif args.score == 'OE':

                    smax = to_np(F.softmax(output, dim=1))
                    OOD_scores_P_out.extend(list(-np.max(smax, axis=1)))


    else:
        if clean_dataset:
            print("computing over clean OOD validation set...\n")
            for aux_out_set in valid_loader_aux_out:

                data = aux_out_set[0]

                if args.ngpu > 0:
                    data = data.cuda()

                # forward
                output = net(data)

                if args.score in ["woods_nn"]:

                    np_out = to_np(output[:, num_classes])
                    np_out_list = list(np_out)
                    OOD_scores_P_out.extend(np_out_list)

                elif args.score in ['energy', 'energy_vos', 'woods']:

                    OOD_scores_P_out.extend(list(-to_np((args.T * torch.logsumexp(output / args.T, dim=1)))))

                elif args.score == 'OE':

                    smax = to_np(F.softmax(output, dim=1))
                    OOD_scores_P_out.extend(list(-np.max(smax, axis=1)))
        else:
            print("computing over mixture for validation...\n")
            for aux_in_set, aux_out_set in zip(valid_loader_aux_in, valid_loader_aux_out):

                data = mix_batches(aux_in_set, aux_out_set)

                if args.ngpu > 0:
                    data = data.cuda()

                # forward
                output = net(data)

                if args.score in ["woods_nn"]:

                    np_out = to_np(output[:, num_classes])
                    np_out_list = list(np_out)
                    OOD_scores_P_out.extend(np_out_list)

                elif args.score in ['energy', 'energy_vos', 'woods']:

                    OOD_scores_P_out.extend(list(-to_np((args.T * torch.logsumexp(output / args.T, dim=1)))))

                elif args.score == 'OE':

                    smax = to_np(F.softmax(output, dim=1))
                    OOD_scores_P_out.extend(list(-np.max(smax, axis=1)))


    # compute FNR and accuracy
    fnr = compute_fnr(np.array(OOD_scores_P_out), np.array(OOD_scores_P0))
    acc = sum(accuracies) / len(accuracies)

    # store and print results
    if test_dataset:
        state['fnr_test'].append(fnr)
        state['test_accuracy'].append(acc)
        state['OOD_scores_P0_test'].append(OOD_scores_P0)
        state['OOD_scores_Ptest'].append(OOD_scores_P_out)

        plt.hist(OOD_scores_P0, alpha=0.5, label='in')
        plt.hist(OOD_scores_P_out, alpha=0.5, label='out')
        plt.legend()
        plt.title('epoch {}, test FNR = {}'.format(epoch, fnr))
        if not os.path.exists('figs_pycharm'):
            os.mkdir('figs_pycharm')
        plt.savefig('figs_pycharm/test_epoch{}.png'.format(epoch))
        plt.clf()
        plt.close()

        wandb.log({"fnr_test": fnr,
                   "test_accuracy": acc,
                   'epoch':epoch})

        print("\n fnr_test {}".format(state['fnr_test']))
        print("test_accuracy {} \n".format(state['test_accuracy']))

    else:
        if clean_dataset:
            state['fnr_valid_clean'].append(fnr)
            state['valid_accuracy_clean'].append(acc)
            state['OOD_scores_P0_valid_clean'].append(OOD_scores_P0)
            state['OOD_scores_PX_valid_clean'].append(OOD_scores_P_out)

            wandb.log({"validation_fnr_clean": fnr,
                       "validation_accuracy_clean": acc,
                       'epoch': epoch})

            print("\n fnr_valid_clean {}".format(state['fnr_valid_clean']))
            print("valid_accuracy_clean {} \n".format(state['valid_accuracy_clean']))
        else:
            state['fnr_valid'].append(fnr)
            state['valid_accuracy'].append(acc)
            state['OOD_scores_P0_valid'].append(OOD_scores_P0)
            state['OOD_scores_PX_valid'].append(OOD_scores_P_out)

            wandb.log({"validation_fnr": fnr,
                       "validation_accuracy": acc,
                       'epoch':epoch})

            print("\n fnr_valid {}".format(state['fnr_valid']))
            print("valid_accuracy {} \n".format(state['valid_accuracy']))


def evaluate_classification_loss_training():
    '''
    evaluate classification loss on training dataset
    '''

    net.eval()
    losses = []
    for in_set in train_loader_in:
        # print('batch', batch_num, '/', min(len(train_loader_in), len(train_loader_out)))
        data = in_set[0]
        target = in_set[1]

        if args.ngpu > 0:
            data, target = data.cuda(), target.cuda()

        # forward
        x = net(data)

        # in-distribution classification accuracy
        x_classification = x[:, :num_classes]
        loss_ce = F.cross_entropy(x_classification, target, reduction='none')

        losses.extend(list(to_np(loss_ce)))

    avg_loss = np.mean(np.array(losses))
    print("average loss fr classification {}".format(avg_loss))

    return avg_loss


def evaluate_energy_logistic_loss():
    '''
    evaluate energy logistic loss on training dataset
    '''

    net.eval()
    sigmoid_energy_losses = []
    logistic_energy_losses = []
    ce_losses = []
    for in_set in train_loader_in:
        # print('batch', batch_num, '/', min(len(train_loader_in), len(train_loader_out)))
        data = in_set[0]
        target = in_set[1]

        if args.ngpu > 0:
            data, target = data.cuda(), target.cuda()

        # forward
        x = net(data)

        # compute energies
        Ec_in = torch.logsumexp(x, dim=1)

        # compute labels
        binary_labels_1 = torch.ones(len(data)).cuda()

        # compute in distribution logistic losses
        logistic_loss_energy_in = F.binary_cross_entropy_with_logits(logistic_regression(
            Ec_in.unsqueeze(1)).squeeze(), binary_labels_1, reduction='none')

        logistic_energy_losses.extend(list(to_np(logistic_loss_energy_in)))

        # compute in distribution sigmoid losses
        sigmoid_loss_energy_in = torch.sigmoid(logistic_regression(
            Ec_in.unsqueeze(1)).squeeze())

        sigmoid_energy_losses.extend(list(to_np(sigmoid_loss_energy_in)))

        # in-distribution classification losses
        x_classification = x[:, :num_classes]
        loss_ce = F.cross_entropy(x_classification, target, reduction='none')

        ce_losses.extend(list(to_np(loss_ce)))

    avg_sigmoid_energy_losses = np.mean(np.array(sigmoid_energy_losses))
    print("average sigmoid in distribution energy loss {}".format(avg_sigmoid_energy_losses))

    avg_logistic_energy_losses = np.mean(np.array(logistic_energy_losses))
    print("average in distribution energy loss {}".format(avg_logistic_energy_losses))

    avg_ce_loss = np.mean(np.array(ce_losses))
    print("average loss fr classification {}".format(avg_ce_loss))

    return avg_sigmoid_energy_losses, avg_logistic_energy_losses, avg_ce_loss

print('Beginning Training\n')

#compute training loss for woods methods
if args.score in [ 'woods_nn', 'woods']:
    full_train_loss = evaluate_classification_loss_training()

###################################################################
# Main loop
###################################################################
min_val_fnr = 1.0
min_val_fnr_clean = 1.0
min_test_fnr = 1.0
state['best_epoch_valid'] = 0
state['best_epoch_valid_clean'] = 0
state['best_epoch_test'] = 0
for epoch in range(0, args.epochs):
    print('epoch', epoch + 1, '/', args.epochs)
    state['epoch'] = epoch

    begin_epoch = time.time()

    train(epoch)
    test(test_dataset=False, clean_dataset=False)  # test on mixed validation set
    test(test_dataset=False, clean_dataset=True)  # test on clean validation set
    test(test_dataset=True)  # test on test dataset

    scheduler.step()

    # check for best epoch based on val fnr - mixed
    if state['fnr_valid'][-1] < min_val_fnr:
        best_epoch_val_old = state['best_epoch_valid']
        state['best_epoch_valid'] = epoch
        min_val_fnr = state['fnr_valid'][-1]

    # check for best epoch based on val fnr - clean
    if state['fnr_valid_clean'][-1] < min_val_fnr_clean:
        best_epoch_val_old = state['best_epoch_valid_clean']
        state['best_epoch_valid_clean'] = epoch
        min_val_fnr_clean = state['fnr_valid_clean'][-1]

    # check for best epoch based on val fnr
    if state['fnr_test'][-1] < min_test_fnr:
        best_epoch_test_old = state['best_epoch_test']
        state['best_epoch_test'] = epoch
        min_test_fnr = state['fnr_test'][-1]

    print('best valid epoch = {}'.format(state['best_epoch_valid']))
    print('best valid epoch (clean) = {}'.format(state['best_epoch_valid_clean']))
    print('best test epoch = {}'.format(state['best_epoch_test']))

    wandb.log({"best_epoch_valid": state['best_epoch_valid'],
               "best_epoch_valid_clean": state['best_epoch_valid_clean'],
               "best_epoch_test": state['best_epoch_test'],
               'epoch': epoch})

    # save model checkpoint
    if args.checkpoints_dir != '' and epoch in [state['best_epoch_valid'], state['best_epoch_valid_clean'], state['best_epoch_test'], args.epochs-1]:
        model_checkpoint_dir = os.path.join(args.checkpoints_dir,
                                            args.dataset,
                                            args.aux_out_dataset,
                                            args.score)
        if not os.path.exists(model_checkpoint_dir):
            os.makedirs(model_checkpoint_dir, exist_ok=True)
        model_filename = '{}_epoch_{}.pt'.format(method_data_name, epoch)
        model_path = os.path.join(model_checkpoint_dir, model_filename)
        print('saving model to {}'.format(model_path))
        torch.save(net.state_dict(), model_path)

        #save path name
        if epoch == state['best_epoch_valid']:
            state['model_loc_valid'] = model_path
        if epoch == state['best_epoch_valid_clean']:
            state['model_loc_valid_clean'] = model_path
        if epoch == state['best_epoch_test']:
            state['model_loc_test'] = model_path
        if epoch == args.epochs-1:
            state['model_loc_last'] = model_path

        if state['best_epoch_valid'] == epoch:
            # delete previous checkpoint
            if best_epoch_val_old not in [epoch, state['best_epoch_test'], state['best_epoch_valid_clean']]:
                print('deleting old best valid epoch')
                model_filename_prev = '{}_epoch_{}.pt'.format(method_data_name, best_epoch_val_old)
                model_path_prev = os.path.join(model_checkpoint_dir, model_filename_prev)
                if os.path.exists(model_path_prev):
                    print('removing {}'.format(model_path_prev))
                    os.remove(model_path_prev)

        if state['best_epoch_valid_clean'] == epoch:
            # delete previous checkpoint
            if best_epoch_val_old not in [epoch, state['best_epoch_test'], state['best_epoch_valid']]:
                print('deleting old best valid epoch (clean)')
                model_filename_prev = '{}_epoch_{}.pt'.format(method_data_name, best_epoch_val_old)
                model_path_prev = os.path.join(model_checkpoint_dir, model_filename_prev)
                if os.path.exists(model_path_prev):
                    print('removing {}'.format(model_path_prev))
                    os.remove(model_path_prev)

        if state['best_epoch_test'] == epoch:
            # delete previous checkpoint
            if best_epoch_test_old not in [epoch, state['best_epoch_valid'], state['best_epoch_valid_clean']]:
                print('deleting old best test epoch')
                model_filename_prev = '{}_epoch_{}.pt'.format(method_data_name, best_epoch_test_old)
                model_path_prev = os.path.join(model_checkpoint_dir, model_filename_prev)
                if os.path.exists(model_path_prev):
                    print('removing {}'.format(model_path_prev))
                    os.remove(model_path_prev)

    for t in range(epoch):
        if t not in [state['best_epoch_valid'], state['best_epoch_valid_clean'], state['best_epoch_test'], args.epochs-1]:
            state['OOD_scores_P0_valid'][t] = 0
            state['OOD_scores_PX_valid'][t] = 0
            state['OOD_scores_P0_valid_clean'][t] = 0
            state['OOD_scores_PX_valid_clean'][t] = 0
            state['OOD_scores_P0_test'][t] = 0
            state['OOD_scores_Ptest'][t] = 0

    # save results to .pkl file
    results_dir = os.path.join(args.results_dir,
                               args.dataset,
                               args.aux_out_dataset,
                               args.score)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
    results_filename = '{}.pkl'.format(method_data_name)
    results_path = os.path.join(results_dir, results_filename)
    with open(results_path, 'wb') as f:
        print('saving results to', results_path)
        pickle.dump(state, f)