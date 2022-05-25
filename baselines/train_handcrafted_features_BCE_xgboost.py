import sys
sys.path.append('../')
import preprocessing.pipelines as pipelines
import utilities.dataset_utils as dataset_utils
import utilities.wrappers as wrappers
import xgboost as xgb
import pandas as pd
import numpy as np
import utilities.paths as paths
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
from utilities.paths import library_folder,model_folder
from train import make_PR_curves
import sys
import copy
if __name__ == '__main__':
    '''
    Script to train and evaluate handcrafted feature baseline on the B-cell epitope data set.
    Five-fold cross-validation is used, i.e. five models are trained.
    '''

    check = False # Check = True to verify installation, =False for full training.
    use_evolutionary = True # True to use evolutionary information (requires hhblits and a sequence database), False otherwise.

    max_depths = [5, 10, 20]
    min_child_weights = [5., 10., 50, 100]
    num_boost_round = 100
    gammas = [0.01, 0.1, 1.0, 5]
    etas = [0.5, 1.0]
    nthread = 20
    try:
        seed = int(sys.argv[1])
    except:
        seed = None

    if use_evolutionary:

        feature_list = [
            'primary', # Amino acid type
            'secondary', # secondary structure (from DSSP, 8 classes)
            'conservation', # Conservation score.
            'pwm', # Position Weight Matrix (21-dimensional vector).
            'asa', # Relative Accessible surface area (from DSSP)
            'residue_depth', # Backbone depth and sidechain depth with respect to the molecular surface (requires MSMS).
            'volume_index', # Surface convexity index (requires MSMS)
            'half_sphere', # Half sphere exposure (Number of residues in upper half-sphere - number of residues in bottom half-sphere)/Coordination index.
            'coordination' # Residue coordination index (13A radius)
        ]

        feature_names = ['aa_%s' % x for x in pipelines.sequence_utils.aa[:20]] + \
                        pipelines.dict_num2name + \
                        ['Conservation', 'ASA', 'Depth1', 'Depth2', 'half_sphere', 'coordination'] + \
                        ['VolIndex_%s' % k for k in range(3)] + \
                        ['pwm_%s' % x for x in pipelines.sequence_utils.aa]

    else:
        feature_list = [
            'primary', # Amino acid type
            'secondary', # secondary structure (from DSSP, 8 classes)
            'asa', # Relative Accessible surface area (from DSSP)
            # 'residue_depth', # Backbone depth and sidechain depth with respect to the molecular surface (requires MSMS).
            # 'volume_index', # Surface convexity index (requires MSMS)
            'half_sphere', # Half sphere exposure (Number of residues in upper half-sphere - number of residues in bottom half-sphere)/Coordination index.
            'coordination' # Residue coordination index (13A radius)
        ]
        # If MSMS or DSSP is not installed, comment out the corresponding features to run.

        feature_names = ['aa_%s' % x for x in pipelines.sequence_utils.aa[:20]] + \
                        pipelines.dict_num2name + \
                        ['ASA', 'Depth1', 'Depth2', 'half_sphere', 'coordination'] + \
                        ['VolIndex_%s' % k for k in range(3)]


    model_name = 'handcrafted_features_%s_xgboost'% (''.join([x[0] for x in feature_list]) )
    if seed is not None:
        model_name += '_%s'%seed
    model_names = [model_name + '_%s'%k for k in range(5)]


    list_datasets = [
    'fold1',
    'fold2',
    'fold3',
    'fold4',
    'fold5',
    ]

    list_dataset_names = [
        'Fold 1',
        'Fold 2',
        'Fold 3',
        'Fold 4',
        'Fold 5'
    ]


    #%% Gather and preprocess each dataset.
    pipeline = pipelines.HandcraftedFeaturesPipeline(feature_list=feature_list)


    list_dataset_locations = [library_folder+'datasets/BCE/labels_%s.txt'% dataset for dataset in list_datasets]
    dataset_table = pd.read_csv(library_folder+'datasets/BCE/table.csv',sep=',')

    list_inputs = []
    list_outputs = []
    list_weights = []

    for dataset,dataset_name,dataset_location in zip(list_datasets,list_dataset_names,list_dataset_locations):
        # Parse label files
        (list_origins,# List of chain identifiers (e.g. [1a3x_A,10gs_B,...])
        list_sequences,# List of corresponding sequences.
        list_resids,#List of corresponding residue identifiers.
        list_labels)  = dataset_utils.read_labels(dataset_location) # List of residue-wise labels

        if check:
            list_origins = list_origins[:10]
            list_sequences = list_sequences[:10]
            list_resids = list_resids[:10]
            list_labels = list_labels[:10]

        inputs,outputs,failed_samples = pipeline.build_processed_dataset(
            'BCE_%s'%(dataset+'_check' if check else dataset),
            list_origins=list_origins, # Mandatory
            list_resids=list_resids, # Optional
            list_labels=list_labels, # Optional
            ncores = 4,
            biounit=False, # Whether to use biological assembly files or the regular pdb files (asymmetric units). True for PPBS data set, False for BCE data set.
            save = True, # Whether to save the results in pickle file format. Files are stored in the pipeline_folder defined in paths.py
            fresh = False, # If fresh = False, attemps to load pickle files first.
        )

        weights = np.array(dataset_table['Sample weight'][ dataset_table['Set'] == dataset_name ] )
        if check:
            weights = weights[:10]
        weights = np.array([weights[b] for b in range(len(weights)) if not b in failed_samples])

        list_inputs.append(inputs)
        list_outputs.append(outputs)
        list_weights.append( weights )


    best_bsts = []
    best_score = 0

    for max_depth in max_depths:
        for min_child_weight in min_child_weights:
            for gamma in gammas:
                for eta in etas:
                    print(max_depth, min_child_weight, gamma, eta)

                    current_bsts = []
                    current_scores = []


                    for k in range(5): # 5-fold training/evaluation.
                        not_k = [i for i in range(5) if i!=k]
                        train_inputs = np.concatenate([list_inputs[i] for i in not_k])
                        train_outputs = np.concatenate([list_outputs[i] for i in not_k])
                        train_weights = np.concatenate([list_weights[i] for i in not_k])

                        test_inputs = list_inputs[k]
                        test_outputs = list_outputs[k]
                        test_weights = list_weights[k]

                        flat_train_inputs = np.concatenate(train_inputs)
                        flat_train_outputs = np.concatenate(train_outputs)
                        flat_train_weights = np.concatenate(
                            [np.ones(len(input_)) * weight for input_, weight in zip(train_inputs, train_weights)])
                        mask = (flat_train_outputs != -1)
                        flat_train_inputs = flat_train_inputs[mask]
                        flat_train_outputs = flat_train_outputs[mask]
                        flat_train_weights = flat_train_weights[mask]

                        flat_test_inputs = np.concatenate(test_inputs)
                        flat_test_outputs = np.concatenate(test_outputs)
                        flat_test_weights = np.concatenate(
                            [np.ones(len(input_)) * weight for input_, weight in zip(test_inputs, test_weights)]
                        )
                        mask = (flat_test_outputs != -1)
                        flat_test_inputs = flat_test_inputs[mask]
                        flat_test_outputs = flat_test_outputs[mask]
                        flat_test_weights = flat_test_weights[mask]


                        print('Starting training for fold %s...' % k)
                        dtrain = xgb.DMatrix(
                            flat_train_inputs,
                            label=flat_train_outputs,
                            weight=flat_train_weights, feature_names=feature_names)

                        dval = xgb.DMatrix(
                            flat_test_inputs,
                            label=flat_test_outputs,
                            weight=flat_test_weights, feature_names=feature_names)

                        params = {'max_depth': max_depth,
                                  'min_child_weight': min_child_weight,
                                  'gamma': gamma,
                                  'eta': eta,
                                  'objective': 'binary:logistic',
                                  'nthread': nthread,
                                  'eval_metric': 'logloss'  # 'aucpr'
                                  }
                        if seed is not None:
                            params['random_state'] = seed

                        bst = xgb.train(
                            params,
                            dtrain,
                            num_boost_round=num_boost_round,
                            evals=[(dval, 'eval')],
                            obj=None,
                            feval=None,
                            maximize=None,
                            early_stopping_rounds=5,
                            verbose_eval=True
                        )
                        score = bst.best_score
                        current_bsts.append(bst)
                        current_scores.append(score)
                    print('Training completed for fold %s! Saving model' % k)
                    current_score = np.mean(current_scores)
                    if current_score >= best_score:
                        print('Found current best %.2f' % current_score)
                        best_score = current_score
                        best_bsts = copy.deepcopy(current_bsts)

        '''
        Note that here, we use a simple random forest classifier and do not run any hyperparameter search.
        In the paper, we used xgboost and extensive hyperparameter search on the validation set. 
        '''

    all_cross_predictions = []
    for k in range(5):
        print('Performing predictions on the test set for fold %s...' % (k + 1))
        best_bsts[k].save_model(model_folder + model_names[k] + '.model')
        all_cross_predictions.append( np.array([best_bsts[k].predict(xgb.DMatrix(inputs_, feature_names=feature_names)) for inputs_ in list_inputs[k]] ) )

    all_cross_predictions = np.concatenate(all_cross_predictions)
    all_labels = np.concatenate(list_outputs)
    all_weights = np.concatenate(list_weights)

    if not os.path.isdir(paths.library_folder + 'plots/'):
        os.mkdir(paths.library_folder + 'plots/')

    fig, ax = make_PR_curves(
        [all_labels],
        [all_cross_predictions],
        [all_weights],
        ['Cross-validation'],
        title='B-cell epitope prediction: %s' % model_name,
        figsize=(10, 10),
        margin=0.05, grid=0.1
        , fs=25)

    fig.savefig(paths.library_folder + 'plots/PR_curve_BCE_%s.png' % model_name, dpi=300)






