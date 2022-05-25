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
import sys

def make_PR_curves(
        all_labels,
        all_predictions,
        all_weights,
        subset_names,
        title = '',
        figsize=(10, 10),
        margin=0.05,grid=0.1
        ,fs=25,legend_fs=15):
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, auc

    nSubsets = len(subset_names)
    subsetColors = ['C%s' % k for k in range(nSubsets)]

    all_PR_curves = []
    all_AUCPRs = []

    for i in range(nSubsets):
        labels = all_labels[i]
        predictions = all_predictions[i]
        weights = all_weights[i]
        weights_repeated = np.array([np.ones(len(label)) * weight for label, weight in zip(labels, weights)], dtype=np.object)
        labels_flat=np.concatenate(labels)
        predictions_flat=np.concatenate(predictions)
        is_nan = np.isnan(predictions_flat) | np.isinf(labels_flat)
        is_missing = np.isnan(labels_flat) | (labels_flat<0)
        count_nan = is_nan.sum()
        if count_nan>0:
            print('Found %s nan predictions in subset %s'%(count_nan,subset_names[i]) )
            predictions_flat[is_nan] = np.nanmedian(predictions_flat)

        precision, recall, _ = precision_recall_curve(labels_flat[~is_missing],predictions_flat[~is_missing],
            sample_weight=np.concatenate(weights_repeated)[~is_missing] )
        all_PR_curves.append((precision,recall) )
        all_AUCPRs.append( auc(recall, precision) )


    fig, ax = plt.subplots(figsize=figsize)
    for i in range(nSubsets):
        ax.plot(all_PR_curves[i][1], all_PR_curves[i][0], color=subsetColors[i],linewidth=2.0,
                label='%s (%.3f)' % (subset_names[i], all_AUCPRs[i]))
    plt.xticks(np.arange(0, 1.0 + grid, grid), fontsize=fs * 2/3)
    plt.yticks(np.arange(0, 1.0 + grid, grid), fontsize=fs * 2/3)
    plt.xlim([0 - margin, 1 + margin])
    plt.ylim([0 - margin, 1 + margin])
    plt.grid()

    plt.legend(fontsize=legend_fs)
    plt.xlabel('Recall', fontsize=fs)
    plt.ylabel('Precision', fontsize=fs)
    plt.title(title,fontsize=fs)
    plt.tight_layout()
    return fig, ax


if __name__ == '__main__':
    '''
    Script to train and evaluate handcrafted feature baseline on the Protein-protein binding site data set.
    '''
    check = False  # Check = True to verify installation, =False for full training.
    use_evolutionary = True  # True to use evolutionary information (requires hhblits and a sequence database), False otherwise.

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
            'secondary', # secondary structure (DSSP, 8 classes)
            'conservation', # Conservation score.
            'pwm', # Position Weight Matrix (21-dimensional vector).
            'asa', # Relative Accessible surface area (DSSP)
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
            'secondary', # secondary structure (DSSP, 8 classes)
            'asa', # Relative Accessible surface area (DSSP)
            # 'residue_depth', # Backbone depth and sidechain depth with respect to the molecular surface (requires MSMS).
            # 'volume_index', # Surface convexity index (requires MSMS)
            'half_sphere', # Half sphere exposure (Number of residues in upper half-sphere - number of residues in bottom half-sphere)/Coordination index.
            'coordination' # Residue coordination index (13A radius)
        ]
    # If MSMS/DSSP is not installed, comment the corresponding features.

        feature_names = ['aa_%s' % x for x in pipelines.sequence_utils.aa[:20]] + \
                        pipelines.dict_num2name + \
                        ['ASA', 'Depth1', 'Depth2', 'half_sphere', 'coordination'] + \
                        ['VolIndex_%s' % k for k in range(3)]

    model_name = 'handcrafted_features_%s_xgboost'% (''.join([x[0] for x in feature_list]) )
    if seed is not None:
        model_name += '_%s'%seed


    list_datasets = [
        'train',
        'validation_70',
        'validation_homology',
        'validation_topology',
        'validation_none',
        'test_70',
        'test_homology',
        'test_topology',
        'test_none',
    ]

    list_dataset_names = [
        'Train',
        'Validation (70\%)',
        'Validation (Homology)',
        'Validation (Topology)',
        'Validation (None)',
        'Test (70\%)',
        'Test (Homology)',
        'Test (Topology)',
        'Test (None)'
    ]

    # %% Gather and preprocess each dataset.

    pipeline = pipelines.HandcraftedFeaturesPipeline(feature_list=feature_list)

    list_dataset_locations = [library_folder+'datasets/PPBS/labels_%s.txt' % dataset for dataset in list_datasets]
    dataset_table = pd.read_csv(library_folder+'datasets/PPBS/table.csv', sep=',')

    list_inputs = []
    list_outputs = []
    list_weights = []

    for dataset, dataset_name, dataset_location in zip(list_datasets, list_dataset_names, list_dataset_locations):
        # Parse label files
        (list_origins,  # List of chain identifiers (e.g. [1a3x_A,10gs_B,...])
         list_sequences,  # List of corresponding sequences.
         list_resids,  # List of corresponding residue identifiers.
         list_labels) = dataset_utils.read_labels(dataset_location)  # List of residue-wise labels

        if check:
            list_origins = list_origins[:10]
            list_sequences = list_sequences[:10]
            list_resids = list_resids[:10]
            list_labels = list_labels[:10]

        '''
        Build processed dataset. For each protein chain, build_processed_chain does the following:
        1. Download the pdb file (biounit=True => Download assembly file, biounit=False => Download asymmetric unit file).
        2. Parse the pdb file.
        3. Construct atom and residue point clouds, determine triplets of indices for each atomic/residue frame.
        4. If evolutionary information is used, build an MSA using HH-blits and calculates a Position Weight Matrix (PWM).
        5. If labels are provided, aligns them onto the residues found in the pdb file.
        '''
        inputs, outputs, failed_samples = pipeline.build_processed_dataset(
            'PPBS_%s' % (dataset + '_check' if check else dataset),
            list_origins=list_origins,  # Mandatory
            list_resids=list_resids,  # Optional
            list_labels=list_labels,  # Optional
            biounit=True,
            # Whether to use biological assembly files or the regular pdb files (asymmetric units). True for PPBS data set, False for BCE data set.
            save=True,
            ncores = 10,
            # Whether to save the results in pickle file format. Files are stored in the pipeline_folder defined in paths.py
            fresh=False,  # If fresh = False, attemps to load pickle files first.
            permissive=True,
            # Will not stop if some examples fail. Errors notably occur when a biological assembly file is updated.
        )

        weights = np.array(dataset_table['Sample weight'][dataset_table['Set'] == dataset_name])
        if check:
            weights = weights[:10]
        weights = np.array([weights[b] for b in range(len(weights)) if not b in failed_samples])

        list_inputs.append(inputs)
        list_outputs.append(outputs)
        list_weights.append(weights)



    train_inputs = list_inputs[0]
    train_outputs = list_outputs[0]
    train_weights = list_weights[0]

    validation_inputs = np.concatenate([list_inputs[i] for i in [1, 2, 3, 4]])
    validation_outputs = np.concatenate([list_outputs[i] for i in [1, 2, 3, 4]])
    validation_weights = np.concatenate([list_weights[i] for i in [1, 2, 3, 4]])

    flat_train_inputs = np.concatenate(train_inputs)
    flat_train_outputs = np.concatenate(train_outputs)
    flat_train_weights = np.concatenate(
            [np.ones(len(input_)) * weight for input_, weight in zip(train_inputs, train_weights)])
    mask = (flat_train_outputs!=-1)
    flat_train_inputs = flat_train_inputs[mask]
    flat_train_outputs = flat_train_outputs[mask]
    flat_train_weights = flat_train_weights[mask]

    flat_validation_inputs = np.concatenate(validation_inputs)
    flat_validation_outputs = np.concatenate(validation_outputs)
    flat_validation_weights = np.concatenate(
        [np.ones(len(input_)) * weight for input_, weight in zip(validation_inputs, validation_weights)]
    )
    mask = (flat_validation_outputs!=-1)
    flat_validation_inputs = flat_validation_inputs[mask]
    flat_validation_outputs = flat_validation_outputs[mask]
    flat_validation_weights = flat_validation_weights[mask]

    dtrain = xgb.DMatrix(
        flat_train_inputs,
        label=flat_train_outputs,
        weight=flat_train_weights, feature_names=feature_names)

    dval = xgb.DMatrix(
        flat_validation_inputs,
        label=flat_validation_outputs,
        weight=flat_validation_weights, feature_names=feature_names)

    best_bst = None
    best_score = 0

    for max_depth in max_depths:
        for min_child_weight in min_child_weights:
            for gamma in gammas:
                for eta in etas:
                    print(max_depth, min_child_weight, gamma, eta)
                    params = {'max_depth': max_depth,
                              'min_child_weight': min_child_weight,
                              'gamma': gamma,
                              'eta': eta,
                              'objective': 'binary:logistic',
                              'nthread': nthread,
                              'eval_metric': 'logloss'
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
                    if score >= best_score:
                        print('Found current best %.2f' % score)
                        best_score = score
                        best_bst = bst

    best_bst.save_model(model_folder + model_name + '.model')


    # %% Predict for test set and evaluate performance.
    print('Performing predictions on the test set...')
    test_predictions = [np.array([best_bst.predict(xgb.DMatrix(inputs_,feature_names=feature_names)) for inputs_ in list_inputs[i]]) for i in [5, 6, 7, 8]]
    test_labels = [list_outputs[i] for i in [5, 6, 7, 8]]

    test_weights = [list_weights[i] for i in [5, 6, 7, 8]]

    print('Evaluating predictions on the test set...')

    if not os.path.isdir(paths.library_folder + 'plots/'):
        os.mkdir(paths.library_folder + 'plots/')

    fig, ax = make_PR_curves(
        test_labels,
        test_predictions,
        test_weights,
        list_dataset_names[5:],
        title='Protein-protein binding site prediction: %s' % model_name,
        figsize=(10, 10),
        margin=0.05, grid=0.1
        , fs=16)

    fig.savefig(paths.library_folder + 'plots/PR_curve_PPBS_%s.png' % model_name, dpi=300)








