import preprocessing.pipelines as pipelines
import utilities.dataset_utils as dataset_utils
import utilities.wrappers as wrappers
import network.scannet as scannet
import pandas as pd
import numpy as np
import utilities.paths as paths
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os


def make_PR_curves(
        all_labels,
        all_predictions,
        all_weights,
        subset_names,
        title = '',
        figsize=(10, 10),
        margin=0.05,grid=0.1
        ,fs=25):
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
                label='%s (AUCPR= %.3f)' % (subset_names[i], all_AUCPRs[i]))
    plt.xticks(np.arange(0, 1.0 + grid, grid), fontsize=fs * 2/3)
    plt.yticks(np.arange(0, 1.0 + grid, grid), fontsize=fs * 2/3)
    plt.xlim([0 - margin, 1 + margin])
    plt.ylim([0 - margin, 1 + margin])
    plt.grid()

    plt.legend(fontsize=fs)
    plt.xlabel('Recall', fontsize=fs)
    plt.ylabel('Precision', fontsize=fs)
    plt.title(title,fontsize=fs)
    plt.tight_layout()
    return fig, ax





if __name__ == '__main__':
    '''
    Script to train and evaluate ScanNet on the Protein-protein binding site data set.
    Model is trained from scratch.
    '''
    check = False # Check = True to verify installation, =False for full training.
    train = False # True to retrain, False to evaluate the model shown in paper.
    use_evolutionary = False # True to use evolutionary information (requires hhblits and a sequence database), False otherwise.
    Lmax_aa = 256 if check else 1024
    ''' 
    Maximum length of the protein sequences.
    Sequences longer than Lmax_aa are truncated, sequences shorter are grouped and processed using the protein serialization trick (see Materials and Methods of paper).
    If memory allows it, use the largest protein length found in the dataset.
    For PPBS, we used Lmax_aa = 1024 in the paper.
    '''
    epochs_max = 2 if check else 100


    if train: # Retrain model.
        model_name = 'ScanNet_PPI_retrained'
        if not use_evolutionary:
            model_name += '_noMSA'
        if check:
            model_name += '_check'
    else: # Evaluate paper model.
        model_name = 'ScanNet_PPI'
        if not use_evolutionary:
            model_name += '_noMSA'



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


    #%% Gather and preprocess each dataset.

    pipeline = pipelines.ScanNetPipeline(
        with_atom=True,
        aa_features='pwm' if use_evolutionary else 'sequence',
    )


    list_dataset_locations = ['datasets/PPBS/labels_%s.txt'% dataset for dataset in list_datasets]
    dataset_table = pd.read_csv('datasets/PPBS/table.csv',sep=',')


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

        '''
        Build processed dataset. For each protein chain, build_processed_chain does the following:
        1. Download the pdb file (biounit=True => Download assembly file, biounit=False => Download asymmetric unit file).
        2. Parse the pdb file.
        3. Construct atom and residue point clouds, determine triplets of indices for each atomic/residue frame.
        4. If evolutionary information is used, build an MSA using HH-blits and calculates a Position Weight Matrix (PWM).
        5. If labels are provided, aligns them onto the residues found in the pdb file.
        '''
        inputs,outputs,failed_samples = pipeline.build_processed_dataset(
            'PPBS_%s'%(dataset+'_check' if check else dataset),
            list_origins=list_origins, # Mandatory
            list_resids=list_resids, # Optional
            list_labels=list_labels, # Optional
            biounit=True, # Whether to use biological assembly files or the regular pdb files (asymmetric units). True for PPBS data set, False for BCE data set.
            save = True, # Whether to save the results in pickle file format. Files are stored in the pipeline_folder defined in paths.py
            fresh = False, # If fresh = False, attemps to load pickle files first.
            permissive=True, # Will not stop if some examples fail. Errors notably occur when a biological assembly file is updated.
        )

        weights = np.array(dataset_table['Sample weight'][ dataset_table['Set'] == dataset_name ] )
        if check:
            weights = weights[:10]
        weights = np.array([weights[b] for b in range(len(weights)) if not b in failed_samples])

        list_inputs.append(inputs)
        list_outputs.append(outputs)
        list_weights.append( weights )





    '''
    Input format:
    [
    input[0]: amino acid-wise triplets of indices for constructing frames ([Naa,3], integer).
    input[1]: amino acid-wise attributes: either one-hot-encoded sequence or position-weight matrix ([Naa,20/21],).
    input[2]: amino acid sequence index ([Naa,1], integer).
    input[3]: amino acid point cloud for constructing frames ([2*Naa+1,3] Calpha,sidechains).
    input[4]: atom-wise triplets of indices for constructing frames ([Natom,3], integer).
    input[5]: amino acid-wise attributes: atom group type ([Natom,1] integer)
    input[6]: atom sequence index ([Natom,1], integer).
    input[7]: atom point cloud for constructing frames ([Natom+Nvirtual atoms,3]).
    ]
    
    Output format:
    output: amino acid-wise binary label, one-hot-encoded as [Naa,2].
    
    Weight format: List of chain-wise weights (optional).
    '''

    if train:

        #%% Initialize model

        train_inputs = list_inputs[0]
        train_outputs = list_outputs[0]
        train_weights = list_weights[0]

        validation_inputs = [np.concatenate([list_inputs[i][j] for i in [1,2,3,4] ] ) for j in range( len(list_inputs[0]) ) ]
        validation_outputs = np.concatenate([list_outputs[i] for i in [1,2,3,4]])
        validation_weights = np.concatenate([list_weights[i] for i in [1,2,3,4]])


        model, extra_params = scannet.initialize_ScanNet(
            train_inputs,
            train_outputs,
            with_atom=True, # Whether to use atomic coordinates or not.
            Lmax_aa=Lmax_aa, # Maximum protein length used for training
            K_aa=16, # Size of neighborhood for amino acid Neighborhood Embedding Module (NEM)
            K_atom=16, # Size of neighborhood for atom Neighborhood Embedding Module (NEM)
            K_graph=32, # Size of neighborhood for Neighborhood Attention Module (NAM)
            Dmax_aa=11., # Cut-off distance for the amino acid NEM. Only used when initializing the aa gaussian kernels.
            Dmax_atom=4., # Cut-off distance for the atom NEM. Only used when initializing the gaussian kernels.
            Dmax_graph=13., # Cut-off distance for the amino acid NAM. Only used when initializing the gaussian kernels.
            N_aa=32, # Number of gaussian kernels for amino acid NEM
            N_atom=32, # Number of gaussian kernels for atom NEM
            N_graph=32, # Number of gaussian kernels for amino acid NAM
            nfeatures_aa=21 if use_evolutionary else 20, # Number of amino acid-wise input attributes.
            nfeatures_atom=12, # Number of atom-wise input attributes (categorical variable).
            nembedding_atom=12, # Dimension of atom attribute embedding. If = nfeatures_atom, use non-trainable one-hot-encoding.
            nembedding_aa=32, # Dimension of amino acid attribute embedding.
            nembedding_graph=1, # Number of values per edge for the NAM.
            dense_pooling=64, # Number of channels for atom -> amino acid pooling operation.
            nattentionheads_pooling=64,  # Number of attention heads for atom -> amino acid pooling operation.
            nfilters_atom=128, # Number of atomic spatio-chemical filters
            nfilters_aa=128, # Number of amino acid spatio-chemical filters
            nfilters_graph=2, # Number of outputs for NAM.
            nattentionheads_graph=1, # Number of attention heads for NAM.
            filter_MLP=[32], # Dimensionality reduction (trainable dense layer) applied after amino acid NEM and before NAM.
            covariance_type_atom='full', # Full or diagonal covariance matrix for atom NEM module
            covariance_type_aa='full', # Full or diagonal covariance matrix for amino acid NEM module
            covariance_type_graph='full', # Full or diagonal covariance matrix for graph NEM module
            activation='relu', # Activation function
            coordinates_atom=['euclidian'], # Local coordinate system used for the atom NEM
            coordinates_aa=['euclidian'], # Local coordinate system used for the amino acid NEM
            frame_aa='triplet_sidechain', # Choice of amino acid frames (backbone-only also supported).
            coordinates_graph=['distance', 'ZdotZ', 'ZdotDelta', 'index_distance'], # Local coordinate system used for the amino acid NAM
            index_distance_max_graph=8, # Maximum sequence distance used.
            l12_atom=2e-3, # Sparse regularization for atom NEM.
            l12_aa=2e-3, # Sparse regularization for amino acid NEM.
            l12_pool=2e-3, # Sparse regularization for atom to amino acid pooling.
            optimizer='adam', # Optimizer.
            batch_size=1, # Batch size.
            epochs=epochs_max,  # Maximum number of epochs
            initial_values_folder = paths.initial_values_folder,
            save_initial_values= False if check else True, # Cache the initial Gaussian kernels for next training.
            n_init=2, # Parameter for initializing the Gaussian kernels. Number of initializations for fitting the GMM model with sklearn. 10 were used for the paper.

        )




        #%% Train!

        extra_params['validation_data'] = (
            validation_inputs, validation_outputs, validation_weights)
        extra_params['callbacks'] = [
            EarlyStopping(monitor='val_categorical_crossentropy', min_delta=0.001, patience=5,
                          verbose=1, mode='min', restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_categorical_crossentropy', factor=0.5,
                              patience=2, verbose=1, mode='min', min_delta=0.001, cooldown=2)
        ]

        history = model.fit(train_inputs, train_outputs,
                            sample_weight=train_weights, **extra_params)
        print('Training completed! Saving model')
        model.save(paths.model_folder +model_name)

    else:
        # No training. Load trained model.
        model = wrappers.load_model(paths.model_folder +model_name,Lmax=Lmax_aa)

    #%% Predict for test set and evaluate performance.

    print('Performing predictions on the test set...')
    test_predictions = [
        model.predict(
            list_inputs[i],
            return_all=False, # Only return the binding site probability p. return_all=True gives [1-p,p] for each residue.
            batch_size=1
        )
        for i in [5,6,7,8]
    ]


    test_labels = [ np.array([
                np.argmax(label, axis=1)[:Lmax_aa] # Back to 0-1 labels from one-hot encoding and truncate to Lmax.
                for label in list_outputs[i]]) for i in [5,6,7,8] ]


    test_weights = [list_weights[i] for i in [5,6,7,8] ]



    print('Evaluating predictions on the test set...')

    if not os.path.isdir(paths.library_folder + 'plots/'):
        os.mkdir(paths.library_folder + 'plots/')

    fig,ax = make_PR_curves(
            test_labels,
            test_predictions,
            test_weights,
            list_dataset_names[5:],
            title = 'Protein-protein binding site prediction: %s'%model_name,
            figsize=(10, 10),
            margin=0.05,grid=0.1
            ,fs=16)

    fig.savefig(paths.library_folder + 'plots/PR_curve_PPBS_%s.png'%model_name,dpi=300)












