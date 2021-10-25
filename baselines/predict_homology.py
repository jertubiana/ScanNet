from utilities import wrappers,dataset_utils,chimera
from preprocessing import PDBio,PDB_processing
from datetime import datetime
import os
from multiprocessing import Pool
from functools import partial
import numpy as np
from keras.layers import Input, TimeDistributed, Masking, Dense, Dot, Lambda, BatchNormalization
from keras.models import Model
from network.embeddings import MaskedBatchNormalization
import tensorflow as tf
from keras.engine.base_layer import Layer
from utilities.paths import homology_folder,predictions_folder,path_to_multiprot,library_folder
import argparse
import predict_bindingsites

if not os.path.isdir(homology_folder):
    os.mkdir(homology_folder)
cache_folder = homology_folder + 'cache/'
if not os.path.isdir(cache_folder):
    os.mkdir(cache_folder)

#%% Multiprot preprocessing
list_aa = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
      'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V',  'W', 'Y']

def parseOutputMultiprot(location):
    solutions = []
    count_solutions = 0
    read_sites = False
    with open(location,'r') as f:
        for line in f:
            if 'Solution Num' in line:
                if count_solutions>0:
                    sites_1 = np.array(sites_1)
                    sites_2 = np.array(sites_2)
                    seqID = (sites_1[:, 1] == sites_2[:, 1]).mean()
                    solutions.append(( sites_1,sites_2,L,RMSD,seqID) )
                count_solutions +=1
                RMSD = None
                seqID = None
                L = None
                read_sites = False
                sites_1 = []
                sites_2 = []
            elif 'RMSD' in line:
                RMSD =  float(line.split(' : ')[1][:-1])
            elif 'Match List (Chain_ID.Res_Type.Res_Num) :' in line:
                L = int(line.split(' : ')[1][:-1])
                read_sites = True
            elif 'End of Match List' in line:
                read_sites = False
            elif read_sites:
                site_1 = line[:9].replace(' ', '').split('.')
                site_2 = line[9:].replace(' ', '').replace('\n', '').split('.')
                if (site_1[1] in list_aa) & (site_2[1] in list_aa):
                    sites_1.append(site_1)
                    sites_2.append(site_2)
    sites_1 = np.array(sites_1)
    sites_2 = np.array(sites_2)

    seqID = (sites_1[:, 1] == sites_2[:, 1]).mean()
    solutions.append(( sites_1, sites_2, L, RMSD, seqID) )
    return solutions


def runMultiprot(pdb_id1, pdb_id2, temporary_folder=homology_folder,
        multiprot_bin=path_to_multiprot):
    if isinstance(pdb_id1,tuple):
        pdb1 = PDBio.getPDB(pdb_id1[0])[0]
        chain1 = pdb_id1[1]
    else:
        pdb1, chain1 = PDBio.getPDB(pdb_id1)
    if isinstance(pdb_id2,tuple):
        pdb2 = PDBio.getPDB(pdb_id2[0])[0]
        chain2 = pdb_id2[1]
    else:
        pdb2, chain2 = PDBio.getPDB(pdb_id2)
    cwd = os.getcwd()
    timestamp = str(datetime.now()).replace(':','_').replace(' ','_')
    work_folder = temporary_folder + 'multiprot_%s_%s_%s/' % (pdb1.split('/')[-1].split('.')[0], pdb2.split('/')[-1].split('.')[0],timestamp)
    if not os.path.isdir(work_folder):
        os.mkdir(work_folder)
    os.chdir(work_folder)
    if chain1 != 'all':
        pdb1 = PDBio.extract_chains(pdb1, chain1, work_folder + 'first.pdb')
    if chain2 != 'all':
        pdb2 = PDBio.extract_chains(pdb2, chain2, work_folder + 'second.pdb')
    command = '%s %s %s' % (multiprot_bin, pdb1, pdb2)
    os.system(command)
    try:
        output = parseOutputMultiprot('2_sol.res')
    except:
        output = [ (np.array([]), np.array([]), 0, 0, 0) ]

    os.chdir(cwd)
    os.system('rm -r %s' % work_folder)
    return output



#%%


class MaskedSoftmax(Layer):
    def __init__(self, axis=-1, eps=1e-6, **kwargs):
        self.supports_masking = True
        self.axis = axis
        self.eps = eps
        super(MaskedSoftmax, self).__init__(**kwargs)
        return

    def call(self, inputs, mask=None):
        outputs = inputs - tf.reduce_max(inputs, axis=self.axis, keepdims=True)
        outputs = tf.exp(outputs)
        if mask is not None:
            outputs *= tf.expand_dims(tf.cast(mask, dtype=tf.float32), axis=-1)
        outputs /= tf.reduce_sum(outputs, axis=self.axis, keepdims=True) + self.eps
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, mask=None):
        return mask


def HomologyWeightLearner(nhits_cut, nalignment_features, MLP=[], baseline_probability=0.2):
    alignment_features = Input(shape=(nhits_cut, nalignment_features))
    alignment_labels = Input(shape=(nhits_cut,))
    masked_alignment_features = Masking()(alignment_features)
    log_weights = masked_alignment_features
    for k, n_filters in enumerate(MLP):
        log_weights = TimeDistributed(Dense(n_filters, use_bias=True, activation='tanh'),
                                      name='MLP_layer_%s' % (k + 1))(log_weights)
    log_weights = TimeDistributed(Dense(1), name='log_weights_unnormalized')(log_weights)
    log_weights = MaskedBatchNormalization(scale=True, axis=-1, name='log_weights')(log_weights)
    weights = MaskedSoftmax(axis=1, eps=1)(log_weights)
    output = Dot(axes=(1, 1), normalize=False)([weights,
                                                Lambda(lambda x: x - baseline_probability)(alignment_labels)]
                                               )
    output = Lambda(lambda x: x + baseline_probability)(output)
    model = Model(inputs=[alignment_features, alignment_labels], outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

def HomologyWeightPredictor(nalignment_features, MLP=[]):
    input_layer = Input(shape=(nalignment_features,))
    intermediate = input_layer
    for k, n_filters in enumerate(MLP):
        intermediate = Dense(n_filters, use_bias=True, activation='tanh', name='MLP_layer_%s' % (k + 1))(intermediate)
    output = Dense(1, name='log_weights_unnormalized')(intermediate)
    output = BatchNormalization(scale=True, axis=-1, name='log_weights')(output)
    predictor = Model(inputs=input_layer, outputs=output)
    return predictor


class HomologyModel():
    def __init__(self, template_labels_file, cache_folder=cache_folder, ncores=4, name='homology_PPBS',baseline_probability=0.2, baseline_log_weight=0., MLP = [10,5], ntemplates_max=None,cutoff_labels=1,biounit=True):
        self.cache_folder = cache_folder

        if not isinstance(template_labels_file,list):
            template_labels_file = [template_labels_file]
        self.template_origins, self.template_sequences, self.template_resids, self.template_labels = [],[],[],[]
        for template_labels_file_ in template_labels_file:
            template_origins, template_sequences, template_resids, template_labels = dataset_utils.read_labels(template_labels_file_)
            self.template_origins += list(template_origins)
            self.template_sequences += list(template_sequences)
            self.template_resids += list(template_resids)
            self.template_labels += list(template_labels)

        self.template_pdbs = [self.cache_folder + template_origin + '.pdb' for template_origin in self.template_origins]
        for template_origin,template_pdb in zip(self.template_origins,self.template_pdbs):
            if not os.path.exists(template_pdb):
                location,chains = PDBio.getPDB(template_origin,biounit=biounit)
                PDBio.extract_chains(location,chains,template_pdb)

        self.template_labels = [template_label >= cutoff_labels for template_label in self.template_labels]
        self.template_resids = [['%s_%s' % (resid[0], resid[1]) for resid in template_resid] for template_resid in
                                self.template_resids]

        if ntemplates_max is not None:
            self.template_origins = self.template_origins[:ntemplates_max]
            self.template_sequences = self.template_sequences[:ntemplates_max]
            self.template_pdbs = self.template_pdbs[:ntemplates_max]
            self.template_resids = self.template_resids[:ntemplates_max]
            self.template_labels = self.template_labels[:ntemplates_max]

        self.name = name
        self.ncores = ncores
        self.baseline_probability = baseline_probability
        self.baseline_log_weight = baseline_log_weight
        self.MLP = MLP
        try:
            self.homology_weight_predictor = wrappers.load_model(homology_folder + 'calibration_%s' % self.name)
        except:
            self.homology_weight_predictor = None

    def get_log_weight(self, L_sequence, L_alignment, RMSD_alignment, seqID_alignment):
        if self.homology_weight_predictor is not None:
            return self.homology_weight_predictor.predict(
                np.array([L_alignment / L_sequence, RMSD_alignment, seqID_alignment])[np.newaxis])[0]
        else:
            return np.log(1e1) * (L_alignment / L_sequence - 0.33) / (1 - 0.33) + np.log(1e1) * (
                        seqID_alignment - 0.2) / (
                           1 - 0.2)

    def fit(self, train_labels_file, nexamples_max=100, nhits_cut=1000,biounit=True):
        if not isinstance(train_labels_file,list):
            train_labels_file = [train_labels_file]
        nfiles = len(train_labels_file)
        train_origins, train_sequences, train_resids, train_labels = [],[],[],[]
        for train_labels_file_ in train_labels_file:
            train_origins_, train_sequences_, train_resids_, train_labels_ = dataset_utils.read_labels(train_labels_file_)
            if nexamples_max is not None:
                train_origins_ = train_origins_[:nexamples_max//nfiles]
                train_sequences_ = train_sequences_[:nexamples_max//nfiles]
                train_resids_ = train_resids_[:nexamples_max//nfiles]
                train_labels_ = train_labels_[:nexamples_max//nfiles]
            train_origins += list(train_origins_)
            train_sequences += list(train_sequences_)
            train_resids += list(train_resids_)
            train_labels += list(train_labels_)
        train_pdbs = [self.cache_folder + train_origin + '.pdb' for train_origin in train_origins]
        for train_origin,train_pdb in zip(train_origins,train_pdbs):
            if not os.path.exists(train_pdb):
                location,chains = PDBio.getPDB(train_origin,biounit=biounit)
                PDBio.extract_chains(location,chains,train_pdb)

        train_Ls = [len(train_label) for train_label in train_labels]
        B = len(train_labels)

        train_hit_alignment_features = []
        train_hit_alignment_labels = []
        for pdb in train_pdbs:
            hit_alignment_features, hit_alignment_labels = self.predict(pdb, return_all=True)
            train_hit_alignment_features.append(hit_alignment_features)
            train_hit_alignment_labels.append(hit_alignment_labels)

        nsites = sum(train_Ls)

        padded_alignment_features = np.zeros([nsites, nhits_cut, 3])
        padded_alignemnt_labels = np.zeros([nsites, nhits_cut])
        target_labels = np.concatenate(train_labels, axis=0) >= 5

        n = 0
        for b in range(B):
            for l in range(train_Ls[b]):
                alignment_features = train_hit_alignment_features[b][l]
                alignment_labels = train_hit_alignment_labels[b][l]
                nhit = len(alignment_features)
                nhit_cut = min(nhit, nhits_cut)
                if nhit > 0:
                    order = np.argsort(alignment_features[:, 0])[::-1]
                    padded_alignment_features[n, :nhit_cut] = alignment_features[order[:nhit_cut]]
                    padded_alignemnt_labels[n, :nhit_cut] = alignment_labels[order[:nhit_cut]]
                n += 1

        homology_weight_learner = HomologyWeightLearner(nhits_cut, 3, MLP=self.MLP,
                                                        baseline_probability=self.baseline_probability)
        homology_weight_learner.fit([padded_alignment_features, padded_alignemnt_labels], target_labels[:, np.newaxis],
                                    epochs=20, batch_size=32)
        self.homology_weight_predictor = wrappers.Predictor_wrapper(HomologyWeightPredictor, 3, MLP=self.MLP)
        for layer in self.homology_weight_predictor.model.layers[1:]:
            layer.set_weights(homology_weight_learner.get_layer(layer.name).get_weights())
        self.homology_weight_predictor.save(homology_folder + 'calibration_%s' % self.name)
        return

    def predict(self, pdb_id, return_all=False,biounit=True):
        if isinstance(pdb_id, tuple):
            pdb = PDBio.getPDB(pdb_id[0],biounit=biounit)[0]
            chain = pdb_id[1]
        else:
            pdb, chain = PDBio.getPDB(pdb_id)
        if chain != 'all':
            timestamp = str(datetime.now()).replace(':', '_').replace(' ', '_')
            pdb = PDBio.extract_chains(pdb, chain, homology_folder + 'tmp_%s.pdb' % timestamp)
            chain = 'all'
            delete_after = True
        else:
            delete_after = False

        struct, chain = PDBio.load_chains(file=pdb,chain_ids= chain)
        sequence = PDB_processing.process_chain(chain)[0]
        resids = PDB_processing.get_PDB_indices(chain, return_chain=True, return_model=True)
        query_resid = ['%s_%s' % (resid[1], resid[2]) for resid in resids]
        L = len(sequence)
        prediction_unnormalized = np.zeros(L)
        total_weights = np.zeros(L)
        partial_multiprot = partial(runMultiprot, pdb)
        pool = Pool(self.ncores)
        all_solutions = pool.map(partial_multiprot, self.template_pdbs)
        pool.close()
        if return_all:
            hit_alignment_features = [[] for _ in range(L)]
            hit_alignment_labels = [[] for _ in range(L)]

        for k, solutions in enumerate(all_solutions):
            template_origin = self.template_origins[k]
            template_sequence = self.template_sequences[k]
            template_resid = self.template_resids[k]
            template_labels = self.template_labels[k]
            print(template_origin,len(template_sequence) )
            for (
                    subset_resids_query, subset_resids_template, L_alignment, RMSD_alignment,
                    seqID_alignment) in solutions:
                if L_alignment > 0:
                    subset_index_query = np.array(
                        [query_resid.index('%s_%s' % (resid[0], resid[2])) for resid in subset_resids_query])
                    subset_index_template = np.array(
                        [template_resid.index('%s_%s' % (resid[0][0], resid[2])) for resid in subset_resids_template])
                    if return_all:
                        for u, v in zip(subset_index_query, subset_index_template):
                            hit_alignment_features[u].append([L_alignment / L, RMSD_alignment, seqID_alignment])
                            hit_alignment_labels[u].append(template_labels[v])
                    else:
                        weight = np.exp(self.get_log_weight(L, L_alignment, RMSD_alignment, seqID_alignment))
                        prediction_unnormalized[subset_index_query] += weight * template_labels[subset_index_template]
                        total_weights[subset_index_query] += weight
        if return_all:
            if delete_after:
                os.system('rm %s' % pdb)
            return [np.array(x) for x in hit_alignment_features], [np.array(x) for x in hit_alignment_labels]
        else:
            weight_baseline = np.exp(self.baseline_log_weight)
            prediction_unnormalized += weight_baseline * self.baseline_probability
            total_weights += weight_baseline
            prediction = prediction_unnormalized / total_weights
            if delete_after:
                os.system('rm %s' % pdb)
            return prediction, sequence, resids



def test_code():
    '''
    A short test script. We create a dummy dataset and check that training is working fine.
    '''
    Lmin = 5
    Lmax = 10
    B = 100
    plabel = 0.2
    nalignment_features = 1
    nhits_min = 0
    nhits_max = 10

    Ls = [np.random.randint(Lmin, Lmax + 1) for _ in range(B)]
    labels = [np.random.rand(L) < plabel for L in Ls]
    nhits = [np.random.randint(nhits_min, nhits_max + 1, size=[L]) for L in Ls]
    hit_alignment_features = [
        [np.random.randn(hit, nalignment_features) for hit in nhit] for nhit in nhits
    ]

    hit_alignment_labels = [
        [np.random.rand(hit) < plabel for hit in nhit] for nhit in nhits
    ]

    # %%
    nhits_cut = 2
    index_sort = 0
    nsites = sum(Ls)
    padded_alignment_features = np.zeros([nsites, nhits_cut, nalignment_features])
    padded_alignemnt_labels = np.zeros([nsites, nhits_cut])
    target_labels = np.concatenate(labels, axis=0)

    n = 0
    for b in range(B):
        for l in range(Ls[b]):
            alignment_features = hit_alignment_features[b][l]
            alignment_labels = hit_alignment_labels[b][l]
            order = np.argsort(alignment_features[:, index_sort])[::-1]
            nhit = len(alignment_features)
            nhit_cut = min(nhit, nhits_cut)
            padded_alignment_features[n, :nhit_cut] = alignment_features[order[:nhit_cut]]
            padded_alignemnt_labels[n, :nhit_cut] = alignment_labels[order[:nhit_cut]]
            n += 1

    model = HomologyWeightLearner(nhits_cut, nalignment_features, MLP=[2, 2], baseline_probability=plabel)
    predictions = model.predict([padded_alignment_features, padded_alignemnt_labels])
    print(predictions.min(), predictions.max())
    model.fit([padded_alignment_features, padded_alignemnt_labels], target_labels[:, np.newaxis], epochs=10,
              batch_size=1)
    predictor = HomologyWeightPredictor(nalignment_features, MLP=[2, 2])
    for layer in predictor.layers[1:]:
        layer.set_weights(model.get_layer(layer.name).get_weights())
    log_weights = predictor.predict(hit_alignment_features[0][1])
    return log_weights


template_databases = {
    'interface': library_folder + 'datasets/PPBS/labels_train.txt',
    'epitope': [library_folder + 'datasets/BCE/labels_fold%s.txt'%k for k in range(1,5)],
    'idp': [library_folder + 'datasets/PIDPBS/labels_fold%s.txt'%k for k in range(1,5)]
}

train_databases = {
    'interface': [library_folder + 'datasets/PPBS/%s'%x for x in ['labels_validation_70.txt','labels_validation_homology.txt','labels_validation_none.txt','labels_validation_topology.txt'] ],
    'epitope': library_folder + 'datasets/BCE/labels_fold5.txt',
    'idp': library_folder + 'datasets/PIDPBS/labels_fold0.txt',
}

names = {
    'interface':'homology_PPBS',
    'epitope':'homology_BCE',
    'idp':'homology_PIDPBS',
}

biounits = {
    'interface':True,
    'epitope':False,
    'idp':False
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict binding sites in PDB files using homology modeling')
    parser.add_argument('input', type=str,
                        help='Three input formats. i) A pdb id (1a3x)\
                   ii) Path to pdb file (structures/1a3x.pdb)\
                   iii) Path to text file containing list of pdb files (one per line) (1a3x \n 2kho \n ...) \
                   For performing prediction only on specfic chains, append "_" and the list of chains. (e.g. 1a3x_AB)')

    parser.add_argument('--predictions_folder', dest='predictions_folder',
                        default=predictions_folder,
                        help='Input name')
    parser.add_argument('--mode', dest='mode',
                        default='interface',
                        help='Prediction mode (interface, epitope)')
    parser.add_argument('--pdb', dest='biounit', action='store_const',
                        const=False, default=True,
                        help='Predict from pdb file (default= predict from biounit file)')
    parser.add_argument('--ncores', dest='ncores',
                        default=1,
                        help='Number of cores')
    parser.add_argument('--no_output', dest='annotate',action='store_const',
                        const=False,
                        default=True,
                        help='Output annotated structure')

    args = parser.parse_args()
    input = args.input
    if '.txt' in input:
        list_inputs = []
        with open(input, 'r') as f:
            for line in f:
                list_inputs.append(line[:-1])
    else:
        list_inputs = [input]

    predictions_folder = args.predictions_folder
    mode = args.mode
    biounit = args.biounit
    ncores = args.ncores
    annotate = args.annotate

    homology_model = HomologyModel(template_databases[mode],ncores=ncores,name=names[mode],biounit=biounits[mode])
    if homology_model.homology_weight_predictor is None:
        print('Model was never trained, need to fit it')
        homology_model.fit(train_databases[mode], nexamples_max=100, nhits_cut=1000)

    for input in list_inputs:
        file,chain = PDBio.getPDB(input,biounit=biounit)
        pdb = file.split('/')[-1].split('.')[0]
        output_folder = predictions_folder + pdb + '_(' + PDBio.format_chain_id(chain) + ')' + '_homology_%s'%mode
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        tmp_location = PDBio.extract_chains(file,chain, output_folder+'tmp.pdb' )
        prediction, sequence, resids = homology_model.predict(tmp_location)
        predict_bindingsites.write_predictions(output_folder+'predictions.csv', resids, sequence, prediction)
        if annotate:
            chimera.annotate_pdb_file(tmp_location,output_folder+'predictions.csv','annotated_%s.pdb' % pdb)
        os.system('rm %s'%tmp_location)