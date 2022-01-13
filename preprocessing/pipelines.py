import numpy as np
from utilities import io_utils
from utilities.paths import pipeline_folder,MSA_folder,structures_folder
from utilities.dataset_utils import align_labels
from multiprocessing import Pool
from functools import partial
import time
import os
try:
    from numba import njit
    from . import protein_frames
    from . import protein_chemistry
    from . import PDB_processing
    from . import sequence_utils
    from . import PDBio
except Exception as e:
    print('Failed to import numba-dependent packages!')
    print(e)



database_locations = {}
database_locations['dockground'] = pipeline_folder + \
    'dockground_database_processed.data'

database_locations['SabDab'] = pipeline_folder + \
    'SabDab_database_processed.data'

database_locations['disprot'] = pipeline_folder + \
    'disprot_database_processed.data'

database_nchunks = {}
database_nchunks['dockground'] = 40
database_nchunks['disprot'] = 1
database_nchunks['SabDab'] = 1

curr_float = np.float32
curr_int = np.int16

dict_dssp2num = {'H': 0, 'B': 1, 'E': 2,
                 'G': 3, 'I': 4, 'T': 5, 'S': 6, '-': 7}
dict_dssp2name = {'H': 'Alpha helix (4-12)', 'B': 'Isolated beta-bridge residue', 'E': 'Strand',
                  'G': '3-10 helix', 'I': 'Pi helix', 'T': 'Turn', 'S': 'Bend', '-': 'None'}
dict_num2name = [dict_dssp2name[key] for key in dict_dssp2name.keys()]

targets_Beff = [10, 50, 100, 200, 300, 400, 500, 1000, 2000, np.inf]

median_asa = 0.24390244  # For completion.
median_backbone_depth = 2.6421
median_sidechain_depth = 2.21609
median_halfsphere_excess_up = -0.1429
median_coordination = 30
median_volumeindex = [-0.6115, -0.4397, -0.2983]


def padd_matrix(matrix, padded_matrix=None, Lmax=800, padding_value=-1):
    if padded_matrix is None:
        ndim = matrix.ndim
        if ndim == 1:
            shape = (Lmax)
        else:
            shape = [Lmax] + list(matrix.shape[1:])
        padded_matrix = np.ones(shape, dtype=matrix.dtype)
    else:
        Lmax = padded_matrix.shape[0]

    L = matrix.shape[0]
    if L > Lmax:
        padded_matrix[:] = matrix[:Lmax]
    else:
        padded_matrix[:L] = matrix
        padded_matrix[L:] = padding_value
    padded_matrix[np.isnan(padded_matrix)] = padding_value
    return padded_matrix

def remove_nan(matrix,padding_value=0.):
    aa_has_nan = np.isnan(matrix).reshape([len(matrix),-1]).max(-1)
    matrix[aa_has_nan] = padding_value
    return matrix

def binarize_variable(matrix, thresholds):
    thresholds = np.array([-np.inf] + list(thresholds) + [np.inf])
    return (matrix[:, np.newaxis] <= thresholds[np.newaxis, 1:]) & (matrix[:, np.newaxis] > thresholds[np.newaxis, :-1])


def categorize_variable(matrix, mini, maxi, n_classes):
    return np.floor((matrix - mini) / ((maxi - mini) /n_classes)).astype(curr_int)


def binarize_categorical(matrix, n_classes, out=None):
    L = matrix.shape[0]
    matrix = matrix.astype(np.int)
    if out is None:
        out = np.zeros([L, n_classes], dtype=np.bool)
    subset = (matrix>=0) & (matrix<n_classes)
    out[np.arange(L)[subset],matrix[subset]] = 1
    return out


def secondary_structure2num(secondary_structure):
    L = len(secondary_structure)
    out = np.zeros(L, dtype=curr_int)
    for l in range(L):
        ss = secondary_structure[l]
        if ss in dict_dssp2num.keys():
            out[l] = dict_dssp2num[ss]
        else:
            out[l] = -1
    return out


def binarize_padd_residue_label(residue_labels, n_classes, Lmax=800):
    B = len(residue_labels)
    Ls = np.array([len(residue_label) for residue_label in residue_labels])
    output = np.zeros([B, Lmax, n_classes], dtype=curr_float)
    for b in range(B):
        for l in range(Ls[b]):
            label = residue_labels[b][l]
            if (label >= 0) & (label < n_classes):
                output[b, l, label] += 1
    return output




class Pipeline():
    def __init__(self, pipeline_name, pipeline_folder=pipeline_folder, *kwargs):
        self.pipeline_name = pipeline_name
        self.pipeline_folder = pipeline_folder
        self.requirements = []
        self.padded = False
        return


    def build_processed_dataset(self,
                                dataset_name,
                                list_origins=None,
                                list_resids=None,
                                list_labels=None,
                                biounit=True,
                                structures_folder=structures_folder,
                                MSA_folder=MSA_folder,
                                pipeline_folder=pipeline_folder,
                                verbose= True,
                                fresh= False,
                                save = True,
                                permissive=True,
                                overwrite=False,
                                ncores = 1
                                ):
        location_processed_dataset = pipeline_folder + dataset_name + '_%s.data' % self.pipeline_name

        found = False
        if not fresh:
            try:
                env = io_utils.load_pickle(location_processed_dataset)
                inputs = env['inputs']
                outputs = env['outputs']
                failed_samples = env['failed_samples']
                found = True
            except:
                found = False
        if not found:
            if verbose:
                print('Processed dataset not found, building it...')

            t = time.time()
            location_raw_dataset = pipeline_folder + dataset_name + '_raw.data'
            condition1 = os.path.exists(location_raw_dataset)
            condition2 = False
            if condition1:
                if verbose:
                    print('Raw dataset %s found' % dataset_name)
                env = io_utils.load_pickle(location_raw_dataset)
                if all(['all_%ss' % requirement in env.keys() for requirement in self.requirements]):
                    if verbose:
                        print('Dataset %s found with all required fields' % dataset_name)
                    condition2 = True

            if condition1 & condition2:
                inputs,outputs,failed_samples = self.process_dataset(env, label_name='all_labels' if 'all_labels' in env.keys() else None,permissive=permissive)
            else:
                if verbose:
                    print('Building and processing dataset %s' % dataset_name)
                inputs,outputs,failed_samples = self.build_and_process_dataset(
                                                      list_origins,
                                                      list_resids=list_resids,
                                                      list_labels=list_labels,
                                                      biounit=biounit,
                                                      structures_folder=structures_folder,
                                                      MSA_folder=MSA_folder,
                                                      verbose=verbose,
                                                      permissive=permissive,
                                                      overwrite=overwrite,
                                                      ncores=ncores
                                                      )
            print('Processed dataset built... (t=%.f s)' % (time.time() - t))
            if save:
                print('Saving processed dataset...')
                t = time.time()
                env = {'inputs': inputs, 'outputs': outputs,'failed_samples':failed_samples}
                io_utils.save_pickle(
                    env, location_processed_dataset)
                print('Processed dataset saved (t=%.f s)' % (time.time() - t))
        return inputs,outputs,failed_samples

    def build_and_process_dataset(self,
                                  list_origins,
                                  list_resids=None,
                                  list_labels=None,
                                  biounit=True,
                                  structures_folder=structures_folder,
                                  MSA_folder=MSA_folder,
                                  verbose=True,
                                  overwrite=False,
                                  permissive=True,
                                  ncores = 1
                                  ):
        B = len(list_origins)
        if (list_labels is not None):
            assert len(list_labels) == B
            has_labels = True
        else:
            has_labels = False

        if (list_resids is not None):
            assert len(list_resids) == B
            has_resids = True
        else:
            has_resids = False

        if ncores>1:
            ncores = min(ncores,B)
            pool = Pool(ncores)
            batch_size = int(np.ceil(B/ncores))
            batch_list_origins = [list_origins[k*batch_size: min( (k+1) * batch_size , B) ] for k in range(ncores)]
            if has_labels:
                batch_list_labels = [list_labels[k * batch_size: min((k + 1) * batch_size, B)] for k in range(ncores)]
            else:
                batch_list_labels = [None for k in range(ncores)]
            if has_resids:
                batch_list_resids = [list_resids[k * batch_size: min((k + 1) * batch_size, B)] for k in range(ncores)]
            else:
                batch_list_resids = [None for k in range(ncores)]
            _build_and_process_dataset = partial(self.build_and_process_dataset,
                                                 biounit=biounit,
                                                 structures_folder=structures_folder,
                                                 MSA_folder=MSA_folder,
                                                 verbose=verbose,
                                                 overwrite=overwrite,
                                                 permissive=permissive,
                                                 ncores = 1)
            batch_outputs = pool.starmap(_build_and_process_dataset,zip(batch_list_origins,batch_list_resids,batch_list_labels))
            pool.close()
            ## Determine if input/output are list.
            input_is_list = False
            output_is_list = False
            ninputs = 1
            noutputs = 1
            for ksuccess in range(ncores):
                if batch_outputs[ksuccess] != []:
                    input_is_list = isinstance(batch_outputs[ksuccess][0],list)
                    ninputs = len(batch_outputs[ksuccess][0])
                    if has_labels:
                        output_is_list = isinstance(batch_outputs[ksuccess][1],list)
                        noutputs = len(batch_outputs[ksuccess][1])
                    break

            if input_is_list:
                inputs = [[] for _ in range(ninputs)]
                for batch_output in batch_outputs:
                    if batch_output[0] != []:
                        for l in range(ninputs):
                            inputs[l] += list(batch_output[0][l])
                for l in range(ninputs):
                    inputs[l] = np.array(inputs[l])
            else:
                inputs = []
                for batch_output in batch_outputs:
                    if batch_output[0] != []:
                        inputs += list(batch_output[0])
                inputs = np.array(inputs)
            if has_labels:
                if output_is_list:
                    outputs = [[] for _ in range(noutputs)]
                    for batch_output in batch_outputs:
                        if batch_output[1] != []:
                            for l in range(noutputs):
                                outputs[l] += list(batch_output[1][l])
                    for l in range(noutputs):
                        outputs[l] = np.array(outputs[l])
                else:
                    outputs = []
                    for batch_output in batch_outputs:
                        if batch_output[1] != []:
                            outputs += list(batch_output[1])
                    outputs = np.array(outputs)
            else:
                outputs = None
            failed_samples = list(np.concatenate([np.array(batch_outputs[k][2],dtype=np.int)+k*batch_size for k in range(ncores)]))
            return inputs,outputs,failed_samples
        else:
            inputs = []
            outputs = []
            failed_samples = []
            for b, origin in enumerate(list_origins):
                if verbose:
                    print('Processing example %s (%s/%s)' % (origin, b, B))

                try:
                    pdbfile, chain_ids = PDBio.getPDB(origin, biounit=biounit, structures_folder=structures_folder)
                    struct, chain_objs = PDBio.load_chains(file=pdbfile, chain_ids=chain_ids)

                    if ('PWM' in self.requirements) | ('conservation' in self.requirements):
                        sequences = [PDB_processing.process_chain(chain_obj)[0] for chain_obj in chain_objs]
                        output_alignments = [MSA_folder + 'MSA_' + '%s_%s_%s' % (
                        PDBio.parse_str(origin)[0].split('/')[-1].split('.')[0], chain_id[0], chain_id[1]) + '.fasta' for chain_id in chain_ids]
                        MSA_files = [sequence_utils.call_hhblits(sequence, output_alignment,overwrite=overwrite) for sequence, output_alignment in
                                     zip(sequences, output_alignments)]
                        if len(MSA_files) == 1:
                            MSA_files = MSA_files[0]
                    else:
                        MSA_files = None

                    if has_labels:
                        labels = list_labels[b]
                        pdb_resids = PDB_processing.get_PDB_indices(chain_objs, return_model=True, return_chain=True)
                        aligned_labels = align_labels(labels,
                                                      pdb_resids,
                                                      label_resids=list_resids[b] if has_resids else None)
                    else:
                        aligned_labels = None

                    input, output = self.process_example(
                        chain_obj=chain_objs,
                        MSA_file=MSA_files,
                        labels=aligned_labels
                    )

                    inputs.append(input)
                    if has_labels:
                        outputs.append(output)
                except Exception as e:
                    print('Failed to process example %s (%s/%s), Error: %s' %(origin,b,B,str(e) ) )
                    if permissive:
                        failed_samples.append(b)
                        continue
                    else:
                        raise ValueError('Failed in non permissive mode')

            if len(inputs)==0:
                # No successful run.
                if has_labels:
                    return [],[],failed_samples
                else:
                    return [],None,failed_samples
            ninputs = len(inputs[0]) if isinstance(inputs[0],list) else 1

            if self.padded:
                if ninputs>1:
                    inputs = [np.stack([input[k] for input in inputs], axis=0)
                              for k in range(ninputs)]
                else:
                    inputs = np.stack(inputs,axis=0)
                if has_labels:
                    outputs = np.stack(outputs,axis=0)
            else:
                if ninputs>1:
                    inputs = [np.array([input[k] for input in inputs])
                              for k in range(ninputs)]
                else:
                    inputs = np.array(inputs)
                if has_labels:
                    outputs = np.array(outputs)
            if has_labels:
                return inputs, outputs,failed_samples
            else:
                return inputs,None,failed_samples

    def process_dataset(self, env,label_name=None,permissive=True):
        return (None,), (None,),None

    def print_progress(self, b):
        if b % 100 == 0:
            print('Processing examples, b=%s' % b)

    def process_example(self,**kwargs):
        return None,None


class HandcraftedFeaturesPipeline(Pipeline):
    def __init__(self,
                 with_gaps=True,
                 Beff=500,
                 pipeline_folder=pipeline_folder,
                 feature_list=[
                     'primary',
                     'secondary',
                     'conservation',
                     'pwm',
                     'asa',
                     'residue_depth',
                     'volume_index',
                     'half_sphere',
                     'coordination'
                 ]
                 ):

        self.feature_list = feature_list
        self.Beff = Beff
        self.with_gaps = with_gaps

        features = ''.join(feature[0] for feature in feature_list)

        pipeline_name = 'pipeline_Handcrafted_features-%s_gaps-%s_Beff-%s' % (
            features, with_gaps, Beff)

        super(HandcraftedFeaturesPipeline, self).__init__(pipeline_name, pipeline_folder=pipeline_folder)

        self.features_dimension = 0
        self.feature_names = []

        if 'primary' in self.feature_list:
            print('primary')
            self.requirements.append('sequence')
            self.feature_names += ['AA %s' %
                                   aa for aa in sequence_utils.aa[:20]]
            self.features_dimension += 20

        if 'secondary' in self.feature_list:
            print('secondary')
            self.requirements.append('secondary_structure')
            self.feature_names += ['SS %s' %
                                   ss for shortname, ss in dict_dssp2name.items()]
            self.features_dimension += 8

        if 'conservation' in self.feature_list:
            print('conservation')
            self.requirements.append('conservation_score')
            self.feature_names.append('Conservation')
            self.features_dimension += 1
        if 'asa' in self.feature_list:
            print('Accessible surface area')
            self.requirements.append('accessible_surface_area')
            self.feature_names.append('Accessible Surface Area')
            self.features_dimension += 1

        if 'residue_depth' in self.feature_list:
            print('Residue Depth')
            self.requirements.append('backbone_depth')
            self.requirements.append('sidechain_depth')
            self.feature_names.append('Residue Backbone Depth')
            self.feature_names.append('Residue SideChain Depth')
            self.features_dimension += 2

        if 'volume_index' in self.feature_list:
            print('Volume index')
            self.feature_names += ['Volume Index %s' %
                                   i for i in range(3)]
            self.features_dimension += 3
            self.requirements.append('volume_index')


        if 'half_sphere' in self.feature_list:
            print('Half sphere exposure')
            self.requirements.append('halfsphere_excess_up')
            self.feature_names.append('Half Sphere Excess up')
            self.features_dimension += 1

        if 'coordination' in self.feature_list:
            print('Coordination')
            self.requirements.append('coordination')
            self.feature_names.append('Coordination')
            self.features_dimension += 1

        if 'pwm' in self.feature_list:
            print('pwm')
            self.requirements.append('PWM')
            if self.with_gaps:
                self.feature_names += ['PWM %s' %
                                       aa for aa in sequence_utils.aa]

                self.features_dimension += 21
            else:
                self.feature_names += ['PWM %s' %
                                       aa for aa in sequence_utils.aa[:20]]

                self.features_dimension += 20
        if 'npwm' in self.feature_list:
            print('normed pwm')
            self.requirements.append('PWM')
            self.requirements.append('conservation')
            if self.with_gaps:
                self.feature_names += ['nPWM %s' %
                                       aa for aa in sequence_utils.aa]
                self.features_dimension += 21
            else:
                self.feature_names += ['nPWM %s' %
                                       aa for aa in sequence_utils.aa[:20]]
                self.features_dimension += 20

    def process_example(self,
                        chain_obj=None,
                        sequence=None,
                        secondary_structure=None,
                        accessible_surface_area=None,
                        backbone_depth=None,
                        sidechain_depth=None,
                        halfsphere_excess_up=None,
                        coordination=None,
                        volume_index=None,
                        MSA_file=None,
                        PWM=None,
                        conservation_score=None,
                        backbone_coordinates=None,
                        atomic_coordinates=None,
                        labels=None):

        missing_features = {
            'sequence': (sequence is None) & ('primary' in self.feature_list),
            'secondary_structure': (secondary_structure is None) & ('secondary' in self.feature_list),
            'conservation': (conservation_score is None) & (('conservation' in self.feature_list) | ('npwm' in self.feature_list) ) ,
            'asa': (accessible_surface_area is None) & ('asa' in self.feature_list),
            'residue_depth':  ( (backbone_depth is None ) | (sidechain_depth is None) ) & ('residue_depth' in self.feature_list),
            'half_sphere':   (halfsphere_excess_up is None) & ('half_sphere' in self.feature_list),
            'coordination': (coordination is None) & ('coordination' in self.feature_list),
            'volume_index':(volume_index is None) & ('volume_index' in self.feature_list),
            'pwm': (PWM is None) & ('pwm' in self.feature_list),
            'npwm':  (PWM is None) & ('npwm' in self.feature_list)
        }

        missing_features['backbone_coordinates'] =  (backbone_coordinates is None) & (
            missing_features['residue_depth'] |
            missing_features['half_sphere'] |
            missing_features['coordination'] |
            missing_features['volume_index']
        )

        missing_features['atomic_coordinates'] =  (atomic_coordinates is None) & (
            missing_features['residue_depth'] |
            missing_features['half_sphere'] |
            missing_features['coordination'] |
            missing_features['volume_index']
        )


        if any([x[1] for x in missing_features.items()]) & (chain_obj is None):
            print(missing_features)
            raise ValueError('Missing features and chain not provided')

        if missing_features['sequence'] | missing_features['backbone_coordinates'] | missing_features['atomic_coordinates']:
            sequence, backbone_coordinates, atomic_coordinates, _, _ = PDB_processing.process_chain(chain_obj)

        if missing_features['secondary_structure'] | missing_features['asa']:
            secondary_structure, accessible_surface_area = PDB_processing.apply_DSSP(chain_obj)
        if missing_features['pwm'] | missing_features['npwm']:
            if (MSA_file is not None):
                if not isinstance(MSA_file,list):
                    PWM = sequence_utils.compute_PWM(MSA_file, gap_threshold=0.3,
                                                     neighbours_threshold=0.1, Beff=self.Beff, WT=0, scaled=False)
                else:
                    PWM = []
                    for MSA_file_ in MSA_file:
                        PWM.append(sequence_utils.compute_PWM(MSA_file_, gap_threshold=0.3,
                                                 neighbours_threshold=0.1, Beff=self.Beff, WT=0, scaled=False) )
                    PWM = np.concatenate(PWM,axis=0)
                # print('Computed MSA, seqlength:%s, MSA_file:%s'%(len(sequence),MSA_file))
            else:
                raise ValueError('Missing PWM or MSA')


        if missing_features['conservation']:
            conservation_score = sequence_utils.conservation_score(PWM,1,Bvirtual=1e-4)

        if any([missing_features['residue_depth'], missing_features['volume_index'] ]):
            backbone_depth, sidechain_depth, volume_index = PDB_processing.analyze_surface(chain_obj,
                                                                                           atomic_coordinates)
        if any([missing_features['half_sphere'],missing_features['coordination']]):
            halfsphere_excess_up, coordination = PDB_processing.ComputeResidueHSE(backbone_coordinates)



        L = len(sequence)
        input_features = np.zeros(
            [L, self.features_dimension], dtype=curr_float)
        index_start = 0
        if 'primary' in self.feature_list:
            binarize_categorical(sequence_utils.seq2num(sequence)[0], 20, out=input_features[:, index_start:index_start + 20])
            index_start += 20
        if 'secondary' in self.feature_list:
            binarize_categorical(secondary_structure2num(secondary_structure), 8, out=input_features[:, index_start:index_start + 8])
            index_start += 8
        if 'conservation' in self.feature_list:
            input_features[:, index_start] = conservation_score
            index_start += 1
        if 'asa' in self.feature_list:
            accessible_surface_area[np.isnan(accessible_surface_area)] = median_asa
            input_features[:,
                           index_start] = accessible_surface_area
            index_start += 1

        if 'residue_depth' in self.feature_list:
            #     """ Sometimes MSMS goes crazy (as in ['1a3y', '0', 'A'],['1af5', '0', 'A'],['1b3r', '0', 'B']),
            #     and produces very few surface vertices, very far away from the true model. I don't know why.
            #     If this is the case, replace by the median. ==> Problem solved. The API included the water molecules and all the other non-residue atoms. Dumb as fuck.
            #     """

            backbone_depth[np.isnan(backbone_depth) | (backbone_depth > 20)] = median_backbone_depth
            sidechain_depth[np.isnan(sidechain_depth) | (sidechain_depth > 20)] = median_sidechain_depth

            input_features[:,index_start] = backbone_depth
            input_features[:,index_start + 1] = sidechain_depth
            index_start += 2
        if 'half_sphere' in self.feature_list:
            halfsphere_excess_up[np.isnan(halfsphere_excess_up)] = median_halfsphere_excess_up
            input_features[:,index_start] = halfsphere_excess_up
            index_start += 1
        if 'coordination' in self.feature_list:
            coordination[np.isnan(coordination)] = median_coordination
            input_features[:,index_start] = coordination

            index_start += 1
        if 'volume_index' in self.feature_list:
            nVolumeIndex = volume_index.shape[-1]
            for i in range(nVolumeIndex):
                volume_index[np.isnan(volume_index[:, i]), i] = median_volumeindex[i]
            input_features[:,index_start:index_start + nVolumeIndex] = volume_index
            index_start += nVolumeIndex

        if 'pwm' in self.feature_list:
            if self.with_gaps:
                length = 21
                input_features[:, index_start:index_start + length] = PWM
            else:
                length = 20
                input_features[:, index_start:index_start + length] = PWM[:,-1]
            index_start += length

        if 'npwm' in self.feature_list:
            if self.with_gaps:
                length = 21
                input_features[:, index_start:index_start + length] = PWM * conservation_score[:,np.newaxis]
            else:
                length = 20
                input_features[:, index_start:index_start + length] = PWM[:,:-1] * conservation_score[:,np.newaxis]
        inputs = input_features
        outputs = labels
        return inputs, outputs


    def process_dataset(self, env, label_name=None,permissive=True):
        if label_name is not None:
            outputs = env[label_name]
        else:
            outputs = None

        failed_samples = []

        B = len(env['all_origins'])
        all_input_features = []
        for b in range(B):
            inputs = dict(
                [
                    (requirement, env['all_%ss' % requirement][b])
                    for requirement in self.requirements
                ]
            )
            try:
                input_features, _ = self.process_example(**inputs)
                all_input_features.append(input_features)
            except Exception as error:
                print('Failed to parse example (%s/%s), Error: %s'%(b,B,str(error) ) )
                if permissive:
                    failed_samples.append(b)
                    continue
                else:
                    raise ValueError('Failed in non permissive mode')

        inputs = np.array(all_input_features)
        outputs = np.array([outputs[b] for b in range(B) if not b in failed_samples])
        return inputs, outputs,failed_samples



class ScanNetPipeline(Pipeline):
    '''
    from preprocessing import PDB_processing,PDBio,pipelines

    pdb = '1a3x'
    chains = 'all'
    struct, chains = PDBio.load_chains(pdb_id=pdb ,chain_ids=chains)

    pipeline = pipelines.ScanNetPipeline(
                     with_aa=True,
                     with_atom=True,
                     aa_features='sequence',
                     atom_features='type',
                     padded=False,
                     Lmax_aa=800
                     )
    [atom_clouds,atom_triplets,atom_attributes,atom_indices,aa_clouds, aa_triplets, aa_attributes, aa_indices] = pipeline.process_example(chains)
    '''

    def __init__(self,
                 pipeline_folder=pipeline_folder,
                 with_aa=True,
                 with_atom=True,
                 aa_features='sequence',
                 atom_features='valency',
                 Beff=500,
                 aa_frames='triplet_sidechain',
                 padded=False,
                 Lmax_aa=800,
                 Lmax_aa_points=None,
                 Lmax_atom=None,
                 Lmax_atom_points=None,
                 ):

        pipeline_name = 'pipeline_ScanNet_aa-%s_atom-%s_frames-%s_Beff-%s' % (
            aa_features if with_aa else 'none',
            atom_features if with_atom else 'none',
            aa_frames,
            Beff,
        )
        if padded:
            pipeline_name += '_padded-%s'%Lmax_aa


        super(ScanNetPipeline, self).__init__(pipeline_name, pipeline_folder=pipeline_folder)
        self.with_aa = with_aa
        self.with_atom = with_atom
        if Lmax_atom is None:
            Lmax_atom = 9 * Lmax_aa
        if Lmax_aa_points is None:
            if aa_frames == 'triplet_backbone':
                Lmax_aa_points = Lmax_aa + 2
            elif aa_frames =='triplet_sidechain':
                Lmax_aa_points = 2*Lmax_aa+1
            elif aa_frames == 'triplet_cbeta':
                Lmax_aa_points = 2*Lmax_aa + 1
            elif aa_frames == 'quadruplet':
                Lmax_aa_points = 2*Lmax_aa+2
        if Lmax_atom_points is None:
            Lmax_atom_points = 11 * Lmax_aa
        self.Lmax_aa = Lmax_aa
        self.Lmax_atom = Lmax_atom
        self.Lmax_aa_points = Lmax_aa_points
        self.Lmax_atom_points = Lmax_atom_points

        self.aa_features = aa_features
        self.atom_features = atom_features

        assert aa_frames in ['triplet_sidechain','triplet_cbeta','triplet_backbone','quadruplet']
        self.aa_frames = aa_frames
        self.Beff = Beff
        self.padded = padded

        self.requirements = ['atom_coordinate', 'atom_id','sequence']

        if self.with_aa:
            if self.aa_features in ['pwm', 'both']:
                self.requirements.append('PWM')


    def process_example(self,
                        chain_obj=None,
                        sequence=None,
                        MSA_file=None,
                        atomic_coordinates=None,
                        atom_ids=None,
                        PWM=None,
                        labels=None,
                        *kwargs
                        ):


        # print('Entering process_example, MSA_file:%s'%MSA_file )
        if chain_obj is not None:
            sequence, backbone_coordinates, atomic_coordinates, atom_ids, atom_types = PDB_processing.process_chain(
                chain_obj)
        # print('Parsed PDB file, seqlength:%s, MSA_file:%s'%(len(sequence),MSA_file))
        if self.with_aa & (self.aa_features in ['pwm', 'both']):
            if isinstance(PWM,list):
                PWM_is_none = (PWM[0] is None)
            else:
                PWM_is_none = (PWM is None)
            if PWM_is_none:
                if (MSA_file is not None):
                    if not isinstance(MSA_file,list):
                        PWM = sequence_utils.compute_PWM(MSA_file, gap_threshold=0.3,
                                                         neighbours_threshold=0.1, Beff=self.Beff, WT=0, scaled=False)
                    else:
                        PWM = []
                        for MSA_file_ in MSA_file:
                            PWM.append(sequence_utils.compute_PWM(MSA_file_, gap_threshold=0.3,
                                                     neighbours_threshold=0.1, Beff=self.Beff, WT=0, scaled=False) )
                        PWM = np.concatenate(PWM,axis=0)
                    # print('Computed MSA, seqlength:%s, MSA_file:%s'%(len(sequence),MSA_file))
                else:
                    raise ValueError('Missing PWM or MSA')
            else:
                if isinstance(PWM,list):
                    PWM = np.concatenate(PWM,axis=0)

        if self.with_aa:
            aa_clouds, aa_triplets, aa_indices = protein_frames.get_aa_frameCloud(atomic_coordinates, atom_ids, verbose=True,method=self.aa_frames)
            if self.aa_features == 'sequence':
                nsequence_features = 20
                aa_attributes = binarize_categorical(
                    sequence_utils.seq2num(sequence)[0], 20)

            elif self.aa_features == 'pwm':
                nsequence_features = 21
                aa_attributes = PWM

            elif self.aa_features == 'both':
                nsequence_features = 41
                aa_attributes = np.concatenate(
                    (
                        binarize_categorical(
                            sequence_utils.seq2num(sequence)[0], 20),
                        PWM
                    ), axis=1)
            else:
                raise ValueError('aa features %s not supported'%self.aa_features)

            aa_attributes = aa_attributes.astype(curr_float)
            # print('Computed aa frame cloud, seqlength:%s, MSA_file:%s' % (len(sequence), MSA_file))

            if self.padded:
                aa_clouds = padd_matrix(aa_clouds, padding_value=0, Lmax=self.Lmax_aa_points)
                aa_triplets = padd_matrix(aa_triplets, padding_value=-1, Lmax=self.Lmax_aa)
                aa_attributes = padd_matrix(aa_attributes, padding_value=0, Lmax=self.Lmax_aa)
                aa_indices = padd_matrix(aa_indices, padding_value=-1, Lmax=self.Lmax_aa)

            else:
                aa_clouds = remove_nan(aa_clouds,padding_value=0.)
                aa_triplets = remove_nan(aa_triplets, padding_value=-1)
                aa_attributes = remove_nan(aa_attributes, padding_value=0.)
                aa_indices = remove_nan(aa_indices, padding_value=-1 )


        if self.with_atom:
            atom_clouds, atom_triplets, atom_attributes, atom_indices = protein_frames.get_atom_frameCloud(sequence, atomic_coordinates,atom_ids)
            atom_clouds, atom_triplets = protein_frames.add_virtual_atoms(atom_clouds, atom_triplets, verbose=True)

            if self.atom_features == 'type':
                natom_features = 4
                atom_attributes = protein_chemistry.index_to_type[atom_attributes]
            elif self.atom_features =='valency':
                atom_attributes = protein_chemistry.index_to_valency[ sequence_utils.seq2num(sequence)[0,atom_indices[:,0]] ,atom_attributes]
                natom_features = 12
            elif self.atom_features == 'id':
                natom_features = 38

            # print('Computed atom frame cloud, seqlength:%s, MSA_file:%s' % (len(sequence), MSA_file))
            if self.padded:
                atom_clouds = padd_matrix(atom_clouds, padding_value=0, Lmax=self.Lmax_atom_points)
                atom_triplets = padd_matrix(atom_triplets, padding_value=-1, Lmax=self.Lmax_atom)
                atom_attributes = padd_matrix(atom_attributes, padding_value=-1, Lmax=self.Lmax_atom)
                atom_indices = padd_matrix(atom_indices, padding_value=-1, Lmax=self.Lmax_atom)

            else:
                atom_clouds = remove_nan(atom_clouds,padding_value=0.)
                atom_attributes = remove_nan(atom_attributes, padding_value=-1)
                atom_triplets = remove_nan(atom_triplets, padding_value=-1)
                atom_indices = remove_nan(atom_indices, padding_value=-1 )

            atom_attributes += 1


        inputs = []
        if self.with_aa:
            inputs += [aa_triplets, aa_attributes,aa_indices,aa_clouds]

        if self.with_atom:
            inputs += [atom_triplets,atom_attributes,atom_indices,atom_clouds]

        if labels is not None:
            if labels.dtype in [np.bool,np.int]:
                outputs = binarize_categorical(
                    labels, 2).astype(curr_int)
            else:
                outputs = labels
            if self.padded:
                outputs = padd_matrix(outputs, Lmax=self.Lmax_aa, padding_value=0)[
                    np.newaxis]
            else:
                outputs = remove_nan(outputs,padding_value = 0.)
        else:
            outputs = None
        # print('Finished!, seqlength:%s, MSA_file:%s'%(len(sequence),MSA_file))
        return inputs, outputs

    def process_dataset(self, env,label_name=None,permissive=True):
        all_sequences = env['all_sequences']
        all_atom_coordinates = env['all_atom_coordinates']
        all_atom_ids = env['all_atom_ids']
        if label_name is not None:
            all_labels = env[label_name]


        if 'PWM' in self.requirements:
            index = targets_Beff.index(self.Beff)
            all_PWMs = np.array([PWM[:, :, index]
                                for PWM in env['all_PWMs']])

        inputs = []
        outputs = []
        failed_samples = []

        B = len(all_sequences)
        for b in range(B):
            self.print_progress(b)
            input2process_example = {
                'atomic_coordinates': all_atom_coordinates[b],
                'atom_ids': all_atom_ids[b],
                'sequence':all_sequences[b]
            }
            if 'PWM' in self.requirements:
                input2process_example['PWM'] = all_PWMs[b]
            if label_name is not None:
                input2process_example['labels'] = all_labels[b]
            try:
                input, output = self.process_example(**input2process_example)
                inputs.append(input)
                outputs.append(output)
            except Exception as error:
                print('Failed to parse example (%s/%s), Error: %s'%(b,B,str(error) ) )
                if permissive:
                    failed_samples.append(b)
                    continue
                else:
                    raise ValueError('Failed in non permissive mode')

        ninputs = len(inputs[0])
        if self.padded:
            inputs = [np.concatenate([input[k] for input in inputs], axis=0)
                      for k in range(ninputs)]
            outputs = np.concatenate(outputs)
        else:
            inputs = [np.array([input[k] for input in inputs])
                      for k in range(ninputs)]
            outputs = np.array(outputs)
        return inputs, outputs,failed_samples
