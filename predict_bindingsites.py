import os
cores = 5  # Set number of CPUs to use!
if __name__ == '__main__':
    os.environ["MKL_NUM_THREADS"] = "%s" % cores
    os.environ["NUMEXPR_NUM_THREADS"] = "%s" % cores
    os.environ["OMP_NUM_THREADS"] = "%s" % cores
    os.environ["OPENBLAS_NUM_THREADS"] = "%s" % cores
    os.environ["VECLIB_MAXIMUM_THREADS"] = "%s" % cores
    os.environ['NUMBA_DEFAULT_NUM_THREADS'] = "%s" % cores
    os.environ["NUMBA_NUM_THREADS"] = "%s" % cores

from preprocessing import pipelines,PDB_processing,sequence_utils,PDBio
from utilities import wrappers, chimera
import numpy as np
import argparse
from keras.models import Model
from utilities.paths import structures_folder,MSA_folder,predictions_folder,path2hhblits,path2sequence_database,model_folder


pipeline_MSA = pipelines.ScanNetPipeline(
    with_aa=True,
    with_atom=True,
    aa_features='pwm',
    atom_features='valency',
    aa_frames='triplet_sidechain',
    Beff=500,
)

pipeline_noMSA = pipelines.ScanNetPipeline(
    with_aa=True,
    with_atom=True,
    aa_features='sequence',
    atom_features='valency',
    aa_frames='triplet_sidechain',
    Beff=500,
)


interface_model_name_MSA = 'ScanNet_interface'
interface_model_MSA = 'ScanNet_PPI'


interface_model_name_noMSA = 'ScanNet_interface_noMSA'
interface_model_noMSA = 'ScanNet_PPI_noMSA'

epitope_model_name_MSA = 'ScanNet_epitope'
epitope_model_MSA = ['ScanNet_PAI_%s'%index for index in range(5)]

epitope_model_name_noMSA = 'ScanNet_epitope_noMSA'
epitope_model_noMSA = ['ScanNet_PAI_noMSA_%s'%index for index in range(5)]


idp_model_name_MSA = 'ScanNet_idp'
idp_model_MSA = ['ScanNet_PIDPI_%s'%index for index in range(5)]

idp_model_name_noMSA = 'ScanNet_idp_noMSA'
idp_model_noMSA = ['ScanNet_PIDPI_noMSA_%s'%index for index in range(5)]



interface_model_folder = model_folder
epitope_model_folder = model_folder
idp_model_folder = model_folder


default_pipeline = pipeline_MSA
default_model = interface_model_MSA
default_model_name = interface_model_name_MSA
model_folder = interface_model_folder

def write_predictions(csv_file, residue_ids, sequence, interface_prediction):
    L = len(residue_ids)
    columns = ['Model','Chain','Residue Index','Sequence']
    if interface_prediction.ndim == 1:
        columns.append('Binding site probability')
    else:
        columns += ['Output %s' %i for i in range(interface_prediction.shape[-1] )]

    with open(csv_file, 'w') as f:
        f.write(','.join(columns) + '\n' )
        for i in range(L):
            string = '%s,%s,%s,%s,' % (residue_ids[i][0],
                                       residue_ids[i][1],
                                       residue_ids[i][2],
                                       sequence[i])
            if interface_prediction.ndim == 1:
                string += '%.3f'%interface_prediction[i]
            else:
                string += ','.join(['%.3f'%value for value in interface_prediction[i]])
            f.write(string + '\n')
    return


def predict_interface_residues(
    query_pdbs='1a3x',
    query_chain_ids=None,
    query_sequences=None,
    query_names=None,
    pipeline=default_pipeline,
    model=default_model,
    model_name=default_model_name,
    model_folder=model_folder,
    structures_folder=structures_folder,
    predictions_folder=predictions_folder,
    query_MSAs=None,
    query_PWMs=None,
    MSA_folder=MSA_folder,
    logfile=None,
    biounit=True,
    assembly=True,
    layer=None,
    use_MSA=True,
    overwrite_MSA=False,
    Lmin=1,
    output_predictions=True,
    aggregate_models=True,
    output_chimera='annotation',
    chimera_thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    permissive=False,
    output_format='numpy'):

    if not os.path.isdir(MSA_folder):
        os.mkdir(MSA_folder)
    if not os.path.isdir(predictions_folder):
        os.mkdir(predictions_folder)
    if use_MSA:
        assert os.path.exists(
            path2hhblits), 'HHblits not found at %s!!' % path2hhblits

    if query_pdbs is not None:
        try:  # Check whether chain_ids is a list of pdb/chains or a single pdb/chain
            assert len(query_pdbs[0]) > 1
        except:
            query_pdbs = [query_pdbs]
        print('Predicting binding sites from pdb structures with %s' %model_name,file=logfile)
        predict_from_pdb = True
        predict_from_sequence = False
        npdbs = len(query_pdbs)

        if query_chain_ids is None:
            query_chain_ids = ['all' for _ in query_pdbs]
        else:
            if not ( (query_chain_ids[0] in ['all','upper','lower'] ) | (isinstance(query_chain_ids[0],list)) ):
                query_chain_ids = [query_chain_ids]

    elif query_sequences is not None:
        try:  # Check whether sequences is a list of pdb/chains or a single pdb/chain
            assert len(query_sequences[0]) > 1
        except:
            query_sequences = [query_sequences]
        print('Predicting interface residues from sequences using %s' %
              model_name, file=logfile)
        predict_from_pdb = False
        predict_from_sequence = True
        nqueries = len(query_sequences)
    else:
        print('No input provided for interface prediction using %s' %
              model_name, file=logfile)
        return

    if query_names is None:
        if predict_from_pdb:
            query_names = []
            for i in range(npdbs):
                pdb = query_pdbs[i].split('/')[-1].split('.')[0]
                query_names.append(pdb)
        elif predict_from_sequence:
            sequence_lengths = [len(sequence) for sequence in query_sequences]
            first_aa = [sequence[:5] for sequence in query_sequences]
            query_names = ['seq_%s_start:%s_L:%s' % (
                i, first_aa[i], sequence_lengths[i]) for i in range(nqueries)]
    if use_MSA:
        if query_MSAs is None:
            query_MSAs = [None for _ in query_names]
        if query_PWMs is None:
            query_PWMs = [None for _ in query_names]


    # Locate pdb files or download from pdb server.
    if predict_from_pdb:
        pdb_file_locations = []
        i = 0
        while i < npdbs:
            pdb_id = query_pdbs[i]
            location,chain = PDBio.getPDB(pdb_id,biounit=biounit,structures_folder=structures_folder)
            if not os.path.exists(location):
                print('i=%s,file:%s not found' %
                      (i, location), file=logfile)
                if permissive & (npdbs > 1):
                    del query_pdbs[i]
                    del query_chain_ids[i]
                    del query_names[i]
                    if use_MSA:
                        del query_MSAs[i]
                        del query_PWMs[i]
                    npdbs -= 1
                else:
                    return
            else:
                pdb_file_locations.append(location)
                i += 1

    # Parse pdb files.
        query_chain_objs = []
        query_chain_names = []
        query_chain_id_is_alls = [query_chain_id == 'all' for query_chain_id in query_chain_ids]

        i = 0
        while i < npdbs:
            try:
                _, chain_objs = PDBio.load_chains(
                    chain_ids= query_chain_ids[i], file=pdb_file_locations[i])

                if query_chain_ids[i] == 'all':
                    query_chain_ids[i] = [(chain_obj.get_full_id()[1], chain_obj.get_full_id()[2])
                                             for chain_obj in chain_objs]
                elif query_chain_ids[i] == 'upper':
                    query_chain_ids[i] = [(chain_obj.get_full_id()[1], chain_obj.get_full_id()[2])
                                             for chain_obj in chain_objs if (chain_obj.get_full_id()[2].isupper() | (chain_obj.get_full_id()[2] == ' ') )]
                elif query_chain_ids[i] == 'lower':
                    query_chain_ids[i] = [(chain_obj.get_full_id()[1], chain_obj.get_full_id()[2])
                                             for chain_obj in chain_objs if chain_obj.get_full_id()[2].islower()]

                query_chain_objs.append(chain_objs)

                query_chain_names.append([query_names[i] + '_%s_%s' %
                                      query_chain_id for query_chain_id in query_chain_ids[i]])

                i += 1
            except:
                print('Failed to parse i=%s,%s, %s' %
                      (i, query_names[i], pdb_file_locations[i]), file=logfile)
                if permissive & (npdbs > 1):
                    del query_pdbs[i]
                    del query_chain_ids[i]
                    del query_names[i]
                    if use_MSA:
                        del query_MSAs[i]
                        del query_PWMs[i]
                    npdbs -= 1
                else:
                    return

        query_sequences = [[PDB_processing.process_chain(chain_obj)[0]
                            for chain_obj in chain_objs] for chain_objs in query_chain_objs]

        if Lmin > 0:
            for i in range(npdbs):
                j = 0
                nsequences = len(query_sequences[i])
                while j < nsequences:
                    sequence = query_sequences[i][j]
                    if len(sequence) < Lmin:
                        print('Chain %s %s from PDB %s is too short (L=%s), discarding.' % (
                        query_chain_ids[i][j][0], query_chain_ids[i][j][1], query_pdbs[i], len(sequence)), file=logfile)
                        del query_sequences[i][j]
                        del query_chain_ids[i][j]
                        del query_chain_objs[i][j]
                        del query_chain_names[i][j]
                        # Assumes that the MSA/PWM input was provided without the small chains...
                        # if use_MSA:
                        #     if (query_MSAs[i] is not None) & isinstance(query_MSAs[i], list):
                        #         del query_MSAs[i][j]
                        #     if (query_PWMs[i] is not None) & isinstance(query_PWMs[i], list):
                        #         del query_PWMs[i][j]
                        nsequences -= 1
                    else:
                        j += 1

        i = 0
        while i < npdbs:
            if not len(query_sequences[i]) > 0:
                print('PDB %s has no chains remaining!' % (query_pdbs[i]))
                if permissive & (npdbs > 1):
                    del query_pdbs[i]
                    del query_sequences[i]
                    del query_chain_ids[i]
                    del query_chain_objs[i]
                    del query_names[i]
                    if use_MSA:
                        del query_MSAs[i]
                        del query_PWMs[i]
                    del query_chain_names[i]
                    npdbs -= 1
                else:
                    return
            else:
                i += 1

        nqueries = npdbs
    else:
        query_chain_names = query_names
        query_chain_objs = [None for _ in query_chain_names]
        if query_chain_ids is None:
            query_chain_ids = [('', '') for _ in query_chain_names]
        nqueries = len(query_names)

    print('List of inputs:', file=logfile)
    for i in range(nqueries):
        print(query_chain_names[i], file=logfile)

    if use_MSA:
        i = 0
        while i < nqueries:
            if query_PWMs[i] is not None:
                if not isinstance(query_PWMs[i], list):
                      query_PWMs[i] = [query_PWMs[i]]
                i +=1
            elif query_MSAs[i] is not None:
                if not isinstance(query_MSAs[i], list):
                      query_MSAs[i] = [query_MSAs[i]]
                for j in range(len(query_MSAs[i])):
                    if not os.path.exists(query_MSAs[i][j]):
                        print('i=%s,file:%s not found' %
                              (i, query_MSAs[i][j]), file=logfile)
                        if permissive & (nqueries > 1):
                            if predict_from_pdb:
                                del query_pdbs[i]
                            del query_sequences[i]
                            del query_chain_ids[i]
                            del query_chain_objs[i]
                            del query_names[i]
                            del query_MSAs[i]
                            del query_PWMs[i]
                            del query_chain_names[i]
                            nqueries -= 1
                            break
                        else:
                            return
                    else:
                          if j == len(query_MSAs[i]) - 1:
                              i += 1
            else:
                query_PWMs[i] = []
                query_MSAs[i] = []
                if not isinstance(query_sequences[i], list):
                    query_sequences[i] = [query_sequences[i]]
                if not isinstance(query_chain_names[i], list):
                    query_chain_names[i] = [query_chain_names[i]]


                for j in range(len(query_sequences[i])):
                    target_location = MSA_folder + 'MSA_' + \
                        query_chain_names[i][j] + '.fasta'
                    sequence = query_sequences[i][j]
                    if not (os.path.exists(target_location) & (~overwrite_MSA)):
                        if sequence in query_sequences[i][:j]:
                          jseen = query_sequences[i][:j].index(sequence)
                          target_location = MSA_folder + 'MSA_' + \
                              query_chain_names[i][jseen] + '.fasta'
                        else:
                            print('i=%s,%s, no MSA found. Building it using HHblits' %
                              (i, query_chain_names[i][j]), file=logfile)
                            sequence_utils.call_hhblits(sequence, target_location,
                                                path2hhblits=path2hhblits, path2sequence_database=path2sequence_database, cores=cores)
                    query_MSAs[i].append(target_location)
                    query_PWMs[i].append(None)
                i+=1

    else:
        if assembly:
            query_MSAs = [None for _ in range(nqueries)]
            query_PWMs = [None for _ in range(nqueries)]
        else:
            query_MSAs = [[None for _ in query_chain_names[i]] for i in range(nqueries)]
            query_PWMs = [[None for _ in query_chain_names[i]] for i in range(nqueries)]

    sequence_lengths = [[len(sequence) for sequence in sequences]
                           for sequences in query_sequences]
    if assembly:
        assembly_lengths = [sum(sequence_length)
                              for sequence_length in sequence_lengths]
        Lmax = max(assembly_lengths)
    else:
        Lmax = max([max(sequence_length)
                 for sequence_length in sequence_lengths])
    Lmax = max(Lmax,32)



    query_residue_ids =[]
    query_sequences=[''.join(sequences) for sequences in query_sequences]
    for i, chain_objs in enumerate(query_chain_objs):
        if chain_objs is not None:
            residue_ids =  PDB_processing.get_PDB_indices(chain_objs, return_chain=True, return_model=True)
        else:
            model_indices=[' ' for _ in query_sequences[i]]
            chain_indices=[' ' for _ in query_sequences[i]]
            residue_indices= ['%s'%i for i in range(1, len(query_sequences[i]) + 1) ]
            residue_ids = np.concatenate(
                np.array(model_indices)[:,np.newaxis],
                np.array(chain_indices)[:,np.newaxis],
                np.array(residue_indices)[:,np.newaxis],
                axis = 1
             )

        query_residue_ids.append( residue_ids)

    print('Loading model %s' % model_name, file=logfile)
    if isinstance(model,list):
        multi_models = True
        model_objs = [wrappers.load_model(model_folder + model_, Lmax=Lmax) for model_ in model]
        model_obj = None
    else:
        multi_models = False
        model_obj = wrappers.load_model(model_folder + model, Lmax=Lmax)
        model_objs = None

    if layer is not None:
        if isinstance(layer,list):
            layer_outputs = []
            for layer_ in layer:
                if layer_ is None:
                    layer_outputs.append(model_obj.model.get_layer('classifier_output').output)
                elif layer_ == 'attention_layer':
                    layer_outputs.append(model_obj.model.get_layer('attention_layer').output[1])
                else:
                    layer_outputs.append(model_obj.model.get_layer(layer_).output)

            model_obj.model = Model(inputs=model_obj.model.inputs,outputs=layer_outputs)
            model_obj.multi_outputs = True
            model_obj.Lmax_output = [int(output.shape[1]) for output in layer_outputs]
        else:
            if layer == 'attention_layer':
                model_obj.model = Model(inputs=model_obj.model.inputs, outputs=model_obj.model.get_layer(layer).output[1])
            else:
                model_obj.model = Model(inputs=model_obj.model.inputs,outputs=model_obj.model.get_layer(layer).output)
        return_all = True
    else:
        return_all = False

    if hasattr(pipeline, 'Lmax'):
        pipeline.Lmax = Lmax
    if hasattr(pipeline, 'Lmax_aa'):
        pipeline.Lmax_aa = Lmax
    if hasattr(pipeline, 'Lmax_atom'):
        pipeline.Lmax_atom = 9* Lmax

    if hasattr(pipeline, 'padded'):
        padded = pipeline.padded
    else:
        padded = True


    if assembly:
        inputs = wrappers.stack_list_of_arrays(
        [pipeline.process_example(chain_obj=chain_obj, sequence=sequence, MSA_file=MSA_file_location,PWM=PWM)[0]
         for chain_obj, sequence, MSA_file_location,PWM in zip(query_chain_objs, query_sequences, query_MSAs,query_PWMs)], padded=padded)
        if multi_models:
            if aggregate_models:
                 query_predictions = model_objs[0].predict(inputs, batch_size=1,return_all=return_all)
                 for model_obj in model_objs[1:]:
                     predictions = model_obj.predict(inputs, batch_size=1,return_all=return_all)
                     query_predictions = [prediction1 + prediction2 for prediction1,prediction2 in zip(query_predictions,predictions)]
                 query_predictions = np.array([prediction/len(model_objs) for prediction in query_predictions])
            else:
                query_predictions = [model_obj.predict(inputs, batch_size=1,return_all=return_all) for model_obj in model_objs]
        else:
            query_predictions = model_obj.predict(inputs, batch_size=1,return_all=return_all)

        if padded:
            query_predictions = wrappers.truncate_list_of_arrays(
                query_predictions, assembly_lengths)

        has_attention_layer = False
        if layer == 'attention_layer':
            has_attention_layer = True
        elif isinstance(layer,list):
            has_attention_layer = 'attention_layer' in layer

        if has_attention_layer:
            '''
            Output the aggregated attention coefficient for each node (= node importance, potential hotspot detector).
            1. Recompute, for each aa, the indices of its K neighbors using Calpha coordinates.
            2. Compute the degree of each aa by summing its contribution to all other amino acids.
            3. Put back into predictions.
            '''
            calpha_coordinates = [inputs[3][n].astype(np.float32)[inputs[0][n].astype(np.int)[..., 0]] for n in range(len(inputs[0]))]
            K_graph = model_obj.kwargs['K_graph']
            neighborhood_graphs = [np.argsort(PDB_processing.distance(calpha_coordinate,calpha_coordinate), axis=1)[:,:K_graph] for calpha_coordinate in calpha_coordinates]
            if layer == 'attention_layer':
                attention_coeffs = query_predictions
            else:
                index = layer.index('attention_layer')
                attention_coeffs = query_predictions[index]

            aggregated_attention_coeffs = []
            sign = np.sign(attention_coeffs[0][:, 0, 0]).mean()
            if sign<0:
                print('Warning, attention coeffs are flipped')
            for attention_coeff,neighborhood_graph in zip(attention_coeffs,neighborhood_graphs):
                aggregated_attention_coeff = np.zeros(len(attention_coeff),dtype=np.float32)
                for s in range( len(attention_coeff) ):
                    aggregated_attention_coeff[neighborhood_graph[s]] += np.maximum(sign*attention_coeff[s][:len(neighborhood_graph[s])],0).mean(
                        -1)  # Attention coefficient has size [N_aa,K_graph,nheads]. average over heads.
                    # aggregated_attention_coeff[neighborhood_graph[s]] += np.abs(attention_coeff[s][:len(neighborhood_graph[s])]).mean(1)  # Attention coefficient has size [N_aa,K_graph,nheads]. average over heads.
                aggregated_attention_coeffs.append(aggregated_attention_coeff)
            aggregated_attention_coeffs = np.array(aggregated_attention_coeffs)
            if layer == 'attention_layer':
                query_predictions = aggregated_attention_coeffs
            else:
                query_predictions[index] = aggregated_attention_coeffs

    else:
        query_predictions = []
        for i in range(nqueries):
            inputs = wrappers.stack_list_of_arrays(
    [pipeline.process_example(chain_obj=chain_obj, sequence=sequence, MSA_file=MSA_file_location,PWM=PWM)[0]
    for chain_obj, sequence, MSA_file_location,PWM in zip(query_chain_objs[i], query_sequences[i], query_MSAs[i],query_PWMs[i])], padded=padded)
            if multi_models:
                if aggregate_models:
                     predictions = model_objs[0].predict(inputs, batch_size=1,return_all=return_all)
                     for model_obj in model_objs[1:]:
                         predictions_ = model_obj.predict(inputs, batch_size=1,return_all=return_all)
                         predictions = [prediction1 + prediction2 for prediction1,prediction2 in zip(predictions,predictions_)]
                     predictions = np.array([prediction/len(model_objs) for prediction in predictions])
                else:
                    predictions = [model_obj.predict(inputs, batch_size=1,return_all=return_all) for model_obj in model_objs]
            else:
                predictions = model_obj.predict(inputs, batch_size=1,return_all=return_all)

            if padded:
                predictions = wrappers.truncate_list_of_arrays(
                    predictions, sequence_lengths[i])

            has_attention_layer = False
            if layer == 'attention_layer':
                has_attention_layer = True
            elif isinstance(layer,list):
                has_attention_layer = 'attention_layer' in layer

            if has_attention_layer:
                '''
                Output the aggregated attention coefficient for each node (= node importance, potential hotspot detector).
                1. Recompute, for each aa, the indices of its K neighbors using Calpha coordinates.
                2. Compute the degree of each aa by summing its contribution to all other amino acids.
                3. Put back into predictions.
                '''

                calpha_coordinates = [inputs[3][n].astype(np.float32)[inputs[0][n].astype(np.int)[..., 0]] for n in range(len(inputs[0]))]
                K_graph = model_obj.kwargs['K_graph']
                neighborhood_graphs = [np.argsort(PDB_processing.distance(calpha_coordinate,calpha_coordinate), axis=1)[:,:K_graph] for calpha_coordinate in calpha_coordinates]
                if layer == 'attention_layer':
                    attention_coeffs = predictions
                else:
                    index = layer.index('attention_layer')
                    attention_coeffs = predictions[index]

                sign = np.sign(attention_coeffs[0][:, 0, 0]).mean()
                if sign < 0:
                    print('Warning, attention coeffs are flipped')
                aggregated_attention_coeffs = []
                for attention_coeff,neighborhood_graph in zip(attention_coeffs,neighborhood_graphs):
                    aggregated_attention_coeff = np.zeros(len(attention_coeff),dtype=np.float32)
                    for s in range( len(attention_coeff) ):
                        aggregated_attention_coeff[neighborhood_graph[s]] += np.maximum(sign*attention_coeff[s][:len(neighborhood_graph[s])],0).mean(-1) # Attention coefficient has size [N_aa,K_graph,nheads]. average over heads.
                        # aggregated_attention_coeff[neighborhood_graph[s]] += np.abs(attention_coeff[s][:len(neighborhood_graph[s])] ).mean(-1)  # Attention coefficient has size [N_aa,K_graph,nheads]. average over heads.
                    aggregated_attention_coeffs.append(aggregated_attention_coeff)
                aggregated_attention_coeffs = np.array(aggregated_attention_coeffs)
                if layer == 'attention_layer':
                    predictions = aggregated_attention_coeffs
                else:
                    predictions[index] = aggregated_attention_coeffs

            if ((isinstance(layer,list) ) | (isinstance(layer,tuple)) | (not aggregate_models) ):
                query_predictions.append(
                    [np.concatenate(prediction, axis=0) for prediction in predictions]
                )
            else:
                query_predictions.append(
                    np.concatenate(predictions, axis=0)
                )
        if ((isinstance(layer, list)) | (isinstance(layer, tuple)) | (not aggregate_models)):
            query_predictions = [ [query_predictions[k][l] for k in range(len(query_predictions))] for l in range(len(query_predictions[0])) ]



    output_folder=predictions_folder + '/'
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)


    if output_predictions:
        for i in range(nqueries):
            res_ids = query_residue_ids[i]
            sequence = query_sequences[i]
            if ((isinstance(layer, list)) | (isinstance(layer, tuple)) | (not aggregate_models)):
                predictions = [query_predictions_[i] for query_predictions_ in query_predictions]
            else:
                predictions = query_predictions[i]
            query_name = query_names[i]
            query_chain = query_chain_ids[i]
            query_chain_id_is_all = query_chain_id_is_alls[i]
            query_pdb = query_pdbs[i]
            file_is_cif = (pdb_file_locations[i][-4:] == '.cif')

            query_output_folder = output_folder+query_name
            if (len(query_pdb) == 4) & biounit:
                query_output_folder += '_biounit'

            if not query_chain_id_is_all:
                query_output_folder += '_(' + PDBio.format_chain_id(query_chain) + ')'

            if not assembly:
                query_output_folder += '_single'

            query_output_folder += '_%s' % model_name

            query_output_folder += '/'
            if not os.path.isdir(query_output_folder):
                os.mkdir(query_output_folder)



            if ((isinstance(layer, list)) | (isinstance(layer, tuple)) | (not aggregate_models)):
                for layer_,prediction in zip(layer,predictions):
                    if layer_ is None:
                        prediction = prediction[:,1]
                        csv_file = query_output_folder + 'predictions_' + query_name + '.csv'
                        chimera_file = query_output_folder + 'chimera_' + query_names[i]
                        annotated_pdb_file = query_output_folder + 'annotated_' + query_names[i] + ('.cif' if file_is_cif else '.pdb')
                    else:
                        csv_file = query_output_folder + 'activity_%s_'%layer_ + query_name + '.csv'
                        chimera_file = query_output_folder + 'chimera_%s'%layer_ + query_names[i]
                        annotated_pdb_file = query_output_folder + 'annotated_%s'%layer_ + query_names[i] + ('.cif' if file_is_cif else '.pdb')
                    write_predictions(csv_file, res_ids,sequence, prediction)
                    if predict_from_pdb & (prediction.ndim == 1):
                        if output_chimera == 'script':
                            chimera.show_binding_sites(
                                query_pdbs[i], csv_file, chimera_file, biounit=biounit, directory='',thresholds=chimera_thresholds)
                        elif output_chimera == 'annotation':
                            if layer_ == 'attention_layer':
                                mini = 0.5
                                maxi = 2.5
                            else:
                                mini = 0
                                maxi = chimera_thresholds[-1]
                            chimera.annotate_pdb_file(pdb_file_locations[i], csv_file, annotated_pdb_file, output_script=True, mini=mini, maxi=maxi,version='surface' if assembly else 'default')
            else:
                if layer is None:
                    csv_file = query_output_folder + 'predictions_' + query_name + '.csv'
                    chimera_file = query_output_folder + 'chimera_' + query_names[i]
                    annotated_pdb_file = query_output_folder + 'annotated_' + query_names[i] + ('.cif' if file_is_cif else '.pdb')
                else:
                    csv_file = query_output_folder + 'activity_%s_' % layer + query_name + '.csv'
                    chimera_file = query_output_folder + 'chimera_%s' % layer + query_names[i]
                    annotated_pdb_file = query_output_folder + 'annotated_%s' % layer + query_names[i] + ('.cif' if file_is_cif else '.pdb')
                write_predictions(csv_file, res_ids, sequence, predictions)
                if predict_from_pdb & (predictions.ndim == 1):
                    if output_chimera == 'script':
                        chimera.show_binding_sites(
                            query_pdbs[i], csv_file, chimera_file, biounit=biounit, directory='',thresholds=chimera_thresholds)
                    elif output_chimera == 'annotation':
                        if layer == 'attention_layer':
                            mini = 0.5
                            maxi = 2.5
                        else:
                            mini = 0
                            maxi = chimera_thresholds[-1]

                        chimera.annotate_pdb_file(pdb_file_locations[i], csv_file, annotated_pdb_file, output_script=True, mini=mini, maxi=maxi,version='surface' if assembly else 'default')

    if output_format == 'dictionary':
        if ((isinstance(layer, list)) | (isinstance(layer, tuple)) | (not aggregate_models)):
            query_dictionary_predictions = [PDB_processing.make_values_dictionary(query_residue_ids[k], [query_predictions[l][k] for l in range(len(query_predictions))])
                                            for k in range(len(query_residue_ids))]
        else:
            query_dictionary_predictions = [PDB_processing.make_values_dictionary(query_residue_id,query_prediction) for query_residue_id,query_prediction in zip(query_residue_ids,query_predictions)]
        return query_pdbs,query_names,query_dictionary_predictions
    else:
        if ((isinstance(layer, list)) | (isinstance(layer, tuple)) | (not aggregate_models)):
            query_predictions = [
                [query_predictions[i][j] for i in range(len(query_predictions))] for j in range(len(query_predictions[0]))]
        return query_pdbs,query_names,query_predictions, query_residue_ids, query_sequences



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Predict binding sites in PDB files using Geometric Neural Network')

    parser.add_argument('input',  type=str,
                   help='Three input formats. i) A pdb id (1a3x)\
                   ii) Path to pdb file (structures/1a3x.pdb)\
                   iii) Path to text file containing list of pdb files (one per line) (1a3x \n 2kho \n ...) \
                   For performing prediction only on specfic chains, append "_" and the list of chains. (e.g. 1a3x_AB)')

    parser.add_argument('--name',dest='name',
                        default='',
                        help='Input name')
                        
                        
    parser.add_argument('--predictions_folder',dest='predictions_folder',
                        default=predictions_folder,
                        help='Input name')

    parser.add_argument('--mode', dest='mode',
                   default='interface',
                   help='Prediction mode (interface, epitope)')
    parser.add_argument('--noMSA', dest='use_MSA', action='store_const',
                  const = False, default = True,
                  help = 'Perform prediction without Multiple Sequence Alignments (less accurate, faster)'
    )
    parser.add_argument('--assembly',dest='assembly',action='store_const',
                 const = True, default = False,
                 help = 'Perform prediction from single chains or from biological assemblies')

    parser.add_argument('--permissive',dest='permissive',action='store_const',
     const=True,default=True,help='Permissive prediction')

    parser.add_argument('--layer', dest='layer',
                   default='',
                   help='Choose output layer')

    parser.add_argument('--pdb',  dest='biounit',action='store_const',
                   const = False, default = True,
                   help='Predict from pdb file (default= predict from biounit file)')


    args = parser.parse_args()

    input = args.input
    query_pdbs = []
    query_chain_ids = []
    if '.txt' in input:
        with open(input,'r') as f:
            for line in f:
                pdb,chain_ids = PDBio.parse_str(line[:-1])
                query_pdbs.append(pdb)
                query_chain_ids.append(chain_ids)
    else:
        query_pdbs, query_chain_ids = PDBio.parse_str(input)

    if args.name != '':
        query_names = [args.name]
    else:
        query_names = None
        
    predictions_folder = args.predictions_folder

    if args.use_MSA:
        pipeline = pipeline_MSA
    else:
        pipeline = pipeline_noMSA

    if args.mode == 'interface':
        model_folder = interface_model_folder
        chimera_thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        if args.use_MSA:
            model_name = interface_model_name_MSA
            model = interface_model_MSA
        else:
            model_name = interface_model_name_noMSA
            model = interface_model_noMSA
    elif args.mode == 'epitope':
        model_folder = epitope_model_folder
        chimera_thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
        if args.use_MSA:
            model_name = epitope_model_name_MSA
            model = epitope_model_MSA
        else:
            model_name = epitope_model_name_noMSA
            model = epitope_model_noMSA

    elif args.mode[:-1] == 'epitope': # epitope1, epitope2, epitope3, epitope4, epitope5
        fold = int(args.mode[-1]) - 1
        model_folder = epitope_model_folder
        chimera_thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
        if args.use_MSA:
            model_name = epitope_model_name_MSA  + str(args.mode[-1])
            model = epitope_model_MSA[fold]
        else:
            model_name = epitope_model_name_noMSA  + str(args.mode[-1])
            model = epitope_model_noMSA[fold]

    elif args.mode == 'idp':
        model_folder = idp_model_folder
        chimera_thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
        if args.use_MSA:
            model_name = idp_model_name_MSA
            model = idp_model_MSA
        else:
            model_name = idp_model_name_noMSA
            model = idp_model_noMSA

    elif args.mode[:-1] == 'idp': # idp1, idp2, idp3, idp4, idp5
        fold = int(args.mode[-1]) - 1
        model_folder = idp_model_folder
        chimera_thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
        if args.use_MSA:
            model_name = idp_model_name_MSA + str(args.mode[-1])
            model = idp_model_MSA[fold]
        else:
            model_name = idp_model_name_noMSA  + str(args.mode[-1])
            model = idp_model_noMSA[fold]
    else:
        raise ValueError('Mode %s not supported'%args.mode)

    if args.layer == '':
        layer = None
    else:
        layer = args.layer
        if '+' in layer:
            layer = layer.split('+')
            for i in range(len(layer)):
                if layer[i] in ['classifier_output','','output','probability']:
                    layer[i] = None

    predict_interface_residues(
        query_pdbs=query_pdbs,
        query_chain_ids=query_chain_ids,
        query_names=query_names,
        pipeline=pipeline,
        model=model,
        model_name=model_name,
        model_folder=model_folder,
        structures_folder=structures_folder,
        predictions_folder=predictions_folder,
        MSA_folder=MSA_folder,
        biounit=args.biounit,
        assembly=args.assembly,
        overwrite_MSA=False,
        permissive=args.permissive,
        use_MSA=args.use_MSA,
        chimera_thresholds=chimera_thresholds,
        layer=layer
    )
