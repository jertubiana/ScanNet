import predict_bindingsites
from preprocessing import PDBio
from utilities.paths import model_folder,structures_folder,MSA_folder



def predict_features(list_queries,layer='SCAN_filter_activity_aa',
                     model='ScanNet_PPI_noMSA',
                     output_format='dictionary',
                     model_folder=model_folder,
                     biounit=False,
                     permissive=False):
    '''
    Usages:
     list_dictionary_features = predict_features(list_queries,output_format='dictionary')
     list_features, list_residueids = predict_features(list_queries,output_format='numpy')
    Example: 
    list_queries = ['1a3x_A','2p6b_AB','1a3y']
    list_dictionary_features = list of residues-level features, each element of the form Nresidues X Nfeatures.

    '''
    if not isinstance(list_queries,list):
        list_queries = [list_queries]
        return_one = True
        permissive = False
    else:
        return_one = False
    query_pdbs = []
    query_chain_ids = []
    nlayers = len(layer) if isinstance(layer,list) else 1

    for query in list_queries:
        pdb,chain_ids = PDBio.parse_str(query)
        query_pdbs.append(pdb)
        query_chain_ids.append(chain_ids)

    

    if 'noMSA' in model:
        pipeline = predict_bindingsites.pipeline_noMSA
        use_MSA = False
    else:
        pipeline = predict_bindingsites.pipeline_MSA
        use_MSA = True

    query_outputs = predict_bindingsites.predict_interface_residues(
    query_pdbs=query_pdbs,
    query_chain_ids=query_chain_ids,
    pipeline=pipeline,
    model=model,
    model_folder=model_folder,
    structures_folder=structures_folder,
    MSA_folder=MSA_folder,
    biounit=biounit,
    assembly=True,
    layer=layer,
    use_MSA=use_MSA,
    overwrite_MSA=False,
    Lmin=1,
    output_chimera=False,
    permissive=permissive,
    output_format = output_format
    )
    if output_format == 'numpy':
        query_pdbs, query_names, query_features, query_residue_ids, query_sequences = query_outputs

        if return_one:
            query_pdbs = query_pdbs[0]
            query_names = query_names[0]
            query_features = query_features[0]
            query_residue_ids = query_residue_ids[0]
            query_sequences = query_sequences[0]
        if permissive:
            return query_pdbs,query_features, query_residue_ids
        else:
            return query_features, query_residue_ids
    elif output_format == 'dictionary':
        query_pdbs, query_names, query_dictionary_features = query_outputs
        if return_one:
            query_pdbs = query_pdbs[0]
            query_names = query_names[0]
            query_dictionary_features = query_dictionary_features[0]
        if permissive:
            return query_pdbs,query_dictionary_features
        else:
            return query_dictionary_features




if __name__ == '__main__':
    model = 'ScanNet_PPI_noMSA' # Protein-protein binding site prediction model without evolutionary information.

    layer_choices = [
        'SCAN_filters_atom_aggregated_activity', # Atomic Neighborhood Embedding Module, *after* pooling. Atomic neighborhoods have radius of about 5 Angstrom.  Size: [Naa,64].
        'all_embedded_attributes_aa', # Embedded residue type or PWM (first 32 channels) + Atomic Neighborhood Embedding Module, *after* pooling (last 64 channels). Size: [Naa,96].
        'SCAN_filter_activity_aa', # Amino Acid Neighborhood Embedding Module. Amino acid neighborhoods have radius of about 11 Angstrom. Size: [Naa,128].
        'SCAN_filters_aa_embedded_1', # Non-linear, 32-dimensional projection of Amino Acid Neighborhood Embedding Module output. Input to the neighborhood attention module. Size: [Naa,32].
        None, # The binding site probabilities Size: ([Naa,])
    ]

    output_format = 'numpy' #'dictionary' # 'numpy'


    # layer = layer_choices[2]
    layer = [layer_choices[1],layer_choices[2],layer_choices[4]] # Multiple layers are supported.


    if output_format == 'dictionary':
        list_names, list_dictionary_features = predict_features(['1a3x_A','1brs_A'],layer=layer,model=model,output_format=output_format,permissive=True)
        print('Dictionary format: Dictionary with residue ids as key and features as items.')
        for k in range(2):
            print('Query',list_names[k])
            for key,item in list(list_dictionary_features[k].items())[:10]:
                if isinstance(item,list):
                    list_shapes = [x.shape for x in item]
                    print('AA' ,key,'Features:',[item_[:5] for item_ in item],'Feature shapes',list_shapes)
                else:
                    print('AA',key, 'Features:',item[:5],'Feature shape',item.shape)
    elif output_format == 'numpy':
        list_names,list_features, list_residue_ids = predict_features(['1a3x_A','1brs_A'],layer=layer,model=model,output_format='numpy',permissive=True)
        print('Numpy format: Numpy arrays with residue ids as key and features as items.')
        for k in range(2):
            print('Query',list_names[k])
            if isinstance(list_features[k],list):
                for feature_ in list_features[k]:
                    print('Features array', feature_[:10, :][:, :5], 'Shape',feature_.shape)
            else:
                print('Features array',list_features[k][:10,:][:,:5])
            print('Residue IDs array',list_residue_ids[k][:10])