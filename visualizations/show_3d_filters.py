from preprocessing import pipelines
import numpy as np
from utilities import wrappers,io_utils
from visualizations import weight_logo_3d
from utilities.paths import model_folder,visualization_folder
import os
from keras.models import Model
import matplotlib

from preprocessing import PDBio,PDB_processing,pipelines,protein_chemistry
from network import neighborhoods
import numpy as np


def activity2percentile(filter_activity, percentiles):
    filter_percentiles = np.zeros_like(filter_activity)
    nfilters = percentiles.shape[-1]
    for n in range(nfilters):
        filter_percentiles[...,n] = np.interp(filter_activity[...,n], percentiles[:,n],np.arange(1,101) )
    return filter_percentiles


def calculate_filter_specificities(model_name,
                                   dataset_name='PPBS_validation_none',
                                   model_folder=model_folder,
                                   dataset_origins = None,
                                   biounit=True,
                                   ncores = 4,
                                   nmax = None,
                                   only_atom=False,
                                   fresh=False,
                                   top_percent = 1,
                                   Lmax = 1024
                                   ):
    def unpack(handcrafted_features):
        seq = handcrafted_features[..., :20]
        ss = handcrafted_features[..., 20:28]
        asa = handcrafted_features[..., 28]
        return seq, ss, asa


    filter_specificities = {}
    path2filter_specificities = visualization_folder + 'filter_specificities_%s.data' % model_name
    if not only_atom:
        if not fresh:
            if os.path.exists(path2filter_specificities):
                return io_utils.load_pickle(path2filter_specificities)

        use_evolutionary = not ('noMSA' in model_name)
        pipeline = pipelines.ScanNetPipeline(
            with_atom=True,
            atom_features = 'id',
            aa_features='pwm' if use_evolutionary else 'sequence',
        )
        handcrafted_pipeline = pipelines.HandcraftedFeaturesPipeline(feature_list=['primary','secondary','asa'])

        if dataset_origins is not None:
            inputs_scannet, _, failed_samples_scannet = pipeline.build_processed_dataset(
                dataset_name,
                list_origins=dataset_origins,  # Mandatory
                biounit=biounit,
                permissive=True,
                ncores=ncores,
                save=True
            )

            inputs_handcrafted,_,failed_samples_handcrafted = handcrafted_pipeline.build_processed_dataset(
                dataset_name,
                list_origins=dataset_origins,  # Mandatory
                biounit=biounit,
                permissive=True,
                ncores=ncores,
                save = True
            )

            if failed_samples_scannet != failed_samples_handcrafted:
                scannet_success_origins = [dataset_origins[i] for i in range(len(dataset_origins)) if not i in failed_samples_scannet]
                handcrafted_success_origins = [dataset_origins[i] for i in range(len(dataset_origins)) if not i in failed_samples_handcrafted]

                mask_scannet = np.array([origin in handcrafted_success_origins for origin in scannet_success_origins])
                mask_handcrafted = np.array([origin in scannet_success_origins for origin in handcrafted_success_origins])

                inputs_scannet = wrappers.slice_list_of_arrays(inputs_scannet,mask_scannet)
                inputs_handcrafted = wrappers.slice_list_of_arrays(inputs_handcrafted, mask_handcrafted)

        else:
            if not isinstance(dataset_name,list):
                dataset_name = [dataset_name]

            all_inputs_scannet = []
            all_inputs_handcrafted = []

            for dataset_name_ in dataset_name:

                inputs_scannet, _, failed_samples_scannet = pipeline.build_processed_dataset(
                    dataset_name_,
                    biounit=biounit,
                    permissive=True,
                    ncores=ncores
                )

                inputs_handcrafted,_,failed_samples_handcrafted = handcrafted_pipeline.build_processed_dataset(
                    dataset_name_,
                    list_origins=dataset_origins,  # Mandatory
                    biounit=biounit,
                    permissive=True,
                    ncores=ncores
                )

                if failed_samples_scannet != failed_samples_handcrafted:
                    scannet_success_origins = [dataset_origins[i] for i in range(len(dataset_origins)) if not i in failed_samples_scannet]
                    handcrafted_success_origins = [dataset_origins[i] for i in range(len(dataset_origins)) if not i in failed_samples_handcrafted]

                    mask_scannet = np.array([origin in handcrafted_success_origins for origin in scannet_success_origins])
                    mask_handcrafted = np.array([origin in scannet_success_origins for origin in handcrafted_success_origins])

                    inputs_scannet = wrappers.slice_list_of_arrays(inputs_scannet,mask_scannet)
                    inputs_handcrafted = wrappers.slice_list_of_arrays(inputs_handcrafted, mask_handcrafted)

                all_inputs_scannet.append(inputs_scannet)
                all_inputs_handcrafted.append(inputs_handcrafted)
            inputs_scannet = wrappers.stack_list_of_arrays(all_inputs_scannet,padded=False)
            inputs_handcrafted = wrappers.stack_list_of_arrays(all_inputs_handcrafted, padded=False)

        mask_identical = []
        for i in range(len(inputs_scannet[0])):
            L_scannet = len(inputs_scannet[0][i])
            L_handcrafted = len(inputs_handcrafted[i])
            if L_scannet != L_handcrafted:
                print('Discrepancy between lengthes, discarding example', L_scannet, L_handcrafted)
                mask_identical.append(False)
            else:
                mask_identical.append(True)
        mask_identical = np.array(mask_identical)
        inputs_scannet = wrappers.slice_list_of_arrays(inputs_scannet, mask_identical)
        inputs_handcrafted = wrappers.slice_list_of_arrays(inputs_handcrafted, mask_identical)

        if Lmax is None:
            Lmax = max([len(x) for x in inputs_handcrafted])

        atom_ids = inputs_scannet[5]
        sequences = [np.argmax(handcrafted_feature[:, :20], axis=-1) for handcrafted_feature in inputs_handcrafted]
        inputs_scannet[5] = [protein_chemistry.index_to_valency[sequence[atom_index[:,0]],atom_id-1]+1 for sequence,atom_index,atom_id in zip(sequences,inputs_scannet[6],inputs_scannet[5])]


        model = wrappers.load_model(model_folder+model_name,Lmax=Lmax)
        layer_outputs = [model.model.get_layer(layer).output for layer in ['SCAN_filter_activity_atom','all_embedded_attributes_aa','SCAN_filter_activity_aa','classifier_output'] ]
        model.model = Model(inputs=model.model.inputs,outputs= layer_outputs
                            )
        model.multi_outputs = True
        model.Lmax_output = [int(output.shape[1]) for output in layer_outputs]
        if nmax is not None:
            nmax = min(len(inputs_handcrafted), nmax)
            inputs_scannet = wrappers.slice_list_of_arrays(inputs_scannet,np.arange(nmax))
            inputs_handcrafted = wrappers.slice_list_of_arrays(inputs_handcrafted,np.arange(nmax))

        [atomic_filter_activities,aggregated_atomic_filter_activities,aminoacid_filter_activities,classifier_output] = model.predict(inputs_scannet,return_all=True)
        atomic_filter_activities_flat = np.concatenate(atomic_filter_activities)
        aggregated_atomic_filter_activities_flat = np.concatenate(aggregated_atomic_filter_activities)
        handcrafted_features_flat = np.concatenate([features[:Lmax] for features in inputs_handcrafted])
        aminoacid_filter_activities_flat = np.concatenate(aminoacid_filter_activities)
        classifier_output_flat = np.concatenate(classifier_output)[:,-1]

        aminoacid = np.concatenate([sequence[:Lmax] for sequence in sequences])
        atom = np.concatenate([atom_id[:Lmax*9] for atom_id in atom_ids])-1
        aminoacid_of_atom = np.concatenate([sequence[input_scannet[:,0]][:Lmax*9] for sequence,input_scannet in zip(sequences,inputs_scannet[-2])])


        W1 = model.model.get_layer('SCAN_filter_input_aa').get_weights()[0]
        W2 = model.model.get_layer('SCAN_filter_input_aa').get_weights()[1]
        gaussian_activities_flat = np.dot(aggregated_atomic_filter_activities_flat, W1) + W2[np.newaxis]

        Ngaussians = gaussian_activities_flat.shape[1]
        nfilters = gaussian_activities_flat.shape[2]

        maxi = int(len(aggregated_atomic_filter_activities_flat) * top_percent/100.)
        top_pos = np.argsort(gaussian_activities_flat, axis=0)[-maxi:]
        top_neg = np.argsort(gaussian_activities_flat, axis=0)[:maxi]

        top_pos_handcrafted_features = handcrafted_features_flat[top_pos]
        top_neg_handcrafted_features = handcrafted_features_flat[top_neg]

        value_pos = np.array(
            [[ np.nanmean(gaussian_activities_flat[top_pos[:, u, v], u, v]) for u in range(Ngaussians)] for v in
             range(nfilters)])  # nfilters X ngaussians
        value_neg = np.array(
            [[ np.nanmean(gaussian_activities_flat[top_neg[:, u, v], u, v]) for u in range(Ngaussians)] for v in
             range(nfilters)])  # nfilters X ngaussians


        seq_pos, ss_pos, asa_pos = unpack(top_pos_handcrafted_features)
        seq_neg, ss_neg, asa_neg= unpack(top_neg_handcrafted_features)

        aa_probability = seq_pos.mean(0)
        aa_probability_neg = seq_neg.mean(0)

        aa_probability = np.swapaxes(aa_probability, 0, 1)
        aa_probability_neg = np.swapaxes(aa_probability_neg, 0, 1)


        ss_probability = ss_pos.mean(0)
        ss_probability_neg = ss_neg.mean(0)
        ss_probability = np.swapaxes(ss_probability, 0, 1)
        ss_probability_neg = np.swapaxes(ss_probability_neg, 0, 1)

        thetas_asa = [0., 0.05, 0.25, 0.5, 1.0]

        asa_probability = np.array([((asa_pos >= thetas_asa[x]) & (asa_pos <= thetas_asa[x + 1])) for x in range(4)]).mean(
            1)
        asa_probability = asa_probability.T

        asa_probability_neg = np.array(
            [((asa_neg >= thetas_asa[x]) & (asa_neg <= thetas_asa[x + 1])) for x in range(4)]).mean(1)
        asa_probability_neg = asa_probability_neg.T





        gaussiankernel_layer = model.model.get_layer('embedded_local_coordinates_aa')
        if gaussiankernel_layer.covariance_type == 'full':
            centers, sqrt_precisions = gaussiankernel_layer.get_weights()
            covariances = [0.1 * np.eye(3) + np.linalg.inv(np.dot(sqrt_precisions[:, :, u].T, sqrt_precisions[:, :, u])) for
                           u in range(sqrt_precisions.shape[-1])]
        else:
            centers, widths = gaussiankernel_layer.get_weights()
            covariances = [0.1 * np.eye(3) + np.diag(widths[:, u]) ** 2 for u in range(widths.shape[-1])]


        filter_specificities['aa_specificity'] = {
            'aa':aa_probability,
            'aa_neg':aa_probability_neg,
            'ss':ss_probability,
            'ss_neg': ss_probability_neg,
            'asa': asa_probability,
            'asa_neg': asa_probability_neg,
            'value':value_pos,
            'value_neg':value_neg,
        }

        filter_specificities['aa_gaussian'] = {'centers':centers,
                              'covariances':covariances
                              }

        filter_specificities['aa_activity'] = {
            'percentiles': np.percentile(aminoacid_filter_activities_flat, np.arange(1, 101),axis=0),
            'mean': aminoacid_filter_activities_flat.mean(0),
            'median': np.median(aminoacid_filter_activities_flat,axis=0),
            'std': aminoacid_filter_activities_flat.std(0),
            'correlation': np.corrcoef(aminoacid_filter_activities_flat.T),
            'correlation2output': np.corrcoef(aminoacid_filter_activities_flat.T,classifier_output_flat)[:-1,-1],
            'conditional_median': np.array([np.median(aminoacid_filter_activities_flat[aminoacid==k,:],axis=0) for k in range(20)]),
        }

        filter_specificities['aggregated_atom_activity'] = {
            'percentiles': np.percentile(aggregated_atomic_filter_activities_flat, np.arange(1, 101),axis=0),
            'mean': aggregated_atomic_filter_activities_flat.mean(0),
            'median': np.median(aggregated_atomic_filter_activities_flat,axis=0),
            'std': aggregated_atomic_filter_activities_flat.std(0),
            'correlation': np.corrcoef(aggregated_atomic_filter_activities_flat.T),
            'correlation2output': np.corrcoef(aggregated_atomic_filter_activities_flat.T,classifier_output_flat)[:-1,-1],
            'conditional_median': np.array([np.median(aggregated_atomic_filter_activities_flat[aminoacid==k,:],axis=0) for k in range(20)]),
        }

        filter_specificities['atom_activity'] = {
            'percentiles': np.percentile(atomic_filter_activities_flat, np.arange(1, 101), axis=0),
            'mean': atomic_filter_activities_flat.mean(0),
            'median': np.median(atomic_filter_activities_flat, axis=0),
            'std': atomic_filter_activities_flat.std(0),
            'correlation': np.corrcoef(atomic_filter_activities_flat.T),
            'conditional_median': np.array(
                [np.median(atomic_filter_activities_flat[atom == k,:],axis=0)for k in range(38)]),
            'conditional_median2':np.array(
                [[np.median(atomic_filter_activities_flat[(atom == k) & (aminoacid_of_atom==j),:],axis=0) if ((atom == k) & (aminoacid_of_atom==j)).sum()>0 else np.zeros(atomic_filter_activities_flat.shape[-1]) for k in range(38)] for j in range(20)]),
        }


    else:
        model = wrappers.load_model(model_folder + model_name)

    atom_embedding_weights = model.model.get_layer('embedded_attributes_atom').get_weights()[0]
    atom_filter_weights = model.model.get_layer('SCAN_filter_input_atom').get_weights()[0]
    atom_filter_offsets = model.model.get_layer('SCAN_filter_input_atom').get_weights()[1]
    atom_gaussian_weights = np.dot(atom_embedding_weights, atom_filter_weights)
    atom_gaussian_weights = atom_gaussian_weights[1:] + atom_filter_offsets[np.newaxis]



    gaussiankernel_layer = model.model.get_layer('embedded_local_coordinates_atom')
    if gaussiankernel_layer.covariance_type == 'full':
        centers, sqrt_precisions = gaussiankernel_layer.get_weights()
        covariances = [0.1 * np.eye(3) + np.linalg.inv(np.dot(sqrt_precisions[:, :, u].T, sqrt_precisions[:, :, u])) for
                       u in range(sqrt_precisions.shape[-1])]
    else:
        centers, widths = gaussiankernel_layer.get_weights()
        covariances = [0.1 * np.eye(3) + np.diag(widths[:, u]) ** 2 for u in range(widths.shape[-1])]



    gaussiankernel_layer = model.model.get_layer('embedded_local_coordinates_atom')
    if gaussiankernel_layer.covariance_type == 'full':
        centers, sqrt_precisions = gaussiankernel_layer.get_weights()
        covariances = [0.1 * np.eye(3) + np.linalg.inv(np.dot(sqrt_precisions[:, :, u].T, sqrt_precisions[:, :, u])) for
                       u in range(sqrt_precisions.shape[-1])]
    else:
        centers, widths = gaussiankernel_layer.get_weights()
        covariances = [0.1 * np.eye(3) + np.diag(widths[:, u]) ** 2 for u in range(widths.shape[-1])]

    filter_specificities['atom_specificity'] = atom_gaussian_weights

    filter_specificities['atom_gaussian'] = {'centers':centers,
                          'covariances':covariances
                          }
    filter_specificities['atom_features'] = 'valency'



    if not only_atom:
        io_utils.save_pickle(filter_specificities,path2filter_specificities)
    return filter_specificities



def plot_atomic_filter(filter_specificities, index, threshold1=0.33, threshold2=0.25,y_offset=0,list_additional_objects = [], sg=None,camera_position=None):
    if camera_position is None:
        camera_position = [0.8, 0.5, 0.8]

    atom_gaussian_weights = filter_specificities['atom_specificity'][:,:,index]
    centers = filter_specificities['atom_gaussian']['centers']
    covariances = filter_specificities['atom_gaussian']['covariances']
    atom_features = filter_specificities['atom_features']

    important_gaussians = np.nonzero(np.abs(atom_gaussian_weights).max(0) > max(threshold2,threshold1 * np.abs(atom_gaussian_weights).max() ) )[0]

    logo_length_pos = np.maximum(atom_gaussian_weights[:, important_gaussians], 0).sum(0)
    logo_length_neg = np.maximum(-atom_gaussian_weights[:, important_gaussians], 0).sum(0)
    for k,u in enumerate(important_gaussians):
        if (atom_gaussian_weights[:,u]<-threshold2).sum()>=10:
            logo_length_pos[k] /= atom_gaussian_weights.shape[0]
            logo_length_neg[k] /= atom_gaussian_weights.shape[0]


    ymax = np.maximum(logo_length_pos,logo_length_neg).max()
    list_ellipsoids = [(centers[:, u], covariances[u]) for u in important_gaussians]


    if atom_features == 'type':
        weight_logo = weight_logo_3d.weight_logo_atom
        list_atoms = weight_logo_3d.list_atoms
        atom_colors = weight_logo_3d.atom_colors
    elif atom_features == 'valency':
        weight_logo = weight_logo_3d.weight_logo_valency
        list_atoms = weight_logo_3d.list_atom_valencies
        atom_colors = weight_logo_3d.valency_colors
    else:
        raise ValueError

    list_figures = [weight_logo(atom_gaussian_weights[:, u], ymax=ymax, threshold=threshold2) for u
                    in important_gaussians]
    list_colors = []
    for u in important_gaussians:
        if atom_gaussian_weights[:, u].max() > threshold2:
            color = atom_colors[list_atoms[np.argmax(atom_gaussian_weights[:, u])]]
        else:
            color = 'gray'
        list_colors.append(color)

    return weight_logo_3d.show_ellipsoids(list_ellipsoids=list_ellipsoids,
                                          list_colors=list_colors,
                                          list_figures=list_figures,
                                          level=1.0, sg=sg,
                                          wireframe=True,
                                          show_frame=True,
                                          fs=1.,
                                          scale=2.0,
                                          offset=[0, y_offset, 0],
                                          list_additional_objects=list_additional_objects,
                                          camera_position=camera_position,
                                          key_light_position=[0.5, 1, 0],
                                          maxi=5)




def plot_aminoacid_filter(filter_specificities,
                        index,
                        show_asa = True,
                        show_ss = False,
                        show_negative=False,
                        threshold1=0.33,
                        scale=5.,
                        list_additional_objects = [],
                        sg=None,
                        camera_position=None
                          ):

    if camera_position is None:
        camera_position = [0.8, 0.5, 0.8]

    centers = filter_specificities['aa_gaussian']['centers']
    covariances = filter_specificities['aa_gaussian']['covariances']
    ngaussians = centers.shape[-1]


    aa = filter_specificities['aa_specificity']['aa'][index]
    if show_negative:
        aa_neg = filter_specificities['aa_specificity']['aa_neg'][index]
    else:
        aa_neg = [None for _ in range(ngaussians)]

    if show_ss:
        ss = filter_specificities['aa_specificity']['ss'][index]
        if show_negative:
            ss_neg = filter_specificities['aa_specificity']['ss_neg'][index]
        else:
            ss_neg =  [None for _ in range(ngaussians)]
    else:
        ss = [None for _ in range(ngaussians)]
        ss_neg = [None for _ in range(ngaussians)]

    if show_asa:
        asa = filter_specificities['aa_specificity']['asa'][index]
        if show_negative:
            asa_neg = filter_specificities['aa_specificity']['asa_neg'][index]
        else:
            asa_neg = [None for _ in range(ngaussians)]
    else:
        asa = [None for _ in range(ngaussians)]
        asa_neg = [None for _ in range(ngaussians)]

    value = filter_specificities['aa_specificity']['value'][index]
    if show_negative:
        value_neg = filter_specificities['aa_specificity']['value_neg'][index]
    else:
        value_neg = value * 0

    maximum = np.maximum(np.abs(value).max(),np.abs(value_neg).max() )
    scaling = value/maximum
    scaling_neg = np.abs(value_neg/maximum)
    important_gaussians = np.nonzero( (scaling > threshold1)  | (scaling_neg > threshold1) )[0]

    list_ellipsoids = [(centers[:, u], covariances[u]) for u in important_gaussians]

    list_figures = [weight_logo_3d.complex_filter_logo(
        aa[u],
        aa_probability_neg=aa_neg[u],
        ss_probability=ss[u],
        ss_probability_neg=ss_neg[u],
        asa_probability=asa[u],
        asa_probability_neg = asa_neg[u],
        scaling_='none',
        scaling=scaling[u],
        scaling_neg=scaling_neg[u]) for u in important_gaussians]

    list_colors = [weight_logo_3d.aa_colors[weight_logo_3d.list_aa[np.argmax(aa[u])]] for u in important_gaussians]

    return weight_logo_3d.show_ellipsoids(list_ellipsoids=list_ellipsoids,
                                          list_colors=list_colors,
                                          list_figures=list_figures,
                                          list_additional_objects = list_additional_objects,
                                          level=1.0, sg=sg,
                                          wireframe=True,
                                          show_frame=True,
                                          fs=1.,
                                          scale=scale,
                                          offset=[0, scale*0.5, 0],
                                          camera_position=camera_position,
                                          key_light_position=[0.5, 1, 0],
                                          maxi=12,dpi=600,crop=False)




def plot_gaussian_kernels(params,sg=None,maxi=5,camera_position=None):
    centers = params['centers']
    covariances = params['covariances']
    K = centers.shape[1]
    list_ellipsoids = [(centers[:, u], covariances[u]) for u in range(K)]
    list_colors = [matplotlib.colors.to_rgb('C%s'%k) for k in range(K)]

    return weight_logo_3d.show_ellipsoids(list_ellipsoids=list_ellipsoids,
                                          list_colors=list_colors,
                                          level=1.0, sg=sg,
                                          wireframe=False,
                                          show_frame=True,
                                          fs=1.,
                                          scale=0.75,
                                          camera_position=camera_position,
                                          key_light_position=[0.5, 1, 0],
                                          opacity=0.3,
                                          maxi=maxi)



def plot_atomic_neighborhood(neighbors, atom_id, index=None,sg=None, camera_position=None):
    if camera_position is None:
        camera_position = [0.8, 0.5, 0.8]
    Kmax = neighbors.shape[0]
    if atom_id.max() > 4:
        atom_features = 'valency'
    else:
        atom_features = 'atom'

    if atom_features == 'type':
        weight_logo = weight_logo_3d.weight_logo_atom
        list_atoms = weight_logo_3d.list_atoms
        atom_colors = weight_logo_3d.atom_colors
    elif atom_features == 'valency':
        weight_logo = weight_logo_3d.weight_logo_valency
        list_atoms = weight_logo_3d.list_atom_valencies
        atom_colors = weight_logo_3d.valency_colors
    else:
        raise ValueError

    list_ellipsoids = [(neighbors[k], np.eye(3) * 0.2 ** 2) for k in range(Kmax)]
    list_colors = [atom_colors[list_atoms[atom_id[k] - 1]] for k in range(Kmax)]
    list_figures = []
    for k in range(Kmax):
        weights = np.zeros(len(list_atoms))
        weights[atom_id[k] - 1] = 4.
        list_figures.append(weight_logo(weights, ymax=4, threshold=0.5, bar=False))

    distances = np.sqrt(((neighbors[:, np.newaxis] - neighbors[np.newaxis, :]) ** 2).sum(-1))
    if index is not None:
        res_distances = np.abs( index[:,np.newaxis,0] - index[np.newaxis,:,0] )
        is_c_or_n =   np.array([id_-1 in [0,1,7,8] for id_ in atom_id])
        mask = (res_distances ==0 ) | ( (res_distances==1) & (is_c_or_n[:,np.newaxis]) & (is_c_or_n[np.newaxis,:]) )
        distances += (1-mask) * 10

    segment_starts,segment_ends =  np.nonzero( (distances > 0) & (distances < 1.7) )
    list_segments = [ [list(neighbors[segment_start]), list(neighbors[segment_end]) ] for segment_start,segment_end in zip(segment_starts,segment_ends) if segment_end>segment_start ]


    return weight_logo_3d.show_ellipsoids(list_ellipsoids=list_ellipsoids,
                                          list_colors=list_colors,
                                          list_figures=list_figures,
                                          list_segments=list_segments,
                                          level=1.0, sg=sg,
                                          wireframe=False,
                                          show_frame=True,
                                          fs=1.,
                                          scale=1.0,
                                          offset=[0., 0.25, 0.0],
                                          camera_position=camera_position,
                                          key_light_position=[0.5, 1, 0],
                                          opacity=0.6,
                                          maxi=5)





