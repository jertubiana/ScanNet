import numpy as np
import bisect
import pandas as pd



def show_binding_sites(
        pdb_files,
        result_files,
        output_name,
        directory='',
        biounit=True,
        thresholds = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1.00],
        colors = [
            'medium blue',
            'blue',
            'cornflower blue',
            'tan',
            'yellow',
            'orange',
            'orange red',
            'red']
        ):


    if isinstance(pdb_files, list):
        # several_files = True
        pass
    else:
        # several_files = False
        pdb_files = [pdb_files]

    if not isinstance(result_files, list):
        result_files = [result_files]

    nstructures = len(pdb_files)

    use_local_files = [len(pdb) != 4 for pdb in pdb_files]
    interface_predictions = [pd.read_csv(result_file, sep=',')
                             if result_file is not None else None
                             for result_file in result_files]

    for n in range(nstructures):
        if interface_predictions is not None:
            interface_predictions[n]['Color'] = [colors[bisect.bisect_left(
                thresholds, probability)] for probability in interface_predictions[n]['Binding site probability']]

    output_full_name = output_name + '.py'
    with open(output_full_name, 'w') as f:
        f.write('import chimera\n')
        f.write('from chimera import runCommand\n')
        for pdb_file, use_local_file in zip(pdb_files, use_local_files):
            if use_local_file:
                f.write("runCommand('open %s%s')\n" % (directory, pdb_file))
            else:
                if biounit:
                    f.write("runCommand('open biounitID:%s.1')\n" % pdb_file)
                else:
                    f.write("runCommand('open pdbID:%s')\n" % pdb_file)
        f.write("runCommand('color dark gray #')\n")

        for n in range(nstructures):
            if interface_predictions[n] is not None:
                L = len(interface_predictions[n])
                if max(interface_predictions[n]['Model']) > 0:
                    add_one = 1
                else:
                    add_one = 0
                for l in range(L):
                    f.write("runCommand('color %s #%s.%s:%s.%s')\n" % (
                        interface_predictions[n]['Color'][l],
                        n,
                        interface_predictions[n]['Model'][l] + add_one,
                        interface_predictions[n]['Residue Index'][l],
                        interface_predictions[n]['Chain'][l]))

    return output_full_name








def annotate_pdb_file(pdb_file,csv_file,output_file,output_script=True,mini=0.0,maxi=0.8,version='default', field = 'Binding site probability'):
    '''
    rangecolor bfactor key 0.0 blue 0.2 yellow 0.4 red
    '''
    interface_predictions = pd.read_csv(csv_file, sep=',')
    resids = ['%s_%s_%s'%(x,y,z) for x,y,z in np.array(interface_predictions[['Model','Chain','Residue Index']]) ]
    probas = np.array(interface_predictions[field])
    model = -1
    multimodel = False
    is_cif = pdb_file[-4:] == '.cif'
    if is_cif:
        list_atom_columns = []
        model_index = None
        chain_index = None
        bfactor_index = None
        resnum_index = None
    with open(output_file,'w') as foutput:
        with open(pdb_file,'r') as finput:
            for line in finput:
                if not is_cif:
                    if (line[:5] == 'MODEL'):
                        model +=1

                if is_cif:
                    if line[:10] == '_atom_site':
                        list_atom_columns.append(line[11:-1])                        

                if line[:4] in ['ATOM','HETA']:
                    if is_cif:
                        line_splitted_nospaces = [y for y in line.split(' ') if y != '']
                        if model_index is None:
                            try:
                                model_index = list_atom_columns.index('pdbx_PDB_model_num')
                            except:
                                model_index = -2
                        if chain_index is None:
                            try:
                                chain_index = list_atom_columns.index('auth_asym_id')
                            except:
                                chain_index = -4
                        if resnum_index is None:
                            try:
                                resnum_index = list_atom_columns.index('auth_seq_id')
                            except:
                                resnum_index = -6
                        if bfactor_index is None:
                            try:
                                bfactor_index = list_atom_columns.index('B_iso_or_equiv')
                            except:
                                bfactor_index = -8


                        model = int(line_splitted_nospaces[model_index]) - 1
                        chain = line_splitted_nospaces[chain_index]
                        try:
                            number = int(line_splitted_nospaces[resnum_index])
                        except:
                            number = '.'
                        b_factor_index = line.index(line_splitted_nospaces[bfactor_index])
                        b_factor_length = len( line_splitted_nospaces[bfactor_index] )
                    else:
                        model = max(model,0)
                        chain = line[21]
                        number = int(line[22:26])

                    identifier = '%s_%s_%s'%(model,chain,number)
                    try:
                        if identifier in resids:
                            index = resids.index(identifier)
                            proba = probas[index]
                        else:
                            proba = -1
                        if is_cif:
                            new_line = line[:b_factor_index] + ('%.4f' % proba)[:b_factor_length] + line[b_factor_index+b_factor_length:]
                        else:
                            new_line = line[:60] + ('  %.2f'% proba)[:6] + line[66:]
                    except Exception as e:
                        print(e,line)
                        new_line = line
                else:
                    new_line = line
                foutput.write(new_line)
    multimodel = (model>0)
    if output_script:
        resids = np.array(interface_predictions[['Model','Chain','Residue Index']])
        if multimodel:
            add_one = 1
        else:
            add_one = 0
        with open(output_file[:-4] + '.py', 'w') as f:
            f.write('import chimera\n')
            f.write('from chimera import runCommand\n')
            f.write("runCommand('open %s')\n" % output_file.split('/')[-1])
            list_chains = [x.split('_') for x in
                           np.unique([str(resids[u, 0]) + '_' + str(resids[u, 1]) for u in range(len(resids))])]
            for chain in list_chains:
                f.write("runCommand('rangecolor bfactor key %s blue %s yellow %s red #0.%s:.%s')\n" % (
                mini, (mini + maxi) / 2, maxi, int(chain[0]) + add_one, chain[1]))


        with open(output_file[:-4] + '.cxc','w') as f:
            f.write('open %s\n'% output_file.split('/')[-1])
#            f.write('graphics bgColor white\n')
            f.write('dssp\n')
            f.write('hide atoms\n')
            f.write('show cartoon\n')
            if not version == 'surface':
                f.write('show surface\n')
            f.write('color gray transparency 10\n')
            list_chains = [x.split('_') for x in
                           np.unique([str(resids[u, 0]) + '_' + str(resids[u, 1]) for u in range(len(resids))])]

            for chain in list_chains:
                if multimodel:
                    if version == 'surface':
                        f.write("surface #1.%s/%s\n" % ( int(chain[0]) + 1, chain[1]) )
                        f.write("color by bfactor #1.%s/%s range %s,%s transparency 0\n" % (
                            int(chain[0]) + 1, chain[1] , mini,maxi ))
                    else:
                        f.write("color by bfactor #1.%s/%s range %s,%s transparency 0\n" % (
                            int(chain[0]) + 1, chain[1] , mini,maxi ))

                else:
                    if version == 'surface':
                        f.write("surface /%s\n" % (  chain[1]) )
                        f.write("color by bfactor /%s range %s,%s transparency 0\n" % (
                             chain[1] , mini,maxi ))
                    else:
                        f.write("color by bfactor /%s range %s,%s  transparency 0\n" % (
                             chain[1] , mini,maxi ))

            if not version == 'surface':
                f.write('hide surface\n')
            f.write('lighting soft\n')
#            f.write('save %s.png format png width 4000 supersample 4 transparentBackground true\n'%output_file[:-4])
    return output_file


def angular_to_cartesian( angles ):
    x,y,z = angles
    cx = np.cos(x)
    cy = np.cos(y)
    cz = np.cos(z)
    sx = np.sin(x)
    sy = np.sin(y)
    sz = np.sin(z)
    rotation = np.zeros([3, 3])
    rotation[0, 0] = cz * cy
    rotation[0, 1] = -sy * sx * cz - sz * cx
    rotation[0, 2] = -sy * cx * cz + sz * sx
    rotation[1, 0] = sz * cy
    rotation[1, 1] = -sy * sx * sz + cx * cz
    rotation[1, 2] = -sy * cx * sz - sx * cz
    rotation[2, 0] = sy
    rotation[2, 1] = cy * sx
    rotation[2, 2] = cy * cx
    return rotation

def cartesian_to_angular(rotation):
    '''
    Test script:
    for k in range(100):
        angles = np.random.rand(3) * (2*np.pi) - np.pi
        rotation = angular_to_cartesian(angles)
        reconstructed_angles = cartesian_to_angular(rotation)
        reconstructed_rotation = angular_to_cartesian(reconstructed_angles)
        print(k,   (reconstructed_angles-angles)/np.pi )
        print(k, (reconstructed_angles + angles) / np.pi)
        print(k,np.abs(reconstructed_rotation-rotation).max())
    '''
    x = np.arctan2(rotation[2,1],rotation[2,2])
    y = np.arctan2(rotation[2,0], rotation[2,1]/np.sin(x) )
    z = np.arctan2( rotation[1,0]/np.cos(y), rotation[0,0]/np.cos(y) )
    return np.array([x,y,z])


def frame_to_center_axis_angle(frame):
    center = frame[0]
    rotation = frame[1:]
    angles = cartesian_to_angular(rotation)
    x,y,z = angles
    axis = np.array([ np.sin(y), np.cos(y) * np.sin(x), np.cos(y) * np.cos(x)])
    angle = z / (2*np.pi) * 360
    return frame,axis,angle
