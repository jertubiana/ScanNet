import pythreejs
import numpy as np
from preprocessing import PDBio,PDB_processing,pipelines,protein_chemistry
from network import neighborhoods

def rgb_to_hex(rgb):
    if isinstance(rgb, str):
        return rgb
    else:
        rgb = np.array(rgb)[:3]
        if rgb.max() < 1:
            rgb *= 256
        rgb = np.floor(rgb).astype(np.int)
        return '#%02x%02x%02x' % (rgb[0], rgb[1], rgb[2])

def orientation2euler(bond_orientation, axis=2):
    from scipy.spatial.transform import Rotation as R
    bond_matrix = np.eye(3)
    bond_matrix[:, axis] = bond_orientation
    permutation = (axis + np.arange(3)) % 3
    bond_matrix = bond_matrix[:, permutation]
    q, r = np.linalg.qr(bond_matrix)
    if np.sign(r[0, 0]) == -1:
        bond_rotation_matrix = -q[:, [0, 2, 1]]
    else:
        bond_rotation_matrix = q
    permutation = np.arange(3)
    permutation[(axis + np.arange(3)) % 3] = np.arange(3)
    bond_rotation_matrix = bond_rotation_matrix[:, permutation]
    euler_rotation = R.from_matrix(bond_rotation_matrix).as_euler('XYZ', degrees=False)
    return euler_rotation


def show_atoms(
        atom_positions,
        atom_types,
        atom_bonds,
        render=True,
        radius_scale=0.2,
        stick_radius=0.75,
        stick_height=0.8,
        camera_position=[0.8, 0.5, 0.8],
        key_light_position=[0.5, 1, 0.0]
):
    default_colors = [
        [210 / 256, 180 / 256, 140 / 256],
        [255 / 256, 0, 0],
        [0, 0, 255 / 256],
        [255 / 256, 255 / 256, 0]
    ]
    VanDerWaalsRadii = np.array([1.70, 1.52, 1.55, 1.80])

    bond_orientations = [atom_positions[atom_bond[1]] - atom_positions[atom_bond[0]] for atom_bond in atom_bonds]
    bond_lengths = [np.sqrt((bond_orientation ** 2).sum(-1)) for bond_orientation in bond_orientations]
    bond_orientations = [bond_orientation / np.sqrt((bond_orientation ** 2).sum(-1, keepdims=True)) for bond_orientation
                         in bond_orientations]

    bond_first_stick_positions = [
        atom_positions[atom_bond[0]] + (bond_length / 2 - stick_height / 2) * bond_orientation
        for atom_bond, bond_length, bond_orientation in zip(atom_bonds, bond_lengths, bond_orientations)]

    bond_second_stick_positions = [
        atom_positions[atom_bond[0]] + (bond_length / 2 + stick_height / 2) * bond_orientation
        for atom_bond, bond_length, bond_orientation in zip(atom_bonds, bond_lengths, bond_orientations)]

    bond_first_stick_colors = [default_colors[atom_types[atom_bond[0]]] for atom_bond in atom_bonds]
    bond_second_stick_colors = [default_colors[atom_types[atom_bond[1]]] for atom_bond in atom_bonds]

    bond_eulers = [orientation2euler(bond_orientation, axis=1) for bond_orientation in bond_orientations]

    if render:
        key_light = pythreejs.DirectionalLight(position=key_light_position, intensity=.3)
        ambient_light = pythreejs.AmbientLight(intensity=.8)
        camera = pythreejs.PerspectiveCamera(position=camera_position)
        controller = pythreejs.OrbitControls(controlling=camera)
        children = [camera, key_light, ambient_light]
    else:
        children = []

    sphere_geometry = [pythreejs.SphereGeometry(radius=radius_scale * radius) for radius in VanDerWaalsRadii]
    stick_geometry = pythreejs.CylinderGeometry(radiusTop=radius_scale *stick_radius,
                                                radiusBottom=radius_scale *stick_radius,
                                                height=stick_height)

    list_spheres = [
        pythreejs.Mesh(geometry=sphere_geometry[atom_type], position=atom_position.tolist(),
                       material=pythreejs.MeshLambertMaterial(color=rgb_to_hex(default_colors[atom_type])))
        for atom_position, atom_type in zip(atom_positions, atom_types)
    ]

    list_first_sticks = [pythreejs.Mesh(geometry=stick_geometry,
                                        position=position.tolist(),
                                        rotation=(euler).tolist() + ['XYZ'],
                                        material=pythreejs.MeshLambertMaterial(color=rgb_to_hex(color))
                                        )
                         for position, euler, color in
                         zip(bond_first_stick_positions, bond_eulers, bond_first_stick_colors)

                         ]

    list_second_sticks = [pythreejs.Mesh(geometry=stick_geometry,
                                         position=position.tolist(),
                                         rotation=(euler).tolist() + ['XYZ'],
                                         material=pythreejs.MeshLambertMaterial(color=rgb_to_hex(color))
                                         )
                          for position, euler, color in
                          zip(bond_second_stick_positions, bond_eulers, bond_second_stick_colors)

                          ]

    children += list_spheres
    children += list_first_sticks
    children += list_second_sticks

    if render:
        scene = pythreejs.Scene(children=children)
        renderer = pythreejs.Renderer(camera=camera, scene=scene, controls=[controller],
                                      alpha=True,
                                      clearOpacity=0,
                                      width=1000, height=1000)
        return renderer
    else:
        return children





def get_neighborhood(
        pdb = '1a3x',
        model = 0,
        chain = 'A',
        resnumber = 1,
        atom = None,
        resindex = None,
        atomindex= None,
        assembly=False,
        biounit=True,
        Kmax = None

):
    if Kmax is None:
        if (atom is not None) | (atomindex is not None):
            Kmax = 32
        else:
            Kmax = 16*9


    filename,_ = PDBio.getPDB(pdb,biounit=biounit)
    if assembly == True:
        chain_ids = 'all'
    elif isinstance(assembly,list):
        chain_ids = assembly
    else:
        chain_ids = [(model,chain)]
    struct, chains = PDBio.load_chains(file=filename,chain_ids=chain_ids)
    pipeline = pipelines.ScanNetPipeline(aa_features='sequence',
                                         atom_features='id'
                                         )

    [aa_triplets, aa_attributes,aa_indices,aa_clouds,
     atom_triplets, atom_ids, atom_indices, atom_clouds],_ = pipeline.process_example(chains)
    atom_ids -=1
    atom_attributes = protein_chemistry.index_to_type[atom_ids]+1


    resids = PDB_processing.get_PDB_indices(chains,return_chain=True,return_model=True)

    if (atom is not None) | (atomindex is not None):
        if (atomindex is not None):
            index = atomindex
        else:
            try:
                index = np.nonzero(
                                   (resids[atom_indices[:,0], 0].astype(np.int) == model) &
                                   (resids[atom_indices[:,0], 1] == chain) &
                                   (resids[atom_indices[:,0], 2].astype(np.int) == resnumber) &
                                   (atom_ids == protein_chemistry.atom_to_index[atom])
                                   )[0][0]
            except:
                raise ValueError('Atom #%s/%s:%s@%s not found' % (model,chain,resnumber,atom) )
    else:
        if resindex is not None:
            index = resindex
        else:
            try:
                index = np.nonzero( (resids[:,0].astype(np.int) == model) &
                                    (resids[:,1]==chain) &
                                    (resids[:,2].astype(np.int)==resnumber)
                                    )[0][0]
            except:
                raise ValueError('Residue #%s/%s:%s not found' % (model, chain, resnumber))


    if (atom is not None) | (atomindex is not None):
        frames = neighborhoods.get_Frames(
            [[atom_triplets[index:index+1]],[atom_clouds]], order='2')
    else:
        frames = neighborhoods.get_Frames(
            [[aa_triplets[index:index+1]],[aa_clouds]], order='2')


    _, atom_types =  neighborhoods.get_LocalNeighborhood([ [frames[0]], [atom_clouds[atom_triplets[:,0]]]  ],{'self_neighborhood':False,'Kmax': Kmax},attributes= [atom_attributes],
                                        )
    atom_positions, atom_triplets =  neighborhoods.get_LocalNeighborhood([ [frames[0]], [atom_clouds[atom_triplets[:,0]]]  ],{'self_neighborhood':False,'Kmax': Kmax},attributes= [atom_triplets],
                                        )

    atom_types = atom_types.astype(np.int)
    atom_triplets = atom_triplets.astype(np.int)


    atom_positions = atom_positions[0][0]
    atom_types = atom_types[0][0] -1 # Inside the code, atom_type = 0 corresponds to virtual atom or empty placeholder.
    atom_triplets = atom_triplets[0][0]

    atom_index = list(atom_triplets[:,0])
    atom_bonds = np.zeros([Kmax,Kmax],dtype=np.bool)

    for triplet in atom_triplets:
        if triplet[1] in atom_index:
            atom_bonds[atom_index.index(triplet[0]),atom_index.index(triplet[1])] = 1
        if triplet[2] in atom_index:
            atom_bonds[atom_index.index(triplet[0]),atom_index.index(triplet[2])] = 1
    atom_bonds += atom_bonds.T
    atom_bonds = list(zip(*np.nonzero(atom_bonds) ) )
    atom_bonds = [(i,j) for i,j in atom_bonds if j>i]
    return atom_positions,atom_types,atom_bonds


