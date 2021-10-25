import os
import Bio.PDB
import numpy as np
import warnings
from numba import njit, prange
from scipy.linalg import eigh
from Bio.PDB import Selection
from Bio.PDB.ResidueDepth import _read_vertex_array, _get_atom_radius
import tempfile
from utilities.paths import structures_folder,path_to_dssp,path_to_msms
from preprocessing.protein_chemistry import list_atoms,list_atoms_types,VanDerWaalsRadii,atom_mass,atom_type_to_index,atom_to_index,index_to_type,atom_type_mass
from preprocessing.protein_chemistry import residue_dictionary,hetresidue_field
from preprocessing import PDBio

#%% Functions for parsing PDB files.

def is_residue(residue):
    try:
        return (residue.get_id()[0] in hetresidue_field) & (residue.resname in residue_dictionary.keys())
    except:
        return False


def is_heavy_atom(atom):
    # Second condition for Rosetta-generated files.
    try:
        return (atom.get_id() in atom_to_index.keys() )
    except:
        return False

def is_hydrogen(atom):
    atomid = atom.get_id()
    if len(atomid)>0:
        cond1 = (atomid[0] == 'H')
        cond2a = (atomid[0] in ['*','0','1','2','3','4','5','6','7','8','9'])
    else:
        cond1 = False
        cond2a = False
    if len(atomid)>1:
        cond2b = atomid[1] == 'H'
    else:
        cond2b = False
    return cond1 | (cond2a & cond2b)




def process_chain(chain):
    sequence = ''
    backbone_coordinates = []
    all_coordinates = []
    all_atoms = []
    all_atom_types = []
    for residue in Selection.unfold_entities(chain, 'R'):
        if is_residue(residue):
            sequence += residue_dictionary[residue.resname]
            residue_atom_coordinates = np.array(
                [atom.get_coord() for atom in residue if is_heavy_atom(atom)])
            residue_atoms = [atom_to_index[atom.get_id()]
                             for atom in residue if is_heavy_atom(atom)  ]
            residue_atom_type = [atom_type_to_index[atom.get_id()[0]]
                                 for atom in residue if is_heavy_atom(atom) ]
            residue_backbone_coordinates = []
            for atom in ['N', 'C', 'CA', 'O', 'CB']:
                try:
                    residue_backbone_coordinates.append(
                        residue_atom_coordinates[residue_atoms.index(atom_to_index[atom])])
                except:
                    residue_backbone_coordinates.append(
                        np.ones(3, dtype=np.float32) * np.nan)
            backbone_coordinates.append(residue_backbone_coordinates)
            all_coordinates.append(residue_atom_coordinates)
            all_atoms.append(residue_atoms)
            all_atom_types.append(residue_atom_type)
    backbone_coordinates = np.array(backbone_coordinates)
    return sequence, backbone_coordinates, all_coordinates, all_atoms, all_atom_types



def fill_cbeta(backbone_coordinates):
    """
        cbeta = dans le plan n,calpha,c. rotation de 2pi/3.
        cbeta - calpha = - (n-calpha) + - (c-calpha)
        cbeta = 3 calpha -n - c
    """
    n = backbone_coordinates[:, 0]
    c = backbone_coordinates[:, 1]
    calpha = backbone_coordinates[:, 2]
    cbeta = backbone_coordinates[:, 4]
    problematic = np.isnan(cbeta).max(1)
#     print(np.nonzero(problematic)[0])
    cbeta[problematic] = 3 * calpha[problematic] - \
        n[problematic] - c[problematic]
    return backbone_coordinates


@njit(parallel=True)
def _get_SideChainCenterofMass(atomic_coordinates, atom_types):
    N = len(atomic_coordinates)
    SideChainCenterofMass = np.zeros((N, 3))
    for n in prange(N):
        atoms = atomic_coordinates[n][4:]
        types = atom_types[n][4:]
        mass = 0
        for u in range(len(atoms)):
            SideChainCenterofMass[n] += atoms[u] * atom_mass[types[u]]
            mass += atom_mass[types[u]]
        SideChainCenterofMass[n] /= mass
    return SideChainCenterofMass


def get_SideChainCenterofMass(atomic_coordinates, atom_types, cbeta_coordinates=None):
    SideChainCenterofMass = _get_SideChainCenterofMass(
        atomic_coordinates, atom_types)
    if cbeta_coordinates is not None:
        glycins_or_missing_side_chain = np.isnan(SideChainCenterofMass[:, 0])
        SideChainCenterofMass[glycins_or_missing_side_chain,
                              :] = cbeta_coordinates[glycins_or_missing_side_chain, :]
    return SideChainCenterofMass




def get_PDB_indices(chain_obj,return_model=False,return_chain=False):
    list_indices = []
    for residue in Selection.unfold_entities(chain_obj, 'R'):
        if is_residue(residue):
            if return_model & return_chain:
                list_indices.append( (residue.get_full_id()[1],residue.get_full_id()[2],residue.get_id()[1]) )
            elif return_chain:
                list_indices.append( (residue.get_full_id()[2],residue.get_id()[1]) )
            else:
                list_indices.append(residue.get_id()[1])
    list_indices = np.array(list_indices)
    return list_indices


#%% Functions for finding interfaces residues and other distance-related features.

@njit(parallel=False)
def distance_residues(all_coordinates):
    N = len(all_coordinates)
    d = np.zeros((N, N))
    for i in prange(N):
        for j in prange(N):
            if i > j:
                d[i, j] = np.inf
                coord1 = all_coordinates[i]
                coord2 = all_coordinates[j]
                n1 = len(coord1)
                n2 = len(coord2)
                for n1_ in range(n1):
                    for n2_ in range(n2):
                        d[i, j] = min(
                            np.sum((coord1[n1_] - coord2[n2_])**2), d[i, j])
    d = d + d.T
    d = np.sqrt(d)
    return d


@njit(parallel=False)
def distance(x1, x2):
    n1 = x1.shape[0]
    n2 = x2.shape[0]
    out = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            out[i, j] = np.sqrt(((x1[i] - x2[j])**2).sum())
    return out


@njit(parallel=False)
def residues_in_contact(atom_coords1, atom_coords2, atoms1, atoms2, method='VdW', threshold=3.0):
    n1 = len(atoms1)
    n2 = len(atoms2)
    in_contact = False
    for i in range(n1):
        for j in range(n2):
            d = np.sqrt(((atom_coords1[i] - atom_coords2[j])**2).sum())
            if method == 'VdW':
                thresh = VanDerWaalsRadii[atoms1[i]] + \
                    VanDerWaalsRadii[atoms2[j]] + threshold
            elif method == 'Heavy':
                thresh = threshold
            if d < thresh:
                in_contact = True
                break
    return in_contact


def find_interface_pairs(calphapos1, list_coordinates1, list_atoms1, calphapos2, list_coordinates2, list_atoms2, method='VdW3'):
    n1 = len(list_coordinates1)
    n2 = len(list_coordinates2)

    if 'VdW' in method:
        threshold = float(method[3:])
        method_ = 'VdW'
    elif 'Calpha' in method:
        threshold = float(method[6:])
        method_ = 'Calpha'
    elif 'Heavy' in method:
        threshold = float(method[5:])
        method_ = 'Heavy'

    if method_ == 'Calpha':
        contact1, contact2 = np.nonzero(
            distance(calphapos1, calphapos2) < threshold)
        list_contacts = np.array(list(zip(contact1, contact2)))
    else:
        # Precompute the Calpha distances first... Only performs the Natoms1 X Natoms2 computation if the Calpha are close enough.
        maybe_in_contacts = distance(calphapos1, calphapos2) < 15.
        list_contacts = []
        for i in range(n1):
            for j in range(n2):
                if maybe_in_contacts[i, j]:
                    if residues_in_contact(list_coordinates1[i], list_coordinates2[j], list_atoms1[i], list_atoms2[j], method=method_, threshold=threshold):
                        list_contacts.append((i, j))
        list_contacts = np.array(list_contacts)
    return list_contacts


def find_nearby_interface_residues(interface_residues, structure):
    if len(interface_residues) > 0:
        calphapos = structure[:, 2, :]
        return np.nonzero(distance(calphapos[interface_residues], calphapos).min(0) < 6.)[0]
    else:
        return np.array([])





#%% Handcrafted geometrical features.

def apply_DSSP(chain_obj, pdbparser=None, io=None, path_to_dssp=path_to_dssp):
    if pdbparser is None:
        pdbparser = Bio.PDB.PDBParser()  # PDB parser; to read pdb files.
    if io is None:
        # PDB IO. To compute solvent accessibility, need to write a PDB file with only the monomers...
        io = Bio.PDB.PDBIO()
    if isinstance(chain_obj,list):
        pdb_id, model, chain = chain_obj[0].get_full_id()
        letters = ['A','B','C','D','E','F','G','H','I','J','K','L',
                  'M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
                  'a','b','c','d','e','f','g','h','i','j','k','l','m','n',
                  'o','p','q','r','s','t','u','v','w','x','y','z',
                  '1','2','3','4','5','6','7','8','9','10']
        chain_objs_copy = []
        for n in range(len(chain_obj)):
            chain_obj_copy = chain_obj[n].copy()
            chain_obj_copy.full_id = ('pdb',0,letters[n])
            chain_obj_copy.id = letters[n]
            chain_objs_copy.append(chain_obj_copy)

        io.set_structure(chain_objs_copy[0])

        for n in range(1,len(chain_obj)):
            io.structure[0].add(chain_objs_copy[n])
    else:
        pdb_id, model, chain = chain_obj.get_full_id()
        # chain id = AB for instance. Only for mmCif file, not compatible with pdbIO.
        if len(chain) > 1:
            chain_obj.id = '!'
        io.set_structure(chain_obj)
        chain_obj.id = chain

    for residue in Selection.unfold_entities(io.structure,'R'):
        for atom in residue:
            atom.disordered_flag = 0

    name = 'tmp_' + pdb_id + \
        '_model_%s_chain_%s' % (model, chain) + '.pdb'



    io.save(name)

    with warnings.catch_warnings(record=True) as w:
        mini_structure = pdbparser.get_structure(name[:-4]  # name of structure
                                                 ,  name)

    mini_model = mini_structure[0]
    dssp = Bio.PDB.DSSP(mini_model,
                        name,
                        dssp=path_to_dssp)
    os.system('rm %s' % name)

    secondary_structure = ''
    accessible_surface_area = []
    for residue in Selection.unfold_entities(mini_model,'R'):
        if is_residue(residue):
            try:
                secondary_structure += dssp[residue.get_full_id()
                                            [2:]][2]
                accessible_surface_area.append(
                    dssp[residue.get_full_id()[2:]][3])
                if accessible_surface_area[-1] == 'NA':
                    accessible_surface_area[-1] = np.nan
            except:
                secondary_structure += '/'
                accessible_surface_area.append(np.nan)

    accessible_surface_area = np.array(
        accessible_surface_area, dtype=np.float32)

    return secondary_structure, accessible_surface_area



def get_surface(chain, MSMS=path_to_msms, probe_radius=1.5):

    # Replace pdb_to_xyzr
    # Make x,y,z,radius file

    xyz_tmp = tempfile.mktemp()
    with open(xyz_tmp, 'w') as pdb_to_xyzr:
        all_coordinates = []
        all_ids = []
        for residue in Selection.unfold_entities(chain,'R'):
            if is_residue(residue):
                for atom in residue:
                    x, y, z = atom.coord
                    # If two atoms have identical coordinates, MSMS breaks.
                    if not (x, y, z) in all_coordinates:
                        radius = _get_atom_radius(atom, rtype='united')
                        print('{:6.3f}\t{:6.3f}\t{:6.3f}\t{:1.2f}'.format(x, y, z, radius),
                              file=pdb_to_xyzr)
                    all_coordinates.append((x, y, z))

    # make surface
    surface_tmp = tempfile.mktemp()
    MSMS = MSMS + " -probe_radius %.2f -if %s -of %s > " + tempfile.mktemp()
    make_surface = MSMS % (probe_radius, xyz_tmp, surface_tmp)
    os.system(make_surface)
    surface_file = surface_tmp + ".vert"
    print(make_surface)
    if not os.path.isfile(surface_file):
        raise RuntimeError("Failed to generate surface file using "
                           "command:\n%s" % make_surface)

    # read surface vertices from vertex file
    surface = _read_vertex_array(surface_file)
    return surface


def get_surface_robust(chain, MSMS=path_to_msms, probe_radius=1.5, ntrials_max=5):
    success = False
    ntrials = 0
    while not (success | (ntrials > ntrials_max)):
        try:
            surface = get_surface(chain, MSMS=MSMS, probe_radius=probe_radius)
            success = True
        except:
            if isinstance(chain,list):
                chainid = chain[0].get_full_id()
            else:
                chainid = chain.get_full_id()
            print('MSMS failed, radius = %.2f, chain id: (%s)' %
                  (probe_radius, chainid) )
            probe_radius = probe_radius + 0.01
            success = False
            ntrials += 1
    if ntrials > ntrials_max:
        raise RuntimeError("Failed to generate surface file using "
                           "command:\n%s")
    return surface


def get_surface_normal(surface, interior_points=None, radius=8., max_nn=10):
    npoints = len(surface)
    d = distance(surface, surface)
    normals = []
    for point in range(npoints):
        subset = np.argsort(d[point])[:max_nn]  # max_nn closest points.
        subset = subset[d[point][subset] < radius]  # within radius.
        relative_locations = surface[subset, :]
        mu = relative_locations.mean(0)
        covariance = np.dot(relative_locations.T, relative_locations) / \
            len(subset) - mu[:, np.newaxis] * mu[np.newaxis, :]
        lam, v = eigh(covariance)
        normals.append(v[:, 0])
    normals = np.array(normals)
    if interior_points is not None:
        closest_points = np.argsort(
            distance(surface, interior_points), axis=1)[:, 1:4]
        sign_dot_product = np.sign(np.sign(
            ((surface[:, np.newaxis] - interior_points[closest_points]) * normals[:, np.newaxis, :]).sum(-1)).mean(-1))
        # If negative, must switch signs.
        normals *= sign_dot_product[:, np.newaxis]
    return normals

def analyze_surface(chain_obj, atom_coordinates,
                    MSMS=path_to_msms,
                    probe_radius=1.5,
                    normal_estimation_radius=8.,
                    normal_estimation_nn=10,
                    nr_ball=5,
                    ntheta_ball=5,
                    nphi_ball=5,
                    volume_index_ball_radii=[5.0, 8.0, 11.0]
                    ):
    """
    Outputs:
    - Volume index (5.0, 8.0, 11.0)
    - Residue Depth (Backbone and Side Chain)
    - Half sphere exposure (up and down) + Coordination number.
    - [ASA]
    - Calcule la surface.
    - Calcule la normale à la surface.
    - Volume index.
    """
    nResidues = len(atom_coordinates)
    surface = get_surface_robust(
        chain_obj, MSMS=MSMS, probe_radius=probe_radius)
    normals = get_surface_normal(surface, interior_points=np.concatenate(
        atom_coordinates), radius=normal_estimation_radius, max_nn=normal_estimation_nn)

    BackboneDepth, SideChainDepth = ComputeResidueDepth(
        atom_coordinates, surface)

    VolumeIndex = np.zeros([nResidues, len(volume_index_ball_radii)])
    unit_ball_points, unit_ball_weights = get_unit_ball(
        nr_ball, ntheta_ball, nphi_ball)
    for i in range(nResidues):
        VolumeIndex[i, :] = compute_convexity_index(np.array(atom_coordinates[i]), surface, normals, ball_radii=volume_index_ball_radii,
                                                    unit_ball_points=unit_ball_points,
                                                    unit_ball_weights=unit_ball_weights).mean(0)
    return BackboneDepth, SideChainDepth, VolumeIndex


def is_inside_surface(query_point, surface, normals):
    """
    inside = -1
    outside = +1
    """
    closest = np.argmin(distance(query_point, surface), axis=1)
    is_inside = np.sign(
        ((-surface[closest] + query_point) * normals[closest]).sum(-1))
    return is_inside


def get_unit_ball(nr, ntheta, nphi):
    radii = np.arange(nr) / (nr - 1)
    thetas = np.arange(ntheta) / (ntheta - 1) * np.pi
    phis = np.arange(nphi) / (nphi - 1) * 2 * np.pi
    grid_r, grid_theta, grid_phi = np.meshgrid(
        radii, thetas, phis, indexing='ij')
    grid_r = grid_r.flatten()
    grid_theta = grid_theta.flatten()
    grid_phi = grid_phi.flatten()
    cartesian = np.zeros([nr * ntheta * nphi, 3])
    cartesian[:, 0] = grid_r * np.sin(grid_theta) * np.cos(grid_phi)
    cartesian[:, 1] = grid_r * np.sin(grid_theta) * np.sin(grid_phi)
    cartesian[:, 2] = grid_r * np.cos(grid_theta)
    dV = grid_r * np.sin(grid_theta)
    return cartesian, dV


def compute_convexity_index(query_points, surface, normals, ball_radii=[2.0],
                            nr=5,
                            ntheta=5,
                            nphi=5,
                            subset_size=20,
                            unit_ball_points=None,
                            unit_ball_weights=None
                            ):
    npoints = len(query_points)
    nradii = len(ball_radii)
    convexity_index = np.zeros([npoints, nradii])

    if (unit_ball_points is None) | (unit_ball_weights is None):
        unit_ball_points, unit_ball_weights = get_unit_ball(nr, ntheta, nphi)

    query_distances = distance(query_points, surface)

    for i in range(npoints):
        #         print(i)
        query_point = query_points[i]
        query_distance = query_distances[i]
        subset = np.argsort(query_distance)[:subset_size]
        surface_subset = surface[subset]
        normals_subset = normals[subset]
        for j, ball_radius in enumerate(ball_radii):
            is_inside = is_inside_surface(
                query_point + ball_radius * unit_ball_points, surface_subset, normals_subset)
            convexity_index[i, j] = (
                is_inside * unit_ball_weights).sum() / unit_ball_weights.sum()
    return np.squeeze(convexity_index)


def ComputeResidueDepth(atom_coordinates, surface):
    nResidues = len(atom_coordinates)
    BackboneDepth = []
    SideChainDepth = []
    for i in range(nResidues):
        nAtoms = len(atom_coordinates[i])
        atomDepth = distance(atom_coordinates[i], surface).min(1)
        if nAtoms >= 4:
            BackboneDepth.append(atomDepth[:4].mean())
            SideChainDepth.append(atomDepth[4:].mean())
        else:  # Missing atoms.
            BackboneDepth.append(atomDepth.mean())
            SideChainDepth.append(atomDepth.mean())
    BackboneDepth = np.array(BackboneDepth)
    SideChainDepth = np.array(SideChainDepth)
    glycins_or_missing_side_chain = np.isnan(SideChainDepth)
    SideChainDepth[glycins_or_missing_side_chain] = BackboneDepth[glycins_or_missing_side_chain]
    return BackboneDepth, SideChainDepth


def ComputeResidueHSE(backbone_coordinates, radius=13):
    nResidues = backbone_coordinates.shape[0]
    fill_cbeta(backbone_coordinates)
    calphas = backbone_coordinates[:, 2, :]
    cbetas = backbone_coordinates[:, -1, :]
    calpha_distances = distance(calphas, calphas)

    HalfSphere_up = np.zeros(nResidues, dtype=np.int)
    HalfSphere_down = np.zeros(nResidues, dtype=np.int)
    Coordination = np.zeros(nResidues, dtype=np.int)
    for i in range(nResidues):
        neighbor_residues = (calpha_distances[i] < radius) & (
            calpha_distances[i] > 0)
        in_upper_plane = np.dot(
            calphas[neighbor_residues] - calphas[i][np.newaxis], cbetas[i] - calphas[i]) >= 0
        HalfSphere_up[i] = in_upper_plane.sum()
        Coordination[i] = neighbor_residues.sum()
        HalfSphere_down[i] = Coordination[i] - HalfSphere_up[i]
    HalfSphere_excess_up = (HalfSphere_up - HalfSphere_down) / Coordination
    return HalfSphere_excess_up, Coordination


def make_values_dictionary(resids,values):
    dictionary_values = {}
    values_is_list = isinstance(values,list)
    for l,resid in enumerate(resids):
        key = (int(resid[0]), resid[1], int(resid[2]) )
        if values_is_list:
            dictionary_values[ key ] = [value[l] for value in values]
        else:
            dictionary_values[ key ] = values[l]
    return dictionary_values

