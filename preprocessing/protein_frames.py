from preprocessing.protein_chemistry import dictionary_covalent_bonds_numba, atom_type_mass, list_atoms, aa_to_index
from numba import njit, types
from numba.typed import List, Dict
import numpy as np


def get_atom_frameCloud(sequence, atom_coordinates, atom_ids):
    atom_clouds = np.concatenate(atom_coordinates, axis=0)
    atom_attributes = np.concatenate(atom_ids, axis=-1)
    atom_triplets = np.array(_get_atom_triplets(sequence, List(atom_ids), dictionary_covalent_bonds_numba),
                             dtype=np.int32)
    atom_indices = np.concatenate([np.ones(len(atom_ids[l]), dtype=np.int32) * l for l in range(len(sequence))],
                                  axis=-1)[:, np.newaxis]
    return atom_clouds, atom_triplets, atom_attributes, atom_indices


@njit(parallel=False, cache=False)
def _get_atom_triplets(sequence, atom_ids, dictionary_covalent_bonds_numba):
    L = len(sequence)
    atom_triplets = List()
    all_keys = List(dictionary_covalent_bonds_numba.keys() )
    current_natoms = 0
    for l in range(L):
        aa = sequence[l]
        atom_id = atom_ids[l]
        natoms = len(atom_id)
        for n in range(natoms):
            id = atom_id[n]
            if (id == 17):  # N, special case, bound to C of previous aa.
                if l > 0:
                    if 0 in atom_ids[l - 1]:
                        previous = current_natoms - len(atom_ids[l - 1]) + atom_ids[l - 1].index(0)
                    else:
                        previous = -1
                else:
                    previous = -1
                if 1 in atom_id:
                    next = current_natoms + atom_id.index(1)
                else:
                    next = -1
            elif (id == 0):  # C, special case, bound to N of next aa.
                if 1 in atom_id:
                    previous = current_natoms + atom_id.index(1)
                else:
                    previous = -1
                if l < L - 1:
                    if 17 in atom_ids[l + 1]:
                        next = current_natoms + natoms + atom_ids[l + 1].index(17)
                    else:
                        next = -1
                else:
                    next = -1

            else:
                key = (aa + '_' + str(id) )
                if key in all_keys:
                    previous_id, next_id, _ = dictionary_covalent_bonds_numba[(aa + '_' + str(id) )]
                else:
                    print('Strange atom', (aa + '_' + str(id) ))
                    previous_id = -1
                    next_id = -1
                if previous_id in atom_id:
                    previous = current_natoms + atom_id.index(previous_id)
                else:
                    previous = -1
                if next_id in atom_id:
                    next = current_natoms + atom_id.index(next_id)
                else:
                    next = -1
            atom_triplets.append((current_natoms + n, previous, next))
        current_natoms += natoms
    return atom_triplets


def get_aa_frameCloud(atom_coordinates, atom_ids, verbose=True, method='triplet_backbone'):
    if method == 'triplet_backbone':
        get_aa_frameCloud_ = _get_aa_frameCloud_triplet_backbone
    elif method == 'triplet_sidechain':
        get_aa_frameCloud_ = _get_aa_frameCloud_triplet_sidechain
    elif method == 'triplet_cbeta':
        get_aa_frameCloud_ = _get_aa_frameCloud_triplet_cbeta
    elif method == 'quadruplet':
        get_aa_frameCloud_ = _get_aa_frameCloud_quadruplet
    aa_clouds, aa_triplets = get_aa_frameCloud_(List(atom_coordinates), List(atom_ids), verbose=verbose)
    aa_indices = np.arange(len(atom_coordinates)).astype(np.int32)[:, np.newaxis]
    aa_clouds = np.array(aa_clouds)
    aa_triplets = np.array(aa_triplets, dtype=np.int32)
    return aa_clouds, aa_triplets, aa_indices


@njit(cache=True, parallel=False)
def _get_aa_frameCloud_triplet_backbone(atom_coordinates, atom_ids, verbose=True):
    L = len(atom_coordinates)
    aa_clouds = List()
    aa_triplets = List()

    for l in range(L):
        atom_coordinate = atom_coordinates[l]
        atom_id = atom_ids[l]
        natoms = len(atom_id)
        if 1 in atom_id:
            calpha_coordinate = atom_coordinate[atom_id.index(1)]
        else:
            if verbose:
                print('Warning, pathological amino acid missing calpha', l)
            calpha_coordinate = atom_coordinate[0]
        aa_clouds.append(calpha_coordinate)

    # Add virtual calpha at beginning and at the end.
    aa_clouds.append(aa_clouds[0] + (aa_clouds[1] - aa_clouds[2]))
    aa_clouds.append(aa_clouds[L - 1] + (aa_clouds[L - 2] - aa_clouds[L - 3]))

    for l in range(L):
        center = l
        if l == 0:
            previous = L
        else:
            previous = l - 1
        if l == L - 1:
            next = L + 1
        else:
            next = l + 1
        aa_triplets.append((center, previous, next))
    return aa_clouds, aa_triplets


@njit(cache=True, parallel=False)
def _get_aa_frameCloud_triplet_sidechain(atom_coordinates, atom_ids, verbose=True):
    L = len(atom_coordinates)
    aa_clouds = List()
    aa_triplets = List()
    count = 0
    for l in range(L):
        atom_coordinate = atom_coordinates[l]
        atom_id = atom_ids[l]
        natoms = len(atom_id)
        if 1 in atom_id:
            calpha_coordinate = atom_coordinate[atom_id.index(1)]
        else:
            if verbose:
                print('Warning, pathological amino acid missing calpha', l)
            calpha_coordinate = atom_coordinate[0]

        center = 1 * count
        aa_clouds.append(calpha_coordinate)
        count += 1
        if count > 1:
            previous = aa_triplets[-1][0]
        else:
            # Need to place another virtual Calpha.
            virtual_calpha_coordinate = 2 * calpha_coordinate - atom_coordinates[1][0]
            aa_clouds.append(virtual_calpha_coordinate)
            previous = 1 * count
            count += 1

        sidechain_CoM = np.zeros(3, dtype=np.float32)
        sidechain_mass = 0.
        for n in range(natoms):
            if not atom_id[n] in [0, 1, 17, 26, 34]:
                mass = atom_type_mass[atom_id[n]]
                sidechain_CoM += mass * atom_coordinate[n]
                sidechain_mass += mass
        if sidechain_mass > 0:
            sidechain_CoM /= sidechain_mass
        else:  # Usually case of Glycin
            #'''
            #TO CHANGE FOR NEXT NETWORK ITERATION... I used the wrong nitrogen when I rewrote the function...
            if l>0:
                if (0 in atom_id) & (1 in atom_id) & (17 in atom_ids[l-1]):  # If C,N,Calpha are here, place virtual CoM
                    sidechain_CoM = 3 * atom_coordinate[atom_id.index(1)] - atom_coordinates[l-1][atom_ids[l-1].index(17)] - \
                                    atom_coordinate[atom_id.index(0)]
                else:
                    if verbose:
                        print('Warning, pathological amino acid missing side chain and backbone', l)
                    sidechain_CoM = atom_coordinate[-1]
            else:
                if verbose:
                    print('Warning, pathological amino acid missing side chain and backbone', l)
                sidechain_CoM = atom_coordinate[-1]
            #'''

            # if (0 in atom_id) & (1 in atom_id) & (17 in atom_id):  # If C,N,Calpha are here, place virtual CoM
            #     sidechain_CoM = 3 * atom_coordinate[atom_id.index(1)] - atom_coordinate[atom_id.index(17)] - \
            #                     atom_coordinate[atom_id.index(0)]
            # else:
            #     if verbose:
            #         print('Warning, pathological amino acid missing side chain and backbone', l)
            #     sidechain_CoM = atom_coordinate[-1]

        aa_clouds.append(sidechain_CoM)
        next = 1 * count
        count += 1
        aa_triplets.append((center, previous, next))
    return aa_clouds, aa_triplets


@njit(cache=True, parallel=False)
def _get_aa_frameCloud_triplet_cbeta(atom_coordinates, atom_ids, verbose=True):
    L = len(atom_coordinates)
    aa_clouds = List()
    aa_triplets = List()
    count = 0
    for l in range(L):
        atom_coordinate = atom_coordinates[l]
        atom_id = atom_ids[l]
        natoms = len(atom_id)
        if 1 in atom_id:
            calpha_coordinate = atom_coordinate[atom_id.index(1)]
        else:
            if verbose:
                print('Warning, pathological amino acid missing calpha', l)
            calpha_coordinate = atom_coordinate[0]

        if 2 in atom_id:
            cbeta_coordinate = atom_coordinate[atom_id.index(2)]
        else:
            if (0 in atom_id) & (1 in atom_id) & (17 in atom_id):  # If C,N,Calpha are here, place virtual CoM
                cbeta_coordinate = 3 * atom_coordinate[atom_id.index(1)] - atom_coordinate[atom_id.index(17)] - \
                                   atom_coordinate[atom_id.index(0)]
            else:
                if verbose:
                    print('Warning, pathological amino acid missing cbeta and backbone', l)
                cbeta_coordinate = atom_coordinate[-1]

        center = 1 * count
        aa_clouds.append(calpha_coordinate)
        count += 1
        if count > 1:
            previous = aa_triplets[-1][0]
        else:
            # Need to place another virtual Calpha.
            virtual_calpha_coordinate = 2 * calpha_coordinate - atom_coordinates[1][0]
            aa_clouds.append(virtual_calpha_coordinate)
            previous = 1 * count
            count += 1

        aa_clouds.append(cbeta_coordinate)
        next = 1 * count
        count += 1
        aa_triplets.append((center, previous, next))
    return aa_clouds, aa_triplets


@njit(cache=True, parallel=False)
def _get_aa_frameCloud_quadruplet(atom_coordinates, atom_ids, verbose=True):
    L = len(atom_coordinates)
    aa_clouds = List()
    aa_triplets = List()

    for l in range(L):
        atom_coordinate = atom_coordinates[l]
        atom_id = atom_ids[l]
        natoms = len(atom_id)
        if 1 in atom_id:
            calpha_coordinate = atom_coordinate[atom_id.index(1)]
        else:
            if verbose:
                print('Warning, pathological amino acid missing calpha', l)
            calpha_coordinate = atom_coordinate[0]
        aa_clouds.append(calpha_coordinate)

    # Add virtual calpha at beginning and at the end.
    aa_clouds.append(aa_clouds[0] + (aa_clouds[1] - aa_clouds[2]))
    aa_clouds.append(aa_clouds[L - 1] + (aa_clouds[L - 2] - aa_clouds[L - 3]))

    count = L + 2

    for l in range(L):
        atom_coordinate = atom_coordinates[l]
        atom_id = atom_ids[l]
        natoms = len(atom_id)

        sidechain_CoM = np.zeros(3, dtype=np.float32)
        sidechain_mass = 0.
        for n in range(natoms):
            if not atom_id[n] in [0, 1, 17, 26, 34]:
                mass = atom_type_mass[atom_id[n]]
                sidechain_CoM += mass * atom_coordinate[n]
                sidechain_mass += mass
        if sidechain_mass > 0:
            sidechain_CoM /= sidechain_mass
        else:  # Usually case of Glycin
            if (0 in atom_id) & (1 in atom_id) & (17 in atom_id):  # If C,N,Calpha are here, place virtual CoM
                sidechain_CoM = 3 * atom_coordinate[atom_id.index(1)] - atom_coordinate[atom_id.index(17)] - \
                                atom_coordinate[atom_id.index(0)]
            else:
                if verbose:
                    print('Warning, pathological amino acid missing side chain and backbone', l)
                sidechain_CoM = atom_coordinate[-1]

        aa_clouds.append(sidechain_CoM)
        center = l
        if l == 0:
            previous = L
        else:
            previous = l - 1
        if l == L - 1:
            next = L + 1
        else:
            next = l + 1
        dipole = L + 2 + l
        aa_triplets.append((center, previous, next, dipole))
    return aa_clouds, aa_triplets


def add_virtual_atoms(atom_clouds, atom_triplets, verbose=True):
    virtual_atom_clouds, atom_triplets = _add_virtual_atoms(atom_clouds, atom_triplets, verbose=verbose)
    if len(virtual_atom_clouds) > 0:
        atom_clouds = np.concatenate([atom_clouds, np.array(virtual_atom_clouds)], axis=0)
    return atom_clouds, atom_triplets


@njit(cache=False)
def _add_virtual_atoms(atom_clouds, atom_triplets, verbose=True):
    natoms = len(atom_triplets)
    virtual_atom_clouds = List()
    count_virtual_atoms = 0
    centers = list(atom_triplets[:, 0])
    for n in range(natoms):
        triplet = atom_triplets[n]
        case1 = (triplet[1] >= 0) & (triplet[2] >= 0)
        case2 = (triplet[1] < 0) & (triplet[2] >= 0)
        case3 = (triplet[1] >= 0) & (triplet[2] < 0)
        case4 = (triplet[1] < 0) & (triplet[2] < 0)
        if case1:  # Atom has at least two covalent bonds.
            continue
        elif case2:  # Atom has one covalent bond. Previous is missing, next is present (Either N-terminal N or missing atom).
            next_triplet = atom_triplets[centers.index(triplet[2])]
            if next_triplet[2] >= 0:  # Next of next is present. Build virtual atom to obtain parallelogram.
                virtual_atom = atom_clouds[next_triplet[0]] - atom_clouds[next_triplet[2]] + atom_clouds[triplet[0]]
            else:  # Next of next is also absent. Pathological case, use absolute x direction...
                if verbose:
                    print('Pathological case, atom has only one bond and its next partner too', triplet[0], triplet[2])
                    # print('Pathological case, atom %s has only one bond and its next partner %s too'%(triplet[0],triplet[2]))
                virtual_atom = atom_clouds[triplet[0]] + np.array([1, 0, 0])
            virtual_atom_clouds.append(virtual_atom)
            triplet[1] = natoms + count_virtual_atoms
            count_virtual_atoms += 1


        elif case3:  # Atom has one covalent bond. Next is missing, previous is present (Either C-terminal C or missing atom).
            previous_triplet = atom_triplets[centers.index(triplet[1])]
            if previous_triplet[1] >= 0:  # Previous of previous is present. Build virtual atom to obtain parallelogram.
                virtual_atom = atom_clouds[previous_triplet[0]] - atom_clouds[previous_triplet[1]] + atom_clouds[
                    triplet[0]]
            else:  # Previous of previous is also absent. Pathological case, use absolute z direction...
                if verbose:
                    print('Pathological case, atom has only one bond and its previous partner too', triplet[0],
                          triplet[1])
                    # print('Pathological case, atom %s has only one bond and its previous partner %s too'%(triplet[0],triplet[1]))
                virtual_atom = atom_clouds[triplet[0]] + np.array([0, 0, 1])
            virtual_atom_clouds.append(virtual_atom)
            triplet[2] = natoms + count_virtual_atoms
            count_virtual_atoms += 1

        elif case4:  # Atom has no covalent bonds. Should never happen, use absolute coordinates.
            if verbose:
                print('Pathological case, atom has no bonds at all', triplet[0])
                # print('Pathological case, atom %s has no bonds at all' %triplet[0])
            virtual_previous_atom = atom_clouds[triplet[0]] + np.array([1, 0, 0])
            virtual_next_atom = atom_clouds[triplet[0]] + np.array([0, 0, 1])
            virtual_atom_clouds.append(virtual_previous_atom)
            virtual_atom_clouds.append(virtual_next_atom)
            triplet[1] = natoms + count_virtual_atoms
            triplet[2] = natoms + count_virtual_atoms + 1
            count_virtual_atoms += 2
    return virtual_atom_clouds, atom_triplets


if __name__ == '__main__':
    import PDB_processing
    import Bio.PDB

    PDB_folder = '/Users/jerometubiana/PDB/'
    pdblist = Bio.PDB.PDBList()
    # pdb = '1a3x'
    pdb = '2kho'
    chain = 'A'
    name = pdblist.retrieve_pdb_file(pdb, pdir=PDB_folder)
    struct, chains = PDBio.load_chains(pdb_id=pdb, chain_ids=[(0, chain)], file=PDB_folder + '%s.cif' % pdb)
    sequence, backbone_coordinates, atom_coordinates, atom_ids, atom_types = PDB_processing.process_chain(chains)

    atom_clouds, atom_triplets, atom_attributes, atom_indices = get_atom_frameCloud(sequence, atom_coordinates,
                                                                                    atom_ids)
    for i in range(20):
        tmp = atom_triplets[i, :]
        center = list_atoms[atom_attributes[tmp[0]]]
        if tmp[1] >= 0:
            previous = list_atoms[atom_attributes[tmp[1]]]
        else:
            previous = 'NONE'
        if tmp[2] >= 0:
            next = list_atoms[atom_attributes[tmp[2]]]
        else:
            next = 'NONE'

        print(i, center, previous, next)

    atom_clouds_filled, atom_triplets_filled = add_virtual_atoms(atom_clouds, atom_triplets, verbose=True)
    aa_clouds, aa_triplets, aa_indices = get_aa_frameCloud(atom_coordinates, atom_ids, verbose=True)
    aa_attributes = np.array([aa_to_index[aa] for aa in sequence], dtype=np.int32)

    inputs2network = [
        aa_triplets,
        aa_indices,
        aa_clouds,
        aa_attributes,
        atom_triplets_filled,
        atom_indices,
        atom_clouds_filled,
    ]

    for input in inputs2network:
        print(input.shape, input.dtype)


