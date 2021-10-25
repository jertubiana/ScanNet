import numpy as np
from numba import types
from numba.typed import List,Dict








list_atoms_types = ['C', 'O', 'N', 'S']  # H
VanDerWaalsRadii = np.array([1.70, 1.52, 1.55, 1.80])  # 1.20

atom_mass = np.array(
    [
        12,  # C
        16,  # O
        14,  # N
        32  # S
    ]
)


atom_type_to_index = dict([(list_atoms_types[i], i)
                           for i in range(len(list_atoms_types))])


list_atoms = ['C', 'CA', 'CB', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'CE3',
              'CG', 'CG1', 'CG2', 'CH2', 'CZ', 'CZ2', 'CZ3', 'N', 'ND1', 'ND2',
              'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'NZ', 'O', 'OD1', 'OD2', 'OE1',
              'OE2', 'OG', 'OG1', 'OH', 'OXT', 'SD', 'SE', 'SG']

atom_to_index = dict([(list_atoms[i], i) for i in range(len(list_atoms))])
atom_to_index['OT1'] = atom_to_index['O']
atom_to_index['OT2'] = atom_to_index['OXT']

index_to_type = np.zeros(38,dtype=np.int)
for atom,index in atom_to_index.items():
    index_to_type[index] = list_atoms_types.index(atom[0])

atom_type_mass = np.zeros( 38 )
for atom,index in atom_to_index.items():
    atom_type_mass[index] = atom_mass[index_to_type[index]]

'''
No exotic amino acids supported yet.
'''

list_aa = [
    'A',
    'C',
    'D',
    'E',
    'F',
    'G',
    'H',
    'I',
    'K',
    'L',
    'M',
    'N',
    'P',
    'Q',
    'R',
    'S',
    'T',
    'V',
    'W',
    'Y'

]

residue_dictionary = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
                      'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
                      'GLY': 'G', 'HIS': 'H', 'HSD':'H','HSE':'H',
                      'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
                      'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M',
                      'MSE': 'M',
                      'PTR':'Y',
                      'TYS':'Y',
                      'SEP':'S',
                      'TPO':'T',
                      'HIP':'H',
}

hetresidue_field = [' '] + ['H_%s'%name for name in residue_dictionary.keys()]

aa_to_index = dict([(list_aa[i],i) for i in range(20)])

'''
!!!! Non-exhaustive, only included the first two bonds.
'''

dictionary_covalent_bonds = {
    'A':{
        'C': ['CA', 'N'],
        'CA': ['N', 'C'],
        'N': ['C', 'CA'],
        'O': ['C', None],
        'OXT':['C',None],
        'CB': ['CA', None]
    },
    'C':{
        'C': ['CA', 'N'],
        'CA': ['N', 'C'],
        'N': ['C', 'CA'],
        'O': ['C', None],
        'OXT': ['C', None],
        'CB': ['CA', 'SG'],
        'SG': ['CB', None]
    },
    'D':{
        'C': ['CA', 'N'],
        'CA': ['N', 'C'],
        'N': ['C', 'CA'],
        'O': ['C', None],
        'OXT': ['C', None],
        'CB': ['CA', 'CG'],
        'CG': ['CB', 'OD1'],
        'OD1': ['CG', None],
        'OD2': ['CG', None],
    },
    'E':{
        'C': ['CA', 'N'],
        'CA': ['N', 'C'],
        'N': ['C', 'CA'],
        'O': ['C', None],
        'OXT': ['C', None],
        'CB': ['CA', 'CG'],
        'CG': ['CB', 'CD'],
        'CD': ['CG', 'OE1'],
        'OE1': ['CD', None],
        'OE2': ['CD', None]
    },
    'F':{
        'C': ['CA', 'N'],
        'CA': ['N', 'C'],
        'N': ['C', 'CA'],
        'O': ['C', None],
        'OXT': ['C', None],
        'CB': ['CA', 'CG'],
        'CG': ['CB', 'CD1'],
        'CD1': ['CG', 'CE1'],
        'CE1': ['CD1', 'CZ'],
        'CZ': ['CE1', 'CE2'],
        'CE2': ['CD2', 'CZ'],
        'CD2': ['CG', 'CE2']
    },
    'G':{
        'C': ['CA', 'N'],
        'CA': ['N', 'C'],
        'N': ['C', 'CA'],
        'O': ['C', None],
        'OXT': ['C', None],
    },
    'H':{
        'C': ['CA', 'N'],
        'CA': ['N', 'C'],
        'N': ['C', 'CA'],
        'O': ['C', None],
        'OXT': ['C', None],
        'CB': ['CA', 'CG'],
        'CG': ['CB', 'ND1'],
        'ND1': ['CG', 'CE1'],
        'CE1': ['ND1', 'NE2'],
        'NE2': ['CE1', 'CD2'],
        'CD2': ['NE2', 'CG']
    },
    'I':{
        'C': ['CA', 'N'],
        'CA': ['N', 'C'],
        'N': ['C', 'CA'],
        'O': ['C', None],
        'OXT': ['C', None],
        'CB': ['CA', 'CG1'],
        'CG1': ['CB', 'CD1'],
        'CG2': ['CB', None],
        'CD1':['CG1',None]
    },
    'K':{
        'C': ['CA', 'N'],
        'CA': ['N', 'C'],
        'N': ['C', 'CA'],
        'O': ['C', None],
        'OXT': ['C', None],
        'CB': ['CA', 'CG'],
        'CG': ['CB', 'CD'],
        'CD': ['CG', 'CE'],
        'CE': ['CD', 'NZ'],
        'NZ': ['CE', None],
    },
    'L':{
        'C': ['CA', 'N'],
        'CA': ['N', 'C'],
        'N': ['C', 'CA'],
        'O': ['C', None],
        'OXT': ['C', None],
        'CB': ['CA', 'CG'],
        'CG': ['CB', 'CD1'],
        'CD1': ['CG', None],
        'CD2': ['CG', None]
    },
    'M':{
        'C': ['CA', 'N'],
        'CA': ['N', 'C'],
        'N': ['C', 'CA'],
        'O': ['C', None],
        'OXT': ['C', None],
        'CB': ['CA', 'CG'],
        'CG': ['CB', 'SD'],
        'SD': ['CG', 'CE'],
        'CE':['SD',None]
    },
    'N':{
        'C': ['CA', 'N'],
        'CA': ['N', 'C'],
        'N': ['C', 'CA'],
        'O': ['C', None],
        'OXT': ['C', None],
        'CB': ['CA', 'CG'],
        'CG': ['CB', 'OD1'],
        'OD1': ['CG', None],
        'ND2': ['CG', None]
    },
    'P':{
        'C': ['CA', 'N'],
        'CA': ['N', 'C'],
        'N': ['C', 'CA'],
        'O': ['C', None],
        'OXT': ['C', None],
        'CB': ['CA', 'CG'],
        'CG': ['CB', 'CD'],
        'CD': ['CG', 'N']
    },
    'Q':{
        'C': ['CA', 'N'],
        'CA': ['N', 'C'],
        'N': ['C', 'CA'],
        'O': ['C', None],
        'OXT': ['C', None],
        'CB': ['CA', 'CG'],
        'CG': ['CB', 'CD'],
        'CD': ['CG', 'OE1'],
        'OE1': ['CD', None],
        'NE2': ['CD', None]
    },
    'R':{
        'C': ['CA', 'N'],
        'CA': ['N', 'C'],
        'N': ['C', 'CA'],
        'O': ['C', None],
        'OXT': ['C', None],
        'CB': ['CA', 'CG'],
        'CG': ['CB', 'CD'],
        'CD': ['CG', 'NE'],
        'NE': ['CD', 'CZ'],
        'CZ': ['NE', 'NH1'],
        'NH1': ['CZ', None],
        'NH2': ['CZ', None]
    },
    'S':{
        'C': ['CA', 'N'],
        'CA': ['N', 'C'],
        'N': ['C', 'CA'],
        'O': ['C', None],
        'OXT': ['C', None],
        'CB': ['CA', 'OG'],
        'OG': ['CB', None],
    },
    'T':{
        'C': ['CA', 'N'],
        'CA': ['N', 'C'],
        'N': ['C', 'CA'],
        'O': ['C', None],
        'OXT': ['C', None],
        'CB': ['CA', 'OG1'],
        'OG1': ['CB', None],
        'CG2': ['CB', None]
    },
    'V':{
        'C': ['CA', 'N'],
        'CA': ['N', 'C'],
        'N': ['C', 'CA'],
        'O': ['C', None],
        'OXT': ['C', None],
        'CB': ['CA', 'CG1'],
        'CG1': ['CB', None],
        'CG2': ['CB', None],
    },
    'W':{
        'C': ['CA', 'N'],
        'CA': ['N', 'C'],
        'N': ['C', 'CA'],
        'O': ['C', None],
        'OXT': ['C', None],
        'CB': ['CA', 'CG'],
        'CG': ['CB', 'CD1'],
        'CD1': ['CG', 'NE1'],
        'NE1': ['CD1', 'CE2'],
        'CD2': ['CG', 'CE2'],
        'CE2': ['CD2', 'CZ2'],
        'CZ2': ['CE2', 'CH2'],
        'CH2': ['CZ2', 'CZ3'],
        'CZ3': ['CH2', 'CE3'],
        'CE3': ['CZ3', 'CD2']
    },
    'Y':{
        'C': ['CA', 'N'],
        'CA': ['N', 'C'],
        'N': ['C', 'CA'],
        'O': ['C', None],
        'OXT': ['C', None],
        'CB': ['CA', 'CG'],
        'CG': ['CB', 'CD1'],
        'CD1': ['CG', 'CE1'],
        'CE1': ['CD1', 'CZ'],
        'CZ': ['CE1', 'CE2'],
        'CE2': ['CD2', 'CZ'],
        'CD2': ['CG', 'CE2'],
        'OH': ['CZ', None]
    }
}


dictionary_covalent_bonds_numba = Dict.empty(
    key_type=types.unicode_type,
    value_type=types.int32[:],
)

for aa,atom_covalent_bonds in dictionary_covalent_bonds.items():
    for atom,bonds in atom_covalent_bonds.items():
        bonds_num = -1 * np.ones(3,dtype=np.int32)
        for l,bond in enumerate(bonds):
            if bond is not None:
                bonds_num[l] = atom_to_index[bond]
        dictionary_covalent_bonds_numba['%s_%s'%(aa, atom_to_index[atom] )] = bonds_num



list_atom_valencies = [
    'C',
    'CH',
    'CH2',
    'CH3',
    'CPi',
    'O',
    'OH',
    'N',
    'NH',
    'NH2',
    'S',
    'SH'
]

dictionary_atom_valencies = {
    'A': {
        'C': 'C',
        'CA': 'CH',
        'CB': 'CH3',
        'O': 'O',
        'OXT': 'OH',
        'N': 'NH'
    },
    'C': {
        'C': 'C',
        'CA': 'CH',
        'CB': 'CH2',
        'O': 'O',
        'OXT': 'OH',
        'N': 'NH',
        'SG': 'SH'
    },
    'D': {
        'C': 'C',
        'CA': 'CH',
        'CB': 'CH2',
        'CG': 'C',
        'O': 'O',
        'OD1': 'O',
        'OD2': 'OH',
        'OXT': 'OH',
        'N': 'NH',
    },
    'E': {
        'C': 'C',
        'CA': 'CH',
        'CB': 'CH2',
        'CG': 'CH',
        'CD': 'C',
        'O': 'O',
        'OE1': 'O',
        'OE2': 'OH',
        'OXT': 'OH',
        'N': 'NH',
    },
    'F': {
        'C': 'C',
        'CA': 'CH',
        'CB': 'CH2',
        'CG': 'CPi',
        'CD1': 'CPi',
        'CD2': 'CPi',
        'CE1': 'CPi',
        'CE2': 'CPi',
        'CZ': 'CPi',
        'N': 'NH',
        'O': 'O',
        'OXT': 'OH',
    },
    'G': {
        'C': 'C',
        'CA': 'CH2',
        'N': 'NH',
        'O': 'O',
        'OXT': 'OH'
    },
    'H': {
        'C': 'C',
        'CA': 'CH',
        'CB': 'CH2',
        'CG': 'CPi',
        'CE1': 'CPi',
        'CD2': 'CPi',
        'N': 'NH',
        'ND1': 'N',
        'ND2': 'NH2', # Supprimer cette ligne pour la prochaine mise a jour.
#        'NE2': 'NH2', # Mettre cette ligne pour la prochaine mise a jour.
        'O': 'O',
        'OXT': 'OH',
    },
    'I': {
        'C': 'C',
        'CA': 'CH',
        'CB': 'CH',
        'CG1': 'CH2',
        'CG2': 'CH3',
        'CD1': 'CH3',
        'N': 'NH',
        'O': 'O',
        'OXT': 'OH'
    },
    'K': {
        'C': 'C',
        'CA': 'CH',
        'N': 'NH',
        'O': 'O',
        'OXT': 'OH',
        'CB': 'CH2',
        'CG': 'CH2',
        'CD': 'CH2',
        'CE': 'CH2',
        'NZ': 'NH2'
    },
    'L': {
        'C': 'C',
        'CA': 'CH',
        'N': 'NH',
        'O': 'O',
        'OXT': 'OH',
        'CB': 'CH2',
        'CG': 'CH',
        'CD1': 'CH3',
        'CD2': 'CH3',
    },
    'M': {
        'C': 'C',
        'CA': 'CH',
        'N': 'NH',
        'O': 'O',
        'OXT': 'OH',
        'CB': 'CH2',
        'CG': 'CH2',
        'SD': 'S',
        'CE': 'CH3'
    },
    'N': {
        'C': 'C',
        'CA': 'CH',
        'N': 'NH',
        'O': 'O',
        'OXT': 'OH',
        'CB': 'CH2',
        'CG': 'C',
        'OD1': 'O',
        'ND2': 'NH2'
    },
    'P': {
        'C': 'C',
        'CA': 'CH',
        'N': 'N',
        'O': 'O',
        'OXT': 'OH',
        'CB': 'CH2',
        'CG': 'CH2',
        'CD': 'CH2'
    },
    'Q': {
        'C': 'C',
        'CA': 'CH',
        'N': 'NH',
        'O': 'O',
        'OXT': 'OH',
        'CB': 'CH2',
        'CG': 'CH2',
        'CD': 'C',
        'OE1': 'O',
        'NE2': 'NH2'
    },
    'R': {
        'C': 'C',
        'CA': 'CH',
        'N': 'NH',
        'O': 'O',
        'OXT': 'OH',
        'CB': 'CH2',
        'CG': 'CH2',
        'CD': 'CH2',
        'NE': 'NH',
        'CZ': 'C',
        'NH1': 'NH',
        'NH2': 'NH2'
    },
    'S': {
        'C': 'C',
        'CA': 'CH',
        'N': 'NH',
        'O': 'O',
        'OXT': 'OH',
        'CB': 'CH2',
        'OG': 'OH',
    },
    'T': {
        'C': 'C',
        'CA': 'CH',
        'N': 'NH',
        'O': 'O',
        'OXT': 'OH',
        'CB': 'CH',
        'OG1': 'OH',
        'CG2': 'CH3'
    },
    'V': {
        'C': 'C',
        'CA': 'CH',
        'N': 'NH',
        'O': 'O',
        'OXT': 'OH',
        'CB': 'CH',
        'CG1': 'CH3',
        'CG2': 'CH3',
    },
    'W': {
        'C': 'C',
        'CA': 'CH',
        'N': 'NH',
        'O': 'O',
        'OXT': 'OH',
        'CB': 'CH2',
        'CG': 'CPi',
        'CD1': 'CPi',
        'NE1': 'NH',
        'CD2': 'CPi',
        'CE2': 'CPi',
        'CZ2': 'CPi',
        'CH2': 'CPi',
        'CZ3': 'CPi',
        'CE3': 'CPi'
    },
    'Y': {
        'C': 'C',
        'CA': 'CH',
        'N': 'NH',
        'O': 'O',
        'OXT': 'OH',
        'CB': 'CH2',
        'CG': 'CPi',
        'CD1': 'CPi',
        'CE1': 'CPi',
        'CZ': 'CPi',
        'CE2': 'CPi',
        'CD2': 'CPi',
        'OH': 'OH'
    }

}


index_to_valency = np.zeros([20, 38], dtype=np.int)
for k, aa in enumerate(list_aa):
    for key, value in dictionary_atom_valencies[aa].items():
        i = list_atoms.index(key)
        j = list_atom_valencies.index(value)
        index_to_valency[k, i] = j
