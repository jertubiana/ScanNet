import os
import Bio.PDB
import warnings
from utilities.paths import structures_folder

from urllib.request import urlretrieve
from urllib.request import urlcleanup
import gzip



def is_PDB_identifier(str):
    return ( (len(str) == 4) & str.isalnum() )

def is_UniProt_identifier(str):
    L = len(str)
    correct_length = L in [6,10]
    if not correct_length:
        return False
    only_alnum = str.isalnum()
    only_upper = (str.upper() == str)
    first_is_letter = str[0].isalpha()
    six_is_digit = str[5].isnumeric()

    valid_uniprot_id = correct_length & only_alnum & only_upper & first_is_letter & six_is_digit
    if L == 10:
        seven_is_letter = str[6].isalpha()
        last_is_digit = str[1].isnumeric()
        valid_uniprot_id = valid_uniprot_id & seven_is_letter & last_is_digit
    return valid_uniprot_id



#%% Function for downloading biounit files.

class myPDBList(Bio.PDB.PDBList):
    PDB_REF = """
    The Protein Data Bank: a computer-based archival file for macromolecular structures.
    F.C.Bernstein, T.F.Koetzle, G.J.B.Williams, E.F.Meyer Jr, M.D.Brice, J.R.Rodgers, O.Kennard, T.Shimanouchi, M.Tasumi
    J. Mol. Biol. 112 pp. 535-542 (1977)
    http://www.pdb.org/.
    """
    def __init__(self,*args, **kwargs):
        kwargs['pdb'] = structures_folder
        super().__init__(*args,**kwargs)
        self.alphafold_server = 'https://alphafold.ebi.ac.uk/' # entry/Q13469
        # self.pdb_server = 'ftp://ftp.ebi.ac.uk/pub/databases/pdb/'
        self.flat_tree = True
        return

    def retrieve_pdb_file(self, code, obsolete=False, pdir=None, file_format=None, overwrite=False):
        """Fetch PDB structure file from PDB server, and store it locally.

        The PDB structure's file name is returned as a single string.
        If obsolete ``==`` True, the file will be saved in a special file tree.

        NOTE. The default download format has changed from PDB to PDBx/mmCif

        :param code: pdb or uniprot ID
         PDB code: 4-symbols structure Id from PDB (e.g. 3J92).
         Uniprot ID: 6 or 10 symbols (e.g. Q8WZ42).

        :type code: string

        :param file_format:
            File format. Available options:

            * "mmCif" (default, PDBx/mmCif file),
            * "pdb" (format PDB),
            * "xml" (PDBML/XML format),
            * "mmtf" (highly compressed),
            * "bundle" (PDB formatted archive for large structure}
            * 'biounit' (format PDB)

        :type file_format: string

        :param overwrite: if set to True, existing structure files will be overwritten. Default: False
        :type overwrite: bool

        :param obsolete:
            Has a meaning only for obsolete structures. If True, download the obsolete structure
            to 'obsolete' folder, otherwise download won't be performed.
            This option doesn't work for mmtf format as obsoleted structures aren't stored in mmtf.
            Also doesn't have meaning when parameter pdir is specified.
            Note: make sure that you are about to download the really obsolete structure.
            Trying to download non-obsolete structure into obsolete folder will not work
            and you face the "structure doesn't exists" error.
            Default: False

        :type obsolete: bool

        :param pdir: put the file in this directory (default: create a PDB-style directory tree)
        :type pdir: string

        :return: filename
        :rtype: string
        """
        file_format = self._print_default_format_warning(
            file_format)  # Deprecation warning

        is_pdb = is_PDB_identifier(code)
        is_uniprot = is_UniProt_identifier(code)
        if not (is_pdb | is_uniprot):
            raise ValueError('Identifier %s is neither a valid PDB or Uniprot ID')

        if is_pdb:
            code = code.lower()


        if is_pdb:
            # Get the compressed PDB structure
            archive = {'pdb': 'pdb%s.ent.gz', 'mmCif': '%s.cif.gz', 'xml': '%s.xml.gz', 'mmtf': '%s',
                       'bundle': '%s-pdb-bundle.tar.gz', 'biounit': '%s.pdb1.gz', 'biounit_mmCif': '%s-assembly1.cif.gz'}
            archive_fn = archive[file_format] % code

            if file_format not in archive.keys():
                raise("Specified file_format %s doesn't exists or is not supported. Maybe a typo. "
                      "Please, use one of the following: mmCif, pdb, xml, mmtf, bundle, biounit" % file_format)

            if file_format in ('pdb', 'mmCif', 'xml'):
                pdb_dir = "divided" if not obsolete else "obsolete"
                file_type = "pdb" if file_format == "pdb" else "mmCIF" if file_format == "mmCif" else "XML"
                url = (self.pdb_server + '/pub/pdb/data/structures/%s/%s/%s/%s' %
                       (pdb_dir, file_type, code[1:3], archive_fn))
            elif file_format == 'bundle':
                url = (self.pdb_server + '/pub/pdb/compatible/pdb_bundle/%s/%s/%s' %
                       (code[1:3], code, archive_fn))
            elif file_format == 'biounit':
                url = (self.pdb_server + '/pub/pdb/data/biounit/PDB/divided/%s/%s' %
                       (code[1:3], archive_fn))
            elif file_format == 'biounit_mmCif':
                url = (self.pdb_server + '/pub/pdb/data/biounit/mmCIF/divided/%s/%s' %
                       (code[1:3], archive_fn))
            else:
                url = ('http://mmtf.rcsb.org/v1.0/full/%s' % code)

        elif is_uniprot:
            assert file_format in ['pdb','mmCif']
            url = self.alphafold_server + '/files/AF-%s-F1-model_v1%s'%(code, '.pdb' if file_format == 'pdb' else '.cif')
            archive_fn = url.split('/')[-1]
        else:
            return


        # Where does the final PDB file get saved?
        if pdir is None:
            path = self.local_pdb if not obsolete else self.obsolete_pdb
            if not self.flat_tree:  # Put in PDB-style directory tree
                path = os.path.join(path, code[1:3])
        else:  # Put in specified directory
            path = pdir
        if not os.access(path, os.F_OK):
            os.makedirs(path)
        filename = os.path.join(path, archive_fn)
        if is_pdb:
            final = {'pdb': 'pdb%s.ent', 'mmCif': '%s.cif', 'xml': '%s.xml',
                     'mmtf': '%s.mmtf', 'bundle': '%s-pdb-bundle.tar', 'biounit': 'pdb%s.bioent', 'biounit_mmCif': '%s_bioentry.cif'}
        elif is_uniprot:
            final = {'pdb':'AF_%s.pdb','mmCif':'AF_%s.cif'}
        else:
            return
        final_file = os.path.join(path, final[file_format] % code)

        # Skip download if the file already exists
        if not overwrite:
            if os.path.exists(final_file):
                if self._verbose:
                    print("Structure exists: '%s' " % final_file)
                return final_file

        # Retrieve the file
        if self._verbose:
            print("Downloading PDB structure '%s'..." % code)
        try:
            urlcleanup()
            urlretrieve(url, filename)
        except IOError:
            print("Desired structure doesn't exists")
            return
        else:
            if is_pdb:
                with gzip.open(filename, 'rb') as gz:
                    with open(final_file, 'wb') as out:
                        out.writelines(gz)
                os.remove(filename)
            else:
                os.rename(filename,final_file)
        return final_file



class ChainSelect(Bio.PDB.Select):
    def __init__(self,selected_chains,*args,**kwargs):
        self.selected_chains = selected_chains
        return super().__init__(*args,**kwargs)
    def accept_model(self,model):
        if self.selected_chains in ['upper','lower','all']:
            return 1
        elif model.id in [x[0] for x in self.selected_chains]:
            return 1
        else:
            return 0
    def accept_chain(self, chain):
        if self.selected_chains == 'all':
            return 1
        elif self.selected_chains == 'upper':
            return int( (chain.get_full_id()[2].isupper() |  (chain.get_full_id()[2]==' ') ) )
        elif self.selected_chains == 'lower':
            return int(chain.get_full_id()[2].islower())
        elif (chain.get_full_id()[1],chain.get_full_id()[2]) in self.selected_chains:
            return 1
        else:
            return 0


def parse_str(str):
    str_split = str.split('_')
    if len(str_split) == 1:
        structure_identifier = str
        chains = None
    else:
        if '.' in str_split[-1]: # Special case, str is a path to a file that includes an underscore. Assume file has extension.
            structure_identifier = str
            chains = None
        else:
            structure_identifier = '_'.join(str_split[:-1])
            chains = str_split[-1]
    if chains is not None:
        if ('+' in chains) | ('-' in chains):
            chain_identifiers = chains.split('+')
            chain_identifiers = [(int(x.split('-')[0]),x.split('-')[1]) if '-' in x else (0,x) for x in  chain_identifiers]
        else:
            chain_identifiers = [(0,x) for x in chains]
    else:
        chain_identifiers = 'all'
    return structure_identifier,chain_identifiers


def format_chain_id(x):
    return '+'.join([ '%s-%s'%(y[0],y[1]) for y in x])



def getPDB(identifier_string,biounit=True,structures_folder=structures_folder,verbose=True):
    structure_id,chain = parse_str(identifier_string)
    is_pdb = is_PDB_identifier(structure_id)
    is_uniprot = is_UniProt_identifier(structure_id)
    if is_uniprot:
        biounit=False

    if not (is_pdb | is_uniprot):
        location = structure_id
        assert os.path.exists(location),'File not found'
    else:
        if is_pdb:
            pdb_id = structure_id.lower()
            if biounit:
                location1 = structures_folder + 'pdb' + pdb_id + '.bioent'
                location2 = structures_folder + pdb_id + '_bioentry.cif'
            else:
                location1 = structures_folder + 'pdb' + pdb_id + '.ent'
                location2 = structures_folder + pdb_id + '.cif'  # New§ format
        else:
            uniprot_id = structure_id
            location1 = structures_folder + 'AF_' + uniprot_id + '.pdb'
            location2 = structures_folder + 'AF_' + uniprot_id + '.cif'

        if os.path.exists(location1):
            location = location1
        elif os.path.exists(location2):
            location = location2
        else:
            pdb_downloader = myPDBList(verbose=verbose)
            if biounit:
                location = pdb_downloader.retrieve_pdb_file(
                    structure_id, pdir=structures_folder, file_format='biounit')
                if location is None:
                    location = pdb_downloader.retrieve_pdb_file(
                        structure_id, pdir=structures_folder, file_format='biounit_mmCif')
            else:
                location = pdb_downloader.retrieve_pdb_file(
                    structure_id, pdir=structures_folder, file_format='pdb')
                if location is None:
                    location = pdb_downloader.retrieve_pdb_file(
                        structure_id, pdir=structures_folder, file_format='mmCif')
    return location,chain


def extract_chains(location,chains,final_location):
    if chains == 'all':
        os.system('scp %s %s'%(location,final_location))
    else:
        with warnings.catch_warnings(record=True) as w:
            if location[-4:] == '.cif':
                parser = Bio.PDB.MMCIFParser()
            else:
                parser = Bio.PDB.PDBParser()
            struct = parser.get_structure('name',location)
            io = Bio.PDB.PDBIO()
            if isinstance(chains,list) & len(chains)==1:
                model,chain = chains[0]
                chain_obj = struct[model][chain]
                if len(chain) > 1:
                    chain_obj.id = chain[0]
                io.set_structure(chain_obj)
                for atom in Bio.PDB.Selection.unfold_entities(io.structure, 'A'):
                    atom.disordered_flag = 0
                io.save(final_location)
            else:
                io.set_structure(struct)
                io.save(final_location, ChainSelect(chains))
    return final_location





def load_chains(pdb_id=None,
                         chain_ids='all',
                         file=None,
                         pdbparser=None, mmcifparser=None,
                         structures_folder=structures_folder,
                         dockground_indexing=False, biounit=True,verbose=True):
    if pdbparser is None:
        pdbparser = Bio.PDB.PDBParser()  # PDB parser; to read pdb files.
    if mmcifparser is None:
        mmcifparser = Bio.PDB.MMCIFParser()

    assert (file is not None) | (pdb_id is not None)

    if (file is None) & (pdb_id is not None):
        file = getPDB(pdb_id, biounit=biounit, structures_folder=structures_folder)[0]
    else:
        pdb_id = 'abcd'

    if file[-4:] == '.cif':
        parser = mmcifparser
    else:
        parser = pdbparser
    if verbose:
        print('Parsing %s'%file)
    with warnings.catch_warnings(record=True) as w:
        structure = parser.get_structure(pdb_id,  file)

    chain_objs = []
    if chain_ids in ['all','lower','upper']:
        for model_obj in structure:
            for chain_obj in model_obj:
                condition1 = (chain_ids == 'all')
                condition2 = ( (chain_ids == 'lower') & chain_obj.id.islower() )
                condition3 = ( (chain_ids == 'upper') & (chain_obj.id.isupper() | (chain_obj.id == ' ') ) )
                if condition1 | condition2 | condition3:
                    chain_objs.append(chain_obj)
    else:
        for model, chain in chain_ids:
            if dockground_indexing & (model > 0):
                model_ = model - 1
            else:
                model_ = model

            chain_obj = structure[model_][chain]
            chain_objs.append(chain_obj)
    return structure, chain_objs
