# List of paths to folders and binaries. All folder paths should finish with slash (/)

# Paths required for prediction.
library_folder = '/path/to/ScanNet/' # Where the Github Repo is located.
structures_folder = '/path/to/PDB/' # Where pdb/mmCIF structures files are stored.
predictions_folder = library_folder + 'predictions/' # Output folder.
model_folder = library_folder + 'models/' # Where the networks as stored as pairs of files (.h5,.data).

# Additional paths required for prediction with evolutionary information.
MSA_folder = library_folder + 'MSA/' # Where multiple sequence alignments are stored.
path2hhblits = None # Path to hhblits binary. Not required if using ScanNet_noMSA networks.
path2sequence_database = None # Path to sequence database Not required if using ScanNet_noMSA networks.

# path2hhblits = '/path/to/hh-suite/build/bin/hhblits'  # Path to hhblits binary. Not required if using ScanNet_noMSA networks.
# path2sequence_database = '/path/to/uniclust30_2018_08/uniclust30_2018_08'  # Path to sequence database Not required if using ScanNet_noMSA networks.

# Additional paths required for training models.
pipeline_folder = library_folder + 'pipelines/' # Where preprocessed datasets are stored.
initial_values_folder = model_folder + 'initial_values/' # Where initial values of the parameters for the gaussian kernels and residue-residue graph edges are stored.

# Additional paths for reproducing baselines.
path_to_dssp = '/path/to/mkdssp' # Path to dssp binary. Only for reproducing handcrafted features baseline performance.
path_to_msms = '/path/to/msms.x86_64Linux2.2.6.1' # Path to msms binary. Only for reproducing handcrafted features baseline performance.

path_to_multiprot = '/path/to/multiprot.Linux'  # Path to multiprot executable. Only relevant for homology baseline.
homology_folder = library_folder + 'homology/'  # Where files are stored for homology baseline.
