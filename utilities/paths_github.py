
# Paths required for prediction.
library_folder = '/Users/jerometubiana/Documents/GitHub/ScanNet/' # Where the Github Repo is located.
structures_folder = '/Users/jerometubiana/PDB/' # Where pdb/mmCIF structures files are stored.
predictions_folder = library_folder + 'predictions/' # Output folder.
model_folder = library_folder + 'models/' # Where the networks as stored as pairs of files (.h5,.data).

# Additional paths required for prediction with evolutionary information.
MSA_folder = '/Users/jerometubiana/ScanNet/MSA/' # Where multiple sequence alignments are stored.
path2hhblits = None # Path to hhblits binary. Not required if using ScanNet_noMSA networks.
path2sequence_database = None # Path to sequence database Not required if using ScanNet_noMSA networks.

# path2hhblits = '/path/to/hh-suite/build/bin/hhblits'  # Path to hhblits binary. Not required if using ScanNet_noMSA networks.
# path2sequence_database = '/path/to/uniclust30_2018_08/uniclust30_2018_08'  # Path to sequence database Not required if using ScanNet_noMSA networks.

# Additional paths required for training models.
pipeline_folder = library_folder + 'pipelines/' # Where preprocessed datasets are stored.
initial_values_folder = model_folder + 'initial_values/' # Where initial values of the parameters for the gaussian kernels and residue-residue graph edges are stored.

# Additional paths for reproducing baselines.
path_to_dssp = '/Users/jerometubiana/Google\ Drive/Scripts/3D_Proteins/xssp-3.0.8/mkdssp' # Path to dssp binary. Only for reproducing handcrafted features baseline performance.
path_to_msms = '/Users/jerometubiana/Google\ Drive/Scripts/3D_Proteins/msms_MacOSX_2.6.1/msms.x86_64Linux2.2.6.1' # Path to msms binary. Only for reproducing handcrafted features baseline performance.
