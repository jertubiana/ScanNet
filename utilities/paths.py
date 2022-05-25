
# List of paths to folders and binaries. All folder paths should finish with slash (/)

mode = 'laptop'
#mode = 'laptop_webserver'
#mode = 'tau'
#mode = 'tau_webserver'


if mode == 'laptop':
    library_folder = '/Users/jerometubiana/Documents/GitHub/ScanNet/'
    structures_folder = '/Users/jerometubiana/PDB/' # Where pdb/mmCIF structures files are stored.
    MSA_folder = library_folder+'MSA/' # Where multiple sequence alignments are stored.
    predictions_folder = library_folder + 'predictions/' # Output folder.
    model_folder = library_folder + 'models/' # Where the networks as stored as pairs of files (.h5,.data).
    pipeline_folder = library_folder + 'pipelines/' # Where preprocessed datasets are stored.
    initial_values_folder = model_folder + 'initial_values/' # Where initial values of the parameters for the gaussian kernels and residue-residue graph edges are stored.
    homology_folder = library_folder + 'baselines/homology/' # Where files are stored for homology baseline.
    visualization_folder = library_folder + 'visualizations/'
    path2hhblits = library_folder # Path to hhblits binary. Not required if using ScanNet_noMSA networks.
    path2sequence_database = None # Path to sequence database Not required if using ScanNet_noMSA networks.
    path_to_dssp = '/Users/jerometubiana/Google\ Drive/Scripts/3D_Proteins/xssp-3.0.8/mkdssp' # Path to dssp binary. Only for reproducing baseline performance.
    path_to_msms = '/Users/jerometubiana/Google\ Drive/Scripts/3D_Proteins/msms_MacOSX_2.6.1/msms.x86_64Linux2.2.6.1' # Path to msms binary. Only for reproducing baseline performance.
    path_to_multiprot = None # Path to multiprot executable. Only relevant for homology baseline

elif mode == 'laptop_webserver':
    library_folder = '/Users/jerometubiana/PostDoc/ScanNet_webserver/v0.2/ScanNet/'
    structures_folder = '/Users/jerometubiana/PostDoc/ScanNet_webserver/v0.2/PDB/' # Where pdb/mmCIF structures files are stored.
    MSA_folder = library_folder+'MSA/' # Where multiple sequence alignments are stored.
    predictions_folder = library_folder + 'predictions/' # Output folder.
    model_folder = library_folder + 'models/' # Where the networks as stored as pairs of files (.h5,.data).
    pipeline_folder = library_folder + 'pipelines/' # Where preprocessed datasets are stored.
    initial_values_folder = model_folder + 'initial_values/' # Where initial values of the parameters for the gaussian kernels and residue-residue graph edges are stored.
    homology_folder = library_folder + 'baselines/homology/' # Where files are stored for homology baseline.
    visualization_folder = library_folder + 'visualizations/'
    path2hhblits = 'null' # Path to hhblits binary. Not required if using ScanNet_noMSA networks.
    path2sequence_database = 'null' # Path to sequence database Not required if using ScanNet_noMSA networks.
    path_to_dssp = '/Users/jerometubiana/Google\ Drive/Scripts/3D_Proteins/xssp-3.0.8/mkdssp' # Path to dssp binary. Only for reproducing baseline performance.
    path_to_msms = '/Users/jerometubiana/Google\ Drive/Scripts/3D_Proteins/msms_MacOSX_2.6.1/msms.x86_64Linux2.2.6.1' # Path to msms binary. Only for reproducing baseline performance.
    path_to_multiprot = None # Path to multiprot executable. Only relevant for homology baseline

elif mode == 'tau':
    library_folder = '/home/iscb/wolfson/jeromet/ScanNet/'
    structures_folder = '/specific/netapp5_2/iscb/wolfson/jeromet/Data/PDB_files/' # Where pdb/mmCIF structures files are stored.
    MSA_folder = '/specific/netapp5_2/iscb/wolfson/jeromet/Data/MSA_test/' # Where multiple sequence alignments are stored.
    predictions_folder = library_folder+'predictions/' # Output folder.
    model_folder = library_folder+'models/' # Where the networks as stored as pairs of files (.h5,.data).
    pipeline_folder = library_folder+'pipelines/' # Where preprocessed datasets are stored.
    initial_values_folder = model_folder + 'initial_values/' # Where initial values of the parameters for the gaussian kernels and residue-residue graph edges are stored.
    homology_folder = library_folder + 'baselines/homology/' # Where files are stored for homology baseline.
    visualization_folder = library_folder + 'visualizations/'
    path2hhblits = '/specific/netapp5_2/iscb/wolfson/sequence_database/hh-suite/build/bin/hhblits' # Path to hhblits binary. Not required if using ScanNet_noMSA networks.
    # path2sequence_database = '/specific/netapp5_2/iscb/wolfson/sequence_database/uniclust30_2018_08/uniclust30_2018_08' # Path to sequence database Not required if using ScanNet_noMSA networks.
    path2sequence_database = '/specific/netapp5_2/iscb/wolfson/sequence_database/uniclust30_2020_06/UniRef30_2020_06' # Path to sequence database Not required if using ScanNet_noMSA networks.
    path_to_dssp = '/specific/a/home/cc/students/cs/jeromet/Drive/Scripts/3D_Proteins/xssp-3.0.9/mkdssp' # Path to dssp binary. Only for reproducing baseline performance.
    path_to_msms = '/specific/a/home/cc/students/cs/jeromet/Drive/Scripts/3D_Proteins/msms/msms.x86_64Linux2.2.6.1' # Path to msms binary. Only for reproducing baseline performance.
    path_to_multiprot = '/home/iscb/wolfson/jeromet/MultiProt/multiprot.Linux' # Path to multiprot executable. Only relevant for homology baseline


elif mode == 'tau_webserver':
    library_folder = '/specific/netapp5_2/iscb/wolfson/ppdock/ScanNet_webserver/ScanNet/'
    structures_folder = '/specific/netapp5_2/iscb/wolfson/ppdock/Data/PDB/' # Where pdb/mmCIF structures files are stored.
    MSA_folder = '/specific/netapp5_2/iscb/wolfson/ppdock/Data/MSA/' # Where multiple sequence alignments are stored.
    predictions_folder = library_folder + 'predictions/' # Output folder.
    model_folder = library_folder + 'models/' # Where the networks as stored as pairs of files (.h5,.data).
    pipeline_folder = '/specific/netapp5_2/iscb/wolfson/ppdock/Data/pipelines/' # Where preprocessed datasets are stored.
    initial_values_folder = model_folder + 'initial_values/' # Where initial values of the parameters for the gaussian kernels and residue-residue graph edges are stored.
    homology_folder = library_folder + 'baselines/homology/' # Where files are stored for homology baseline.
    visualization_folder = library_folder + 'visualizations/'
    path2hhblits = '/specific/netapp5_2/iscb/wolfson/sequence_database/hh-suite/build/bin/hhblits' # Path to hhblits binary. Not required if using ScanNet_noMSA networks.
    # path2sequence_database = '/specific/netapp5_2/iscb/wolfson/sequence_database/uniclust30_2018_08/uniclust30_2018_08' # Path to sequence database Not required if using ScanNet_noMSA networks.
    path2sequence_database = '/specific/netapp5_2/iscb/wolfson/sequence_database/uniclust30_2020_06/UniRef30_2020_06' # Path to sequence database Not required if using ScanNet_noMSA networks.
    path_to_dssp = '/specific/a/home/cc/students/cs/jeromet/Drive/Scripts/3D_Proteins/xssp-3.0.9/mkdssp' # Path to dssp binary. Only for reproducing baseline performance.
    path_to_msms = '/specific/a/home/cc/students/cs/jeromet/Drive/Scripts/3D_Proteins/msms/msms.x86_64Linux2.2.6.1' # Path to msms binary. Only for reproducing baseline performance.
    path_to_multiprot = None # Path to multiprot executable. Only relevant for homology baseline




