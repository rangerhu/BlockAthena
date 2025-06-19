# === Set Module Settings Here for BlockAthena ===

# Temporal analysis settings
DEFAULT_BIN_SIZE = 100
DEFAULT_TIME_GAP = 3600  # seconds (used in motif segmentation)

# Wavelet transform
WAVELET_TYPE = 'morl'
MIN_SCALE = 1
MAX_SCALE = 64
NUM_SCALES = 32

# CHGM temporal proximity threshold
DELTA_T = 600  # 10 minutes in seconds

# ERA residual depth
RESIDUAL_DEPTH = 3

# Feature dimensions
FUSION_HIDDEN_DIM = 128
HYPERGRAPH_EMBED_DIM = 64

# Storage paths
DATA_DIR = 'data'
RAW_DIR = f'{DATA_DIR}/raw'
PROCESSED_DIR = f'{DATA_DIR}/processed'
