"""
Global Constants for Transformer Architecture and Runtime.
"""

# Special Token Identifiers (GPT-2 Tokenizer standard)
PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 50256
EOS_TOKEN_ID = 50256
UNK_TOKEN_ID = 50256

# Mathematical and Numerical Stability Constants
EPSILON = 1e-8
INF = float("inf")
NEG_INF = -1e9

# Model Hyperparameter Bounds & Defaults
MIN_SEQ_LEN = 1
MAX_SEQ_LEN = 2048
DEFAULT_D_MODEL = 768
DEFAULT_N_HEADS = 12
DEFAULT_N_LAYERS = 12
DEFAULT_VOCAB_SIZE = 50257

# Default Asset and Serialization Paths
DEFAULT_WEIGHTS_PATH = "weights/model_weights.pt"
DEFAULT_VOCAB_PATH = "weights/vocab.json"
DEFAULT_MERGES_PATH = "weights/merges.txt"
DEFAULT_CONFIG_PATH = "config/model_config.json"

# Server and Network Defaults
DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_SERVER_PORT = 8080
DEFAULT_TIMEOUT = 30
