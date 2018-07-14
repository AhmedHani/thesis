import os
from os import getcwd

############ PROJECT DIR ############

ROOT = './'
SUBROOT = '../'
############ END PROJECT DIR ##########

############ DATASETS VARS ############

DATASETS_BASE_DIR = os.path.join(SUBROOT, 'rawdata')

############ END DATASETS VARS #########


############ QUORA DATASET VARS ###########

QUORA_QUESTIONS_PAIRS_DOWNLOAD_LINK = 'http://qim.ec.quoracdn.net/quora_duplicate_questions.tsv'

QUORA_QUESTIONS_PAIRS_DIR  = os.path.join(DATASETS_BASE_DIR, 'quora')
QUORA_QUESTIONS_PAIRS_RAW  = os.path.join(QUORA_QUESTIONS_PAIRS_DIR, 'raw.tsv')
QUORA_QUESTIONS_PAIRS_MOD  = os.path.join(QUORA_QUESTIONS_PAIRS_DIR, 'mod.tsv')
QUORA_QUESTIONS_PAIRS_META = os.path.join(QUORA_QUESTIONS_PAIRS_DIR, 'metadata.txt')

############ END QUORA DATASET VARS ###########


############ SNLI DATASET VARS ################

SNLI_DOWNLOAD_LINK = 'https://nlp.stanford.edu/projects/snli/snli_1.0.zip'

SNLI_DIR       = os.path.join(DATASETS_BASE_DIR, 'snli')
SNLI_ZIP       = os.path.join(SNLI_DIR, 'snli_1.0.zip')
SNLI_FILES     = os.path.join(SNLI_DIR, 'snli_1.0')

SNLI_TRAIN_RAW = os.path.join(SNLI_FILES, 'snli_1.0_train.jsonl')
SNLI_DEV_RAW   = os.path.join(SNLI_FILES, 'snli_1.0_dev.jsonl')
SNLI_TEST_RAW  = os.path.join(SNLI_FILES, 'snli_1.0_test.jsonl')
SNLI_TRAIN_MOD = os.path.join(SNLI_DIR, 'mod_train.tsv')
SNLI_DEV_MOD   = os.path.join(SNLI_DIR, 'mod_dev.tsv')
SNLI_TEST_MOD  = os.path.join(SNLI_DIR, 'mod_test.tsv')
SNLI_META      = os.path.join(SNLI_DIR, 'metadata.txt')

############ END SNLI DATASET VARS #############


############ PPDB Dataset ######################

PPDB_DOWNLOAD_LINK = 'http://nlpgrid.seas.upenn.edu/PPDB/eng/ppdb-2.0-s-phrasal.gz'

PPDB_DIR  = os.path.join(DATASETS_BASE_DIR, 'ppdb')
PPDB_GZIP = os.path.join(PPDB_DIR, 'ppdb-2.0-s-phrasal.gz')
PPDB_RAW  = os.path.join(PPDB_DIR, 'ppdb-2.0-s-phrasal')
PPDB_MOD  = os.path.join(PPDB_DIR, 'mod.tsv')
PPDB_META = os.path.join(PPDB_DIR, 'metadata.txt')

############ END PPDB DATASET VARS #############

