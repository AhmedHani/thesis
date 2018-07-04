import os
from os import getcwd

############ PROJECT DIR ############

ROOT = './'
SUBROOT = '../'
############ END PROJECT DIR ##########

############ DATASETS VARS ############

DATASETS_BASE_DIR = os.path.join(SUBROOT, 'rawdata')

############ END DATASETS VARS


############ QUORA DATASET VARS ###########

QUORA_QUESTIONS_PAIRS_DOWNLOAD_LINK = 'http://qim.ec.quoracdn.net/quora_duplicate_questions.tsv'

QUORA_QUESTIONS_PAIRS_DIR  = os.path.join(DATASETS_BASE_DIR, 'quora')
QUORA_QUESTIONS_PAIRS_RAW  = os.path.join(QUORA_QUESTIONS_PAIRS_DIR, 'raw.tsv')
QUORA_QUESTIONS_PAIRS_MOD  = os.path.join(QUORA_QUESTIONS_PAIRS_DIR, 'mod.tsv')
QUORA_QUESTIONS_PAIRS_META = os.path.join(QUORA_QUESTIONS_PAIRS_DIR, 'metadata.txt')

############ END QUORA DATASET VARS ###########



