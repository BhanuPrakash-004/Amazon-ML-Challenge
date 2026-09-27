"""Combined feature matrix (V3 = V2-light + enriched). Column order frozen."""
import pandas as pd

from .features_basic import basic_features, BASIC_COLS
from .features_lexical import lexical_features, LEX_COLS
from .features_address import address_features, ADDR_COLS
from .features_semantic import semantic_features, SEM_COLS
from .features_block import block_features, BLOCK_COLS
from .features_entity import entity_features, ENT_COLS

ALL_COLS = BASIC_COLS + LEX_COLS + ADDR_COLS + SEM_COLS + BLOCK_COLS + ENT_COLS


def features_all(pairs: pd.DataFrame) -> pd.DataFrame:
    parts = [basic_features(pairs), lexical_features(pairs), address_features(pairs),
             semantic_features(pairs), block_features(pairs), entity_features(pairs)]
    return pd.concat(parts, axis=1)[ALL_COLS]
