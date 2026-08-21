"""IsharaConnect 4-Tier BdSL Dataset Ingestion Suite."""

from dataset.ingestors.tier1_fingerspelling_ingestor import Tier1FingerspellingIngestor
from dataset.ingestors.tier2_islr_ingestor import Tier2ISLRIngestor
from dataset.ingestors.tier3_cslr_ingestor import Tier3CSLRIngestor
from dataset.ingestors.tier4_slt_ingestor import Tier4SLTIngestor

__all__ = [
    "Tier1FingerspellingIngestor",
    "Tier2ISLRIngestor",
    "Tier3CSLRIngestor",
    "Tier4SLTIngestor",
]
