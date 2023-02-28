import torch
from transformers.data.data_collator import *
import numpy as np
from dataclasses import dataclass


@dataclass
class DataCollatorTFIDF:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        tfidf_features = [feature["tfidf_feature"] for feature in features]
        tfidf_features = torch.tensor(tfidf_features, dtype=torch.float32)
        batch["tfidf_feature"] = tfidf_features

        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        return batch
