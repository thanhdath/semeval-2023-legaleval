import torch
from transformers.data.data_collator import *
import numpy as np

class DataCollatorChunking:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # features: {chunks: [], attention_masks: {}}
        batch = {
            'chunks': [],
            'attention_masks': [],
            'doc_ids': [], 
            'labels': []
        }
        for feature in features:
            batch['chunks'].extend(feature['chunks'])
            batch['attention_masks'].extend(feature['attention_masks'])
            batch['doc_ids'].extend(feature['doc_ids'])
            if 'label' in feature:
                batch['labels'].append(feature['label'])

        batch['chunks'] = torch.tensor(batch['chunks'], dtype=torch.int64)
        batch['attention_masks'] = torch.tensor(batch['attention_masks'], dtype=torch.int64)
        batch['doc_ids'] = np.array(batch['doc_ids'], dtype=np.int64)

        if len(batch['labels']) == 0:
            del batch['labels']
        else:
            batch['labels'] = torch.tensor(batch['labels'], dtype=torch.int64)
        
        return batch
