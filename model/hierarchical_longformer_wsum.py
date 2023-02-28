from transformers.models.longformer.modeling_longformer import *
from transformers.models.longformer.modeling_longformer import (
    _CHECKPOINT_FOR_DOC,
    _CONFIG_FOR_DOC
)

import numpy as np
from torch.autograd import Function
from torch_scatter import scatter_softmax

def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src

def scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    # Source: https://github.com/rusty1s/pytorch_scatter
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)


class HierarchicalLongformerClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, cls_embeddings, **kwargs):
        hidden_states = self.dropout(cls_embeddings)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output = self.out_proj(hidden_states)
        return output

class HierarchicalAttentionLongformerForSequenceClassification(LongformerPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.longformer = LongformerModel(config, add_pooling_layer=False)
#         self.classifier = HierarchicalLongformerClassificationHead(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        self.linear = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size, 1),
#             nn.Tanh(),
        )

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(LONGFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="jpwahle/longformer-base-plagiarism-detection",
        output_type=LongformerSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="'ORIGINAL'",
        expected_loss=5.44,
    )
    def forward(
        self,
        chunks: Optional[torch.Tensor] = None,
        attention_masks: Optional[torch.Tensor] = None,
        doc_ids: Optional[np.int64] = None,  # use to merge chunks
#         input_ids: Optional[torch.Tensor] = None,
#         attention_mask: Optional[torch.Tensor] = None,
        global_attention_mask: Optional[torch.Tensor] = None,
#         head_mask: Optional[torch.Tensor] = None,
#         token_type_ids: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.Tensor] = None,
#         inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, LongformerSequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = True

        if global_attention_mask is None:
#             logger.info("Initializing global attention on CLS token...")
            global_attention_mask = torch.zeros_like(chunks)
            # global attention on cls token
            global_attention_mask[:, 0] = 1
        
        outputs = self.longformer(
            chunks,
            attention_mask=attention_masks,
            global_attention_mask=global_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        cls_embeddings = outputs.last_hidden_state[:, 0]

        # convert doc_ids from [3,3,3,5,6,6,6,6] to [0,0,0,1,2,2,2,2]
        counts = np.unique(doc_ids, return_counts=True)[1]
        nums = np.arange(counts.shape[0])
        doc_ids = np.repeat(nums, counts)
        doc_ids = torch.tensor(doc_ids, dtype=torch.int64).to(cls_embeddings.get_device())
        
        attention_logits = self.linear(cls_embeddings)
        attention_weights = scatter_softmax(attention_logits, doc_ids, dim=0)
        
        pooled_output = scatter_sum(cls_embeddings * attention_weights, doc_ids, dim=0)
        logits = self.classifier(pooled_output)
       
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        return LongformerSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            global_attentions=outputs.global_attentions,
        )
