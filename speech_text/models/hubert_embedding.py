
from typing import Optional, Union
from transformers import HubertModel, HubertPreTrainedModel
from transformers.modeling_outputs import CausalLMOutput
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import MultipleNegativesRankingLoss
import random
import torch
import torch.nn.functional as F

class L1CosLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, inputs, targets):
        l1 = F.l1_loss(input=inputs, target=targets, reduction="none").mean(-1)
        cos = F.sigmoid(F.cosine_similarity(inputs, targets, dim=-1)).log()
        # minus because we maximize the cosine similarity
        return (l1 - cos).mean()

class HubertMatryoshka(torch.nn.Module):
    def __init__(self, hubert_model: HubertPreTrainedModel, dims: list[int], training_mode: str = "clip"):
        super().__init__()
        self.hubert = hubert_model
        self.training_mode = training_mode
        base_dim = self.hubert.config.hidden_size * 12
        
        if self.training_mode == "clip":
            self.matryoshka_layers = torch.nn.ModuleDict({
                str(d): torch.nn.Linear(base_dim, d) for d in dims
            })

            self.pooling_parameters = torch.nn.Parameter(torch.ones(1, base_dim))
        else:
            self.projector = torch.nn.Linear(in_features=base_dim, out_features=max(dims), bias=False)
            self.pooling_parameters = torch.nn.Conv1d(
                in_channels=max(dims),
                out_channels=max(dims),
                kernel_size=2,
                stride=2,
                padding=0)


    def forward(
            self,
            input_values: Optional[torch.Tensor],
            attention_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            labels: Optional[torch.Tensor] = None,
            dim: int = 256
        ) -> torch.Tensor:
        
        outputs = self.hubert(
            input_values,
            attention_mask=None,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = torch.cat(outputs.hidden_states[1:], dim=-1)
        if self.training_mode == "clip":
            q = self.pooling_parameters.expand(hidden_states.size(0), -1, -1)
            return F.scaled_dot_product_attention(q, hidden_states, hidden_states).squeeze(1)
        hidden_states = self.projector(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.pooling_parameters(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states
    
class HubertForEmbedding(HubertPreTrainedModel):
    def __init__(self,
                 config,
                 text_model_path_or_name: str,
                 matryoshka_dims: list[int],
                 loss = "contrastive",
                 training_mode: str = "clip"):
        super().__init__(config)
        self.hubert = HubertModel(config)

        self.matryoshka_dims = matryoshka_dims

        self._keys_to_ignore_on_save = set()

        self.loss_name = loss
        if loss == "regression":
            self.loss = L1CosLoss()
        elif loss == "contrastive":
            self.loss = MultipleNegativesRankingLoss(None)
        else:
            raise ValueError(f"Unknown loss: {loss}")
        self.text_model_path_or_name = text_model_path_or_name
        self.text_embedding_model = None
        self.training_mode = training_mode
        self.freeze_text_model = True
        # Initialize weights and apply final processing
        self.post_init()
    
    def load_embedding_model(self) -> SentenceTransformer:
        dims = self.matryoshka_dims if self.training_mode == "clip" else [max(self.matryoshka_dims)]
        self.hubert = HubertMatryoshka(self.hubert, dims, self.training_mode)
        self.text_embedding_model = SentenceTransformer(
            self.text_model_path_or_name, self.device,
            # model_kwargs={"attn_implementation": "flash_attention_2", "torch_dtype": "bfloat16"}
            )
        if self.training_mode == "clip":
            for k in self.text_embedding_model.state_dict().keys():
                self._keys_to_ignore_on_save.add('text_embedding_model.' + k)
        
        if self.training_mode == "clip" or self.freeze_text_model:
            for param in self.text_embedding_model.transformers_model.parameters():
                param.requires_grad = False

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.hubert.hubert.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.hubert.hubert.feature_extractor._freeze_parameters()

    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for param in self.hubert.hubert.parameters():
            param.requires_grad = False

    def forward(
        self,
        input_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        labels_attention_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        input_ids_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ) -> Union[tuple, CausalLMOutput]:

        speech_embedding = self.hubert(
            input_values,
            attention_mask=None,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )
        
        if self.training_mode != "clip":
            if input_ids is not None:
                token_embeddings = self.text_embedding_model.transformers_model.embed_tokens(input_ids)
                bs, sql, *_ = speech_embedding.shape
                speech_embedding = torch.cat([token_embeddings, speech_embedding], dim=1)
                input_ids_attention_mask = torch.cat(
                    [input_ids_attention_mask, torch.ones((bs, sql)).to(input_ids_attention_mask.device)], dim=1
                    ).to(torch.long)
            inputs = {"inputs_embeds": speech_embedding, "attention_mask": input_ids_attention_mask} if input_ids_attention_mask is not None else {"inputs_embeds": speech_embedding}
            speech_embedding = self.text_embedding_model(inputs)["sentence_embedding"]

        target_embedding = self.text_embedding_model({"input_ids": labels, "attention_mask": labels_attention_mask})["sentence_embedding"]
        
        if self.loss_name == "regression":
            if self.training_mode == "clip":
                losses = [
                    self.loss(self.hubert.matryoshka_layers[str(dim)](speech_embedding), target_embedding[..., :dim]) \
                        for dim in self.matryoshka_dims
                        ]
            else:
                losses = [
                    self.loss(speech_embedding[..., :dim], target_embedding[..., :dim]) \
                        for dim in self.matryoshka_dims
                        ]

        else:
            if self.training_mode == "clip":
                losses = [
                    self.loss.compute_loss_from_embeddings(
                        [self.hubert.matryoshka_layers[str(dim)](speech_embedding), target_embedding[..., :dim]], None) \
                    for dim in self.matryoshka_dims
                ]
            else:
                losses = [
                    self.loss.compute_loss_from_embeddings([speech_embedding[..., :dim], target_embedding[..., :dim]], None) \
                    for dim in self.matryoshka_dims
                ]
        
        loss = torch.stack(losses).mean()

        if not return_dict:
            return (loss,)

        return CausalLMOutput(
            loss=loss, logits=None, hidden_states=None, attentions=None
        )