from typing import Optional
try:
    from typing import Self
except ImportError:
    from typing_extensions import Self
import torch
import torch.nn.functional as F
from transformers import HubertPreTrainedModel, Wav2Vec2Processor, HubertModel, AutoConfig
import torchaudio
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Module
from pathlib import Path

class HubertMatryoshka(HubertPreTrainedModel):
    def __init__(self, config, dims: list[int], mode: str="clip"):
        super().__init__(config)
        self.hubert = HubertModel(config)
        base_dim = self.hubert.config.hidden_size * 12
        self.mode = mode

        if self.mode == "clip":
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
                padding=0
                )


    def forward(
            self,
            input_values: Optional[torch.Tensor],
            dim: int = 1024
        ) -> torch.Tensor:
        
        outputs = self.hubert(
            input_values,
            attention_mask=None,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = torch.cat(outputs.hidden_states[1:], dim=-1)
        if self.mode == "clip":
            q = self.pooling_parameters.expand(hidden_states.size(0), -1, -1)
            pooled = F.scaled_dot_product_attention(q, hidden_states, hidden_states).squeeze(1)
            return self.matryoshka_layers[str(dim)](pooled)
        hidden_states = self.projector(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.pooling_parameters(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states

class OolelEmbed(Module):
    save_in_root: bool = True

    def __init__(self, text_embedding_model_path: str, speech_embedding_model_path: str, prompt: str=None, mode: str="lf", truncate_dim: int=None, task: str=None) -> None:
        super().__init__()
        self.truncate_dim = truncate_dim
        self.text_embedding_model_path = text_embedding_model_path
        self.speech_embedding_model_path = speech_embedding_model_path
        self.mode = mode
        self.processor = Wav2Vec2Processor.from_pretrained(speech_embedding_model_path)
        
        self.text_embedding_model = SentenceTransformer(text_embedding_model_path)

        self.speech_embedding_model = HubertMatryoshka.from_pretrained(
            speech_embedding_model_path,
            dims=[1024, 512, 256, 128],
            mode=mode,
            )

        self.tokenize_fn = self.text_embedding_model[0].tokenize
        self.prompt_ids = None
        if prompt is not None:
            self.prompt_ids = self.tokenize_fn([prompt])["input_ids"]

        self.dimension = self.text_embedding_model.get_sentence_embedding_dimension()
        self.task = task
        self.sample_rate = 16_000

    def forward(self, features: dict[str, torch.Tensor], *args, **kwargs) -> dict[str, torch.Tensor]:
        if "input_values" in features:
            if self.mode != "clip":
                output_features = self.speech_embedding_model(**features)
                if self.prompt_ids is not None:
                    prompt_embed = self.text_embedding_model.transformers_model.embed_tokens(self.prompt_ids.to(output_features.device))
                    prompt_embed = prompt_embed.expand(output_features.shape[0], -1, -1)
                    output_features = torch.cat([prompt_embed, output_features], dim=1)

                features["inputs_embeds"] = output_features
                output_features = self.text_embedding_model(features)["sentence_embedding"][..., :self.truncate_dim]
            else:
                output_features = self.speech_embedding_model(**features, dim=self.truncate_dim)
            features["sentence_embedding"] = output_features
            return features

        if "input_ids" in features:
            output_features = self.text_embedding_model(features)["sentence_embedding"][..., :self.truncate_dim]
            features["sentence_embedding"] = output_features
            return features

        
    def load_audio(self, audio_list):
        audio_arrays = []
        audio_list = audio_list if isinstance(audio_list, list) else [audio_list]
        for audio in audio_list:
            waveform, sr = torchaudio.load(str(audio))

            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            waveform = waveform.squeeze().numpy()
        
            audio_arrays.append(waveform)
        input_values = self.processor(
            audio_arrays,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        return input_values

    def tokenize(self, inputs, **kwargs) -> dict[str, torch.Tensor]:
        suffix = Path(inputs).suffix if not isinstance(inputs, list) else Path(inputs[0]).suffix
        if suffix in {".wav", ".flac", ".mp3"}:
            return self.load_audio(inputs)
        return self.tokenize_fn(inputs)

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

    def get_max_seq_length(self) -> int | None:
        return 2048

    def save(self, output_path: str, *args, safe_serialization: bool = True, **kwargs) -> None:
        self.text_embedding_model.save(f"{output_path}/oolel_mrl")
        self.processor.save_pretrained(f"{output_path}/hubert_mrl")
        self.speech_embedding_model.save_pretrained(f"{output_path}/hubert_mrl")

    @classmethod
    def load(
        cls,
        model_name_or_path: str,
        subfolder: str = "",
        token: bool | str | None = None,
        cache_folder: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
        **kwargs,
    ) -> Self:
        local_path = cls.load_dir_path(
            model_name_or_path=model_name_or_path,
            subfolder=subfolder,
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )
        return cls(local_path)