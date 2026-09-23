"""
XTTS-v2 GPT Fine-Tuning and Inference Engine.

Wraps Coqui TTS GPTTrainer and XTTS model inference with explicit VRAM
protection safeguards for 4GB laptop GPUs:
    - Micro-batching (batch_size=1)
    - Gradient accumulation (grad_accum=2 or 4)
    - Mixed precision where supported
    - Explicit cache clearing before and after operations
    - Checkpoint loading and candidate speech generation
"""

import gc
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import soundfile as sf
import torch
import torchaudio

from trainer import Trainer, TrainerArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig
from TTS.tts.models.xtts import Xtts, XttsAudioConfig
from TTS.utils.manage import ModelManager

from .config import (
    CHECKPOINTS_DIRECTORY,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_GRAD_ACCUM_STEPS,
    DEFAULT_LANGUAGE,
    DEFAULT_LEARNING_RATE,
    DEVICE,
    DVAE_CHECKPOINT_LINK,
    MAX_AUDIO_LENGTH,
    MEL_NORM_LINK,
    TOKENIZER_FILE_LINK,
    XTTS_CHECKPOINT_LINK,
    XTTS_CONFIG_LINK,
    XTTS_SAMPLE_RATE,
)


def clear_gpu_memory():
    """Explicitly deallocate cached GPU memory and invoke Python GC."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@dataclass
class TrainingResult:
    """Artifacts produced by an XTTS fine-tuning run."""
    user_id: str
    round_number: int
    output_dir: Path
    best_model_path: Path
    config_path: Path
    vocab_path: Path
    speaker_reference_path: Path
    training_success: bool
    error_message: Optional[str] = None


class XTTSTrainer:
    """
    Manages XTTS-v2 GPT fine-tuning and fine-tuned model inference.
    """

    def __init__(
        self,
        base_models_dir: Optional[Path] = None,
        device: str = DEVICE,
    ):
        self.device = device
        self.base_models_dir = base_models_dir or (CHECKPOINTS_DIRECTORY / "base_xtts_files")
        self.base_models_dir.mkdir(parents=True, exist_ok=True)
        self._loaded_ft_model: Optional[Xtts] = None
        self._loaded_ft_model_path: Optional[str] = None

    def ensure_base_model_files(self) -> Tuple[Path, Path, Path, Path, Path]:
        """
        Ensure base XTTS-v2 model checkpoint, config, vocab, DVAE, and mel stats exist locally.
        """
        dvae_file = self.base_models_dir / "dvae.pth"
        mel_norm_file = self.base_models_dir / "mel_stats.pth"
        vocab_file = self.base_models_dir / "vocab.json"
        xtts_checkpoint = self.base_models_dir / "model.pth"
        xtts_config_file = self.base_models_dir / "config.json"

        download_list = []
        if not dvae_file.exists():
            download_list.append(DVAE_CHECKPOINT_LINK)
        if not mel_norm_file.exists():
            download_list.append(MEL_NORM_LINK)
        if not vocab_file.exists():
            download_list.append(TOKENIZER_FILE_LINK)
        if not xtts_checkpoint.exists():
            download_list.append(XTTS_CHECKPOINT_LINK)
        if not xtts_config_file.exists():
            download_list.append(XTTS_CONFIG_LINK)

        if download_list:
            print(f"Downloading required XTTS base model files to {self.base_models_dir}...")
            ModelManager._download_model_files(
                download_list,
                str(self.base_models_dir),
                progress_bar=True
            )

        return (
            xtts_checkpoint,
            xtts_config_file,
            vocab_file,
            dvae_file,
            mel_norm_file,
        )

    def train_user_model(
        self,
        user_id: str,
        round_number: int,
        train_csv_path: Path,
        eval_csv_path: Path,
        output_round_dir: Path,
        num_epochs: int = DEFAULT_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        grad_accum: int = DEFAULT_GRAD_ACCUM_STEPS,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        language: str = DEFAULT_LANGUAGE,
    ) -> TrainingResult:
        """
        Run XTTS-v2 GPT fine-tuning on user dataset.
        """
        clear_gpu_memory()

        output_round_dir.mkdir(parents=True, exist_ok=True)
        raw_training_out = output_round_dir / "run"

        try:
            (
                xtts_checkpoint,
                xtts_config_file,
                vocab_file,
                dvae_file,
                mel_norm_file,
            ) = self.ensure_base_model_files()

            # Dataset configuration
            dataset_config = BaseDatasetConfig(
                formatter="coqui",
                dataset_name=f"ft_{user_id}_r{round_number}",
                path=str(train_csv_path.parent),
                meta_file_train=str(train_csv_path.resolve()),
                meta_file_val=str(eval_csv_path.resolve()),
                language=language,
            )

            # Model arguments
            model_args = GPTArgs(
                max_conditioning_length=132300,
                min_conditioning_length=66150,
                debug_loading_failures=False,
                max_wav_length=MAX_AUDIO_LENGTH,
                max_text_length=200,
                mel_norm_file=str(mel_norm_file.resolve()),
                dvae_checkpoint=str(dvae_file.resolve()),
                xtts_checkpoint=str(xtts_checkpoint.resolve()),
                tokenizer_file=str(vocab_file.resolve()),
                gpt_num_audio_tokens=1026,
                gpt_start_audio_token=1024,
                gpt_stop_audio_token=1025,
                gpt_use_masking_gt_prompt_approach=True,
                gpt_use_perceiver_resampler=True,
            )

            audio_config = XttsAudioConfig(
                sample_rate=XTTS_SAMPLE_RATE,
                dvae_sample_rate=XTTS_SAMPLE_RATE,
                output_sample_rate=24000,
            )

            config = GPTTrainerConfig(
                epochs=num_epochs,
                output_path=str(raw_training_out),
                model_args=model_args,
                run_name=f"GPT_XTTS_{user_id}_r{round_number}",
                project_name="CARE_XTTS_Personalization",
                run_description=f"XTTS Personalization for {user_id} Round {round_number}",
                dashboard_logger="tensorboard",
                logger_uri=None,
                audio=audio_config,
                batch_size=batch_size,
                batch_group_size=48,
                eval_batch_size=batch_size,
                num_loader_workers=2,
                eval_split_max_size=256,
                print_step=25,
                plot_step=50,
                log_model_step=50,
                save_step=500,
                save_n_checkpoints=1,
                save_checkpoints=True,
                print_eval=False,
                optimizer="AdamW",
                optimizer_wd_only_on_weights=True,
                optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
                lr=learning_rate,
                lr_scheduler="MultiStepLR",
                lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1},
                test_sentences=[],
            )

            model = GPTTrainer.init_from_config(config)

            train_samples, eval_samples = load_tts_samples(
                [dataset_config],
                eval_split=True,
                eval_split_max_size=config.eval_split_max_size,
                eval_split_size=config.eval_split_size,
            )

            trainer = Trainer(
                TrainerArgs(
                    restore_path=None,
                    skip_train_epoch=False,
                    start_with_eval=False,
                    grad_accum_steps=grad_accum,
                ),
                config,
                output_path=str(raw_training_out),
                model=model,
                train_samples=train_samples,
                eval_samples=eval_samples,
            )

            trainer.fit()

            # Identify best checkpoint produced
            trainer_out = Path(trainer.output_path)
            candidate_best = trainer_out / "best_model.pth"
            if not candidate_best.exists():
                candidate_best = trainer_out / "checkpoint.pth"

            # Copy standard assets into round directory
            final_best = output_round_dir / "best_model.pth"
            final_config = output_round_dir / "config.json"
            final_vocab = output_round_dir / "vocab.json"

            shutil.copy(str(candidate_best), str(final_best))
            shutil.copy(str(xtts_config_file), str(final_config))
            shutil.copy(str(vocab_file), str(final_vocab))

            # Select speaker reference from train samples (longest text)
            samples_len = [len(item["text"].split()) for item in train_samples]
            longest_idx = samples_len.index(max(samples_len))
            speaker_ref_rel = train_samples[longest_idx]["audio_file"]
            speaker_ref_src = train_csv_path.parent / speaker_ref_rel
            speaker_ref_dst = output_round_dir / "speaker_reference.wav"
            shutil.copy(str(speaker_ref_src), str(speaker_ref_dst))

            del model, trainer, train_samples, eval_samples
            clear_gpu_memory()

            return TrainingResult(
                user_id=user_id,
                round_number=round_number,
                output_dir=output_round_dir,
                best_model_path=final_best,
                config_path=final_config,
                vocab_path=final_vocab,
                speaker_reference_path=speaker_ref_dst,
                training_success=True,
            )

        except Exception as exc:
            clear_gpu_memory()
            return TrainingResult(
                user_id=user_id,
                round_number=round_number,
                output_dir=output_round_dir,
                best_model_path=output_round_dir / "best_model.pth",
                config_path=output_round_dir / "config.json",
                vocab_path=output_round_dir / "vocab.json",
                speaker_reference_path=output_round_dir / "speaker_reference.wav",
                training_success=False,
                error_message=str(exc),
            )

    def load_fine_tuned_model(
        self,
        checkpoint_path: Path,
        config_path: Path,
        vocab_path: Path,
    ) -> Xtts:
        """
        Load fine-tuned XTTS model checkpoint into memory for inference.
        """
        chk_key = str(checkpoint_path.resolve())
        if self._loaded_ft_model is not None and self._loaded_ft_model_path == chk_key:
            return self._loaded_ft_model

        clear_gpu_memory()

        config = XttsConfig()
        config.load_json(str(config_path))
        model = Xtts.init_from_config(config)
        model.load_checkpoint(
            config,
            checkpoint_path=str(checkpoint_path),
            vocab_path=str(vocab_path),
            eval=True,
            use_deepspeed=False
        )

        if self.device == "cuda" and torch.cuda.is_available():
            model = model.cuda()

        self._loaded_ft_model = model
        self._loaded_ft_model_path = chk_key
        return self._loaded_ft_model

    def generate_fine_tuned_speech(
        self,
        text: str,
        reference_audio_path: Path,
        output_path: Path,
        checkpoint_path: Path,
        config_path: Path,
        vocab_path: Path,
        language: str = DEFAULT_LANGUAGE,
    ) -> Path:
        """
        Generate synthesized speech using a fine-tuned XTTS checkpoint.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        model = self.load_fine_tuned_model(checkpoint_path, config_path, vocab_path)

        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
            audio_path=str(reference_audio_path),
            gpt_cond_len=model.config.gpt_cond_len,
            max_ref_length=model.config.max_ref_len,
            sound_norm_refs=model.config.sound_norm_refs,
            load_sr=XTTS_SAMPLE_RATE,
        )

        out = model.inference(
            text=text,
            language=language,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=model.config.temperature,
            length_penalty=model.config.length_penalty,
            repetition_penalty=model.config.repetition_penalty,
            top_k=model.config.top_k,
            top_p=model.config.top_p,
        )

        wav_tensor = torch.tensor(out["wav"]).unsqueeze(0)
        torchaudio.save(str(output_path), wav_tensor, 24000)

        if not output_path.exists():
            raise RuntimeError(f"Failed to produce fine-tuned audio at {output_path}")

        return output_path
