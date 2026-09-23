"""
Personalized Voice Manager with Automatic Zero-Shot Fallback.

High-level voice synthesis controller that dynamically routes generation requests:
    1. Checks if an active promoted fine-tuned model checkpoint exists for the user.
    2. If YES: Executes speech synthesis using the fine-tuned model.
       If fine-tuned synthesis encounters any runtime failure:
           -> Automatically catches exception and executes Zero-Shot XTTS fallback.
    3. If NO: Directly routes to existing production Zero-Shot XTTS cloning pipeline.

Preserves full backward compatibility with VoiceCloningManager.
"""

from pathlib import Path
from typing import Optional, Union

from Voice_Cloning.config import DEFAULT_LANGUAGE
from Voice_Cloning.utils import ensure_directory, validate_text, validate_user_id
from Voice_Cloning.voice_cloning_manager import VoiceCloningManager

from .checkpoint_manager import CheckpointManager
from .trainer import XTTSTrainer


class PersonalizedVoiceManager:
    """
    Orchestrates personalized voice synthesis with automatic zero-shot fallback.
    """

    def __init__(
        self,
        checkpoint_manager: Optional[CheckpointManager] = None,
        zero_shot_manager: Optional[VoiceCloningManager] = None,
        trainer: Optional[XTTSTrainer] = None,
    ):
        print("Initializing Personalized Voice Manager...")
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()
        self.zero_shot_manager = zero_shot_manager or VoiceCloningManager()
        self.trainer = trainer or XTTSTrainer()
        print("Personalized Voice Manager initialized successfully.")

    def get_user_model_status(self, user_id: str) -> dict:
        """Return information about the active model for a user."""
        user_id = validate_user_id(user_id)
        state = self.checkpoint_manager.load_active_model_state(user_id)
        return {
            "user_id": user_id,
            "primary_model": state.primary_model,
            "fallback_model": state.fallback_model,
            "active_round": state.active_round,
            "has_promoted_checkpoint": state.primary_model == "fine_tuned" and state.checkpoint_path is not None,
        }

    def generate_for_user(
        self,
        user_id: str,
        text: str,
        output_path: Optional[Union[str, Path]] = None,
        language: str = DEFAULT_LANGUAGE,
    ) -> Path:
        """
        Generate speech for a user using their promoted fine-tuned model if available,
        or seamlessly fall back to Zero-Shot XTTS.

        Parameters
        ----------
        user_id : str
            Target user identifier.
        text : str
            Text string to synthesize.
        output_path : str or Path, optional
            Destination audio path. If None, default path is generated.
        language : str
            Language code (default: 'en').

        Returns
        -------
        Path
            Path to the synthesized audio file.
        """
        user_id = validate_user_id(user_id)
        text = validate_text(text)

        if output_path is None:
            output_path = self.zero_shot_manager.get_default_output_path(user_id)
        else:
            output_path = Path(output_path)
            ensure_directory(output_path.parent)

        active_state = self.checkpoint_manager.load_active_model_state(user_id)

        # -------------------------------------------------------------
        # 1. Attempt Fine-Tuned Generation if Promoted
        # -------------------------------------------------------------
        if (
            active_state.primary_model == "fine_tuned"
            and active_state.checkpoint_path
            and Path(active_state.checkpoint_path).exists()
            and active_state.config_path
            and Path(active_state.config_path).exists()
            and active_state.vocab_path
            and Path(active_state.vocab_path).exists()
        ):
            print(f"\n[PersonalizedVoiceManager] Using fine-tuned XTTS model for user '{user_id}' (Round {active_state.active_round})...")
            try:
                # Reference audio: use stored speaker reference or default top-1 reference
                if active_state.speaker_reference_path and Path(active_state.speaker_reference_path).exists():
                    ref_audio = Path(active_state.speaker_reference_path)
                else:
                    ref_audio = self.zero_shot_manager.reference_manager.get_reference_audio(user_id)

                generated_audio = self.trainer.generate_fine_tuned_speech(
                    text=text,
                    reference_audio_path=ref_audio,
                    output_path=output_path,
                    checkpoint_path=Path(active_state.checkpoint_path),
                    config_path=Path(active_state.config_path),
                    vocab_path=Path(active_state.vocab_path),
                    language=language,
                )

                print(f"[PersonalizedVoiceManager] Fine-tuned generation successful: {generated_audio}")
                return generated_audio

            except Exception as exc:
                print(f"\n[WARNING] Fine-tuned generation failed for user '{user_id}': {exc}")
                print("[PersonalizedVoiceManager] Falling back to Zero-Shot XTTS generation...")

        # -------------------------------------------------------------
        # 2. Zero-Shot Fallback / Default Primary
        # -------------------------------------------------------------
        print(f"[PersonalizedVoiceManager] Generating speech via Zero-Shot XTTS for user '{user_id}'...")
        generated_audio = self.zero_shot_manager.generate_for_user(
            user_id=user_id,
            text=text,
            output_path=output_path,
            language=language,
        )

        return generated_audio
