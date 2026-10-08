"""Settings, read from environment variables (or a .env file)."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    data_dir: Path = Path("data")
    # When set, the web UI asks for this password (HTTP basic auth, any username).
    app_password: str = ""

    # Speech-to-text: "openai" (hosted Whisper) or "whisperx" (local GPU).
    transcriber: str = "openai"
    openai_api_key: str = ""
    openai_transcribe_model: str = "whisper-1"

    # Vocal / background separation: "demucs" (local) or "none".
    separator: str = "demucs"

    # Translation with Claude.
    anthropic_api_key: str = ""
    claude_model: str = "claude-opus-5-5"
    claude_effort: str = "medium"

    # English voice with ElevenLabs.
    elevenlabs_api_key: str = ""
    elevenlabs_model: str = "eleven_multilingual_v2"
    # Used when voice cloning is off or fails.
    elevenlabs_default_voice_id: str = "21m00Tcm4TlvDq8ikWAM"
    clone_voice: bool = True

    # Lip sync with sync.so. Needs public_base_url so sync.so can fetch the files.
    sync_api_key: str = ""
    sync_model: str = "lipsync-2"  # cheapest current model; "sync-3" or "lipsync-2-pro" for higher quality
    public_base_url: str = ""

    # Timing: how much faster than normal a dubbed line may be played to fit its slot.
    max_speedup: float = 1.35


@lru_cache
def get_settings() -> Settings:
    return Settings()
