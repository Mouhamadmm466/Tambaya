import logging
import os
import tempfile
import time
from dataclasses import dataclass

from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

# Segments with avg_logprob below this are too uncertain to trust.
_MIN_AVG_LOG_PROB = -1.2
# Segments with no_speech_prob above this are likely silence or noise.
_MAX_NO_SPEECH_PROB = 0.6


@dataclass
class TranscriptionResult:
    text: str
    language: str
    avg_log_prob: float
    no_speech_prob: float
    duration_ms: int

    @property
    def is_usable(self) -> bool:
        return (
            bool(self.text)
            and self.avg_log_prob >= _MIN_AVG_LOG_PROB
            and self.no_speech_prob <= _MAX_NO_SPEECH_PROB
        )


class WhisperTranscriber:
    def __init__(self):
        model_size = os.environ.get("WHISPER_MODEL_SIZE", "large-v3")
        compute_type = os.environ.get("WHISPER_COMPUTE_TYPE", "float16")
        logger.info("Loading faster-whisper %s (%s) ...", model_size, compute_type)
        self._model = WhisperModel(model_size, device="cuda", compute_type=compute_type)
        logger.info("Whisper model ready.")

    def transcribe(self, audio_bytes: bytes) -> TranscriptionResult:
        """Synchronous — call via asyncio.to_thread() from async handlers."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name

        try:
            t0 = time.monotonic()
            segments_gen, _ = self._model.transcribe(
                tmp_path,
                language="ha",
                beam_size=5,
                vad_filter=True,
            )
            segments = list(segments_gen)
            duration_ms = int((time.monotonic() - t0) * 1000)
        finally:
            os.unlink(tmp_path)

        text = " ".join(seg.text.strip() for seg in segments).strip()

        if segments:
            avg_log_prob = sum(s.avg_logprob for s in segments) / len(segments)
            no_speech_prob = max(s.no_speech_prob for s in segments)
        else:
            avg_log_prob = -999.0
            no_speech_prob = 1.0

        return TranscriptionResult(
            text=text,
            language="ha",
            avg_log_prob=avg_log_prob,
            no_speech_prob=no_speech_prob,
            duration_ms=duration_ms,
        )
