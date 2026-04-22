-- Phase 3: add transcription metadata columns to call_logs.
-- Run manually: docker exec -it namu_db psql -U namu_user -d namu_tambaya -f /path/to/001_add_transcription_columns.sql

ALTER TABLE call_logs ADD COLUMN IF NOT EXISTS transcription_succeeded BOOLEAN;
ALTER TABLE call_logs ADD COLUMN IF NOT EXISTS transcription_time_ms INTEGER;
ALTER TABLE call_logs ADD COLUMN IF NOT EXISTS no_speech_prob FLOAT;
ALTER TABLE call_logs ADD COLUMN IF NOT EXISTS avg_log_prob FLOAT;
