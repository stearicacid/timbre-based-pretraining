import logging
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import pretty_midi
from mido import MidiFile

logger = logging.getLogger(__name__)


def tick_to_second(ticks: int, ticks_per_beat: int, tempo_microseconds_per_beat: int) -> float:
    seconds_per_beat = tempo_microseconds_per_beat / 1_000_000
    beats = ticks / ticks_per_beat
    return beats * seconds_per_beat


def parse_midi_notes_mido(midi_file_path: str) -> List[Dict[str, float]]:
    """Parse MIDI with mido and fallback to pretty_midi for malformed files."""
    notes: List[Dict[str, float]] = []
    try:
        mid = MidiFile(midi_file_path, clip=True)
        ticks_per_beat = mid.ticks_per_beat
        current_tempo = 500000  # 120 BPM
        active_notes = {}
        current_program = defaultdict(lambda: 0)

        for track in mid.tracks:
            track_time_in_ticks = 0
            track_tempo = current_tempo

            for msg in track:
                track_time_in_ticks += msg.time

                if msg.type == "set_tempo":
                    track_tempo = msg.tempo
                    current_tempo = msg.tempo
                elif msg.type == "program_change":
                    current_program[msg.channel] = msg.program
                elif msg.type == "note_on" and msg.velocity > 0:
                    if getattr(msg, "channel", 0) == 9:
                        continue
                    key = (msg.note, msg.channel)
                    if key not in active_notes:
                        active_notes[key] = {
                            "onset_ticks": track_time_in_ticks,
                            "velocity": msg.velocity,
                            "program": current_program[msg.channel],
                            "tempo": track_tempo,
                            "channel": msg.channel,
                        }
                elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                    if getattr(msg, "channel", 0) == 9:
                        continue
                    key = (msg.note, msg.channel)
                    if key in active_notes:
                        note_info = active_notes.pop(key)
                        onset_seconds = tick_to_second(
                            note_info["onset_ticks"], ticks_per_beat, note_info["tempo"]
                        )
                        offset_seconds = tick_to_second(
                            track_time_in_ticks, ticks_per_beat, track_tempo
                        )
                        notes.append(
                            {
                                "onset": onset_seconds,
                                "offset": offset_seconds,
                                "pitch": msg.note,
                                "velocity": note_info["velocity"],
                                "program": note_info["program"],
                                "channel": note_info["channel"],
                            }
                        )

        final_time_seconds = tick_to_second(track_time_in_ticks, ticks_per_beat, current_tempo)
        for key, note_info in active_notes.items():
            if note_info.get("channel", 0) == 9:
                continue
            onset_seconds = tick_to_second(note_info["onset_ticks"], ticks_per_beat, note_info["tempo"])
            notes.append(
                {
                    "onset": onset_seconds,
                    "offset": final_time_seconds,
                    "pitch": key[0],
                    "velocity": note_info["velocity"],
                    "program": note_info["program"],
                    "channel": note_info["channel"],
                }
            )
        return notes
    except Exception:
        logger.warning("mido parse failed for %s; falling back to pretty_midi", midi_file_path)

    try:
        pm = pretty_midi.PrettyMIDI(midi_file_path)
        for inst in pm.instruments:
            if inst.is_drum:
                continue
            for note in inst.notes:
                notes.append(
                    {
                        "onset": note.start,
                        "offset": note.end,
                        "pitch": note.pitch,
                        "velocity": note.velocity,
                        "program": inst.program,
                        "channel": 0,
                    }
                )
        return notes
    except Exception as exc:
        logger.warning("Failed to parse MIDI %s: %s", midi_file_path, exc)
        return []


def parse_split_info(split_info_data) -> List[Dict[str, str]]:
    """Normalize split info to [{'path': ..., 'split': ...}, ...]."""
    parsed: List[Dict[str, str]] = []

    if isinstance(split_info_data, list):
        for item in split_info_data:
            if isinstance(item, dict) and "path" in item:
                entry = item.copy()
                entry.setdefault("split", "train")
                parsed.append(entry)
            elif isinstance(item, str):
                parsed.append({"path": item, "split": "train"})
            else:
                logger.warning("Ignoring unknown split item: %r", item)
        return parsed

    if isinstance(split_info_data, dict):
        for split_name, items in split_info_data.items():
            if not isinstance(items, Sequence):
                logger.warning("Split '%s' is not a list; skipped", split_name)
                continue
            for item in items:
                if isinstance(item, dict) and "path" in item:
                    entry = item.copy()
                    entry["split"] = split_name
                    parsed.append(entry)
                elif isinstance(item, str):
                    parsed.append({"path": item, "split": split_name})
                else:
                    logger.warning("Ignoring unknown item in split '%s': %r", split_name, item)
        return parsed

    raise TypeError(f"Unsupported split_info format: {type(split_info_data)}")


def load_dismiss_midis(path: Optional[str]) -> set:
    if not path:
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}
