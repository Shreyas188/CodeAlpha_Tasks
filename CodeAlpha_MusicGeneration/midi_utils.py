"""
CodeAlpha - Music Generation with AI
MIDI utilities: Convert note sequences to MIDI files
Author: CodeAlpha Intern
"""
import os
from midiutil import MIDIFile

# MIDI note number mapping: note name -> MIDI number
NOTE_TO_MIDI_NUM = {
    "C3": 48, "C#3": 49, "Db3": 49, "D3": 50, "D#3": 51, "Eb3": 51,
    "E3": 52, "F3": 53, "F#3": 54, "Gb3": 54, "G3": 55, "G#3": 56,
    "Ab3": 56, "A3": 57, "A#3": 58, "Bb3": 58, "B3": 59,
    "C4": 60, "C#4": 61, "Db4": 61, "D4": 62, "D#4": 63, "Eb4": 63,
    "E4": 64, "F4": 65, "F#4": 66, "Gb4": 66, "G4": 67, "G#4": 68,
    "Ab4": 68, "A4": 69, "A#4": 70, "Bb4": 70, "B4": 71,
    "C5": 72, "C#5": 73, "Db5": 73, "D5": 74, "D#5": 75, "Eb5": 75,
    "E5": 76, "F5": 77, "F#5": 78, "Gb5": 78, "G5": 79, "G#5": 80,
    "Ab5": 80, "A5": 81, "A#5": 82, "Bb5": 82, "B5": 83,
}


def sequence_to_midi(
    sequence,
    output_path: str,
    tempo: int = 120,
    velocity: int = 80,
    channel: int = 0,
    program: int = 0  # 0=Piano, 40=Violin, 73=Flute
):
    """
    Convert a list of (note_name, duration) tuples to a MIDI file.
    
    Args:
        sequence: List of (note_name, duration) tuples
                  note_name can be 'rest' for silence
        output_path: Path to save the MIDI file (.mid)
        tempo: BPM (beats per minute)
        velocity: Note velocity (0-127)
        channel: MIDI channel (0-15)
        program: MIDI instrument program number
    """
    midi = MIDIFile(1)  # 1 track
    track = 0
    time = 0  # Start time in beats
    
    midi.addTempo(track, 0, tempo)
    midi.addProgramChange(track, channel, 0, program)
    
    for note_name, duration in sequence:
        if note_name == "rest":
            # Just advance time for rests
            time += duration
            continue
        
        midi_num = NOTE_TO_MIDI_NUM.get(note_name)
        if midi_num is None:
            # Try common enharmonic lookup
            enharmonics = {
                "Db": "C#", "Eb": "D#", "Fb": "E",
                "Gb": "F#", "Ab": "G#", "Bb": "A#", "Cb": "B"
            }
            for flat, sharp in enharmonics.items():
                if flat in note_name:
                    alt = note_name.replace(flat, sharp)
                    midi_num = NOTE_TO_MIDI_NUM.get(alt)
                    if midi_num:
                        break
        
        if midi_num is None:
            # Default to middle C
            midi_num = 60
        
        midi.addNote(track, channel, midi_num, time, duration, velocity)
        time += duration
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    with open(output_path, "wb") as f:
        midi.writeFile(f)
    
    return output_path, time  # Return path and total duration in beats


def get_midi_info(sequence, tempo: int = 120):
    """Calculate MIDI info for display."""
    total_beats = sum(dur for _, dur in sequence)
    total_seconds = (total_beats / tempo) * 60
    note_count = sum(1 for note, _ in sequence if note != "rest")
    rest_count = sum(1 for note, _ in sequence if note == "rest")
    
    return {
        "total_beats": total_beats,
        "total_seconds": total_seconds,
        "note_count": note_count,
        "rest_count": rest_count,
        "minutes": int(total_seconds // 60),
        "seconds": int(total_seconds % 60)
    }


def play_midi_windows(midi_path: str):
    """Play a MIDI file on Windows using the built-in MIDI player."""
    import subprocess
    import sys
    
    if sys.platform == "win32":
        try:
            os.startfile(os.path.abspath(midi_path))
            return True
        except Exception as e:
            print(f"Could not open MIDI file: {e}")
            return False
    return False
