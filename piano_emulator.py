"""
Piano Emulator - Reads hand commands from a text file and plays/visualizes them.

"""

import pygame
import numpy as np
import sys
import os
import math
import time
import threading
from pathlib import Path

pygame.mixer.pre_init(44100, -16, 2, 512)  # stereo
pygame.init()

# ─── Constants ────────────────────────────────────────────────────────────────
SCREEN_W, SCREEN_H = 1200, 700
FPS = 60

# Piano layout: 54 keys (0=C up to 53=F), spanning ~4.5 octaves
NUM_KEYS = 54

# White/black key pattern in an octave: C C# D D# E F F# G G# A A# B
IS_BLACK = [False, True, False, True, False, False, True, False, True, False, True, False]

PIANO_Y = SCREEN_H - 220
KEY_HEIGHT_WHITE = 200
KEY_HEIGHT_BLACK = 130
KEY_WIDTH_WHITE = 28
KEY_WIDTH_BLACK = 18

# Colors – dark concert hall aesthetic
BG_COLOR       = (8, 8, 12)
PIANO_BG       = (15, 15, 20)
WHITE_KEY      = (240, 238, 230)
WHITE_KEY_EDGE = (180, 178, 170)
BLACK_KEY      = (22, 20, 25)
BLACK_KEY_EDGE = (40, 38, 45)

LEFT_ACTIVE    = (255, 200, 80)   # gold – left hand
RIGHT_ACTIVE   = (100, 200, 255)  # cyan – right hand
BOTH_ACTIVE    = (200, 120, 255)  # purple – both

PARTICLE_COLORS_L = [(255, 220, 100), (255, 180, 50),  (255, 240, 130)]
PARTICLE_COLORS_R = [(80,  190, 255), (120, 220, 255), (60,  160, 220)]

FONT_PATH = None  # will use pygame default

# ─── Audio synthesis ──────────────────────────────────────────────────────────
SAMPLE_RATE = 44100
NOTE_CACHE = {}

def midi_to_freq(midi_note):
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))

def key_to_midi(key_index):
    # Key 0 = C3 (midi 48)
    return 48 + key_index

def synthesize_note(midi_note, duration_s=2.0, volume=0.5):
    """Piano-like tone using additive synthesis + envelope."""
    freq = midi_to_freq(midi_note)
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), endpoint=False)

    # Additive harmonics with natural falloff
    wave = np.zeros(len(t))
    harmonics = [1, 2, 3, 4, 5, 6, 8]
    amps     = [1.0, 0.5, 0.25, 0.15, 0.08, 0.04, 0.02]
    for h, a in zip(harmonics, amps):
        if freq * h < SAMPLE_RATE / 2:
            wave += a * np.sin(2 * np.pi * freq * h * t)

    # ADSR envelope
    attack  = min(0.01, duration_s * 0.05)
    decay   = min(0.05, duration_s * 0.1)
    sustain = 0.6
    release = min(0.3, duration_s * 0.4)

    envelope = np.ones(len(t))
    atk_s = int(attack * SAMPLE_RATE)
    dec_s = int(decay * SAMPLE_RATE)
    rel_s = int(release * SAMPLE_RATE)

    if atk_s > 0:
        envelope[:atk_s] = np.linspace(0, 1, atk_s)
    if dec_s > 0 and atk_s + dec_s < len(t):
        envelope[atk_s:atk_s+dec_s] = np.linspace(1, sustain, dec_s)
    if atk_s + dec_s < len(t):
        envelope[atk_s+dec_s:max(atk_s+dec_s, len(t)-rel_s)] = sustain
    if rel_s > 0 and len(t) - rel_s >= 0:
        envelope[len(t)-rel_s:] = np.linspace(sustain, 0, rel_s)

    wave *= envelope * volume

    # Normalize and convert to 16-bit
    wave = np.clip(wave, -1, 1)
    mono = (wave * 32767).astype(np.int16)
    # Make stereo (2D) to work regardless of whether mixer is mono or stereo
    audio = np.column_stack((mono, mono))
    return audio

def get_sound(midi_note, duration_s=1.5):
    key = (midi_note, round(duration_s, 2))
    if key not in NOTE_CACHE:
        audio = synthesize_note(midi_note, duration_s)
        sound = pygame.sndarray.make_sound(audio)
        NOTE_CACHE[key] = sound
    return NOTE_CACHE[key]

# ─── Key layout calculation ───────────────────────────────────────────────────
def build_key_layout():
    """Return list of (x, y, w, h, is_black) for each of the 54 keys."""
    layout = []
    white_x = 0
    white_count = 0

    # First pass: white keys
    for i in range(NUM_KEYS):
        octave_pos = i % 12
        if not IS_BLACK[octave_pos]:
            layout.append({'x': white_x, 'y': PIANO_Y,
                           'w': KEY_WIDTH_WHITE, 'h': KEY_HEIGHT_WHITE,
                           'is_black': False, 'index': i})
            white_x += KEY_WIDTH_WHITE
            white_count += 1
        else:
            layout.append(None)  # placeholder

    PIANO_WIDTH = white_x
    PIANO_OFFSET_X = (SCREEN_W - PIANO_WIDTH) // 2

    # Resolve black key positions
    white_x = 0
    for i in range(NUM_KEYS):
        octave_pos = i % 12
        if IS_BLACK[octave_pos]:
            # Black key sits between the whites on either side
            # Find the previous white key's x
            prev_white_x = white_x - KEY_WIDTH_WHITE
            bx = PIANO_OFFSET_X + prev_white_x + KEY_WIDTH_WHITE - KEY_WIDTH_BLACK // 2
            layout[i] = {'x': bx, 'y': PIANO_Y,
                         'w': KEY_WIDTH_BLACK, 'h': KEY_HEIGHT_BLACK,
                         'is_black': True, 'index': i}
        else:
            if layout[i] is not None:
                layout[i]['x'] += PIANO_OFFSET_X
            white_x += KEY_WIDTH_WHITE

    return layout, PIANO_OFFSET_X, PIANO_WIDTH

# ─── Command parsing ──────────────────────────────────────────────────────────
def parse_commands(filepath):
    """
    Each line: type, data0, data1, data2, data3, data4 (comma-separated)
    Type 1 = Set Tempo, data0 = BPM
    Type 0 = Play note:
      data0 = hand (0=left, 1=right)
      data1 = fingers active (decimal of 6-bit binary)
      data2 = key position (base key for the hand)
      data3 = duration (1=whole, 2=half, 4=quarter, 8=eighth, 16=sixteenth)
      data4 = unused
    """
    commands = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Support both space and comma separated
            if ',' in line:
                raw_parts = line.split(',')
            else:
                raw_parts = line.split()
            if len(raw_parts) < 4:
                continue
            cmd_type = int(raw_parts[0])
            if cmd_type == 0:
                # Play note: type hand fingers base_key duration
                hand = int(raw_parts[1])  # file: 0=left, 1=right
                # Finger byte: accept 6-bit binary string (e.g. "111111") or decimal
                finger_str = raw_parts[2]
                if all(c in '01' for c in finger_str) and len(finger_str) == 6:
                    fingers = int(finger_str, 2)
                else:
                    fingers = int(finger_str)
                base_key = int(raw_parts[3]) #AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA THIS IS THE OFFSET
                dur_val = int(raw_parts[4]) if len(raw_parts) > 4 else 4
                commands.append([cmd_type, hand, fingers, base_key, dur_val])
            elif cmd_type == 1:
                # Set tempo
                bpm = int(raw_parts[1])
                commands.append([cmd_type, bpm])
            else:
                commands.append([int(x) for x in raw_parts])
    return commands

IS_BLACK = [False, True, False, True, False, False, True, False, True, False, True, False]

# ── Finger spacing (cumulative semitone offsets from finger 1) ────────────────
# RH: all fingers a half step apart → offsets 0,1,2,3,4,5
# LH: half, whole, half, whole, half between consecutive fingers
#     f1→f2 = 1, f2→f3 = 2, f3→f4 = 1, f4→f5 = 2, f5→f6 = 1
#     cumulative offsets from f1: 0, 1, 3, 4, 6, 7
RH_FINGER_OFFSETS = [0, 1, 2, 3, 4, 5]
LH_FINGER_OFFSETS = [0, 1, 3, 4, 6, 7]

def fingers_to_keys(base_key, finger_byte, hand):
    """
    finger_byte is a 6-bit number. Bit 5 (MSB) = finger 1 (leftmost).
    base_key is the key where the leftmost *active* finger sits.
    If no fingers are active, base_key is treated as finger 1's position.
    hand: 0 = left (LH_FINGER_OFFSETS), 1 = right (RH_FINGER_OFFSETS).
    """
    offsets = LH_FINGER_OFFSETS if hand == 0 else RH_FINGER_OFFSETS

    # Find which finger is the leftmost active one (highest bit = finger 1)
    leftmost_active_f = None
    for f in range(6):
        bit = 5 - f
        if finger_byte & (1 << bit):
            leftmost_active_f = f
            break

    # Shift so that leftmost active finger lands on base_key.
    # If no finger is active, finger 1 sits at base_key (offset 0).
    anchor_offset = offsets[leftmost_active_f] if leftmost_active_f is not None else 0
    finger1_key = base_key - anchor_offset

    active_keys = []
    all_keys = []
    for f in range(6):
        k = finger1_key + offsets[f]
        all_keys.append(k if 0 <= k < NUM_KEYS else None)
        bit = 5 - f
        if 0 <= k < NUM_KEYS and (finger_byte & (1 << bit)):
            active_keys.append(k)
    return active_keys, all_keys

# ─── Particles ────────────────────────────────────────────────────────────────
class Particle:
    def __init__(self, x, y, color, key_x, key_w):
        self.x = x + np.random.randint(-key_w//3, key_w//3)
        self.y = y
        self.vy = -np.random.uniform(2, 6)
        self.vx = np.random.uniform(-1, 1)
        self.size = np.random.uniform(3, 8)
        self.life = 1.0
        self.decay = np.random.uniform(0.015, 0.04)
        self.color = color

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.12  # gravity
        self.life -= self.decay
        self.size *= 0.97

    def draw(self, surf):
        if self.life <= 0 or self.size < 0.5:
            return
        alpha = int(self.life * 255)
        r, g, b = self.color
        s = max(1, int(self.size))
        # Draw glow
        for offset in range(3, 0, -1):
            glow_surf = pygame.Surface((s*2+offset*4, s*2+offset*4), pygame.SRCALPHA)
            ga = max(0, int(alpha * 0.3 / offset))
            pygame.draw.circle(glow_surf, (r, g, b, ga),
                               (s+offset*2, s+offset*2), s+offset*2)
            surf.blit(glow_surf, (int(self.x)-s-offset*2, int(self.y)-s-offset*2))
        dot_surf = pygame.Surface((s*2, s*2), pygame.SRCALPHA)
        pygame.draw.circle(dot_surf, (r, g, b, alpha), (s, s), s)
        surf.blit(dot_surf, (int(self.x)-s, int(self.y)-s))


class FallingNote:
    """Rousseau-style falling note bar."""
    def __init__(self, key_x, key_w, is_black, color, fall_duration_frames):
        self.x = key_x
        self.w = key_w - 2
        self.h = max(10, fall_duration_frames * 0.4)
        self.y = PIANO_Y - 20  # start just above piano
        self.vy = 4.5
        self.color = color
        self.alpha = 220
        self.is_black = is_black

    def update(self):
        self.y -= self.vy  # rise upward
        self.alpha = max(0, self.alpha - 1.5)

    def draw(self, surf):
        if self.alpha <= 0:
            return
        r, g, b = self.color
        note_surf = pygame.Surface((self.w, int(self.h)), pygame.SRCALPHA)
        # Gradient fill
        for row in range(int(self.h)):
            fade = 1.0 - row / self.h * 0.3
            row_alpha = int(self.alpha * fade)
            pygame.draw.line(note_surf, (r, g, b, row_alpha), (0, row), (self.w, row))
        # Bright top edge
        pygame.draw.line(note_surf, (min(255,r+60), min(255,g+60), min(255,b+60), self.alpha),
                         (0, 0), (self.w, 0), 2)
        surf.blit(note_surf, (int(self.x), int(self.y - self.h)))


# ─── Main App ─────────────────────────────────────────────────────────────────
class PianoEmulator:
    def __init__(self, filepath):
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("Piano Emulator")
        self.clock = pygame.time.Clock()

        self.key_layout, self.piano_offset_x, self.piano_width = build_key_layout()
        self.commands = parse_commands(filepath)
        self.filepath = filepath

        self.bpm = 120
        self.active_keys = {}  # key_index -> 'left'|'right'|'both'
        self.particles = []
        self.falling_notes = []

        # Hand position state (always updated, even for rests)
        # Each entry: {'base_key': int, 'all_keys': list, 'is_rest': bool}
        self.hand_state = {0: None, 1: None}  # 0=left, 1=right

        self.playing = False
        self.play_thread = None
        self.cmd_index = 0
        self.done = False

        # Font setup
        self.font_large = pygame.font.SysFont("Georgia", 28, bold=True)
        self.font_med   = pygame.font.SysFont("Georgia", 18)
        self.font_small = pygame.font.SysFont("Courier New", 13)
        self.font_title = pygame.font.SysFont("Georgia", 42, bold=True)

        # Pre-synthesize a few notes for speed
        self._preload_sounds()

        self.status_msg = "Press SPACE to play"
        self.current_cmd_display = ""

        # Reflections surface
        self.piano_surface = None

        print(f"Loaded {len(self.commands)} commands from {filepath}")
        print(f"Starting playback at {self.bpm} BPM")

    def _preload_sounds(self):
        print("Pre-synthesizing piano sounds...")
        for i in range(0, NUM_KEYS, 6):
            get_sound(key_to_midi(i), 1.5)
        print("Ready.")

    def beat_duration(self, note_value):
        """Duration in seconds for a note value (1=whole, 2=half, 4=quarter...)"""
        beats_per_whole = 4.0
        whole_duration = beats_per_whole * (60.0 / self.bpm)
        return whole_duration / note_value

    def _build_timed_events(self):
        """Pre-compute start times for each command so both hands play together.
        Each hand has its own timeline; tempo changes apply globally."""
        bpm = self.bpm
        hand_time = {0: 0.0, 1: 0.0}  # seconds
        timed = []
        for cmd in self.commands:
            if cmd[0] == 1:
                bpm = cmd[1]
                timed.append((min(hand_time.values()), cmd))
            elif cmd[0] == 0:
                hand = cmd[1]
                start = hand_time[hand]
                dur_val = cmd[4]
                whole_dur = 4.0 * (60.0 / bpm)
                dur_s = whole_dur / dur_val
                timed.append((start, cmd, dur_s))
                hand_time[hand] = start + dur_s
        timed.sort(key=lambda x: x[0])
        return timed

    def play_sequence(self):
        """Runs in a background thread, fires commands at computed timestamps."""
        timed = self._build_timed_events()
        t_start = time.time()

        for entry in timed:
            if not self.playing:
                break

            start_time = entry[0]
            cmd = entry[1]

            # Wait until it's time
            now = time.time() - t_start
            if start_time > now:
                time.sleep(start_time - now)

            if cmd[0] == 1:
                self.bpm = cmd[1]
                self.status_msg = f"Tempo: {self.bpm} BPM"
                self.current_cmd_display = f"SET TEMPO {self.bpm} BPM"
                continue

            # Play note command
            hand     = cmd[1]
            fingers  = cmd[2]
            base_key = cmd[3]
            dur_val  = cmd[4]
            duration_s = entry[2]

            active, all_keys = fingers_to_keys(base_key, fingers, hand)
            hand_str = "LEFT" if hand == 0 else "RIGHT"
            dur_names = {1:"whole", 2:"half", 4:"quarter", 8:"eighth", 16:"sixteenth"}
            dur_name = dur_names.get(dur_val, str(dur_val))
            spacing = f"LH offsets {LH_FINGER_OFFSETS}" if hand == 0 else f"RH offsets {RH_FINGER_OFFSETS}"
            self.current_cmd_display = (f"{hand_str} hand | fingers={bin(fingers)[2:].zfill(6)} "
                                        f"| base={base_key} | {dur_name} | {spacing}")

            # Update hand position (even for rests)
            self.hand_state[hand] = {
                'base_key': base_key,
                'all_keys': all_keys,
                'is_rest': fingers == 0
            }

            # Clear previous active notes for this hand
            color_side = 'left' if hand == 0 else 'right'
            to_remove = [k for k, v in self.active_keys.items() if v == color_side]
            for k in to_remove:
                del self.active_keys[k]

            if fingers > 0:
                for k in active:
                    if k in self.active_keys:
                        self.active_keys[k] = 'both'
                    else:
                        self.active_keys[k] = color_side

                    kl = self.key_layout[k]
                    if kl:
                        cx = kl['x'] + kl['w'] // 2
                        cy = kl['y'] + (kl['h'] * 0.8 if not kl['is_black'] else kl['h'] * 0.9)
                        pcolors = PARTICLE_COLORS_L if hand == 0 else PARTICLE_COLORS_R
                        hcolor  = LEFT_ACTIVE  if hand == 0 else RIGHT_ACTIVE
                        for _ in range(8):
                            self.particles.append(
                                Particle(cx, cy, pcolors[np.random.randint(len(pcolors))],
                                         kl['x'], kl['w']))
                        fall_frames = int(duration_s * FPS)
                        self.falling_notes.append(
                            FallingNote(kl['x'] + 1, kl['w'], kl['is_black'],
                                        hcolor, fall_frames))

                    sound = get_sound(key_to_midi(k), max(0.2, duration_s * 0.9))
                    sound.play()

                self.status_msg = f"Playing: {len(active)} note(s)"
            else:
                self.status_msg = f"Rest ({dur_name})"

        self.playing = False
        self.active_keys.clear()
        self.hand_state = {0: None, 1: None}
        self.status_msg = "Playback complete. Press SPACE to replay."
        self.done = True

    def start_playback(self):
        if self.play_thread and self.play_thread.is_alive():
            return
        self.playing = True
        self.done = False
        self.active_keys.clear()
        self.hand_state = {0: None, 1: None}
        self.particles.clear()
        self.falling_notes.clear()
        self.play_thread = threading.Thread(target=self.play_sequence, daemon=True)
        self.play_thread.start()

    def stop_playback(self):
        self.playing = False
        self.active_keys.clear()
        self.hand_state = {0: None, 1: None}
        pygame.mixer.stop()

    def draw_background(self):
        self.screen.fill(BG_COLOR)
        # Subtle vignette effect
        for i in range(8):
            alpha = 20 - i * 2
            if alpha <= 0: break
            rect = pygame.Rect(i*8, i*6, SCREEN_W-i*16, SCREEN_H-i*12)
            s = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
            pygame.draw.rect(s, (0, 0, 0, alpha), (0, 0, SCREEN_W, SCREEN_H))
            pygame.draw.rect(s, (0, 0, 0, 0), rect)
            self.screen.blit(s, (0, 0))

    def draw_piano(self):
        # Piano body shadow
        shadow_rect = pygame.Rect(self.piano_offset_x - 4, PIANO_Y - 4,
                                  self.piano_width + 8, KEY_HEIGHT_WHITE + 8)
        pygame.draw.rect(self.screen, (5, 5, 8), shadow_rect, border_radius=4)

        # White keys first
        for kl in self.key_layout:
            if kl is None or kl['is_black']:
                continue
            idx = kl['index']
            color = WHITE_KEY

            active_state = self.active_keys.get(idx)
            if active_state == 'left':
                color = LEFT_ACTIVE
            elif active_state == 'right':
                color = RIGHT_ACTIVE
            elif active_state == 'both':
                color = BOTH_ACTIVE

            rect = pygame.Rect(kl['x'], kl['y'], kl['w'] - 1, kl['h'])
            pygame.draw.rect(self.screen, color, rect, border_radius=2)
            pygame.draw.rect(self.screen, WHITE_KEY_EDGE, rect, 1, border_radius=2)

            # Key number label (every C)
            if idx % 12 == 0:
                octave = idx // 12
                lbl = self.font_small.render(f"C{octave+3}", True, (100, 100, 110))
                self.screen.blit(lbl, (kl['x'] + 2, kl['y'] + kl['h'] - 18))

        # Black keys on top
        for kl in self.key_layout:
            if kl is None or not kl['is_black']:
                continue
            idx = kl['index']
            color = BLACK_KEY

            active_state = self.active_keys.get(idx)
            if active_state == 'left':
                color = tuple(min(255, c + 100) for c in LEFT_ACTIVE)
                color = (int(LEFT_ACTIVE[0]*0.8), int(LEFT_ACTIVE[1]*0.7), int(LEFT_ACTIVE[2]*0.2))
            elif active_state == 'right':
                color = (int(RIGHT_ACTIVE[0]*0.3), int(RIGHT_ACTIVE[1]*0.6), int(RIGHT_ACTIVE[2]*0.9))
            elif active_state == 'both':
                color = (int(BOTH_ACTIVE[0]*0.7), int(BOTH_ACTIVE[1]*0.4), int(BOTH_ACTIVE[2]*0.9))

            rect = pygame.Rect(kl['x'], kl['y'], kl['w'], kl['h'])
            pygame.draw.rect(self.screen, color, rect, border_radius=2)
            pygame.draw.rect(self.screen, BLACK_KEY_EDGE, rect, 1, border_radius=2)

            # Shine line on black keys
            if not active_state:
                shine = pygame.Rect(kl['x'] + 3, kl['y'] + 4, 3, kl['h'] // 3)
                pygame.draw.rect(self.screen, (55, 52, 60), shine, border_radius=1)

    def draw_particles(self):
        for p in self.particles:
            p.draw(self.screen)

    def draw_falling_notes(self):
        for fn in self.falling_notes:
            fn.draw(self.screen)

    def draw_hand_positions(self):
        """Draw a bracket above the piano showing where each hand is positioned.
        Filled dot = actively playing. Dim dot = positioned but resting."""
        for hand_id, state in self.hand_state.items():
            if state is None:
                continue

            # hand 0 = LEFT (gold), hand 1 = RIGHT (cyan)
            color     = LEFT_ACTIVE  if hand_id == 0 else RIGHT_ACTIVE
            label_txt = "L"          if hand_id == 0 else "R"
            is_rest   = state['is_rest']
            all_keys  = state['all_keys']

            active_set = {k for k, v in self.active_keys.items()
                          if v == ('left' if hand_id == 0 else 'right') or v == 'both'}

            valid_kls = [self.key_layout[k] for k in all_keys
                         if k is not None and self.key_layout[k] is not None]
            if not valid_kls:
                continue

            span_x1 = min(kl['x'] for kl in valid_kls) - 4
            span_x2 = max(kl['x'] + kl['w'] for kl in valid_kls) + 4
            span_y  = PIANO_Y - 10

            bracket_alpha = 130 if is_rest else 210
            r, g, b = color
            line_surf = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
            pygame.draw.line(line_surf, (r, g, b, bracket_alpha),
                             (span_x1, span_y), (span_x2, span_y), 2)
            pygame.draw.line(line_surf, (r, g, b, bracket_alpha),
                             (span_x1, span_y - 5), (span_x1, span_y + 5), 2)
            pygame.draw.line(line_surf, (r, g, b, bracket_alpha),
                             (span_x2, span_y - 5), (span_x2, span_y + 5), 2)
            self.screen.blit(line_surf, (0, 0))

            label_surf = self.font_small.render(label_txt, True, color)
            self.screen.blit(label_surf, (span_x1 - label_surf.get_width() - 4,
                                          span_y - label_surf.get_height() // 2))

            for k in all_keys:
                if k is None:
                    continue
                kl = self.key_layout[k]
                if kl is None:
                    continue
                cx = kl['x'] + kl['w'] // 2
                finger_active = (not is_rest) and (k in active_set)
                if finger_active:
                    pygame.draw.circle(self.screen, color, (cx, span_y), 6)
                    pygame.draw.circle(self.screen, (255, 255, 255), (cx, span_y), 2)
                else:
                    dot_surf = pygame.Surface((16, 16), pygame.SRCALPHA)
                    dr = (80, 70, 30) if hand_id == 0 else (30, 60, 80)
                    pygame.draw.circle(dot_surf, (*dr, 140), (8, 8), 5)
                    pygame.draw.circle(dot_surf, (r, g, b, 80), (8, 8), 5, 1)
                    self.screen.blit(dot_surf, (cx - 8, span_y - 8))

    def draw_hud(self):
        # Title
        title = self.font_title.render("File Tester", True, (220, 200, 150))
        self.screen.blit(title, (SCREEN_W // 2 - title.get_width() // 2, 20))

        # Subtitle / file
        sub = self.font_med.render(f"  {os.path.basename(self.filepath)}  ", True, (120, 110, 100))
        self.screen.blit(sub, (SCREEN_W // 2 - sub.get_width() // 2, 68))

        # BPM display
        bpm_surf = self.font_large.render(f"♩ = {self.bpm}", True, (200, 180, 100))
        self.screen.blit(bpm_surf, (30, 20))

        # Status
        status_surf = self.font_med.render(self.status_msg, True, (160, 200, 160))
        self.screen.blit(status_surf, (30, 56))

        # Current command
        cmd_surf = self.font_small.render(self.current_cmd_display, True, (100, 140, 180))
        self.screen.blit(cmd_surf, (30, 82))

        # Controls legend
        controls = [
            ("SPACE", "Play / Restart"),
            ("S", "Stop"),
            ("ESC", "Quit"),
        ]
        cx = SCREEN_W - 200
        for i, (key, desc) in enumerate(controls):
            k_surf = self.font_small.render(key, True, (200, 200, 100))
            d_surf = self.font_small.render(f"  {desc}", True, (120, 120, 140))
            self.screen.blit(k_surf, (cx, 20 + i * 20))
            self.screen.blit(d_surf, (cx + k_surf.get_width(), 20 + i * 20))

        # Legend dots
        for color, label, offset in [(LEFT_ACTIVE, "Left hand", 0),
                                      (RIGHT_ACTIVE, "Right hand", 90),
                                      (BOTH_ACTIVE, "Both hands", 180)]:
            pygame.draw.circle(self.screen, color, (SCREEN_W//2 - 90 + offset, 110), 7)
            ls = self.font_small.render(label, True, (150, 150, 160))
            self.screen.blit(ls, (SCREEN_W//2 - 75 + offset, 103))

        # Progress bar
        if self.commands:
            progress = self.cmd_index / max(1, len(self.commands))
            bar_w = self.piano_width
            bar_x = self.piano_offset_x
            bar_y = PIANO_Y - 18
            pygame.draw.rect(self.screen, (30, 30, 40), (bar_x, bar_y, bar_w, 5), border_radius=2)
            prog_w = int(bar_w * min(1.0, progress))
            if prog_w > 0:
                pygame.draw.rect(self.screen, (180, 160, 80), (bar_x, bar_y, prog_w, 5), border_radius=2)

    def draw_reflection(self):
        """Subtle piano key reflection below the piano."""
        refl_h = 40
        refl_surf = pygame.Surface((self.piano_width, refl_h), pygame.SRCALPHA)
        for kl in self.key_layout:
            if kl is None or kl['is_black']:
                continue
            idx = kl['index']
            active_state = self.active_keys.get(idx)
            if active_state:
                color = (LEFT_ACTIVE if active_state == 'left' else
                         RIGHT_ACTIVE if active_state == 'right' else BOTH_ACTIVE)
                for row in range(refl_h):
                    alpha = int((1 - row / refl_h) * 60)
                    pygame.draw.line(refl_surf, (*color, alpha),
                                     (kl['x'] - self.piano_offset_x, row),
                                     (kl['x'] - self.piano_offset_x + kl['w'] - 1, row))
        self.screen.blit(refl_surf, (self.piano_offset_x, PIANO_Y + KEY_HEIGHT_WHITE))

    def run(self):
        running = True
        while running:
            self.clock.tick(FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.stop_playback()
                        time.sleep(0.05)
                        self.start_playback()
                    elif event.key == pygame.K_s:
                        self.stop_playback()
                        self.status_msg = "Stopped. Press SPACE to play."

            # Update particles
            self.particles = [p for p in self.particles if p.life > 0]
            for p in self.particles:
                p.update()

            # Update falling notes
            self.falling_notes = [fn for fn in self.falling_notes if fn.alpha > 0]
            for fn in self.falling_notes:
                fn.update()

            # Draw everything
            self.draw_background()
            self.draw_falling_notes()
            self.draw_piano()
            self.draw_hand_positions()
            self.draw_reflection()
            self.draw_particles()
            self.draw_hud()

            pygame.display.flip()

        self.stop_playback()
        pygame.quit()


# ─── Entry point ──────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        # Search in the same directory as this script for any .txt file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(script_dir, "commands.txt"),
        ]
        filepath = None
        # First try known names
        for c in candidates:
            if os.path.exists(c):
                filepath = c
                break
        # Then grab the first .txt found next to the script
        if not filepath:
            for f in sorted(os.listdir(script_dir)):
                if f.endswith(".txt"):
                    filepath = os.path.join(script_dir, f)
                    break
        if not filepath:
            print("Usage: python piano_emulator.py <commands_file.txt>")
            print(f"No .txt command file found in {script_dir}")
            sys.exit(1)
        print(f"Using command file: {filepath}")

    emulator = PianoEmulator(filepath)
    emulator.run()

if __name__ == "__main__":
    main()