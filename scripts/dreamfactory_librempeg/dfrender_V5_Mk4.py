#!/usr/bin/env python3
"""Render a DreamFactory movie with explicit audio playback policies.

Usage: python3 dfrender_V5_Mk4_r6.py input.move output.mkv
No audio flags: strict slot replacement, scene interruption, and movie-end cuts.
--overlap-sounds: preserve finite SFX across replacements and scene changes.
--finish-audio: preserve finite tails stopped only by movie EOF and hold the
last image. Use both flags for Mk2/r5's overlapping finite-sound presentation.
Loop priority/replacement is retained in every mode; no retriggers are suppressed.

The patched ffmpeg/ffprobe pair is found in the current directory, beside this
script, or on PATH. MKV defaults to FFV1/FLAC; MP4 uses native MPEG-4/AAC.
VFR timestamps, continuous PCM mixing, and the FFmpeg aresample fallback remain.
These selectable policies do not establish full original-executable fidelity.
"""

import argparse
from array import array
import json
import math
import os
import re
import shutil
import shlex
import subprocess
import sys
import tempfile
import wave
from fractions import Fraction


def run_probe(ffprobe, path):
    result = subprocess.run(
        [ffprobe, "-v", "error", "-of", "json", "-show_streams",
         "-show_format", path], capture_output=True, text=True)
    if result.returncode:
        sys.exit(f"ffprobe failed:\n{result.stderr}")
    return json.loads(result.stdout)



def filter_help(ffmpeg, name):
    """Return filter help text, or an empty string when the filter is unavailable."""
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-h", f"filter={name}"],
        capture_output=True, text=True)
    output = (result.stdout or "") + (result.stderr or "")
    if result.returncode or not re.search(
            rf"^Filter {re.escape(name)}$", output, re.MULTILINE):
        return ""
    return output


def select_resampler(ffmpeg):
    """Prefer aresample; fall back to Librempeg's ardftsrc."""
    if filter_help(ffmpeg, "aresample"):
        return "aresample"
    if filter_help(ffmpeg, "ardftsrc"):
        return "ardftsrc"
    sys.exit("ffmpeg has neither the aresample nor ardftsrc audio resampler")


def uses_named_pan(ffmpeg):
    """Detect Librempeg's named pan options versus FFmpeg's args syntax."""
    help_text = filter_help(ffmpeg, "pan")
    return bool(re.search(r"^\s+layout\s+<", help_text, re.MULTILINE) and
                re.search(r"^\s+mix\s+", help_text, re.MULTILINE))



def video_encoder_time_base(streams):
    """Return the demuxed video time base as an encoder rational."""
    video = next((stream for stream in streams
                  if stream.get("codec_type") == "video"), None)
    time_base = (video or {}).get("time_base")
    if not time_base or "/" not in time_base:
        return None
    try:
        num, den = (int(part) for part in time_base.split("/", 1))
    except (TypeError, ValueError):
        return None
    if num <= 0 or den <= 0:
        return None
    return f"{num}:{den}"


def needs_even_padding(vcodec, streams):
    """libx264 4:2:0 output requires even coded dimensions."""
    if vcodec != "libx264":
        return False
    video = next((stream for stream in streams
                  if stream.get("codec_type") == "video"), None)
    if not video:
        return False
    try:
        width = int(video.get("width") or 0)
        height = int(video.get("height") or 0)
    except (TypeError, ValueError):
        return False
    return bool((width & 1) or (height & 1))


def add_video_encoding_options(command, streams, vcodec, filters=None):
    """Preserve the demuxed VFR clock and apply codec-only geometry fixes."""
    filters = list(filters or [])
    if needs_even_padding(vcodec, streams):
        filters.append("pad=ceil(iw/2)*2:ceil(ih/2)*2")

    if filters:
        command += ["-vf", ",".join(filters)]

    command += ["-fps_mode:v", "passthrough"]
    if vcodec != "copy":
        time_base = video_encoder_time_base(streams)
        if time_base:
            command += ["-enc_time_base:v", time_base]
    command += ["-c:v", vcodec]
    return command

def detect_version(info):
    names = set((info.get("format", {}).get("format_name") or "").split(","))
    if "cfdf_d5" in names:
        return 5
    if "cfdf" in names:
        return 4
    return None


def tags(stream):
    return {key.lower(): value for key, value in
            (stream.get("tags", {}) or {}).items()}


def number(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def seconds(stream, field):
    ticks = stream.get(field + "_ts")
    time_base = stream.get("time_base")
    if ticks is not None and time_base and "/" in time_base:
        num, den = time_base.split("/", 1)
        try:
            return Fraction(int(ticks) * int(num), int(den))
        except (TypeError, ValueError, ZeroDivisionError):
            pass
    try:
        return Fraction(str(stream.get(field) or 0))
    except (TypeError, ValueError, ZeroDivisionError):
        return Fraction(0)


def classify(streams, filename, include_untimed):
    base = os.path.basename(filename)
    audio = [stream for stream in streams if stream.get("codec_type") == "audio"]
    selected = []
    for index, stream in enumerate(audio):
        metadata = tags(stream)
        role = metadata.get("timeline")
        if role is None:
            if (stream.get("disposition", {}).get("default") or
                    metadata.get("title") == base):
                role = "background"
            elif seconds(stream, "start_time") > 0:
                role = "sfx"
            else:
                role = "untimed"
        if role in ("background", "sfx") or (include_untimed and role == "untimed"):
            if metadata.get("cfdf_d5_start_ticks") is not None:
                start = Fraction(number(metadata["cfdf_d5_start_ticks"]), 60)
            elif metadata.get("cfdf_d5_start_us") is not None:
                start = Fraction(number(metadata["cfdf_d5_start_us"]), 1000000)
            else:
                start = seconds(stream, "start_time")
            selected.append({
                "audio_index": index,
                "stream": stream,
                "tags": metadata,
                "role": role,
                "start": start,
                "duration": seconds(stream, "duration"),
            })
    return selected


def video_duration(streams):
    videos = [stream for stream in streams if stream.get("codec_type") == "video"]
    return max((seconds(stream, "duration") for stream in videos), default=0.0)


def d5_schedule(selected, movie_end, overlap_sounds=False):
    scheduled = []
    backgrounds = sorted((item for item in selected
                          if item["role"] == "background"),
                         key=lambda item: (item["start"], item["audio_index"]))
    effects = sorted((item for item in selected if item["role"] == "sfx"),
                     key=lambda item: (item["start"], item["audio_index"]))
    scene_starts = {
        (Fraction(number(item["tags"]["cfdf_d5_scene_start"]), 60)
         if item["tags"].get("cfdf_d5_scene_start") is not None
         else item["start"])
        for item in backgrounds
    }
    scene_starts.update(Fraction(number(item["tags"].get("cfdf_d5_scene_start")), 60)
                        for item in effects
                        if item["tags"].get("cfdf_d5_scene_start") is not None)
    scene_starts = sorted(start for start in scene_starts if start > 0)

    def scene_end(start):
        return next((boundary for boundary in scene_starts if boundary > start),
                    movie_end or math.inf)

    for index, item in enumerate(backgrounds):
        if item["tags"].get("cfdf_d5_end_us") is not None:
            tagged_end = Fraction(number(item["tags"]["cfdf_d5_end_us"]), 1000000)
        else:
            tagged_end = Fraction(number(item["tags"].get("cfdf_d5_end")), 60)
        end = tagged_end or (backgrounds[index + 1]["start"]
                             if index + 1 < len(backgrounds) else movie_end)
        if end <= item["start"]:
            end = item["start"] + item["duration"]
        if not number(item["tags"].get("cfdf_d5_loop")):
            end = min(end, item["start"] + item["duration"])
        record = dict(item)
        record.update(slot="2", key=None, end=end, dropped=False)
        scheduled.append(record)

    # Slots own finite sounds and loops alike. Rejected requests do not play.
    pool = [None, None]
    fixed = None
    active_scene = None

    def stop(record, when):
        if record is not None and record["end"] >= when:
            record["end"] = when
            record["interrupted"] = True

    def replace_sound(record, when):
        # Overlap preserves finite tails across replacement and scene changes.
        if record is not None and overlap_sounds and not record["loop"]:
            return
        stop(record, when)

    for item in effects:
        scene = number(item["tags"].get("cfdf_d5_scene"), -1)
        if active_scene is None:
            active_scene = scene
        elif scene != active_scene:
            for occupant in pool:
                replace_sound(occupant, item["start"])
            replace_sound(fixed, item["start"])
            pool = [None, None]
            fixed = None
            active_scene = scene

        loop = number(item["tags"].get("cfdf_d5_loop")) != 0
        boundary = (Fraction(number(item["tags"]["cfdf_d5_scene_end"]), 60)
                    if "cfdf_d5_scene_end" in item["tags"]
                    else scene_end(item["start"]))
        end = boundary if loop else min(boundary, item["start"] + item["duration"])
        if overlap_sounds and not loop:
            end = item["start"] + item["duration"]
        slot = item["tags"].get("cfdf_d5_slot", "independent")
        record = dict(item)
        record.update(slot=slot,
                      key=number(item["tags"].get("cfdf_d5_key")),
                      end=end, dropped=False, loop=loop)

        if slot == "3":
            replace_sound(fixed, item["start"])
            fixed = record
            scheduled.append(record)
            continue

        if slot != "pool":
            scheduled.append(record)
            continue

        for index in range(2):
            if pool[index] is not None and pool[index]["end"] <= item["start"]:
                pool[index] = None

        match = next((index for index in range(2)
                      if pool[index] is not None and
                      pool[index]["key"] == record["key"]), None)
        if match is not None:
            replace_sound(pool[match], item["start"])
            pool[match] = record
            record["slot"] = str(match)
            scheduled.append(record)
            continue

        free = next((index for index in range(2) if pool[index] is None), None)
        if free is not None:
            pool[free] = record
            record["slot"] = str(free)
            scheduled.append(record)
            continue

        replace = 0 if pool[0]["key"] < pool[1]["key"] else 1
        if record["key"] > pool[replace]["key"]:
            replace_sound(pool[replace], item["start"])
            pool[replace] = record
            record["slot"] = str(replace)
            scheduled.append(record)
        elif overlap_sounds and not loop:
            # Match the optional Mk2 finite-voice policy without taking a slot.
            record["slot"] = "pool-preserved"
            scheduled.append(record)
        else:
            record["dropped"] = True

    for item in selected:
        if item["role"] == "untimed":
            record = dict(item)
            record.update(slot="independent", key=None, start=0.0,
                          end=item["duration"], dropped=False)
            scheduled.append(record)

    for record in scheduled:
        if not math.isfinite(record["end"]):
            record["end"] = movie_end or record["start"] + record["duration"]
        # Untimed catalog export remains an explicit additive CLI option.
        if movie_end and record["role"] != "untimed":
            record["end"] = min(record["end"], movie_end)

    records = sorted((record for record in scheduled
                      if record["end"] > record["start"]),
                     key=lambda record: (record["start"], record["audio_index"]))

    # A selected background must be represented. Effects can be rejected or
    # shortened by the slot scheduler.
    background_ids = {item["audio_index"] for item in backgrounds}
    rendered_background_ids = {record["audio_index"] for record in records
                               if record["role"] == "background"}
    if background_ids - rendered_background_ids:
        raise RuntimeError("background track was omitted from the resolved mix")

    return records


def finish_audio(records, movie_end):
    """Preserve finite sources stopped only by movie EOF, not earlier events."""
    for record in records:
        if (record["role"] == "untimed" or record.get("interrupted") or
                number(record["tags"].get("cfdf_d5_loop")) or
                number(record["tags"].get("cfdf_d5_playlist_loop"))):
            continue
        natural_end = record["start"] + record["duration"]
        # Theme end tags are rounded to microseconds; trigger ticks are exact.
        if (record["start"] < movie_end < natural_end and
                abs(record["end"] - movie_end) <= Fraction(1, 1000000)):
            record["end"] = natural_end


def ds_pan(value):
    if value <= 0:
        return -10000
    if value >= 255:
        return 10000
    if value == 128:
        return 0
    if value < 128:
        return int(-1000.0 * math.log(128.0 / value))
    return int(1000.0 * math.log(127.0 / (255.0 - value)))


def pan_gains(value):
    pan = ds_pan(value)
    if pan <= -10000:
        return 1.0, 0.0
    if pan >= 10000:
        return 0.0, 1.0
    if pan < 0:
        return 1.0, 10.0 ** (pan / 2000.0)
    return 10.0 ** (-pan / 2000.0), 1.0


def sample_position(when, rate):
    """Convert an authored Fraction timestamp to the nearest output sample."""
    return max(0, round(when * rate))


def d5_clip_filter(record, rate, resampler, named_pan):
    """Build the local (zero-based) filter chain for one scheduled D5 clip."""
    stream = record["stream"]
    source_rate = number(stream.get("sample_rate"), rate)
    start_sample = sample_position(record["start"], rate)
    end_sample = sample_position(record["end"], rate)
    length_samples = max(1, end_sample - start_sample)
    duration_samples = max(1, round(record["duration"] * rate))

    if resampler == "ardftsrc":
        # ardftsrc passes equal-rate frames through unchanged, so force a
        # supported floating format on both sides and rebuild the local clock.
        chain = f"aformat=f=fltp,ardftsrc={rate},aformat=f=fltp,asetpts=N/SR/TB"
    else:
        # FFmpeg/older Librempeg keep aresample.  Mk4 materializes each clip
        # independently, so authored placement is applied by the slot file,
        # not by the clip's packet timestamps.
        chain = f"aresample={rate},asetpts=N/SR/TB"

    if number(record["tags"].get("cfdf_d5_loop")):
        source_loop_start = number(record["tags"].get("cfdf_d5_loop_start"))
        loop_start = round(source_loop_start * rate / source_rate)
        loop_start = min(max(loop_start, 0), duration_samples - 1)
        loop_size = duration_samples - loop_start
        chain += f",aloop=loop=-1:size={loop_size}:start={loop_start}"

    pan_value = number(record["tags"].get("cfdf_d5_pan"), 128)
    left, right = pan_gains(pan_value)
    if named_pan:
        pan = (f"pan=layout=stereo:"
               f"mix=c0={left:.9f}*c0|c1={right:.9f}*c0")
    else:
        pan = (f"pan=stereo|"
               f"c0={left:.9f}*c0|c1={right:.9f}*c0")

    return (f"{chain},atrim=end_sample={length_samples},{pan},"
            f"asetpts=N/SR/TB,aformat=f=s16:cl=stereo")


def render_d5_clip(ffmpeg, input_path, record, rate, resampler,
                   named_pan, clip_path):
    """Decode one scheduled D5 stream to a local zero-based PCM WAV."""
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "warning", "-y",
        "-i", input_path,
        "-map", f"0:a:{record['audio_index']}",
        "-af", d5_clip_filter(record, rate, resampler, named_pan),
        "-f", "wav", "-c:a", "pcm_s16le", clip_path,
    ]
    result = subprocess.run(command)
    if result.returncode:
        title = record["tags"].get("title", "")
        raise RuntimeError(
            f"failed to decode D5 stream a:{record['audio_index']} {title!r}")


def materialize_d5_audio(ffmpeg, input_path, records, rate, resampler,
                         named_pan, render_end, directory):
    """Place complete occurrences on one sample clock; sum before clipping.

    Decode independently to avoid queueing future streams in the demuxer.
    The temporary signed 64-bit accumulator keeps overlapping finite voices
    without saturating intermediate sums. Only the final PCM mix is clipped.
    """
    target_samples = max(1, sample_position(render_end, rate),
                         *(sample_position(record["end"], rate) for record in records))
    accumulator_path = os.path.join(directory, "mix.acc")
    output_path = os.path.join(directory, "mix.wav")
    frame_bytes = 2 * array("q").itemsize
    with open(accumulator_path, "w+b") as accumulator:
        accumulator.truncate(target_samples * frame_bytes)
        for record in records:
            clip_path = os.path.join(directory, "clip.wav")
            render_d5_clip(ffmpeg, input_path, record, rate, resampler,
                           named_pan, clip_path)
            start = sample_position(record["start"], rate)
            end = sample_position(record["end"], rate)
            expected = max(1, end - start)
            with wave.open(clip_path, "rb") as clip:
                if (clip.getnchannels(), clip.getsampwidth(),
                        clip.getframerate(), clip.getcomptype()) != (2, 2, rate, "NONE"):
                    raise RuntimeError("decoded clip has unexpected PCM parameters")
                actual = clip.getnframes()
                if actual + 8 < expected:
                    raise RuntimeError(
                        f"stream a:{record['audio_index']} decoded {actual} samples; "
                        f"schedule requires {expected}")
                remaining = min(actual, expected)
                accumulator.seek(start * frame_bytes)
                while remaining:
                    count = min(65536, remaining)
                    data = clip.readframes(count)
                    if len(data) != count * 4:
                        raise RuntimeError("decoded clip ended before its frame count")
                    source = array("h")
                    source.frombytes(data)
                    if sys.byteorder != "little":
                        source.byteswap()
                    position = accumulator.tell()
                    mixed = array("q")
                    mixed.frombytes(accumulator.read(count * frame_bytes))
                    for index, value in enumerate(source):
                        mixed[index] += value
                    accumulator.seek(position)
                    accumulator.write(mixed.tobytes())
                    remaining -= count
        accumulator.seek(0)
        with wave.open(output_path, "wb") as output:
            output.setparams((2, 2, rate, 0, "NONE", "not compressed"))
            while True:
                data = accumulator.read(65536 * frame_bytes)
                if not data:
                    break
                mixed = array("q")
                mixed.frombytes(data)
                samples = array("h", (max(-32768, min(32767, value)) for value in mixed))
                if sys.byteorder != "little":
                    samples.byteswap()
                output.writeframesraw(samples.tobytes())
    return output_path


def d5_mix_graph():
    return ("[1:a]asetnsamples=n=4096:p=0,asetpts=N/SR/TB,"
            "aformat=f=s16:cl=stereo[mix]")


def additive_filter(selected, rate, resampler):
    filters = []
    labels = []
    for index, item in enumerate(selected):
        if resampler == "aresample":
            filters.append(
                f"[0:a:{item['audio_index']}]"
                f"aresample={rate}:async=1:first_pts=0[s{index}]")
        else:
            start_samples = max(0, round(item["start"] * rate))
            filters.append(
                f"[0:a:{item['audio_index']}]"
                f"aformat=f=fltp,ardftsrc={rate},"
                "aformat=f=fltp,asetpts=N/SR/TB,"
                f"adelay={start_samples}S:all=1[s{index}]")
        labels.append(f"[s{index}]")
    if len(labels) == 1:
        filters.append(
            f"{labels[0]}asetnsamples=n=4096:p=0,"
            "asetpts=N/SR/TB,aformat=f=s16[mix]")
    else:
        filters.append(
            f"{''.join(labels)}amix=inputs={len(labels)}:normalize=0:duration=longest,"
            "asetnsamples=n=4096:p=0,asetpts=N/SR/TB,aformat=f=s16[mix]")
    return ";".join(filters)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("output", nargs="?")
    parser.add_argument("--audio-only", action="store_true")
    parser.add_argument("--include-untimed", action="store_true",
                        help="include unscheduled catalog sounds in compatibility mode; "
                             "rejected for V5 movie rendering")
    parser.add_argument("--overlap-sounds", action="store_true",
                        help="preserve finite SFX tails across replacements and scene "
                             "changes, including finite requests rejected by slot priority; "
                             "loops keep the strict slot policy; movie EOF still cuts audio")
    parser.add_argument("--finish-audio", action="store_true",
                        help="let finite sounds cut only by movie EOF finish, holding "
                             "the final image; earlier interruptions are preserved")
    parser.add_argument("--list", action="store_true",
                        help="print the resolved schedule and command without running it")
    parser.add_argument("--acodec", default=None)
    parser.add_argument("--vcodec", default=None)
    parser.add_argument("--ffmpeg", default=None)
    parser.add_argument("--ffprobe", default=None)
    args = parser.parse_args()
    if args.ffmpeg is None:
        candidates = [os.path.abspath("ffmpeg"),
                      os.path.join(os.path.dirname(os.path.abspath(__file__)), "ffmpeg")]
        args.ffmpeg = next((path for path in candidates if os.path.isfile(path) and
                           os.access(path, os.X_OK)), "ffmpeg")
    mp4 = os.path.splitext(args.output or "")[1].lower() in (".mp4", ".m4a", ".mov")
    args.acodec = args.acodec or ("aac" if mp4 else "flac")
    args.vcodec = args.vcodec or ("mpeg4" if mp4 else "ffv1")
    if args.ffprobe is None:
        executable = shutil.which(args.ffmpeg)
        sibling = os.path.join(os.path.dirname(executable or args.ffmpeg), "ffprobe")
        args.ffprobe = sibling if os.path.isfile(sibling) else "ffprobe"

    info = run_probe(args.ffprobe, args.input)
    streams = info.get("streams", [])
    version = detect_version(info)
    selected = classify(streams, args.input, args.include_untimed)
    have_video = any(stream.get("codec_type") == "video" for stream in streams)

    if version == 5 and have_video:
        if tags(info.get("format", {})).get("cfdf_d5_timeline") != "2":
            sys.exit("Mk4 requires the D5 timeline update for movies "
                     "(complete disk themes, both hold flags, and scene-end metadata).")

    if (args.overlap_sounds or args.finish_audio) and (version != 5 or not have_video):
        sys.exit("--overlap-sounds and --finish-audio require a Version 5 movie")

    if not selected:
        if have_video and not args.audio_only:
            output = args.output or os.path.splitext(args.input)[0] + "_render.mkv"
            command = [args.ffmpeg, "-hide_banner", "-y", "-i", args.input,
                       "-map", "0:v"]
            add_video_encoding_options(command, streams, args.vcodec)
            command.append(output)
            if args.list:
                print(" ".join(shlex.quote(part) for part in command))
                return
            sys.exit(subprocess.run(command).returncode)
        sys.exit("no scheduled audio streams found")

    rate = max(number(item["stream"].get("sample_rate")) for item in selected)
    if rate <= 0:
        sys.exit("scheduled audio streams have no usable sample rate")
    movie_end = video_duration(streams)
    have_d5_slots = any(item["tags"].get("cfdf_d5_slot") for item in selected)
    resampler = select_resampler(args.ffmpeg)
    named_pan = uses_named_pan(args.ffmpeg)

    if (args.overlap_sounds or args.finish_audio) and not have_d5_slots:
        sys.exit("audio playback options require D5 slot metadata")
    if args.finish_audio and movie_end <= 0:
        sys.exit("--finish-audio requires a finite movie duration")
    render_end = movie_end

    use_d5_slots = version == 5 and have_d5_slots
    records = None
    if use_d5_slots:
        if any(item["role"] == "untimed" for item in selected):
            sys.exit("unscheduled catalog sounds are not part of V5 movie playback; "
                     "extract individual streams with ffmpeg for catalog diagnostics")

        records = d5_schedule(selected, movie_end, args.overlap_sounds)
        if args.finish_audio:
            finish_audio(records, movie_end)
        render_end = max([movie_end] + [record["end"] for record in records])
        for record in records:
            title = record["tags"].get("title", "")
            key = "" if record["key"] is None else f" key={record['key']}"
            print(f"{float(record['start']):10.6f}..{float(record['end']):10.6f}  "
                  f"slot={record['slot']:<11}{key:<15} {title}")
    else:
        if version == 5:
            print("warning: no cfdf_d5 slot metadata; using additive compatibility mix",
                  file=sys.stderr)
        graph = additive_filter(selected, rate, resampler)
        if not graph:
            sys.exit("resolved schedule contains no audible intervals")

    output = args.output
    if not output:
        stem = os.path.splitext(args.input)[0]
        output = stem + ("_render.mka" if args.audio_only or not have_video
                         else "_render.mkv")

    def finish_command(command):
        command += ["-map", "[mix]", "-c:a", args.acodec]
        if args.acodec == "aac":
            command += ["-b:a", "192k"]
        if have_video and not args.audio_only:
            command += ["-map", "0:v"]
            video_filters = []
            if use_d5_slots and render_end > movie_end:
                if args.vcodec == "copy":
                    sys.exit("finite audio tails need video encoding to hold the last frame; "
                             "choose --vcodec ffv1 or another video encoder")
                # Bound cloned frames by the extended audio endpoint. Round up
                # on the movie clock so output limiting cannot remove an audio sample.
                end_ticks = math.ceil(render_end * 60)
                video_filters.append("tpad=stop_mode=clone:stop=-1")
                command += ["-t", f"{end_ticks / 60:.9f}"]
            add_video_encoding_options(
                command, streams, args.vcodec, video_filters)
            if args.vcodec == "ffv1":
                command += ["-level", "3"]
            elif args.vcodec == "mpeg4":
                command += ["-q:v", "2"]
        if os.path.splitext(output)[1].lower() in (".mp4", ".m4a", ".mov"):
            command += ["-movflags", "+faststart"]
        command.append(output)
        return command

    if use_d5_slots:
        if args.list:
            print(f"\nPlan: overlap-sounds={args.overlap_sounds}, "
                  f"finish-audio={args.finish_audio}; continuous PCM output.")
            command = [args.ffmpeg, "-hide_banner", "-y", "-i", args.input,
                       "-i", "<complete-mix.wav>", "-filter_complex", d5_mix_graph()]
            finish_command(command)
            print(" ".join(shlex.quote(part) for part in command))
            return
        try:
            with tempfile.TemporaryDirectory(prefix="dfrender-v5-") as directory:
                mix_path = materialize_d5_audio(
                    args.ffmpeg, args.input, records, rate, resampler,
                    named_pan, render_end, directory)
                command = [args.ffmpeg, "-hide_banner", "-y", "-i", args.input,
                           "-i", mix_path, "-filter_complex", d5_mix_graph()]
                finish_command(command)
                sys.exit(subprocess.run(command).returncode)
        except (RuntimeError, OSError, wave.Error) as exc:
            sys.exit(str(exc))

    command = [args.ffmpeg, "-hide_banner", "-y", "-i", args.input,
               "-filter_complex", graph]
    finish_command(command)
    if args.list:
        print("\n" + " ".join(shlex.quote(part) for part in command))
        return
    sys.exit(subprocess.run(command).returncode)


if __name__ == "__main__":
    main()
