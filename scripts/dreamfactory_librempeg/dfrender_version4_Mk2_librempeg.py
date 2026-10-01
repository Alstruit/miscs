#!/usr/bin/env python3
"""dfrender — DreamFactory movie "editor render".

This is the V4-oriented renderer. It keeps the demuxer's authored timeline as
the source of truth: the renderer never invents sound duration that librempeg
did not expose.

Large timelines are rendered in two stages. Each scheduled audio stream is
decoded independently to local PCM, then placed on one continuous mix clock
before the final video/audio mux. This follows the queue-safe V5 materializing
architecture without importing V5 slot arbitration into V4.

Stream roles come from the demuxer's timeline= metadata tag:
  background  assembled playlist track
  sfx         frame-triggered one-shot
  untimed     director sound never scheduled on the linear timeline

The optional Mk3-style switches are deliberately conservative:
  --overlap-sounds
      Keep any overlap already present in the demuxed SFX streams. V4.0 group-0
      replacement may already have truncated a sound at the next trigger, so
      this option cannot restore samples that are no longer exposed.
  --finish-audio
      If a scheduled audio stream actually extends beyond the video duration,
      let that exposed audio finish and hold the last video frame. It does not
      synthesize a tail that the demuxer already truncated.

Usage:
  dfrender.py INPUT [OUTPUT] [--audio-only] [--include-untimed]
              [--overlap-sounds] [--finish-audio] [--list]
              [--acodec flac] [--vcodec ffv1] [--ffmpeg X] [--ffprobe Y]
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


def probe(ffprobe, path):
    result = subprocess.run(
        [ffprobe, "-v", "error", "-of", "json", "-show_streams",
         "-show_format", path],
        capture_output=True, text=True)
    if result.returncode != 0:
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

def stream_seconds(stream, field):
    try:
        return float(stream.get(field) or 0)
    except (TypeError, ValueError):
        return 0.0


def classify(streams, filename, include_untimed):
    """Return ([(audio_relative_index, stream, role)], have_video)."""
    base = os.path.basename(filename)
    audio = [stream for stream in streams
             if stream.get("codec_type") == "audio"]
    video = [stream for stream in streams
             if stream.get("codec_type") == "video"]
    selected = []

    for index, stream in enumerate(audio):
        tags = stream.get("tags", {}) or {}
        role = tags.get("timeline")

        if role is None:
            if (stream.get("disposition", {}).get("default") or
                    tags.get("title") == base):
                role = "background"
            elif stream_seconds(stream, "start_time") > 0:
                role = "sfx"
            else:
                role = "untimed"

        if (role in ("background", "sfx") or
                (include_untimed and role == "untimed")):
            selected.append((index, stream, role))

    return selected, bool(video)


def video_duration(streams):
    return max(
        (stream_seconds(stream, "duration")
         for stream in streams if stream.get("codec_type") == "video"),
        default=0.0)


def audio_end(stream):
    return (stream_seconds(stream, "start_time") +
            stream_seconds(stream, "duration"))


def exposed_sfx_overlaps(selected):
    """Count overlaps that are actually present in the demuxed streams."""
    effects = sorted(
        ((stream_seconds(stream, "start_time"),
          audio_end(stream), index)
         for index, stream, role in selected
         if role == "sfx" and stream_seconds(stream, "duration") > 0),
        key=lambda item: (item[0], item[2]))

    overlaps = 0
    active_end = -1.0
    for start, end, _index in effects:
        if start < active_end:
            overlaps += 1
        active_end = max(active_end, end)

    return overlaps


def sample_position(seconds, rate):
    """Convert an authored timestamp to the nearest output sample."""
    return max(0, round(seconds * rate))


def v4_clip_filter(stream, rate, resampler):
    """Build one local clip while preserving Mk2 resampler timing."""
    start_samples = sample_position(
        stream_seconds(stream, "start_time"), rate)
    duration = stream_seconds(stream, "duration")
    duration_samples = sample_position(duration, rate) if duration > 0 else 0

    if resampler == "aresample":
        # Keep Mk2's global first_pts behavior so mixed-rate clips retain the
        # same resampler phase, then remove the generated leading silence.
        chain = f"aresample={rate}:async=1:first_pts=0"
    else:
        # Same rule for the Librempeg fallback: build the authored delay first,
        # then trim it away after the filter has seen the original timing.
        chain = (f"aformat=f=fltp,ardftsrc={rate},"
                 "aformat=f=fltp,asetpts=N/SR/TB,"
                 f"adelay={start_samples}S:all=1")

    end_samples = start_samples + duration_samples
    trim = f",atrim=start_sample={start_samples}"
    if duration_samples > 0:
        trim += f":end_sample={end_samples}"
    return f"{chain}{trim},asetpts=N/SR/TB,aformat=f=s16:cl=mono"


def render_v4_clip(ffmpeg, input_path, audio_index, stream, rate,
                   resampler, clip_path):
    """Decode one scheduled stream independently to local PCM."""
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "warning", "-y",
        "-i", input_path,
        "-map", f"0:a:{audio_index}",
        "-af", v4_clip_filter(stream, rate, resampler),
        "-f", "wav", "-c:a", "pcm_s16le", clip_path,
    ]
    result = subprocess.run(command)
    if result.returncode:
        title = (stream.get("tags", {}) or {}).get("title", "")
        raise RuntimeError(
            f"failed to decode V4 stream a:{audio_index} {title!r}")


def materialize_v4_audio(ffmpeg, input_path, selected, rate, resampler,
                         minimum_end, directory):
    """Decode streams independently, place them on one clock, then mix.

    Keeping only one DreamFactory audio stream active per decode avoids
    filling demuxer packet queues with packets for many future-timed streams.
    The signed 64-bit accumulator preserves additive overlap until the final
    s16 clip, matching the old amix=normalize=0 path without one huge graph.
    """
    target_samples = max(
        1,
        sample_position(minimum_end, rate),
        *(sample_position(audio_end(stream), rate)
          for _index, stream, _role in selected),
    )
    accumulator_path = os.path.join(directory, "mix.acc")
    output_path = os.path.join(directory, "mix.wav")
    frame_bytes = array("q").itemsize

    with open(accumulator_path, "w+b") as accumulator:
        accumulator.truncate(target_samples * frame_bytes)

        for audio_index, stream, _role in selected:
            clip_path = os.path.join(directory, "clip.wav")
            render_v4_clip(
                ffmpeg, input_path, audio_index, stream, rate, resampler,
                clip_path)

            start = sample_position(
                stream_seconds(stream, "start_time"), rate)
            duration = stream_seconds(stream, "duration")
            expected = (sample_position(duration, rate)
                        if duration > 0 else None)

            with wave.open(clip_path, "rb") as clip:
                if (clip.getnchannels(), clip.getsampwidth(),
                        clip.getframerate(), clip.getcomptype()) != (1, 2, rate, "NONE"):
                    raise RuntimeError(
                        "decoded V4 clip has unexpected PCM parameters")

                actual = clip.getnframes()
                if expected is not None and actual + 8 < expected:
                    raise RuntimeError(
                        f"stream a:{audio_index} decoded {actual} samples; "
                        f"schedule requires {expected}")

                remaining = (min(actual, expected)
                             if expected is not None else actual)
                required_samples = start + remaining
                if required_samples > target_samples:
                    target_samples = required_samples
                    accumulator.truncate(target_samples * frame_bytes)
                accumulator.seek(start * frame_bytes)

                while remaining:
                    count = min(65536, remaining)
                    data = clip.readframes(count)
                    if len(data) != count * 2:
                        raise RuntimeError(
                            "decoded V4 clip ended before its frame count")

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
            output.setparams((1, 2, rate, 0, "NONE", "not compressed"))
            while True:
                data = accumulator.read(65536 * frame_bytes)
                if not data:
                    break
                mixed = array("q")
                mixed.frombytes(data)
                samples = array(
                    "h",
                    (max(-32768, min(32767, value)) for value in mixed),
                )
                if sys.byteorder != "little":
                    samples.byteswap()
                output.writeframesraw(samples.tobytes())

    return output_path


def continuous_mix_graph():
    """Normalize the materialized PCM clock for the final encode/mux."""
    return ("[1:a]asetnsamples=n=4096:p=0,asetpts=N/SR/TB,"
            "aformat=f=s16:cl=mono[mix]")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input")
    parser.add_argument("output", nargs="?")
    parser.add_argument(
        "--audio-only", action="store_true",
        help="render only the mixed audio track")
    parser.add_argument(
        "--include-untimed", action="store_true",
        help="also mix director sounds never scheduled on the linear timeline")
    parser.add_argument(
        "--overlap-sounds", action="store_true",
        help="retain SFX overlap already exposed by the demuxer; "
             "cannot restore V4.0 tails truncated at replacement")
    parser.add_argument(
        "--finish-audio", action="store_true",
        help="if exposed scheduled audio extends past the video, let it finish "
             "and hold the last video frame")
    parser.add_argument(
        "--list", action="store_true",
        help="print the timeline and ffmpeg command without running it")
    parser.add_argument("--acodec", default="flac")
    parser.add_argument("--vcodec", default="ffv1")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default=None)
    args = parser.parse_args()
    if args.ffprobe is None:
        executable = shutil.which(args.ffmpeg)
        sibling = os.path.join(os.path.dirname(executable or args.ffmpeg), "ffprobe")
        args.ffprobe = sibling if os.path.isfile(sibling) else "ffprobe"

    info = probe(args.ffprobe, args.input)
    streams = info.get("streams", [])
    selected, have_video = classify(
        streams, args.input, args.include_untimed)

    if not selected:
        if have_video and not args.audio_only:
            output = (args.output or
                      os.path.splitext(args.input)[0] + "_render.mkv")
            command = [
                args.ffmpeg, "-hide_banner", "-y", "-i", args.input,
                "-map", "0:v",
            ]
            add_video_encoding_options(command, streams, args.vcodec)
            if args.vcodec == "ffv1":
                command += ["-level", "3"]
            command.append(output)

            print(f"{args.input}: silent movie (no scheduled audio); "
                  "rendering video only")
            if args.list:
                print("\n" + " ".join(
                    shlex.quote(part) for part in command))
                return
            sys.exit(subprocess.run(command).returncode)

        sys.exit("no scheduled audio streams found (nothing to render)")

    tagged = any(
        (stream.get("tags", {}) or {}).get("timeline")
        for stream in streams if stream.get("codec_type") == "audio")
    if not tagged:
        print(
            "warning: input has no timeline tags; falling back to the "
            "legacy stream heuristic",
            file=sys.stderr)

    rate = max(
        int(stream.get("sample_rate") or 0)
        for _index, stream, _role in selected)
    if rate <= 0:
        sys.exit("scheduled audio streams have no usable sample rate")

    print(
        f"{args.input}: timeline ({len(selected)} scheduled clip(s)), "
        f"mix at {rate} Hz")
    for index, stream, role in selected:
        start = stream_seconds(stream, "start_time")
        duration = stream_seconds(stream, "duration")
        title = (stream.get("tags", {}) or {}).get("title", "")
        default = (" *default*"
                   if stream.get("disposition", {}).get("default") else "")
        print(
            f"  a:{index:<3} [{role:10}] at {start:8.3f}s  "
            f"dur {duration:8.3f}s  "
            f"{int(stream.get('sample_rate') or 0):5d} Hz  "
            f"{title}{default}")

    movie_end = video_duration(streams)
    if have_video and not args.audio_only:
        print(f"  video: {movie_end:.3f}s "
              "(includes any hold already represented by the demuxer)")

    if args.overlap_sounds:
        overlaps = exposed_sfx_overlaps(selected)
        if overlaps:
            print(
                f"  overlap: {overlaps} SFX overlap(s) are already present "
                "in the demuxed timeline")
        else:
            print(
                "warning: --overlap-sounds found no demuxer-exposed SFX "
                "tails to extend across replacement; V4.0 group-0 samples "
                "already truncated by the demuxer cannot be recovered here",
                file=sys.stderr)

    resampler = select_resampler(args.ffmpeg)

    render_end = movie_end
    if args.finish_audio and have_video and not args.audio_only:
        scheduled_ends = [
            audio_end(stream)
            for _index, stream, role in selected
            if role != "untimed"
        ]
        if scheduled_ends:
            render_end = max([movie_end] + scheduled_ends)

        if render_end <= movie_end + 1e-9:
            print(
                "warning: --finish-audio found no exposed scheduled audio "
                "past the video end; a tail already truncated by the demuxer "
                "cannot be reconstructed by the renderer",
                file=sys.stderr)

    output = args.output
    if not output:
        stem = os.path.splitext(args.input)[0]
        output = stem + (
            "_render.mka"
            if args.audio_only or not have_video
            else "_render.mkv")

    def finish_command(command):
        command += ["-map", "[mix]", "-c:a", args.acodec]

        if have_video and not args.audio_only:
            command += ["-map", "0:v"]
            video_filters = []

            if args.finish_audio and render_end > movie_end + 1e-9:
                if args.vcodec == "copy":
                    sys.exit(
                        "--finish-audio needs video encoding to hold the last "
                        "frame; choose --vcodec ffv1 or another video encoder")
                video_filters.append("tpad=stop_mode=clone:stop=-1")
                command += ["-t", f"{render_end:.9f}"]

            add_video_encoding_options(
                command, streams, args.vcodec, video_filters)
            if args.vcodec == "ffv1":
                command += ["-level", "3"]

        command.append(output)
        return command

    if args.list:
        print(
            f"\nPlan: decode {len(selected)} scheduled V4 stream(s) "
            "independently, materialize one continuous PCM mix, then mux video.")
        command = [
            args.ffmpeg, "-hide_banner", "-y", "-i", args.input,
            "-i", "<complete-mix.wav>",
            "-filter_complex", continuous_mix_graph(),
        ]
        finish_command(command)
        print(" ".join(shlex.quote(part) for part in command))
        return

    try:
        with tempfile.TemporaryDirectory(prefix="dfrender-v4-") as directory:
            minimum_mix_end = (render_end
                               if have_video and not args.audio_only else 0.0)
            mix_path = materialize_v4_audio(
                args.ffmpeg, args.input, selected, rate, resampler,
                minimum_mix_end, directory)
            command = [
                args.ffmpeg, "-hide_banner", "-y", "-i", args.input,
                "-i", mix_path,
                "-filter_complex", continuous_mix_graph(),
            ]
            finish_command(command)
            result = subprocess.run(command)
    except (RuntimeError, OSError, wave.Error) as exc:
        sys.exit(str(exc))

    if result.returncode == 0:
        print(f"rendered -> {output}")
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
