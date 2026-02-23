"""Local test for blend video generator — Morph-centered, TikTok-optimized."""
import json
import sys
sys.path.insert(0, "/Users/yosuke/dev/cao/backend")

from app.services.blend_video_generator import BlendVideoGenerator, CONFIG, TOTAL_DURATION

def main():
    gen = BlendVideoGenerator()

    with open("tests/test_images/current.png", "rb") as f:
        current = f.read()
    with open("tests/test_images/result.png", "rb") as f:
        result = f.read()

    print(f"current: {len(current)} bytes")
    print(f"result: {len(result)} bytes")

    # Test decode + fit
    before_img = gen._fit(gen._decode(current))
    print(f"Fitted before shape: {before_img.shape}, dtype: {before_img.dtype}")

    after_img = gen._fit(gen._decode(result))
    print(f"Fitted after shape: {after_img.shape}, dtype: {after_img.dtype}")

    # Print config
    print(f"\n=== Configuration ===")
    print(f"Timeline: Before {CONFIG['before_hold_sec']}s → Morph {CONFIG['morph_sec']}s "
          f"→ After {CONFIG['after_hold_sec']}s → Bridge {CONFIG['loop_bridge_sec']}s")
    print(f"Total duration: {TOTAL_DURATION}s")
    print(f"FPS: {CONFIG['fps']}, CRF: {CONFIG['crf']}")
    print(f"Flash: enabled={CONFIG['flash_enabled']}, opacity={CONFIG['flash_opacity']}")
    print(f"Loop bridge blend max: {CONFIG['loop_bridge_blend_max']}")

    # Generate video
    print(f"\n=== Generating morph-centered video ===")
    video_result = gen.generate(current, None, result)
    print(f"Video: {len(video_result.data)} bytes, type={video_result.content_type}, ext={video_result.extension}")
    print(f"Duration: {video_result.duration}s")
    print(f"Metadata: {json.dumps(video_result.metadata, indent=2)}")

    # Calculate bitrate
    bitrate_kbps = (len(video_result.data) * 8) / (video_result.duration * 1000)
    print(f"Bitrate: {bitrate_kbps:.0f} kbps")

    # Save output
    out_path = "tests/test_images/blend_morph" + video_result.extension
    with open(out_path, "wb") as f:
        f.write(video_result.data)
    print(f"Saved to {out_path}")

    # Verify with ffprobe if available
    import shutil
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        import subprocess
        print(f"\n=== ffprobe analysis ===")
        cmd = [ffprobe, "-v", "error", "-show_format", "-show_streams", out_path]
        result_probe = subprocess.run(cmd, capture_output=True, text=True)
        # Extract key info
        for line in result_probe.stdout.split("\n"):
            for key in ["bit_rate", "duration", "codec_name", "width", "height", "r_frame_rate"]:
                if line.startswith(f"{key}="):
                    print(f"  {line}")

    print(f"\nDone! Inspect the output video:")
    print(f"  {out_path}")
    print(f"\nChecklist:")
    print(f"  [ ] Before static 0.5s at start")
    print(f"  [ ] Morph starts at 0.5s, runs 2.0s with ease-in-out")
    print(f"  [ ] Flash visible at ~2.4-2.5s")
    print(f"  [ ] After static from 2.5s to 4.0s")
    print(f"  [ ] Loop bridge 4.0s-4.5s (After→Before crossfade)")
    print(f"  [ ] Seamless loop when played on repeat")
    print(f"  [ ] Labels: 'Before' visible then fades, 'After' fades in after morph")
    print(f"  [ ] Watermark at bottom-right")
    print(f"  [ ] Bitrate >= 800kbps (got {bitrate_kbps:.0f}kbps)")

if __name__ == "__main__":
    main()
