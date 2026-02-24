"""Local test for blend video generator — Gap-maximized, TikTok-optimized."""
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
    print(f"Timeline: Before {CONFIG['before_hold_sec']}s → Transition {CONFIG['transition_sec']}s "
          f"→ After {CONFIG['after_hold_sec']}s → Bridge {CONFIG['loop_bridge_sec']}s")
    print(f"Total duration: {TOTAL_DURATION}s")
    print(f"FPS: {CONFIG['fps']}, CRF: {CONFIG['crf']}")
    print(f"Transition style: {CONFIG['transition_style']}")
    print(f"Enhancement before: {CONFIG['enhance_before']}")
    print(f"Enhancement after: {CONFIG['enhance_after']}")
    print(f"Quality gate thresholds: {CONFIG['quality_gate']}")

    # Test quality gate
    print(f"\n=== Quality Gate ===")
    quality = gen._quality_gate(before_img, after_img)
    print(f"Face diff: {quality['face_diff']}")
    print(f"Verdict: {quality['verdict']}")
    if "warning" in quality:
        print(f"Warning: {quality['warning']}")

    # Test enhancement
    print(f"\n=== Enhancement ===")
    enhanced_before = gen._enhance_before(before_img)
    enhanced_after = gen._enhance_after(after_img)
    print(f"Enhanced before mean: {enhanced_before.mean():.1f} (original: {before_img.mean():.1f})")
    print(f"Enhanced after mean: {enhanced_after.mean():.1f} (original: {after_img.mean():.1f})")

    # Generate video with blur transition (default)
    print(f"\n=== Generating blur transition video ===")
    video_result = gen.generate(current, None, result, transition_style="blur")
    print(f"Video: {len(video_result.data)} bytes, type={video_result.content_type}, ext={video_result.extension}")
    print(f"Duration: {video_result.duration}s")
    print(f"Metadata: {json.dumps(video_result.metadata, indent=2)}")

    # Calculate bitrate
    bitrate_kbps = (len(video_result.data) * 8) / (video_result.duration * 1000)
    print(f"Bitrate: {bitrate_kbps:.0f} kbps")

    # Save output
    out_path = "tests/test_images/blend_blur" + video_result.extension
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
        for line in result_probe.stdout.split("\n"):
            for key in ["bit_rate", "duration", "codec_name", "width", "height", "r_frame_rate"]:
                if line.startswith(f"{key}="):
                    print(f"  {line}")

    print(f"\nDone! Inspect the output video:")
    print(f"  {out_path}")
    print(f"\nChecklist:")
    print(f"  [ ] Before static 2.0s at start (enhanced: darker/desaturated)")
    print(f"  [ ] Blur transition at 2.0s, runs 0.3s")
    print(f"  [ ] After static from 2.3s to 4.5s (enhanced: brighter/saturated)")
    print(f"  [ ] Loop bridge 4.5s-5.0s (After→Before crossfade)")
    print(f"  [ ] Seamless loop when played on repeat")
    print(f"  [ ] NO text labels")
    print(f"  [ ] Watermark at bottom-right (60px, 30% opacity)")
    print(f"  [ ] Bitrate >= 800kbps (got {bitrate_kbps:.0f}kbps)")
    print(f"  [ ] Quality gate verdict: {quality['verdict']}")

if __name__ == "__main__":
    main()
