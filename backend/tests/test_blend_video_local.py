"""Local test for blend video generator — v8 4-phase timeline."""
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
    print(f"\n=== Configuration (v8) ===")
    print(f"Timeline: Before {CONFIG['before_hold_sec']}s → Flash {CONFIG['flash_sec']}s "
          f"→ Bounce {CONFIG['bounce_sec']}s → After {CONFIG['after_hold_sec']}s")
    print(f"Total duration: {TOTAL_DURATION}s (no loop bridge)")
    print(f"FPS: {CONFIG['fps']}, CRF: {CONFIG['crf']}")
    print(f"Transition style: {CONFIG['transition_style']}")
    print(f"Motion style: {CONFIG['motion_style']}")
    print(f"Before zoom: 1.0 → {CONFIG['before_zoom_end_scale']}")
    print(f"After bounce: {CONFIG['after_bounce_start_scale']} → ~{CONFIG['after_bounce_settle_scale']}")
    print(f"After zoom-out: {CONFIG['after_bounce_settle_scale']} → {CONFIG['after_zoom_out_end_scale']}")
    print(f"Enhancement before: {CONFIG['enhance_before']}")
    print(f"Enhancement after: {CONFIG['enhance_after']}")

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

    # Test zoom (including zoom-out border fix)
    print(f"\n=== Zoom Test ===")
    zoomed = gen._apply_zoom(before_img, 1.08)
    print(f"Zoom 1.08x shape: {zoomed.shape}")
    zoomed_out = gen._apply_zoom(after_img, 0.96)
    print(f"Zoom 0.96x shape: {zoomed_out.shape} (no black borders)")

    # Test bounce spring (vertical + zoom)
    print(f"\n=== Bounce Spring Test ===")
    start = CONFIG['after_bounce_start_scale']
    settle = CONFIG['after_bounce_settle_scale']
    amp_px = CONFIG['bounce_amplitude_px']
    for t in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
        spring = gen._bounce_spring(t)
        zoom_scale = settle + (start - settle) * spring
        offset_y = -amp_px * spring
        print(f"  t={t:.1f}: spring={spring:+.3f}, zoom={zoom_scale:.4f}, offset_y={offset_y:+.0f}px")

    # Generate video with flash transition (default)
    print(f"\n=== Generating v8 flash transition + zoom motion video ===")
    video_result = gen.generate(current, None, result, transition_style="flash", motion_style="zoom")
    print(f"Video: {len(video_result.data)} bytes, type={video_result.content_type}, ext={video_result.extension}")
    print(f"Duration: {video_result.duration}s")
    print(f"Metadata: {json.dumps(video_result.metadata, indent=2)}")

    # Calculate bitrate
    bitrate_kbps = (len(video_result.data) * 8) / (video_result.duration * 1000)
    print(f"Bitrate: {bitrate_kbps:.0f} kbps")

    # Save output
    out_path = "tests/test_images/blend_flash_zoom" + video_result.extension
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
    print(f"\nChecklist (v8):")
    print(f"  [ ] Before zoom-in 0.0–2.0s (scale 1.0 → 1.08, ease-in)")
    print(f"  [ ] White flash 2.0–2.5s with white 'hold' (~0.1s pure white)")
    print(f"  [ ] After bounce 2.5–3.0s (vertical ±80px + zoom 1.12 → ~1.05)")
    print(f"  [ ] After zoom-out 3.0–5.5s (scale 1.05 → 1.0, perceived pull-back)")
    print(f"  [ ] Final frame is After image (NOT Before)")
    print(f"  [ ] Enhancement: Before darker/desaturated, After brighter/saturated (natural)")
    print(f"  [ ] Watermark bottom-right, 60px, 30% opacity")
    print(f"  [ ] NO text labels")
    print(f"  [ ] Bitrate >= 800kbps (got {bitrate_kbps:.0f}kbps)")
    print(f"  [ ] Quality gate verdict: {quality['verdict']}")

if __name__ == "__main__":
    main()
