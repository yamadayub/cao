"""Local test for blend video generator."""
import sys
sys.path.insert(0, "/Users/yosuke/dev/cao/backend")

from app.services.blend_video_generator import BlendVideoGenerator

def main():
    gen = BlendVideoGenerator()

    with open("tests/test_images/current.png", "rb") as f:
        current = f.read()
    with open("tests/test_images/result.png", "rb") as f:
        result = f.read()

    print(f"current: {len(current)} bytes")
    print(f"result: {len(result)} bytes")

    # Test decode + fit
    import cv2
    import numpy as np
    before_img = gen._fit(gen._decode(current))
    print(f"Fitted before shape: {before_img.shape}, dtype: {before_img.dtype}")
    print(f"  min={before_img.min()}, max={before_img.max()}, mean={before_img.mean():.1f}")

    after_img = gen._fit(gen._decode(result))
    print(f"Fitted after shape: {after_img.shape}, dtype: {after_img.dtype}")
    print(f"  min={after_img.min()}, max={after_img.max()}, mean={after_img.mean():.1f}")

    # Test individual frames
    for t_sec in [0.0, 0.5, 1.0, 1.2, 1.4, 2.0, 2.4, 2.7, 3.0]:
        frame = gen._render_frame(t_sec, before_img, after_img)
        print(f"  t={t_sec:.1f}s: shape={frame.shape}, min={frame.min()}, max={frame.max()}, mean={frame.mean():.1f}")

    # Generate full video
    print("\nGenerating video...")
    video_result = gen.generate(current, None, result)
    print(f"Video: {len(video_result.data)} bytes, type={video_result.content_type}, ext={video_result.extension}")

    # Save to file for inspection
    out_path = "tests/test_images/blend_output" + video_result.extension
    with open(out_path, "wb") as f:
        f.write(video_result.data)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
