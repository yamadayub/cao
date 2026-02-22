"""Local test for blend video generator."""
import sys
sys.path.insert(0, "/Users/yosuke/dev/cao/backend")

from app.services.blend_video_generator import BlendVideoGenerator

def main():
    gen = BlendVideoGenerator()

    with open("tests/test_images/current.png", "rb") as f:
        current = f.read()
    with open("tests/test_images/ideal.png", "rb") as f:
        ideal = f.read()
    with open("tests/test_images/result.png", "rb") as f:
        result = f.read()

    print(f"current: {len(current)} bytes")
    print(f"ideal: {len(ideal)} bytes")
    print(f"result: {len(result)} bytes")

    # Test decode + fit
    import cv2
    import numpy as np
    cur_img = gen._fit(gen._decode(current))
    print(f"Fitted current shape: {cur_img.shape}, dtype: {cur_img.dtype}")
    print(f"  min={cur_img.min()}, max={cur_img.max()}, mean={cur_img.mean():.1f}")

    ideal_img = gen._fit(gen._decode(ideal))
    print(f"Fitted ideal shape: {ideal_img.shape}, dtype: {ideal_img.dtype}")
    print(f"  min={ideal_img.min()}, max={ideal_img.max()}, mean={ideal_img.mean():.1f}")

    result_img = gen._fit(gen._decode(result))
    print(f"Fitted result shape: {result_img.shape}, dtype: {result_img.dtype}")
    print(f"  min={result_img.min()}, max={result_img.max()}, mean={result_img.mean():.1f}")

    # Test individual frames
    for t_sec in [0.0, 0.5, 1.0, 1.3, 1.5, 2.0, 2.5, 3.5, 5.0]:
        frame = gen._render_frame(t_sec, cur_img, ideal_img, result_img)
        print(f"  t={t_sec:.1f}s: shape={frame.shape}, min={frame.min()}, max={frame.max()}, mean={frame.mean():.1f}")

    # Generate full video
    print("\nGenerating video...")
    video_result = gen.generate(current, ideal, result)
    print(f"Video: {len(video_result.data)} bytes, type={video_result.content_type}, ext={video_result.extension}")

    # Save to file for inspection
    out_path = "tests/test_images/blend_output" + video_result.extension
    with open(out_path, "wb") as f:
        f.write(video_result.data)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
