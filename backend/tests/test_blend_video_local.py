"""Local test for blend video generator — TikTok-optimized patterns."""
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
    before_img = gen._fit(gen._decode(current))
    print(f"Fitted before shape: {before_img.shape}, dtype: {before_img.dtype}")

    after_img = gen._fit(gen._decode(result))
    print(f"Fitted after shape: {after_img.shape}, dtype: {after_img.dtype}")

    # ── Pattern A ──────────────────────────
    print("\n=== Pattern A (4.0s loop-optimized) ===")
    print("Generating video...")
    result_a = gen.generate(current, None, result, pattern="A")
    print(f"Video: {len(result_a.data)} bytes, type={result_a.content_type}, ext={result_a.extension}")
    print(f"Duration: {result_a.duration}s")
    print(f"Metadata: {result_a.metadata}")

    out_a = "tests/test_images/blend_A" + result_a.extension
    with open(out_a, "wb") as f:
        f.write(result_a.data)
    print(f"Saved to {out_a}")

    # ── Pattern B ──────────────────────────
    print("\n=== Pattern B (6.0s morph showcase) ===")
    print("Generating video...")
    result_b = gen.generate(current, None, result, pattern="B")
    print(f"Video: {len(result_b.data)} bytes, type={result_b.content_type}, ext={result_b.extension}")
    print(f"Duration: {result_b.duration}s")
    print(f"Metadata: {result_b.metadata}")

    out_b = "tests/test_images/blend_B" + result_b.extension
    with open(out_b, "wb") as f:
        f.write(result_b.data)
    print(f"Saved to {out_b}")

    print("\nDone! Inspect the output videos:")
    print(f"  {out_a} — verify snap cut at 1.5s, loop bridge, labels, watermark")
    print(f"  {out_b} — verify snap cut, hard cut, slow morph, loop bridge")

if __name__ == "__main__":
    main()
