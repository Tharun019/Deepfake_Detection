WEIGHTS = {
    "metadata": 0.30,
    "content": 0.45,
    "binary": 0.25,
}
FAKE_THRESHOLD = 0.35
def fuse_scores(metadata: float, content: float, binary: float) -> dict:
    """Weighted average of layer scores; returns verdict and confidence."""
    confidence = (
        metadata * WEIGHTS["metadata"]
        + content * WEIGHTS["content"]
        + binary * WEIGHTS["binary"]
    )
    is_fake = confidence >= FAKE_THRESHOLD
    verdict = "DEEPFAKE DETECTED" if is_fake else "LIKELY AUTHENTIC"
    return {
        "verdict": verdict,
        "is_fake": is_fake,
        "confidence": round(confidence, 4),
        "layer_scores": {
            "metadata": round(metadata, 4),
            "content": round(content, 4),
            "binary": round(binary, 4),
        },
    }
