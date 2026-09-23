import json

from PIL import Image

from scripts.train_lora import to_example
from sentinel.vlm_client import SYSTEM_PROMPT, USER_PROMPT

LABEL = {"source_type": "campfire", "smoke_color": "white", "attended": "yes",
         "near_structures": False, "near_road": True, "size_estimate": "small",
         "description": "Small campfire with people nearby."}


def _row(tmp_path):
    p = tmp_path / "crop.jpg"
    Image.new("L", (8, 6), 128).save(p)  # greyscale on purpose: to_example must convert to RGB
    return {"image": str(p), "label": LABEL}


def test_to_example_is_prompt_completion_matching_runtime_prompt(tmp_path):
    # prompt/completion format -> TRL computes loss on the JSON answer only
    ex = to_example(_row(tmp_path))
    assert set(ex) == {"images", "prompt", "completion"}
    assert [m["role"] for m in ex["prompt"]] == ["system", "user"]
    assert ex["prompt"][0]["content"] == [{"type": "text", "text": SYSTEM_PROMPT}]
    assert ex["prompt"][1]["content"] == [{"type": "image"}, {"type": "text", "text": USER_PROMPT}]
    assert ex["completion"] == [
        {"role": "assistant", "content": [{"type": "text", "text": json.dumps(LABEL)}]}]


def test_to_example_loads_one_rgb_image(tmp_path):
    ex = to_example(_row(tmp_path))
    assert len(ex["images"]) == 1
    assert ex["images"][0].mode == "RGB"
    assert ex["images"][0].size == (8, 6)
