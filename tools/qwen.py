import torch
from PIL import Image

from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig  # 4bt量化




MODEL_PATH = "/opt/data/private/code/pretrain/qwen/Qwen3.8-27B" # -FP8

IMAGE_PATH = "/opt/data/private/data/test_data/111.jpg"


print("Loading model...")


model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)


processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)


print("Model loaded.")



# =========================
# Load image
# =========================

image = Image.open(
    IMAGE_PATH
).convert("RGB")



# =========================
# Build conversation
# =========================

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": IMAGE_PATH
            },
            {
                "type": "text",
                "text": "请描述这张图片的内容。"
            }
        ]
    }
]



# =========================
# Apply chat template
# =========================

text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)



# =========================
# Process text + image
# =========================

inputs = processor(
    text=[text],
    images=[image],
    return_tensors="pt",
    padding=True
)



# move inputs to GPU

inputs = {
    key: value.to(model.device)
    for key, value in inputs.items()
}


# =========================
# Generate
# =========================

with torch.no_grad():

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False
    )



# =========================
# Remove input tokens
# =========================

generated_ids_trimmed = [
    output_ids[len(input_ids):]
    for input_ids, output_ids in zip(
        inputs["input_ids"],
        generated_ids
    )
]



# =========================
# Decode
# =========================

response = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)



print("\n================ RESULT ================\n")

print(response[0])