"""Quick connectivity test for CrEval API."""
from openai import OpenAI

client = OpenAI(api_key="0", base_url="http://127.0.0.1:8000/v1")

# List models
models = client.models.list()
for m in models.data:
    print(f"Model: {m.id}")

# Test one inference
resp = client.chat.completions.create(
    model=models.data[0].id,
    messages=[
        {"role": "system", "content": "你是一个语言创意评估专家。请评估两个回复的创意程度。"},
        {"role": "user", "content": (
            "[[DATA FIELD START]]\n"
            "### Query:\n写一首关于春天的诗\n"
            "### Response 1:\n春天来了，花儿开了。\n"
            "### Response 2:\n春风轻抚，万物苏醒，花开如海。\n"
            "[[DATA FIELD END]]\n"
            "请注意：以\"更有创意的回复是：Response 1\"或\"更有创意的回复是：Response 2\"或\"二者的创意程度相当。\"结尾。"
        )},
    ],
    temperature=0.0,
    max_tokens=32,
)
print(f"Response: {resp.choices[0].message.content}")
