from openai import OpenAI

client = OpenAI(
  api_key="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"
)

completion = client.chat.completions.create(
  model="gpt-4o-mini",
  store=True,
  messages=[
    {"role": "user", "content": "write a haiku about ai"}
  ]
)

print(completion.choices[0].message);
