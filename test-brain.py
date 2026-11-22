import ollama

response = ollama.chat(
    model="qwen3:14b",
    messages=[
        {"role": "user", "content": "In one sentence: what should an email triage assistant be good at?"}
    ],
)

print(response.message.content)