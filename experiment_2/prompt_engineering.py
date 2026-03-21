from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# TECHNIQUE 1: Zero-shot prompting

print("=" * 50)
print("TECHNIQUE 1: Zero-shot")
print("=" * 50)

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "user", "content": "Classify this as positive or negative: 'I love coding!'"}
    ]
)
print(response.choices[0].message.content)


# TECHNIQUE 2: Few-shot prompting

print("\n" + "=" * 50)
print("TECHNIQUE 2: Few-shot")
print("=" * 50)

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "user",
            "content": """Classify these as positive or negative:

'I love coding!' → positive
'This is boring' → negative
'Python is amazing!' → positive
'I hate bugs' → negative

Now classify this:
'FastAPI is really fun to use!' → """
        }
    ]
)
print(response.choices[0].message.content)



# TECHNIQUE 3: System prompt

print("\n" + "=" * 50)
print("TECHNIQUE 3: System Prompt")
print("=" * 50)

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "system",
            "content": "You are an expert Python teacher who explains things in very simple words for beginners. Always use simple analogies and keep answers short."
        },
        {
            "role": "user",
            "content": "What is an API?"
        }
    ]
)
print(response.choices[0].message.content)