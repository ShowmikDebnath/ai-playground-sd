from groq import Groq
from dotenv import load_dotenv
import os

# Load API key from .env
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Configure Groq client
client = Groq(api_key=api_key)

# Send a message and get response
# response = client.chat.completions.create(
#     model="llama-3.3-70b-versatile",
#     messages=[
#         {"role": "user", "content": "What is Python programming language?"}
#     ]
# )

# Print the response
# print(response.choices[0].message.content)

print("AI Chat — type 'exit' to quit")
print("───────────────────────────────")


#chat loop

while True:
    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        print("Good Bye!")
        break

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": user_input}
        ]
    )

    print("\n SD-AI:", response.choices[0].message.content)
    