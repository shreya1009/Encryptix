def chatbot_response(user_input):
    user_input = user_input.lower()

    # Greeting
    if "hello" in user_input or "hi" in user_input:
        return "Hello! How can I assist you today?"

    # Asking about the bot
    elif "who are you" in user_input or "what are you" in user_input:
        return "I am a simple rule-based chatbot designed to assist with your queries. Created by Mohamed Benhasan "
    elif "who is mohamed benhasan" in user_input or "mohamed benhasan" in user_input:
        return "Mohamed Benhasan is an Industrial Computer Engineering student in ENET'Com "
    # Farewell
    elif "bye" in user_input or "goodbye" in user_input:
        return "Goodbye! Have a great day!"

    # Asking about functionality
    elif "what can you do" in user_input or "help" in user_input:
        return "I can respond to greetings, tell you about myself, and provide basic information."

    # Default response for unrecognized input
    else:
        return "I'm sorry, I don't understand that. Can you please rephrase?"

# Example interaction loop
if __name__ == "__main__":
    print("Welcome to the Chatbot. Type 'bye' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["bye", "goodbye"]:
            print("Chatbot: Goodbye! Have a great day!")
            break
        response = chatbot_response(user_input)
        print(f"Chatbot: {response}")
