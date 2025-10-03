import ollama

convo = []

def get_response():
    global convo
    response = ollama.chat("llama3.2:1b", messages=convo, stream=True)
    complete_response = ""
    for chunk in response:
        print(chunk['message']['content'], end='', flush=True)
        complete_response += chunk['message']['content']

    convo.append({"role": "assistant", "content": complete_response})
    print()

def main():
    global convo
    print("Welcome to the Ollama Chat Interface!")
    print("Type 'exit' to quit.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        convo.append({"role": "user", "content": user_input})
        get_response()

if __name__ == "__main__":
    main()