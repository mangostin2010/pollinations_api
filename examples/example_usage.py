import json
from pollinations_api import PollinationsAPI, PollinationsAPIError

def example_chat():
    print("\n=== Example: Basic Chat Completion ===")
    api = PollinationsAPI(referrer="example-app")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the FIFA World Cup in 2018?"}
    ]
    try:
        response = api.openai_chat_completion(model="openai", messages=messages)
        answer = response['choices'][0]['message']['content']
        print("Assistant:", answer)
    except PollinationsAPIError as e:
        print("Error:", e)

def example_streaming_chat():
    print("\n=== Example: Streaming Chat Completion ===")
    api = PollinationsAPI(referrer="example-app")
    messages = [
        {"role": "user", "content": "Write a short poem about the sea."}
    ]
    try:
        stream = api.openai_chat_completion(model="openai", messages=messages, stream=True)
        print("Assistant (streaming): ", end="", flush=True)
        for chunk in stream:
            # chunk is dict or raw string, here expect dict with choices
            if isinstance(chunk, dict):
                content = chunk.get('choices', [{}])[0].get('delta', {}).get('content')
                if content:
                    print(content, end="", flush=True)
            else:
                # raw text chunk
                print(chunk, end="", flush=True)
        print("\n--- End of stream ---")
    except PollinationsAPIError as e:
        print("Error:", e)

def example_function_calling():
    print("\n=== Example: Function Calling ===")
    api = PollinationsAPI(referrer="example-app")

    # Define a dummy weather function as tool
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"]
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]

    messages = [{"role": "user", "content": "What's the weather in New York?"}]

    try:
        # 1. Initial call
        response = api.openai_chat_completion(
            model="openai",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        message = response["choices"][0]["message"]
        print("Model response:", message.get("content", ""))

        tool_calls = message.get("tool_calls")
        if tool_calls:
            tool_call = tool_calls[0]
            func_name = tool_call["function"]["name"]
            func_args = json.loads(tool_call["function"]["arguments"])

            print(f"Function call requested: {func_name} with args {func_args}")

            # 2. Execute the function (dummy implementation)
            if func_name == "get_current_weather":
                # Mock response
                func_result = json.dumps({
                    "location": func_args.get("location"),
                    "temperature": "22°C",
                    "condition": "Sunny"
                })
            else:
                func_result = json.dumps({"error": "Unknown function"})

            # 3. Append the tool call and function result to messages
            updated_messages = messages + [
                message,
                {
                    "tool_call_id": tool_call["id"],
                    "role": "tool",
                    "name": func_name,
                    "content": func_result
                }
            ]

            # 4. Second API call with function result
            final_response = api.openai_chat_completion(
                model="openai",
                messages=updated_messages
            )
            final_answer = final_response["choices"][0]["message"]["content"]
            print("Assistant final answer:", final_answer)
        else:
            print("No function call requested by model.")

    except PollinationsAPIError as e:
        print("Error:", e)

def example_generate_image():
    print("\n=== Example: Generate Image ===")
    api = PollinationsAPI(referrer="example-app")
    prompt = "A futuristic city skyline at sunset, vibrant colors"
    try:
        api.generate_image(prompt, width=512, height=512, nologo=True, save_to="city.jpg")
        print("Image saved as city.jpg")
    except PollinationsAPIError as e:
        print("Error:", e)

def example_generate_text_get():
    print("\n=== Example: Generate Text (GET) ===")
    api = PollinationsAPI(referrer="example-app")
    prompt = "Explain black holes in simple terms"
    try:
        text = api.generate_text(prompt, model="openai")
        print("Generated text:", text)
    except PollinationsAPIError as e:
        print("Error:", e)

def example_tts_get():
    print("\n=== Example: Text-to-Speech (GET) ===")
    api = PollinationsAPI(referrer="example-app")
    text = "Hello from Pollinations AI text-to-speech."
    try:
        api.tts_get(text, voice="nova", save_to="hello.mp3")
        print("Audio saved as hello.mp3")
    except PollinationsAPIError as e:
        print("Error:", e)

if __name__ == "__main__":
    example_chat()
    example_streaming_chat()
    example_function_calling()
    example_generate_image()
    example_generate_text_get()
    example_tts_get()
