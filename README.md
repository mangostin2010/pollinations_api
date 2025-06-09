# PollinationsAPI Python Client

A developer-friendly Python SDK for interacting with the [Pollinations API](https://pollinations.ai/), supporting multimodal AI features: image generation, text generation (including OpenAI-compatible chat), vision (image analysis), speech-to-text, and text-to-speech.

---

## Features

- **Image Generation:** Create images from text prompts with multiple model options.
- **Text Generation:** Generate or complete text, including OpenAI-compatible chat and streaming.
- **Vision (Image Analysis):** Describe or answer questions about images (from URL or local files).
- **Speech-to-Text:** Transcribe audio files to text.
- **Text-to-Speech:** Synthesize speech from text, save as MP3.
- **Model Listing:** List available image, text, and voice models.
- **Real-time Feeds:** Subscribe to live image/text generation events via SSE.

## Installation

Install via pip:

```bash
pip install pollinations-api
pip install requests
# For SSE real-time feeds (not needed for streaming chat/text completions):
pip install sseclient-py
```

## Quick Start

```python
from pollinations_api import PollinationsAPI, PollinationsAPIError

api = PollinationsAPI(referrer="my-app")  # Optionally add a backend Bearer token

# Image Generation
try:
    img_bytes = api.generate_image("A beautiful sunset over ocean", width=512, height=512, nologo=True, save_to="sunset.jpg")
    print("Image saved to sunset.jpg")
except PollinationsAPIError as e:
    print(e)

# Text Generation
try:
    text = api.generate_text("Explain quantum computing simply", model="openai")
    print("Text generation result:", text)
except PollinationsAPIError as e:
    print(e)

# OpenAI-Compatible Chat (non-streaming)
try:
    response = api.openai_chat_completion(
        model="openai",
        messages=[
            {"role": "system", "content": "You are an assistant."},
            {"role": "user", "content": "Who won the world cup in 2018?"}
        ]
    )
    print("Chat completion:", response['choices'][0]['message']['content'])
except PollinationsAPIError as e:
    print(e)
```

---

## Usage Examples

### 1. Image Generation

```python
api.generate_image(
    prompt="A cyberpunk cityscape at night",
    width=768,
    height=512,
    nologo=True,
    model="flux",
    save_to="cityscape.png"
)
```

### 2. Text Generation

- **Simple completion:**

    ```python
    result = api.generate_text("Write a haiku about AI")
    print(result)
    ```

- **Streaming responses:**

    ```python
    for chunk in api.generate_text("Tell a story about a robot", stream=True):
        print(chunk, end="", flush=True)
    ```

### 3. OpenAI-Compatible Chat Completion

**Non-streaming:**

```python
response = api.openai_chat_completion(
    model="openai",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Summarize the plot of Inception."}
    ]
)
print(response['choices'][0]['message']['content'])
```

**Streaming:**

```python
for chunk in api.openai_chat_completion(
    model="openai",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Summarize the plot of Inception."}
    ],
    stream=True
):
    print(chunk, end="", flush=True)
```
*No `sseclient-py` is needed for OpenAI-style streaming!*

### 4. Vision (Image Analysis)

- **Analyze an image URL:**

    ```python
    result = api.analyze_image_url(
        image_url="https://example.com/cat.jpg",
        question="What is in this image?"
    )
    print(result)
    ```

- **Analyze a local image:**

    ```python
    result = api.analyze_local_image("local_image.png")
    print(result)
    ```

### 5. Speech-to-Text

```python
transcript = api.transcribe_audio("sample.wav")
print("Transcription:", transcript)
```

### 6. Text-to-Speech

- **GET (short texts):**

    ```python
    api.tts_get("Hello world!", voice="nova", save_to="hello.mp3")
    ```

- **POST (longer texts):**

    ```python
    api.tts_post("This is a long text to synthesize.", voice="alloy", save_to="long.mp3")
    ```

### 7. List Models

```python
print("Image models:", api.list_image_models())
print("Text models:", api.list_text_models())
```

### 8. Real-time Feeds (SSE)

- **Image feed:**
  (Requires `sseclient-py`)

    ```python
    for event in api.connect_image_feed():
        print(event)
    ```

- **Text feed:**
  (Requires `sseclient-py`)

    ```python
    for event in api.connect_text_feed():
        print(event)
    ```

---

## Authentication

- **Backend Token:** Pass a Bearer token to the constructor.
- **Frontend Referrer:** Pass a referrer string to the constructor.

## Error Handling

All API errors are raised as `PollinationsAPIError`. Use try-except blocks to handle errors gracefully.

## Requirements

- Python 3.7+
- `requests`
- `sseclient-py` (only for real-time GET feeds, not for chat streaming)

## License

MIT

---

## Notes

- For best results, see official documentation: [https://github.com/pollinations/pollinations/blob/master/APIDOCS.md](https://github.com/pollinations/pollinations/blob/master/APIDOCS.md)
- Issues and contributions are welcome!

---

**Special Notes:**
- For OpenAI-style streaming (chat completions), you do **not** need `sseclient-py`, as the SDK parses streamed events directly.
- For real-time image/text feeds (`connect_image_feed`, `connect_text_feed`), `sseclient-py` is required.

---