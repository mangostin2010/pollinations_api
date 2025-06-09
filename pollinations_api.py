#######################################################

import requests
import urllib.parse
import base64
import json
from typing import Optional, List, Dict, Any, Union, Generator

# SSE streaming support requires sseclient-py
try:
    import sseclient
except ImportError:
    sseclient = None


class PollinationsAPIError(Exception):
    pass


class PollinationsAPI:
    IMAGE_BASE = "https://image.pollinations.ai"
    TEXT_BASE = "https://text.pollinations.ai"
    OPENAI_ENDPOINT = f"{TEXT_BASE}/openai"

    def __init__(self, token: Optional[str] = None, referrer: Optional[str] = None, timeout: int = 30):
        """
        :param token: Bearer token for backend authentication (optional)
        :param referrer: Referrer string for frontend authentication (optional)
        :param timeout: HTTP request timeout in seconds
        """
        self.token = token
        self.referrer = referrer
        self.timeout = timeout
        self.session = requests.Session()
        if token:
            self.session.headers.update({"Authorization": f"Bearer {token}"})

    def _build_headers(self, extra_headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        headers = extra_headers.copy() if extra_headers else {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        headers["User-Agent"] = "PollinationsAPI-Python/1.0"
        return headers

    def _add_auth_to_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if self.token:
            # Prefer bearer token in header; optionally support token param
            # Here we rely on header so do nothing
            pass
        elif self.referrer:
            params["referrer"] = self.referrer
        return params

    # ------------------- Image Generation -------------------

    def generate_image(
        self,
        prompt: str,
        model: Optional[str] = None,
        seed: Optional[int] = None,
        width: Optional[int] = 1024,
        height: Optional[int] = 1024,
        nologo: bool = False,
        private: bool = False,
        enhance: bool = False,
        safe: bool = False,
        transparent: bool = False,
        save_to: Optional[str] = None
    ) -> bytes:
        """
        Generate image from text prompt (GET request).

        :param prompt: Text prompt for image generation
        :param model: Image model, default 'flux'
        :param seed: Seed integer for reproducibility
        :param width: Image width (default 1024)
        :param height: Image height (default 1024)
        :param nologo: Disable Pollinations logo overlay
        :param private: Prevent image from public feed
        :param enhance: Enhance prompt with LLM
        :param safe: Enable strict NSFW filtering (throws error if detected)
        :param transparent: Transparent background (only gptimage model supports)
        :param save_to: If specified, save image bytes to this file
        :return: Image bytes
        """
        if not prompt:
            raise ValueError("Prompt is required for image generation")

        encoded_prompt = urllib.parse.quote(prompt)
        url = f"{self.IMAGE_BASE}/prompt/{encoded_prompt}"

        params = {}
        if model:
            params["model"] = model
        if seed is not None:
            params["seed"] = seed
        if width:
            params["width"] = width
        if height:
            params["height"] = height
        if nologo:
            params["nologo"] = "true"
        if private:
            params["private"] = "true"
        if enhance:
            params["enhance"] = "true"
        if safe:
            params["safe"] = "true"
        if transparent:
            params["transparent"] = "true"

        params = self._add_auth_to_params(params)

        try:
            resp = self.session.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            content_type = resp.headers.get("Content-Type", "")
            if not content_type.startswith("image"):
                raise PollinationsAPIError(f"Expected image response, got Content-Type: {content_type}\nBody: {resp.text}")
            if save_to:
                with open(save_to, "wb") as f:
                    f.write(resp.content)
            return resp.content

        except requests.RequestException as e:
            raise PollinationsAPIError(f"Error generating image: {e}")

    def list_image_models(self) -> List[str]:
        """
        List available image generation models.

        :return: List of model names
        """
        url = f"{self.IMAGE_BASE}/models"
        try:
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            raise PollinationsAPIError(f"Error listing image models: {e}")

    # ------------------- Text Generation (GET) -------------------

    def generate_text(
        self,
        prompt: str,
        model: Optional[str] = "openai",
        seed: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        json_mode: bool = False,
        system: Optional[str] = None,
        stream: bool = False,
        private: bool = False,
    ) -> Union[str, Generator[str, None, None]]:
        """
        Generate text from prompt (GET request).

        :param prompt: Text prompt (required)
        :param model: Model name
        :param seed: Seed int for reproducibility
        :param temperature: Randomness 0.0-3.0
        :param top_p: Diversity 0.0-1.0
        :param presence_penalty: -2.0 to 2.0 penalty
        :param frequency_penalty: -2.0 to 2.0 penalty
        :param json_mode: Receive response as JSON string (then parse)
        :param system: System prompt (URL-encoded)
        :param stream: If True, returns a generator streaming SSE text chunks
        :param private: Prevent response from public feed
        :return: Generated text string or generator yielding strings if stream=True
        """
        if not prompt:
            raise ValueError("Prompt is required for text generation")

        encoded_prompt = urllib.parse.quote(prompt)
        url = f"{self.TEXT_BASE}/{encoded_prompt}"

        params = {}
        if model:
            params["model"] = model
        if seed is not None:
            params["seed"] = seed
        if temperature is not None:
            params["temperature"] = temperature
        if top_p is not None:
            params["top_p"] = top_p
        if presence_penalty is not None:
            params["presence_penalty"] = presence_penalty
        if frequency_penalty is not None:
            params["frequency_penalty"] = frequency_penalty
        if json_mode:
            params["json"] = "true"
        if system:
            params["system"] = urllib.parse.quote(system)
        if stream:
            params["stream"] = "true"
        if private:
            params["private"] = "true"

        params = self._add_auth_to_params(params)

        try:
            if stream:
                if sseclient is None:
                    raise PollinationsAPIError("sseclient-py is required for streaming support. Install with `pip install sseclient-py`")

                resp = self.session.get(url, params=params, timeout=self.timeout, stream=True)
                resp.raise_for_status()
                client = sseclient.SSEClient(resp)
                for event in client.events():
                    if event.data:
                        if event.data == "[DONE]":
                            break
                        yield event.data
            else:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                resp.raise_for_status()
                if json_mode:
                    # API returns *string* JSON; parse it
                    try:
                        return json.loads(resp.text)
                    except json.JSONDecodeError:
                        raise PollinationsAPIError("API returned invalid JSON string")
                else:
                    return resp.text

        except requests.RequestException as e:
            raise PollinationsAPIError(f"Error generating text: {e}")

    def list_text_models(self) -> Union[List[Any], Dict[str, Any]]:
        """
        List available text models (including voices for audio).

        :return: JSON response (list or dict)
        """
        url = f"{self.TEXT_BASE}/models"
        try:
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            raise PollinationsAPIError(f"Error listing text models: {e}")

    # ------------------- OpenAI Compatible POST (Text, Vision, STT, Function Calling) -------------------

    def openai_chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        *,
        seed: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        stream: bool = False,
        json_mode: Optional[bool] = None,
        response_format: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        private: bool = False,
        referrer: Optional[str] = None,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> Union[Dict[str, Any], Generator[Dict[str, Any], None, None]]:
        """
        Call the OpenAI compatible POST endpoint.

        :param model: Model ID (required)
        :param messages: List of message dicts with roles and content
        :param seed: Seed int
        :param temperature: float
        :param top_p: float
        :param presence_penalty: float
        :param frequency_penalty: float
        :param stream: If True, return generator with streaming chunks
        :param json_mode: Legacy boolean for response_format JSON object
        :param response_format: Dict to set response_format, e.g. {"type": "json_object"}
        :param tools: List of function/tool definitions for function calling
        :param tool_choice: Control how tools are used (e.g. "auto" or specific function)
        :param private: True to keep private
        :param referrer: Optional referrer override
        :param extra_payload: Additional keys to add to payload
        :return: JSON response or generator yielding chunks if stream=True
        """
        url = self.OPENAI_ENDPOINT
        headers = self._build_headers({"Content-Type": "application/json"})
        if stream:
            headers["Accept"] = "text/event-stream"

        payload = {
            "model": model,
            "messages": messages,
        }
        if seed is not None:
            payload["seed"] = seed
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if presence_penalty is not None:
            payload["presence_penalty"] = presence_penalty
        if frequency_penalty is not None:
            payload["frequency_penalty"] = frequency_penalty
        if stream:
            payload["stream"] = True
        if json_mode is not None:
            # legacy alias
            payload["jsonMode"] = json_mode
        if response_format:
            payload["response_format"] = response_format
        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice
        if private:
            payload["private"] = True
        if referrer:
            payload["referrer"] = referrer
        elif self.referrer:
            payload["referrer"] = self.referrer

        if extra_payload:
            payload.update(extra_payload)

        try:
            if stream:
                if sseclient is None:
                    raise PollinationsAPIError("sseclient-py is required for streaming support. Install with `pip install sseclient-py`")
                resp = self.session.post(url, headers=headers, json=payload, stream=True, timeout=self.timeout)
                resp.raise_for_status()
                client = sseclient.SSEClient(resp)
                for event in client.events():
                    if event.data:
                        if event.data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(event.data)
                            yield chunk
                        except json.JSONDecodeError:
                            # Non-JSON chunk, yield raw data
                            yield event.data
            else:
                resp = self.session.post(url, headers=headers, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                return resp.json()

        except requests.RequestException as e:
            raise PollinationsAPIError(f"Error in openai_chat_completion: {e}")

    # ------------------- Vision (Image Input) -------------------

    @staticmethod
    def encode_image_base64(image_path: str) -> Optional[str]:
        try:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except FileNotFoundError:
            print(f"Image file not found: {image_path}")
            return None

    def analyze_image_url(self, image_url: str, question: str = "Describe this image:", model: str = "openai", max_tokens: int = 500) -> Optional[Dict[str, Any]]:
        """
        Analyze image from URL using vision capabilities.

        :param image_url: URL of the image
        :param question: Question prompt about the image
        :param model: Vision capable model (default 'openai')
        :param max_tokens: Max tokens for response
        :return: JSON response or None on error
        """
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ],
            "max_tokens": max_tokens
        }
        try:
            response = self.session.post(self.OPENAI_ENDPOINT, headers={"Content-Type": "application/json"}, json=payload, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error analyzing image URL: {e}")
            return None

    def analyze_local_image(self, image_path: str, question: str = "Describe this image:", model: str = "openai", max_tokens: int = 500) -> Optional[Dict[str, Any]]:
        """
        Analyze local image file (base64 encoded).

        :param image_path: Path to local image file
        :param question: Question prompt about the image
        :param model: Vision capable model
        :param max_tokens: Max tokens for response
        :return: JSON response or None on error
        """
        base64_image = self.encode_image_base64(image_path)
        if not base64_image:
            return None

        image_format = image_path.split(".")[-1].lower()
        if image_format not in ['jpeg', 'jpg', 'png', 'gif', 'webp']:
            print(f"Warning: Unknown image format '{image_format}', defaulting to jpeg")
            image_format = "jpeg"

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": f"data:image/{image_format};base64,{base64_image}"}}
                    ]
                }
            ],
            "max_tokens": max_tokens
        }
        try:
            response = self.session.post(self.OPENAI_ENDPOINT, headers={"Content-Type": "application/json"}, json=payload, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error analyzing local image: {e}")
            return None

    # ------------------- Speech-to-Text (Audio Input) -------------------

    @staticmethod
    def encode_audio_base64(audio_path: str) -> Optional[str]:
        try:
            with open(audio_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except FileNotFoundError:
            print(f"Audio file not found: {audio_path}")
            return None

    def transcribe_audio(
        self,
        audio_path: str,
        question: str = "Transcribe this audio",
        model: str = "openai-audio"
    ) -> Optional[str]:
        """
        Transcribe audio file to text.

        :param audio_path: Path to audio file (wav or mp3 recommended)
        :param question: Prompt text to precede audio content
        :param model: Model name, default 'openai-audio'
        :return: Transcribed text or None on error
        """
        base64_audio = self.encode_audio_base64(audio_path)
        if not base64_audio:
            return None
        audio_format = audio_path.split(".")[-1].lower()
        if audio_format not in ['mp3', 'wav']:
            print(f"Warning: Unsupported audio format '{audio_format}'. Supported: mp3, wav.")

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "input_audio", "input_audio": {"data": base64_audio, "format": audio_format}}
                    ]
                }
            ]
        }
        try:
            response = self.session.post(self.OPENAI_ENDPOINT, headers={"Content-Type": "application/json"}, json=payload, timeout=self.timeout)
            response.raise_for_status()
            resp_json = response.json()
            transcription = resp_json.get("choices", [{}])[0].get("message", {}).get("content")
            return transcription
        except requests.RequestException as e:
            print(f"Error transcribing audio: {e}")
            return None

    # ------------------- Text-to-Speech (TTS) -------------------

    def tts_get(
        self,
        text: str,
        voice: str = "alloy",
        save_to: Optional[str] = None,
        model: str = "openai-audio"
    ) -> bytes:
        """
        Text-to-Speech via GET request (best for short texts).

        :param text: Text to synthesize
        :param voice: Voice name (default 'alloy')
        :param save_to: Optional path to save audio (mp3)
        :param model: Model name (must be 'openai-audio')
        :return: Audio bytes
        """
        if not text:
            raise ValueError("Text is required for TTS GET")

        encoded_text = urllib.parse.quote(text)
        url = f"{self.TEXT_BASE}/{encoded_text}"
        params = {"model": model, "voice": voice}
        params = self._add_auth_to_params(params)

        try:
            resp = self.session.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            content_type = resp.headers.get("Content-Type", "")
            if 'audio/mpeg' not in content_type:
                raise PollinationsAPIError(f"Expected audio/mpeg response, got {content_type}\nBody: {resp.text}")
            if save_to:
                with open(save_to, "wb") as f:
                    f.write(resp.content)
            return resp.content

        except requests.RequestException as e:
            raise PollinationsAPIError(f"Error making TTS GET request: {e}")

    def tts_post(
        self,
        text: str,
        voice: str = "alloy",
        model: str = "openai-audio",
        private: bool = False,
        save_to: Optional[str] = None,
        extra_messages: Optional[List[Dict[str, str]]] = None,
        audio_format: str = "mp3"
    ) -> Optional[bytes]:
        """
        Text-to-Speech via POST (OpenAI compatible) for longer texts.

        :param text: Text to synthesize
        :param voice: Voice name
        :param model: Must be 'openai-audio'
        :param private: Keep private feed
        :param save_to: Path to save the audio file (mp3)
        :param extra_messages: List of additional messages to prepend (e.g., system messages)
        :param audio_format: Audio format for returned audio data (default 'mp3')
        :return: Audio bytes or None on failure
        """
        if not text:
            raise ValueError("Text is required for TTS POST")

        messages = extra_messages[:] if extra_messages else []
        messages.append({"role": "user", "content": text})

        payload = {
            "model": model,
            "modalities": ["text", "audio"],
            "audio": {"voice": voice, "format": audio_format},
            "messages": messages,
            "private": private
        }
        if self.referrer:
            payload["referrer"] = self.referrer

        try:
            resp = self.session.post(self.OPENAI_ENDPOINT, headers={"Content-Type": "application/json"}, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()

            # Extract base64 audio data
            audio_b64 = data['choices'][0]['message'].get('audio', {}).get('data')
            if not audio_b64:
                print("No audio data found in response")
                return None

            audio_bytes = base64.b64decode(audio_b64)
            if save_to:
                with open(save_to, "wb") as f:
                    f.write(audio_bytes)
            return audio_bytes

        except requests.RequestException as e:
            print(f"Error in TTS POST request: {e}")
            return None

    # ------------------- SSE Feeds -------------------

    def connect_image_feed(self) -> Generator[Dict[str, Any], None, None]:
        """
        Connect to the image feed SSE stream.
        Requires sseclient-py.

        :yield: Parsed JSON event data for each image
        """
        if sseclient is None:
            raise PollinationsAPIError("sseclient-py is required for SSE feed. Install with `pip install sseclient-py`")

        url = f"{self.IMAGE_BASE}/feed"
        while True:
            try:
                resp = self.session.get(url, timeout=self.timeout, stream=True, headers={"Accept": "text/event-stream"})
                resp.raise_for_status()
                client = sseclient.SSEClient(resp)
                for event in client.events():
                    if event.data:
                        try:
                            yield json.loads(event.data)
                        except json.JSONDecodeError:
                            continue
            except requests.RequestException as e:
                print(f"Image feed connection error: {e}. Reconnecting...")
            except KeyboardInterrupt:
                print("Interrupted by user, stopping image feed")
                break

    def connect_text_feed(self) -> Generator[Dict[str, Any], None, None]:
        """
        Connect to the text feed SSE stream.
        Requires sseclient-py.

        :yield: Parsed JSON event data for each text
        """
        if sseclient is None:
            raise PollinationsAPIError("sseclient-py is required for SSE feed. Install with `pip install sseclient-py`")

        url = f"{self.TEXT_BASE}/feed"
        while True:
            try:
                resp = self.session.get(url, timeout=self.timeout, stream=True, headers={"Accept": "text/event-stream"})
                resp.raise_for_status()
                client = sseclient.SSEClient(resp)
                for event in client.events():
                    if event.data:
                        try:
                            yield json.loads(event.data)
                        except json.JSONDecodeError:
                            continue
            except requests.RequestException as e:
                print(f"Text feed connection error: {e}. Reconnecting...")
            except KeyboardInterrupt:
                print("Interrupted by user, stopping text feed")
                break


# ------------------- Usage Examples -------------------

if __name__ == "__main__":
    api = PollinationsAPI(referrer="my-app")

    # Example 1: Generate image and save
    try:
        img_bytes = api.generate_image("A beautiful sunset over ocean", width=512, height=512, nologo=True, save_to="sunset.jpg")
        print("Image saved to sunset.jpg")
    except PollinationsAPIError as e:
        print(e)

    # Example 2: Generate text completion
    try:
        text = api.generate_text("Explain quantum computing simply", model="openai")
        print("Text generation result:", text)
    except PollinationsAPIError as e:
        print(e)

    # Example 3: OpenAI compatible chat (non-streaming)
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

    # Example 4: Transcribe audio
    transcript = api.transcribe_audio("sample.wav")
    if transcript:
        print("Transcription:", transcript)

    # Example 5: Text to speech (GET)
    try:
        audio_data = api.tts_get("Hello world, this is a test.", voice="nova", save_to="hello.mp3")
        print("Audio saved to hello.mp3")
    except PollinationsAPIError as e:
        print(e)

    # Example 6: List models
    try:
        image_models = api.list_image_models()
        print("Image models:", image_models)
        text_models = api.list_text_models()
        print("Text models:", text_models)
    except PollinationsAPIError as e:
        print(e)