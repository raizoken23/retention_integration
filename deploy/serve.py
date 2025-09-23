# serve a power retention model with OpenAI-compatible API (async streaming)
"""
OpenAI-compatible API server for Power Retention models with streaming.

Usage:
    python serve_power_async.py --model ./models/powercoder --tokenizer bigcode/starcoder2-3b --port 8000

Dependencies:
    Install with: uv sync --group server

API Endpoints:
    POST /v1/chat/completions - OpenAI-compatible chat completions (supports stream)
    POST /v1/completions      - OpenAI-compatible text completions (supports stream)
    GET /v1/models - List available models
    GET /health - Health check
"""
import argparse
import json
import logging
import threading
import time
import uuid
from typing import Any, Dict, Generator, List, Optional

import models.powercoder  # noqa: F401
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import (  # type: ignore
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    TextIteratorStreamer,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# OpenAI-compatible request/response models (non-streaming shapes)
class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    max_new_tokens: Optional[int] = 1024
    temperature: Optional[float] = 1.5
    top_p: Optional[float] = 0.2
    stream: Optional[bool] = False
    skip_special_tokens: Optional[bool] = True
    repetition_penalty: Optional[float] = 1.2


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage


# OpenAI-compatible text completion models (non-streaming shapes)
class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_new_tokens: Optional[int] = 1024
    temperature: Optional[float] = 1.5
    top_p: Optional[float] = 0.9
    stream: Optional[bool] = False
    skip_special_tokens: Optional[bool] = True
    repetition_penalty: Optional[float] = 1.2


class CompletionChoice(BaseModel):
    index: int
    text: str
    finish_reason: str


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Usage


def parse_args():
    parser = argparse.ArgumentParser(
        description="Serve a power retention model with OpenAI-compatible API (async streaming)"
    )
    # model config
    parser.add_argument("--model", type=str, default="./models/powercoder")
    parser.add_argument("--tokenizer", type=str, default="bigcode/starcoder2-3b")
    # server config
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()


def _generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 1024,
    temperature: float = 1.5,
    top_p: float = 0.2,
    repetition_penalty: float = 1.2,
    skip_special_tokens: bool = True,
) -> Dict[str, Any]:
    """Generate text with timing and token statistics (non-streaming)."""
    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)  # type: ignore
    input_length = inputs.input_ids.shape[1]

    config = GenerationConfig(
        do_sample=True,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        bos_token_id=tokenizer.bos_token_id,  # type: ignore
        eos_token_id=tokenizer.eos_token_id,  # type: ignore
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,  # type: ignore
    )

    with torch.no_grad():
        outputs = model.generate(  # type: ignore
            inputs.input_ids,
            generation_config=config,
            attention_mask=inputs.attention_mask,
        )

    end_time = time.time()
    generation_time = end_time - start_time

    generated_tokens = outputs[0][input_length:]

    prompt_tokens = int(inputs.attention_mask.sum().item())
    if getattr(tokenizer, "pad_token_id", None) is not None:  # type: ignore
        completion_tokens = int(
            (generated_tokens != tokenizer.pad_token_id).sum().item()  # type: ignore
        )
    else:
        completion_tokens = int(generated_tokens.shape[0])

    full_text = tokenizer.decode(
        outputs[0], skip_special_tokens=skip_special_tokens  # type: ignore
    )
    completion_text = tokenizer.decode(
        generated_tokens, skip_special_tokens=skip_special_tokens  # type: ignore
    )

    tokens_per_second = (
        completion_tokens / generation_time if generation_time > 0 else 0
    )

    logger.info(
        f"Generated {completion_tokens} tokens in {generation_time:.2f}s "
        f"({tokens_per_second:.2f} tokens/s)"
    )

    return {
        "full_text": full_text,
        "completion_text": completion_text,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "generation_time": generation_time,
        "tokens_per_second": tokens_per_second,
    }


# Global variables for model and tokenizer
model: Optional[AutoModelForCausalLM] = None
tokenizer: Optional[AutoTokenizer] = None


def _sse(data: Dict[str, Any]) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _chat_stream_sse(
    request_id: str,
    model_name: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    skip_special_tokens: bool,
) -> Generator[str, None, None]:
    assert model is not None and tokenizer is not None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)  # type: ignore

    config = GenerationConfig(
        do_sample=True,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        bos_token_id=tokenizer.bos_token_id,  # type: ignore
        eos_token_id=tokenizer.eos_token_id,  # type: ignore
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,  # type: ignore
    )

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_special_tokens=skip_special_tokens,
        skip_prompt=True,
    )

    def generate_func():
        with torch.no_grad():
            model.generate(  # type: ignore
                inputs.input_ids,
                generation_config=config,
                attention_mask=inputs.attention_mask,
                streamer=streamer,
            )

    thread = threading.Thread(target=generate_func)
    thread.start()

    created_ts = int(time.time())
    sent_role = False
    completion_text = ""

    try:
        logger.info(f"[chat_stream] start streaming id={request_id}")
        for text in streamer:
            if not text:
                continue
            chunk: Dict[str, Any] = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": None,
                    }
                ],
            }
            if not sent_role:
                chunk["choices"][0]["delta"]["role"] = "assistant"
                sent_role = True
            chunk["choices"][0]["delta"]["content"] = text
            completion_text += text
            logger.info(
                f"[chat_stream] send {len(text)} chars (total={len(completion_text)})"
            )
            yield _sse(chunk)
    except Exception as e:
        logger.exception(f"[chat_stream] error: {e}")
    finally:
        thread.join()

    done_chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created_ts,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
    }
    logger.info(f"[chat_stream] done id={request_id} total_chars={len(completion_text)}")
    yield _sse(done_chunk)
    yield "data: [DONE]\n\n"


def _text_stream_sse(
    request_id: str,
    model_name: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    skip_special_tokens: bool,
) -> Generator[str, None, None]:
    assert model is not None and tokenizer is not None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)  # type: ignore

    config = GenerationConfig(
        do_sample=True,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        bos_token_id=tokenizer.bos_token_id,  # type: ignore
        eos_token_id=tokenizer.eos_token_id,  # type: ignore
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,  # type: ignore
    )

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_special_tokens=skip_special_tokens,
        skip_prompt=True,
    )

    def generate_func():
        with torch.no_grad():
            model.generate(  # type: ignore
                inputs.input_ids,
                generation_config=config,
                attention_mask=inputs.attention_mask,
                streamer=streamer,
            )

    thread = threading.Thread(target=generate_func)
    thread.start()

    created_ts = int(time.time())

    try:
        logger.info(f"[text_stream] start streaming id={request_id}")
        for text in streamer:
            if not text:
                continue
            chunk = {
                "id": request_id,
                "object": "text_completion.chunk",
                "created": created_ts,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "text": text,
                        "finish_reason": None,
                    }
                ],
            }
            logger.info(f"[text_stream] send {len(text)} chars")
            yield _sse(chunk)
    except Exception as e:
        logger.exception(f"[text_stream] error: {e}")
    finally:
        thread.join()

    done_chunk = {
        "id": request_id,
        "object": "text_completion.chunk",
        "created": created_ts,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "text": "",
                "finish_reason": "stop",
            }
        ],
    }
    yield _sse(done_chunk)
    yield "data: [DONE]\n\n"


def create_app() -> FastAPI:
    """Create FastAPI application with OpenAI-compatible endpoints (streaming)."""
    app = FastAPI(title="Power Retention Model Server (Async)", version="1.0.0")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        """OpenAI-compatible chat completions endpoint supporting streaming."""
        if model is None or tokenizer is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Convert messages to prompt (simple concatenation, same as non-streaming)
        prompt = ""
        for message in request.messages:
            if message.role == "system":
                prompt += f"System: {message.content}\n"
            elif message.role == "user":
                prompt += f"User: {message.content}\n"
            elif message.role == "assistant":
                prompt += f"Assistant: {message.content}\n"

        if request.stream:
            response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
            logger.info(f"Streaming chat completion {response_id}")

            def event_generator():
                yield from _chat_stream_sse(
                    request_id=response_id,
                    model_name=request.model,
                    prompt=prompt,
                    max_new_tokens=request.max_new_tokens or 1024,
                    temperature=request.temperature or 1.5,
                    top_p=request.top_p or 0.2,
                    repetition_penalty=request.repetition_penalty or 1.2,
                    skip_special_tokens=request.skip_special_tokens or True,
                )

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        try:
            result = _generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=request.max_new_tokens or 1024,
                temperature=request.temperature or 1.5,
                top_p=request.top_p or 0.2,
                skip_special_tokens=request.skip_special_tokens or True,
                repetition_penalty=request.repetition_penalty or 1.2,
            )

            response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
            choice = Choice(
                index=0,
                message=Message(role="assistant", content=result["completion_text"]),
                finish_reason="stop",
            )
            usage = Usage(
                prompt_tokens=result["prompt_tokens"],
                completion_tokens=result["completion_tokens"],
                total_tokens=result["total_tokens"],
            )

            return ChatCompletionResponse(
                id=response_id,
                created=int(time.time()),
                model=request.model,
                choices=[choice],
                usage=usage,
            )
        except Exception as e:
            logger.error(f"Error generating completion: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/v1/completions")
    async def completions(request: CompletionRequest):
        """OpenAI-compatible text completions endpoint supporting streaming."""
        logger.info(f"Completions request: {request}")
        if model is None or tokenizer is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        if request.stream:
            response_id = f"cmpl-{uuid.uuid4().hex[:8]}"
            logger.info(f"Streaming text completion {response_id}")

            def event_generator():
                yield from _text_stream_sse(
                    request_id=response_id,
                    model_name=request.model,
                    prompt=request.prompt,
                    max_new_tokens=request.max_new_tokens or 1024,
                    temperature=request.temperature or 1.5,
                    top_p=request.top_p or 0.9,
                    repetition_penalty=request.repetition_penalty or 1.2,
                    skip_special_tokens=request.skip_special_tokens or True,
                )

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        try:
            result = _generate(
                model=model,
                tokenizer=tokenizer,
                prompt=request.prompt,
                max_new_tokens=request.max_new_tokens or 1024,
                temperature=request.temperature or 1.5,
                top_p=request.top_p or 0.9,
                skip_special_tokens=request.skip_special_tokens or True,
                repetition_penalty=request.repetition_penalty or 1.2,
            )

            response_id = f"cmpl-{uuid.uuid4().hex[:8]}"
            choice = CompletionChoice(
                index=0,
                text=result["completion_text"],
                finish_reason="stop",
            )
            usage = Usage(
                prompt_tokens=result["prompt_tokens"],
                completion_tokens=result["completion_tokens"],
                total_tokens=result["total_tokens"],
            )

            logger.info(
                f"Request {response_id}: {result['tokens_per_second']:.2f} tokens/s\n "
                f"completion_text: {result['completion_text']}"
            )
            return CompletionResponse(
                id=response_id,
                created=int(time.time()),
                model=request.model,
                choices=[choice],
                usage=usage,
            )
        except Exception as e:
            logger.error(f"Error generating completion: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/v1/models")
    async def list_models():
        """List available models."""
        return {
            "object": "list",
            "data": [
                {
                    "id": "power-retention",
                    "object": "./models/powercoder",
                    "created": int(time.time()),
                    "owned_by": "manifest-ai",
                }
            ],
        }

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy"}

    return app


def main():
    """Initialize and start the server (async streaming)."""
    global model, tokenizer

    GREEN = "\033[32m"
    BLUE = "\033[34m"
    RESET = "\033[0m"

    args = parse_args()
    print(f"{BLUE}args: {GREEN}{args.__dict__}{RESET}")

    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Set up model
    model.eval()
    model = model.to("cuda")  # type: ignore

    if args.compile:
        logger.info("Compiling model...")
        model = torch.compile(model)

    logger.info("Model loaded successfully!")

    # Create and run server
    app = create_app()
    logger.info(f"Starting server on {args.host}:{args.port}")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
