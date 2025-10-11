import urllib.parse
import base64
import json
import os
import threading
import re
from typing import Optional, List
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Form, Body, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from openai import OpenAI
import uvicorn
import dotenv


dotenv.load_dotenv()

app = FastAPI(title="Skin Analyzer API", version="1.0.0")


def get_client(api_key: Optional[str] = None, base_url: Optional[str] = None) -> OpenAI:
    """Create an OpenAI/OpenRouter client. Both API key and base_url must be provided
    either via parameters or environment variables. This avoids hard-coded endpoints.
    """
    # Use environment variable for API key if not provided
    key = api_key or os.getenv("OPENROUTER_API_KEY")
    base = base_url or os.getenv("OPENROUTER_BASE_URL")
    if not key:
        raise ValueError("OpenRouter API key not set. Please set the OPENROUTER_API_KEY environment variable.")
    return OpenAI(base_url=base, api_key=key)


def build_messages(image_data_url: str) -> list:
    return [
        {
            "role": "system",
            "content": (
                "You are an AI dermatologist. Analyze an input image and return ONLY raw JSON. "
                "Never include markdown, comments, or explanations.\n\n"
                "Rules:\n"
                "- If the image is NOT of human skin, return:{\n"
                "\"predictions\": [\"not_skin\"],\n"
                "\"accuracy\": 0.0,\n"
                "\"recommendations\": [],\n"
                "\"next_steps\": [],\n"
                "\"cure_available\": false\n}"
                "\n- If the image IS human skin, return:{\n"
                "\"predictions\": [\"condition_1\"],\n"
                "\"accuracy\": 0.95,\n"
                "\"recommendations\": [\"...\"],\n"
                "\"next_steps\": [\"...\"],\n"
                "\"cure_available\": true\n}"
                "\nReturn ONLY valid JSON. Do not return markdown, explanations, or any text before or after the JSON. Output must be raw JSON only."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this image strictly as instructed and return raw JSON only."},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        },
    ]


def try_parse_json(text: str):
    # Best-effort: strip code fences and whitespace, then parse
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        # remove possible language tag
        if "\n" in cleaned:
            cleaned = cleaned.split("\n", 1)[1]
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # As last resort, return raw string
        return {"raw": text}


@app.get("/")
def health() -> dict:
    return {
        "status": "ok",
        "service": "skin-analyzer",
        "api_endpoints": [
            {
                "endpoint": "/conversation",
                "method": "POST",
                "description": "Conversational AI endpoint. Accepts prompt via query or JSON body.",
                "curl_examples": [
                    "curl.exe -X POST \"http://127.0.0.1:8000/conversation?prompt=What%20is%20the%20best%20skincare%20routine%20for%20oily%20skin%3F\"",
                    "curl.exe -X POST http://127.0.0.1:8000/conversation -H \"Content-Type: application/json\" -d '{\"prompt\": \"What is the best skincare routine for oily skin?\"}'"
                ]
            },
            {
                "endpoint": "/diagnostic_questions",
                "method": "POST",
                "description": "Returns five diagnostic questions for suspected skin diseases.",
                "curl_examples": [
                    "curl.exe -X POST \"http://127.0.0.1:8000/diagnostic_questions?suspected_diseases=acne,eczema,psoriasis\""
                ]
            },
            {
                "endpoint": "/analyze_by_path",
                "method": "POST",
                "description": "Analyze an image by absolute path on the server.",
                "curl_examples": [
                    "curl.exe -X POST \"http://127.0.0.1:8000/analyze_by_path?image=C:\\path\\to\\image.jpg\""
                ]
            },
            {
                "endpoint": "/diagnosis_result",
                "method": "POST",
                "description": "Returns diagnosis result based on suspected diseases, questions, and user response.",
                "curl_examples": [
                    "curl.exe -X POST \"http://127.0.0.1:8000/diagnosis_result?suspected_diseases=acne,eczema,psoriasis&questions=Does%20your%20skin%20itch%3F%20Is%20it%20dry%3F&response=Yes%2C%20it%20itches%20and%20is%20very%20dry\""
                ]
            }
        ]
    }


@app.post("/analyze_by_path")
def analyze_by_path(
    image: str = Query(..., description="Absolute path to an image file on the server"),
    api_key: Optional[str] = Query(default=None, description="Optional OpenRouter API key override"),
    base_url: Optional[str] = Query(default=None, description="Optional OpenRouter base URL override"),
    model: Optional[str] = Query(default=None, description="OpenRouter model identifier (required if OPENROUTER_IMAGE_MODEL env is not set)"),
) -> JSONResponse:
    """Analyze an image that already exists on the server filesystem.

    WARNING: this endpoint reads files from the server's filesystem. Do NOT expose
    this endpoint in production unless you trust the callers. This is intended for
    local testing only.
    """
    try:
        if not os.path.isabs(image):
            raise HTTPException(status_code=400, detail="Please provide an absolute image path.")
        if not os.path.exists(image):
            raise HTTPException(status_code=404, detail=f"File not found: {image}")

        with open(image, "rb") as f:
            contents = f.read()
        if not contents:
            raise HTTPException(status_code=400, detail="File is empty")


        mime_type = "image/png"
        # try to infer simple mime type from extension
        _, ext = os.path.splitext(image.lower())
        if ext in [".jpg", ".jpeg"]:
            mime_type = "image/jpeg"
        elif ext == ".png":
            mime_type = "image/png"
        elif ext == ".gif":
            mime_type = "image/gif"

        base64_image = base64.b64encode(contents).decode("utf-8")
        image_data_url = f"data:{mime_type};base64,{base64_image}"

        # Hard-coded model for now
        chosen_model = "google/gemma-3-27b-it:free"

        client = get_client()
        try:
            completion = client.chat.completions.create(
                model=chosen_model,
                messages=build_messages(image_data_url),
            )
            raw = completion.choices[0].message.content
            data = try_parse_json(raw)
            return JSONResponse(content=data)
        except Exception as e:
            if hasattr(e, 'status_code') and e.status_code == 404:
                return JSONResponse(status_code=404, content={"detail": f"Model '{chosen_model}' not available. Please check model name or use a different model."})
            if '404' in str(e) and 'No endpoints found' in str(e):
                return JSONResponse(status_code=404, content={"detail": f"Model '{chosen_model}' not available. Please check model name or use a different model."})
            raise HTTPException(status_code=500, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# For local dev with: uvicorn API_Calls.main:app --reload


# Conversation manager: simple file-backed conversation and question counter
class ConversationManager:
    def __init__(self, convo_file: str = "conversation_history.txt", counter_file: str = "question_counter.txt"):
        self.convo_file = convo_file
        self.counter_file = counter_file
        self.lock = threading.Lock()

    def read_conversation(self) -> str:
        try:
            with open(self.convo_file, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return ""

    def append_conversation(self, user: str, ai: str) -> None:
        try:
            with self.lock:
                with open(self.convo_file, "a", encoding="utf-8") as f:
                    f.write(f"User: {user}\nAI: {ai}\n")
        except Exception:
            pass

    def read_counter(self) -> int:
        try:
            with open(self.counter_file, "r", encoding="utf-8") as f:
                return int(f.read().strip() or "0")
        except Exception:
            return 0

    def increment_counter(self, by: int = 1) -> int:
        with self.lock:
            val = self.read_counter() + by
            try:
                with open(self.counter_file, "w", encoding="utf-8") as f:
                    f.write(str(val))
            except Exception:
                pass
            return val


def sanitize(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<\|.*?\|>", "", text)
    text = re.sub(r'\{"role":.*\}\s*', "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Instruction used for conversational replies (kept from previous script)
INSTRUCTION = (
    "You're an empathetic AI skin companion-like their dermatologist best friend. "
    "Speak short, warm, natural: one or two sentences. Ask questions that feel real-Any allergies? "
    "Rough day?-only when needed, then wait. Learn their whole life: skin type, schedule, triggers (sugar? dairy?), "
    "allergies, sleep, hormones, diet. Never dump info. When ready, output JSON only-no extra chatter: "
    "{ concern: acne, recommendation: salicylic cleanser, reasons: cuts oil, clears pores, alternatives: [benzoyl peroxide, niacinamide serum], confidence: 9, user_profile: { allergies: [fragrance], triggers: [dairy], schedule: work till 8, routine: AM: wash + moisturize. PM: serum + sleep } } "
    "Backend reads JSON, pulls safe products from your database, shows user. Then: daily check-in, mornings-like, Hey, how'd last night go? Skin calmer? "
    "If they say Stressed, no time, reply: Cool-moisturize in the car, thirty seconds. You got this. Celebrate: You stuck with it-proud of you. "
    "Nudge: Forgot? No stress-just try tonight. Make it fun-tie it to their coffee, playlist, commute. You're not a doctor. "
    "You're the buddy who cares, knows skincare, keeps them going."
)


conv_manager = ConversationManager()

@app.post("/conversation")
async def conversation_endpoint(request: Request) -> JSONResponse:
    """Accepts ONLY JSON body. Query params are not supported.
    JSON body format example:
      {"prompt": "I have combination skin.", "summary": "Evening flare", "model": "openai/gpt-oss-20b:free"}
    """
    if not request.headers.get("content-type", "").startswith("application/json"):
        raise HTTPException(status_code=400, detail="Content-Type must be application/json and body must be valid JSON.")
    try:
        raw_body = await request.body()
        if not raw_body or not raw_body.strip():
            raise HTTPException(status_code=400, detail="Missing JSON body.")
        try:
            body_data = json.loads(raw_body)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Malformed JSON body.")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Malformed or missing JSON body.")

    prompt_val = body_data.get("prompt")
    summary_val = body_data.get("summary")
    api_key_val = body_data.get("api_key")
    model_val = body_data.get("model")

    if not prompt_val or not isinstance(prompt_val, str) or not prompt_val.strip():
        raise HTTPException(status_code=400, detail="Missing 'prompt' in JSON body.")

    user_input_clean = sanitize(prompt_val)
    summary_note = sanitize(summary_val) if summary_val else None

    prev_convo = conv_manager.read_conversation()
    qcount = conv_manager.read_counter()
    composite = (
        f"{INSTRUCTION}\n\nPrevious conversation:\n{prev_convo}\n\n"
        f"Question Counter: {qcount} (AI can only ask a few important questions)\n"
    )
    if summary_note:
        composite += f"Summary: {summary_note}\n"
    composite += f"User: {user_input_clean}"

    chosen_model = model_val or os.environ.get("OPENROUTER_CHAT_MODEL", "openai/gpt-oss-20b:free")
    try:
        client = get_client(api_key_val, base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"))
        completion = client.chat.completions.create(
            extra_body={},
            model=chosen_model,
            messages=[{"role": "user", "content": composite}],
        )
        raw = completion.choices[0].message.content
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    reply_clean = sanitize(raw)
    num_questions = len(re.findall(r"\?", reply_clean))
    new_qcount = (
        conv_manager.increment_counter(num_questions)
        if num_questions > 0
        else conv_manager.read_counter()
    )
    conv_manager.append_conversation(user_input_clean, reply_clean)
    return JSONResponse(
        content={
            "reply": reply_clean,
            "raw": raw,
            "question_counter": new_qcount,
            "model_used": chosen_model,
        }
    )


@app.get("/conversation")
def get_conversation() -> JSONResponse:
    """Return the current conversation history and question counter."""
    data = conv_manager.read_conversation()
    qcount = conv_manager.read_counter()
    return JSONResponse(content={"conversation": data, "question_counter": qcount})



@app.post("/diagnostic_questions")
async def diagnostic_questions(request: Request) -> JSONResponse:
    """Accepts ONLY JSON body. Example:
    {"suspected_diseases": ["acne", "eczema", "psoriasis"]}
    """
    if not request.headers.get("content-type", "").startswith("application/json"):
        raise HTTPException(status_code=400, detail="Content-Type must be application/json and body must be valid JSON.")
    try:
        raw_body = await request.body()
        if not raw_body or not raw_body.strip():
            raise HTTPException(status_code=400, detail="Missing JSON body.")
        try:
            body = json.loads(raw_body)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Malformed JSON body.")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Malformed or missing JSON body.")

    diseases_list = body.get("suspected_diseases")
    if not diseases_list or not isinstance(diseases_list, list) or not all(isinstance(d, str) and d.strip() for d in diseases_list):
        raise HTTPException(status_code=400, detail="'suspected_diseases' must be a non-empty list of strings in the JSON body.")

    diseases_list = [d.strip() for d in diseases_list if d.strip()]
    diseases_str = ", ".join(diseases_list)
    prompt = (
        f"Given these suspected skin diseases: {diseases_str}. "
        "Generate five clinically relevant diagnostic questions to help distinguish between them. "
        "Each question should be based on symptoms, triggers, progression, or visual characteristics, "
        "and should help eliminate or confirm specific diseases from the list. "
        "Return ONLY a JSON object with keys 'suspected_diseases' and 'questions'. "
        "No Markdown, no explanations, no text outside the JSON."
    )
    model_name = body.get("model") or "openai/gpt-oss-20b:free"
    api_key = body.get("api_key")
    try:
        client = get_client(api_key)
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = completion.choices[0].message.content
        try:
            data = json.loads(raw)
            if (
                isinstance(data.get("suspected_diseases"), list)
                and isinstance(data.get("questions"), list)
            ):
                return JSONResponse(content=data)
        except Exception:
            pass
        questions = re.findall(r"(?:^|\n)[\d\-\*\.]?\s*([A-Z][^\n\r\?]+\?)", raw)
        if not questions:
            questions = [line.strip() for line in raw.splitlines() if line.strip().endswith("?")]
        return JSONResponse(content={"suspected_diseases": diseases_list, "questions": questions[:5]})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/diagnosis_result")
async def diagnosis_result(request: Request) -> JSONResponse:
    """Accepts ONLY JSON body. Example:
    {"suspected_diseases": ["acne", "eczema", "psoriasis"], "questions": "...", "response": "..."}
    """
    if not request.headers.get("content-type", "").startswith("application/json"):
        raise HTTPException(status_code=400, detail="Content-Type must be application/json and body must be valid JSON.")
    try:
        raw_body = await request.body()
        if not raw_body or not raw_body.strip():
            raise HTTPException(status_code=400, detail="Missing JSON body.")
        try:
            body = json.loads(raw_body)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Malformed JSON body.")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Malformed or missing JSON body.")

    diseases_list = body.get("suspected_diseases")
    questions = body.get("questions")
    response = body.get("response")
    model_name = body.get("model") or "openai/gpt-oss-20b:free"
    api_key = body.get("api_key")

    if not diseases_list or not isinstance(diseases_list, list) or not all(isinstance(d, str) and d.strip() for d in diseases_list):
        raise HTTPException(status_code=400, detail="'suspected_diseases' must be a non-empty list of strings in the JSON body.")
    if not questions or not isinstance(questions, str) or not questions.strip():
        raise HTTPException(status_code=400, detail="'questions' must be a non-empty string in the JSON body.")
    if not response or not isinstance(response, str) or not response.strip():
        raise HTTPException(status_code=400, detail="'response' must be a non-empty string in the JSON body.")

    diseases_list = [d.strip() for d in diseases_list if d.strip()]
    prompt = (
        f"Suspected skin diseases: {', '.join(diseases_list)}.\n"
        f"Questions asked: {questions}\n"
        f"User's response: {response}\n\n"
        "Based on the user's answers, determine which skin disease is most likely and which are unlikely. "
        "Explain why, briefly, in clinical reasoning terms (symptoms, triggers, appearance, or pattern). "
        "Return ONLY a JSON object with:\n"
        "- 'most_likely_disease': the single best-matching disease,\n"
        "- 'reasoning': short text explaining why it fits best,\n"
        "- 'accuracy_scores': a dictionary with each disease name as key and a confidence percentage (0–100) as value.\n"
        "No Markdown, no extra text, only valid JSON output."
    )
    try:
        client = get_client(api_key)
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = completion.choices[0].message.content
        try:
            data = json.loads(raw)
            if (
                "most_likely_disease" in data
                and "reasoning" in data
                and "accuracy_scores" in data
            ):
                return JSONResponse(content=data)
        except Exception:
            pass
        most_likely = None
        reasoning = None
        accuracy_scores = {}
        match = re.search(r"most_likely.*?:\s*(\w+)", raw, re.IGNORECASE)
        if match:
            most_likely = match.group(1)
        reasoning_match = re.search(r"reasoning.*?:\s*(.+)", raw, re.IGNORECASE)
        if reasoning_match:
            reasoning = reasoning_match.group(1)
        for d in diseases_list:
            conf_match = re.search(fr"{d}\D*(\d+)%", raw, re.IGNORECASE)
            if conf_match:
                accuracy_scores[d] = int(conf_match.group(1))
        return JSONResponse(content={
            "most_likely_disease": most_likely or "Unknown",
            "reasoning": reasoning or "No reasoning extracted",
            "accuracy_scores": accuracy_scores or {d: 0 for d in diseases_list}
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    

if __name__ == "__main__":
    # Run this module directly (correct target 'main:app')
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

