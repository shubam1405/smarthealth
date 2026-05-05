import google.generativeai as genai
from app.core.config import settings
from app.core.logs import logger

genai.configure(api_key=settings.GEMINI_API_KEY)

SYSTEM_PROMPT = """You are MediBot, a knowledgeable and compassionate AI medical assistant integrated into the MediSense Smart Healthcare Platform.

Your role:
- Help patients understand their symptoms, medications, and general health queries
- Assist doctors with quick clinical references, drug information, and differential considerations
- Provide clear, accurate, and empathetic health information

Rules you must strictly follow:
1. ALWAYS recommend consulting a qualified doctor for diagnosis or treatment decisions
2. NEVER prescribe medications or provide specific dosages as medical advice
3. If someone describes a medical emergency (chest pain, difficulty breathing, stroke symptoms etc.), immediately tell them to call emergency services (108 in India / 911 in US)
4. Be empathetic and supportive — many users may be anxious about their health
5. Keep responses concise but complete — use bullet points for clarity when listing symptoms or steps
6. If patient context is provided, use it to personalize responses
7. Do not make up symptoms, drug names, or statistics — if unsure, say so

You are not a replacement for professional medical care. Always make this clear when relevant."""

# Try these models in order until one works
MODEL_PRIORITY = [
    "gemini-1.5-flash-8b",   # lightest quota usage
    "gemini-1.5-flash",
    "gemini-2.0-flash",
]


def build_gemini_history(history: list[dict]) -> list[dict]:
    """
    Convert our message format to Gemini's format.
    IMPORTANT: Gemini uses 'model' not 'assistant' for bot turns.
    Also ensures history alternates user/model correctly.
    """
    gemini_history = []
    for msg in history:
        # Map 'assistant' → 'model' which is what Gemini expects
        role = "user" if msg["role"] == "user" else "model"
        gemini_history.append({
            "role": role,
            "parts": [{"text": msg["content"]}]
        })
    return gemini_history


def chat_with_gemini(
    message: str,
    history: list[dict],
    patient_context: dict | None = None,
) -> str:
    """
    Send a message to Gemini. Tries multiple models in priority order.
    history should NOT include the current message — we send it via chat.send_message().
    """
    # Build patient context prefix (only inject on first message)
    context_prefix = ""
    if patient_context and len(history) <= 1:
        parts = []
        if patient_context.get("full_name"):         parts.append(f"Patient: {patient_context['full_name']}")
        if patient_context.get("gender"):             parts.append(f"Gender: {patient_context['gender']}")
        if patient_context.get("blood_group"):        parts.append(f"Blood group: {patient_context['blood_group']}")
        if patient_context.get("allergies"):          parts.append(f"Known allergies: {patient_context['allergies']}")
        if patient_context.get("active_medications"): parts.append(f"Current medications: {patient_context['active_medications']}")
        if parts:
            context_prefix = f"[Patient context — {', '.join(parts)}]\n\n"

    # history passed here is everything EXCEPT the current message
    gemini_history = build_gemini_history(history)
    full_message = context_prefix + message if context_prefix else message

    last_error = None
    for model_name in MODEL_PRIORITY:
        try:
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=SYSTEM_PROMPT
            )
            chat = model.start_chat(history=gemini_history)
            response = chat.send_message(full_message)
            logger.info(f"Gemini responded | model={model_name} | history_len={len(history)}")
            return response.text

        except Exception as e:
            err_str = str(e)
            is_quota = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower()
            is_unavailable = "404" in err_str or "not found" in err_str.lower()

            if is_quota or is_unavailable:
                logger.warning(f"Model {model_name} skipped: {'quota' if is_quota else 'not found'}")
                last_error = e
                continue

            # Any other error (auth, malformed request, etc.) — log and raise immediately
            logger.error(f"Gemini error on {model_name}: {err_str}")
            raise

    logger.error(f"All Gemini models exhausted. Last error: {last_error}")
    raise last_error
