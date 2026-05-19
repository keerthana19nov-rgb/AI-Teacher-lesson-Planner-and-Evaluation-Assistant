import os
from queue import Queue, Empty
from threading import Thread

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

import ollama
from huggingface_hub import InferenceClient

from knowledge_base import load_uploaded_file

load_dotenv()

# =========================
# CONFIGURATION
# =========================

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:latest")
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
OLLAMA_REQUEST_TIMEOUT = float(os.getenv("OLLAMA_REQUEST_TIMEOUT", "300"))

HUGGINGFACE_API_TOKEN = os.getenv(
    "HUGGINGFACE_API_TOKEN",
    ""
)

HUGGINGFACE_MODEL = os.getenv(
    "HUGGINGFACE_MODEL",
    "google/flan-t5-large"
)
HUGGINGFACE_REQUEST_TIMEOUT = float(os.getenv("HUGGINGFACE_REQUEST_TIMEOUT", "30"))

OLLAMA_CLIENT = ollama.Client(host=OLLAMA_API_URL)

# =========================
# FLASK APP
# =========================

app = Flask(
    __name__,
    static_folder='.',
    static_url_path='/static',
    template_folder='.',
)

app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

# =========================
# KNOWLEDGE STORAGE
# =========================

knowledge_store = {
    "text": ""
}

# =========================
# SIMPLE FALLBACK ANSWERS
# =========================

def get_fallback_answer(question: str, language: str = "english") -> str:
    """Generate a simple answer for common questions if AI fails."""

    q_lower = question.lower()
    language_name = {
        "english": "English",
        "kannada": "Kannada",
        "hindi": "Hindi",
        "telugu": "Telugu",
        "tamil": "Tamil"
    }.get(language, "English")

    if "what is python" in q_lower or "python programming" in q_lower:
        base = "Python is a high-level programming language known for its simplicity and readability. It's widely used in web development, data science, AI, and education. Python emphasizes code readability with significant indentation."
    elif "what is ai" in q_lower or "artificial intelligence" in q_lower:
        base = "Artificial Intelligence is computer systems designed to perform tasks that typically require human intelligence, such as learning from experience, recognizing patterns, and understanding language."
    elif "what is machine learning" in q_lower:
        base = "Machine Learning is a branch of AI where systems learn from data without being explicitly programmed."
    elif "hello" in q_lower or "hi" in q_lower or "hey" in q_lower:
        base = "Hello! I'm EduGenAI Pro, your AI educational assistant. I can help you with education, coding, science, math, and answer almost any question."
    elif "what is education" in q_lower:
        base = "Education is the process of learning knowledge, skills, values, and habits through study, teaching, or experience."
    else:
        base = f"I understand you're asking: '{question}'. I will do my best to answer that in {language_name}."

    translations = {
        "english": base,
        "kannada": {
            "Python is a high-level programming language known for its simplicity and readability. It's widely used in web development, data science, AI, and education. Python emphasizes code readability with significant indentation.":
                "Python ಒಂದು ಹೆಚ್ಚು ಮಟ್ಟದ ಪ್ರೋಗ್ರಾಮಿಂಗ್ ಭಾಷೆಯಾಗಿದ್ದು, ಅದು ಸರಳತೆ ಮತ್ತು ಓದಲು ಸುಲಭ ಎನ್ನುವ ಲಕ್ಷಣಗಳಿಗಾಗಿ ಖ್ಯಾತಿಯಾಗಿದೆ. ಇದು ವೆಬ್ ಅಭಿವೃದ್ಧಿ, ಡೇಟಾ ಸೈನ್ಸ್, AI, ಮತ್ತು ಶಿಕ್ಷಣದಲ್ಲಿ ವ್ಯಾಪಕವಾಗಿ ಬಳಕೆಯಾಗುತ್ತದೆ.",
            "Artificial Intelligence is computer systems designed to perform tasks that typically require human intelligence, such as learning from experience, recognizing patterns, and understanding language.":
                "ಕೃತಕ ಬುದ್ಧಿಮತ್ತೆ ಎಂದರೆ ಅನುಭವದಿಂದ ಕಲಿಯುವುದು, ಮಾದರಿಗಳನ್ನು ಗುರುತಿಸುವುದು ಮತ್ತು ಭಾಷೆಯನ್ನು ಅರ್ಥಮಾಡಿಕೊಳ್ಳುವುದು ಮುಂತಾದ ಮಾನವ ಬುದ್ಧಿಮತ್ತೆಯನ್ನು ಅವಲಂಬಿಸುವ ಕಾರ್ಯಗಳನ್ನು ನಿರ್ವಹಿಸಲು ರೂಪಿಸಲ್ಪಟ್ಟ ಕಂಪ್ಯೂಟರ್ ವ್ಯವಸ್ಥೆಗಳಾಗಿವೆ.",
            "Machine Learning is a branch of AI where systems learn from data without being explicitly programmed.":
                "ಮಷಿನ್ ಲೆರ್ನಿಂಗ್ ಒಂದು AI ಶಾಖೆಯಾಗಿದೆ, ಇದರಲ್ಲಿ ವ್ಯವಸ್ಥೆಗಳು ಸ್ಪಷ್ಟವಾಗಿ ಪ್ರೋಗ್ರಾಮ್ ಮಾಡದೆ ಡೇಟಾದಿಂದ ಕಲಿಯುತ್ತವೆ.",
            "Hello! I'm EduGenAI Pro, your AI educational assistant. I can help you with education, coding, science, math, and answer almost any question.":
                "ಹೆಲೋ! ನಾನು EduGenAI Pro, ನಿಮ್ಮ AI ಶೈಕ್ಷಣಿಕ ಸಹಾಯಕ. ನಾನು ಶಿಕ್ಷಣ, ಕೋಡಿಂಗ್, ವಿಜ್ಞಾನ, ಗಣಿತ ಮತ್ತು ಯಾವುದೇ ಪ್ರಶ್ನೆಗಳಿಗೆ ಸಹಾಯ ಮಾಡಬಹುದು.",
            "Education is the process of learning knowledge, skills, values, and habits through study, teaching, or experience.":
                "ಶಿಕ್ಷಣವು ಅಭ್ಯಾಸ, ಕಲಿಕೆ ಅಥವಾ ಅನುಭವದ ಮೂಲಕ ಜ್ಞಾನ, ಕೌಶಲ್ಯಗಳು, ಮೌಲ್ಯಗಳು ಮತ್ತು ಆಚಾರಗಳನ್ನು ಕಲಿಯುವ ಪ್ರಕ್ರಿಯೆಯಾಗಿರುತ್ತದೆ.",
            base:
                "ನಾನು ನಿಮ್ಮ ಪ್ರಶ್ನೆಯನ್ನು ಗುರುತಿಸಿಕೊಂಡು, ಅದಕ್ಕಾಗಿ ಉತ್ತಮ ಉತ್ತರವನ್ನು ನೀಡಲು ಪ್ರಯತ್ನಿಸುತ್ತೇನೆ."
        },
        "hindi": {
            "Python is a high-level programming language known for its simplicity and readability. It's widely used in web development, data science, AI, and education. Python emphasizes code readability with significant indentation.":
                "Python एक उच्च-स्तरीय प्रोग्रामिंग भाषा है जो अपनी सरलता और पठनीयता के लिए जानी जाती है। इसे वेब विकास, डेटा विज्ञान, AI और शिक्षा में व्यापक रूप से उपयोग किया जाता है।",
            "Artificial Intelligence is computer systems designed to perform tasks that typically require human intelligence, such as learning from experience, recognizing patterns, and understanding language.":
                "कृत्रिम बुद्धिमत्ता कंप्यूटर सिस्टम हैं जो अनुभव से सीखने, पैटर्न पहचानने और भाषा को समझने जैसे आमतौर पर मानवीय बुद्धिमत्ता वाले कार्य करने के लिए तैयार किए गए हैं।",
            "Machine Learning is a branch of AI where systems learn from data without being explicitly programmed.":
                "मशीन लर्निंग AI की एक शाखा है जहां सिस्टम स्पष्ट रूप से प्रोग्राम किए बिना डेटा से सीखते हैं।",
            "Hello! I'm EduGenAI Pro, your AI educational assistant. I can help you with education, coding, science, math, and answer almost any question.":
                "नमस्ते! मैं EduGenAI Pro हूं, आपका AI शैक्षिक सहायक। मैं आपको शिक्षा, कोडिंग, विज्ञान, गणित में मदद कर सकता हूं और लगभग किसी भी प्रश्न का उत्तर दे सकता हूं।",
            "Education is the process of learning knowledge, skills, values, and habits through study, teaching, or experience.":
                "शिक्षा अध्ययन, शिक्षण या अनुभव के माध्यम से ज्ञान, कौशल, मूल्यों और आदतों को सीखने की प्रक्रिया है।",
            base:
                "मैं आपके प्रश्न को समझ रहा हूँ और इसका उत्तर देने का प्रयास कर रहा हूँ।"
        },
        "telugu": {
            "Python is a high-level programming language known for its simplicity and readability. It's widely used in web development, data science, AI, and education. Python emphasizes code readability with significant indentation.":
                "Python ఒక హై-లెవల్ ప్రోగ్రామింగ్ భాష, ఇది దాని సరళత్వం మరియు చదవడానికి సులభత కోసం ప్రసిద్ధి చెందింది. ఇది వెబ్ అభివృద్ధి, డేటా సైన్స్, AI, మరియు విద్యలో విస్తృతంగా ఉపయోగించబడుతుంది.",
            "Artificial Intelligence is computer systems designed to perform tasks that typically require human intelligence, such as learning from experience, recognizing patterns, and understanding language.":
                "కృత్రిమ మేధస్సు అనేది అనుభవం నుంచి నేర్చుకోవడం, నమూనాలను గుర్తించడం, భాషను అర్థముచేసుకోవడం వంటి మానవ మేధస్సు అవసరమైన పనులు చేయడానికి రూపకల్పన చేయబడిన కంప్యూటర్ సిస్టమ్‌లు.",
            "Machine Learning is a branch of AI where systems learn from data without being explicitly programmed.":
                "మిషిన్ లెర్నింగ్ అనేది AI యొక్క ఒక శాఖ, ఇందులో సిస్టమ్స్ స్పష్టంగా ప్రోగ్రామ్ చేయకుండానే డేటా నుండి నేర్చుకుంటాయి.",
            "Hello! I'm EduGenAI Pro, your AI educational assistant. I can help you with education, coding, science, math, and answer almost any question.":
                "హలో! నేను EduGenAI Pro, మీ AI విద్య సహాయకుడు. నేను విద్య, కోడింగ్, శాస్త్రం, గణితం మరియు దాదాపు ఏ ప్రశ్నకైనా సహాయం చేయగలను.",
            "Education is the process of learning knowledge, skills, values, and habits through study, teaching, or experience.":
                "విద్య అనేది అధ్యయనం, బోధన లేదా అనుభవం ద్వారా జ్ఞానం, నైపుణ్యాలు, విలువలు మరియు అలవాట్లను నేర్చుకునే ప్రక్రియ.",
            base:
                "నేను మీ ప్రశ్నను గ్రహించి, దానికి సమాధానం ఇవ్వడానికి ప్రయత్నిస్తున్నాను."
        },
        "tamil": {
            "Python is a high-level programming language known for its simplicity and readability. It's widely used in web development, data science, AI, and education. Python emphasizes code readability with significant indentation.":
                "Python என்பது அதன் எளிமை மற்றும் வாசிப்புப் பொருத்தளவிற்காகப் புகழ்பெற்ற ஒரு உயர் நிலை நிரலாக்க மொழி. இது வலை வளர்ச்சி, தரவு அறிவியல், AI மற்றும் கல்வியில் பரவலாகப் பயன்படுத்தப்படுகிறது.",
            "Artificial Intelligence is computer systems designed to perform tasks that typically require human intelligence, such as learning from experience, recognizing patterns, and understanding language.":
                "கிறுமை நுண்ணறிவு என்பது அனுபவத்திலிருந்து கற்றுக் கொள்ளுதல், முறைமைகளை அடையாளம் காணுதல் மற்றும் மொழியை புரிந்துகொள்ளுதல் போன்ற மனித நுண்ணறிவைத் தேவைப்படுத்தும் பணிகளை செய்ய வடிவமைக்கப்பட்ட கணினி அமைப்புகளாகும்.",
            "Machine Learning is a branch of AI where systems learn from data without being explicitly programmed.":
                "மெஷின் லெர்னிங் என்பது AI யின் ஒரு கிளை, இதில் அமைப்புகள் தெளிவாகக் கோடிடப்படாமல் தரவிலிருந்து கற்கின்றன.",
            "Hello! I'm EduGenAI Pro, your AI educational assistant. I can help you with education, coding, science, math, and answer almost any question.":
                "வணக்கம்! நான் EduGenAI Pro, உங்கள் AI கல்வி உதவியாளர். நான் கல்வி, குறியீட்டு, அறிவியல், கணிதத்தில் உங்களுக்கு உதவி செய்யவும், எந்தவொரு கேள்விக்கும் பதிலளிக்கவும் முடியும்.",
            "Education is the process of learning knowledge, skills, values, and habits through study, teaching, or experience.":
                "கல்வி என்பது படிப்பு, கற்பித்தல் அல்லது அனுபவத்தின் மூலம் அறிவு, திறன்கள், மதிப்புகள் மற்றும் பழக்கங்களை கற்கும் செயல்முறை.",
            base:
                "நான் உங்கள் கேள்வியைப் புரிந்து கொண்டு, இதற்கு பதில் சொல்ல முயற்சிக்கிறேன்."
        }
    }

    language_map = translations.get(language, {})
    return language_map.get(base, base)

# =========================
# ROUTES      
# =========================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify(success=False, error="No file uploaded."), 400

    uploaded_file = request.files["file"]
    if uploaded_file.filename == "":
        return jsonify(success=False, error="No file selected."), 400

    try:
        content = load_uploaded_file(uploaded_file)
        knowledge_store["text"] += "\n\n" + content
        return jsonify(
            success=True,
            filename=uploaded_file.filename,
            message="File uploaded and indexed successfully.",
        )

    except ValueError as exc:

        return jsonify(
            success=False,
            error=str(exc)
        ), 400

    except Exception as exc:

        return jsonify(
            success=False,
            error=f"Upload failed: {exc}"
        ), 500


@app.route("/clear", methods=["POST"])
def clear_knowledge():
    knowledge_store["text"] = ""
    return jsonify(success=True, message="Knowledge store cleared.")


# =========================
# PROMPT BUILDER
# =========================

def build_prompt(question: str, language: str = "english") -> str:
    """Construct the prompt using available knowledge context."""
    context = ""
    if knowledge_store["text"]:
        context = f"Context from uploaded materials:\n{knowledge_store['text']}\n\n"

    language_name = {
        "english": "English",
        "kannada": "Kannada",
        "hindi": "Hindi",
        "telugu": "Telugu",
        "tamil": "Tamil"
    }.get(language, "English")

    prompt = f"""You are EduGenAI Pro, an advanced educational AI assistant.
Answer clearly, accurately, and helpfully.
Use uploaded notes first when available.
Answer in {language_name}.

{context}User Question: {question}

Answer:
"""
    return prompt


# =========================
# HELPERS
# =========================

def run_with_timeout(func, timeout: float):
    """Run a blocking function in a worker thread and return its result or None on timeout."""

    result_queue = Queue()

    def worker():
        try:
            result_queue.put((True, func()))
        except Exception as exc:
            result_queue.put((False, exc))

    thread = Thread(target=worker, daemon=True)
    thread.start()

    try:
        success, payload = result_queue.get(timeout=timeout)
        if success:
            return payload
        raise payload
    except Empty:
        print(f"[DEBUG] Query timed out after {timeout} seconds.", flush=True)
        return None


# =========================
# OLLAMA QUERY
# =========================

def query_ollama(question: str, language: str = "english") -> str:

    def call():
        response = OLLAMA_CLIENT.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are EduGenAI Pro, a helpful educational assistant. Use the provided context to answer questions concisely and accurately."
                },
                {
                    "role": "user",
                    "content": build_prompt(question, language)
                }
            ],
            stream=False,
            options={
                "num_predict": 512,
                "temperature": 0.7,
            }
        )

        answer = None
        if isinstance(response, dict):
            answer = response.get("message", {}).get("content")
        else:
            message = getattr(response, "message", None)
            if isinstance(message, dict):
                answer = message.get("content")
            elif hasattr(message, "content"):
                answer = message.content

        if answer:
            return answer.strip()
        return None

    try:
        return run_with_timeout(call, timeout=OLLAMA_REQUEST_TIMEOUT)
    except Exception as exc:
        print(f"[DEBUG] Ollama error: {exc}", flush=True)
        return None


# =========================
# HUGGING FACE QUERY
# =========================

def query_huggingface(question: str, language: str = "english") -> str:

    try:

        if not HUGGINGFACE_API_TOKEN:
            return None

        client = InferenceClient(
            provider="hf-inference",
            token=HUGGINGFACE_API_TOKEN,
            timeout=HUGGINGFACE_REQUEST_TIMEOUT,
        )

        result = client.text_generation(
            prompt=build_prompt(question, language),
            model=HUGGINGFACE_MODEL,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            return_full_text=False,
        )

        if not result:
            return None

        if isinstance(result, (list, tuple)) and len(result) > 0:
            result = result[0]

        if isinstance(result, dict):
            answer = result.get("generated_text") or result.get("text") or result.get("output")
        else:
            answer = getattr(result, "generated_text", None) or str(result)

        if answer:
            answer = answer.strip()
            prompt_prefix = build_prompt(question, language)
            if answer.startswith(prompt_prefix):
                answer = answer[len(prompt_prefix) :].strip()
            return answer

        return None

    except Exception as exc:
        print(f"[DEBUG] HuggingFace error: {exc}", flush=True)
        return None


def query_both(question: str, language: str = "english") -> tuple[str, str] | tuple[None, None]:
    queue = Queue()

    def run_query(source_name, func):
        try:
            answer = func(question, language)
            if answer and answer.strip():
                queue.put((source_name, answer.strip()))
        except Exception:
            pass

    threads = [
        Thread(target=run_query, args=("ollama", query_ollama)),
        Thread(target=run_query, args=("huggingface", query_huggingface)),
    ]

    for thread in threads:
        thread.daemon = True
        thread.start()

    try:
        return queue.get(timeout=30)
    except Empty:
        return None, None


# =========================
# CHAT API
# =========================

@app.route("/chat", methods=["POST"])
def chat():

    data = request.json or {}

    question = data.get(
        "question",
        ""
    ).strip()

    source = data.get(
        "source",
        "ollama"
    )

    language = data.get(
        "language",
        "english"
    ).strip().lower()

    if language not in {"english", "kannada", "hindi", "telugu", "tamil"}:
        language = "english"

    if not question:
        return jsonify(
            answer="Please type a question."
        ), 400

    if source not in {
        "ollama",
        "huggingface",
        "both"
    }:
        source = "ollama"

    # Try to get AI response
    answer = None
    answer_source = "fallback"

    if source == "both":
        answer, answer_source = query_both(question, language)
        if not answer or not answer.strip():
            answer = None
            answer_source = "fallback"

    elif source == "ollama":
        answer = query_ollama(question, language)
        if answer and answer.strip():
            answer_source = "ollama"
        else:
            answer = query_huggingface(question, language)
            if answer and answer.strip():
                answer_source = "huggingface"

    else:
        if HUGGINGFACE_API_TOKEN:
            answer = query_huggingface(question, language)
            if answer and answer.strip():
                answer_source = "huggingface"
            else:
                answer = query_ollama(question, language)
                if answer and answer.strip():
                    answer_source = "ollama"
        else:
            answer = query_ollama(question, language)
            if answer and answer.strip():
                answer_source = "ollama"

    # =====================
    # FALLBACK
    # =====================
    if not answer or not answer.strip():
        answer = get_fallback_answer(question, language)
        answer_source = "fallback"

    return jsonify(answer=answer, source=answer_source)


# =========================
# RUN APP
# =========================

if __name__ == "__main__":
    app.run(debug=True)
