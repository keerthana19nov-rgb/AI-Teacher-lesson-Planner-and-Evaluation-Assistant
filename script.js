document.getElementById("fileUpload").addEventListener("change", uploadFile);

let recognition = null;
let isListening = false;

function getSpeechRecognitionLanguage() {
    const languageSelect = document.getElementById("language-source");
    return {
        english: 'en-US',
        kannada: 'kn-IN',
        hindi: 'hi-IN',
        telugu: 'te-IN',
        tamil: 'ta-IN'
    }[languageSelect?.value] || 'en-US';
}

function setupSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const voiceButton = document.getElementById("voice-button");

    if (!SpeechRecognition) {
        if (voiceButton) {
            voiceButton.disabled = true;
            voiceButton.title = "Voice input is not supported in this browser.";
        }
        return;
    }

    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.lang = getSpeechRecognitionLanguage();
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    recognition.addEventListener("result", (event) => {
        const transcript = Array.from(event.results)
            .map(result => result[0].transcript)
            .join(" ")
            .trim();

        if (transcript) {
            const input = document.getElementById("user-input");
            input.value = transcript;
            sendMessage();
        }
    });

    recognition.addEventListener("start", () => {
        isListening = true;
        updateVoiceButton();
        addMessage("Listening... Speak your question now.", "bot");
    });

    const languageSelect = document.getElementById("language-source");
    if (languageSelect) {
        languageSelect.addEventListener("change", () => {
            if (recognition) {
                recognition.lang = getSpeechRecognitionLanguage();
            }
        });
    }

    recognition.addEventListener("end", () => {
        isListening = false;
        updateVoiceButton();
    });

    recognition.addEventListener("error", (event) => {
        isListening = false;
        updateVoiceButton();
        console.error("Speech recognition error:", event.error);
        addMessage(`Voice error: ${event.error}. Please try typing.`, "bot");
    });

    window.toggleVoice = toggleVoice;
}

function updateVoiceButton() {
    const voiceButton = document.getElementById("voice-button");
    if (!voiceButton) return;

    if (isListening) {
        voiceButton.textContent = "⏹️";
        voiceButton.title = "Stop listening";
        voiceButton.classList.add("listening");
    } else {
        voiceButton.textContent = "🎤";
        voiceButton.title = "Speak your question";
        voiceButton.classList.remove("listening");
    }
}

function toggleVoice() {
    if (!recognition) {
        addMessage("Voice input is not supported by this browser.", "bot");
        return;
    }

    if (isListening) {
        recognition.stop();
        return;
    }

    recognition.lang = getSpeechRecognitionLanguage();
    recognition.start();
}

setupSpeechRecognition();

async function sendMessage() {
    const input = document.getElementById("user-input");
    const message = input.value.trim();

    if (!message) return;

    addMessage(message, "user");
    input.value = "";

    // Create and show loading indicator
    const chatBox = document.getElementById("chat-box");
    const loadingDiv = document.createElement("div");
    loadingDiv.classList.add("message", "bot", "loading-container");
    loadingDiv.innerHTML = '<div class="spinner"></div><span>EduGenAI is thinking...</span>';
    chatBox.appendChild(loadingDiv);
    chatBox.scrollTop = chatBox.scrollHeight;

    const source = document.getElementById("model-source").value;
    const language = document.getElementById("language-source").value;
    let response;
    try {
        response = await fetch("/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ question: message, source, language }),
        });
    } catch (error) {
        if (chatBox.contains(loadingDiv)) chatBox.removeChild(loadingDiv);
        addMessage("Network error. Please check your connection.", "bot");
        return;
    }

    // Remove loading indicator before displaying the final answer
    if (chatBox.contains(loadingDiv)) chatBox.removeChild(loadingDiv);

    let data;
    try {
        data = await response.json();
    } catch (error) {
        addMessage("Sorry, I could not get an answer.", "bot");
        return;
    }

    const answerText = data.answer || "Sorry, I could not get an answer.";
    const sourceText = data.source ? formatModelSource(data.source) : null;
    addMessage(sourceText ? `${answerText}\n\n${sourceText}` : answerText, "bot");

    // Speak the answer back to the user
    speak(answerText);
}

function speak(text) {
    if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel(); // Stop any current speech
        const utterance = new SpeechSynthesisUtterance(text);
        window.speechSynthesis.speak(utterance);
    }
}

function formatModelSource(source) {
    if (source === "both") {
        return "Source: Ollama + Hugging Face";
    }
    if (source === "ollama") {
        return "Source: Ollama";
    }
    if (source === "huggingface") {
        return "Source: Hugging Face";
    }
    return "Source: fallback";
}

function addMessage(text, type) {
    const chatBox = document.getElementById("chat-box");
    const div = document.createElement("div");

    div.classList.add("message");
    div.classList.add(type);
    div.innerText = text;

    chatBox.appendChild(div);
    chatBox.scrollTop = chatBox.scrollHeight;
}

async function newChat() {
    if ('speechSynthesis' in window) window.speechSynthesis.cancel();
    await fetch("/clear", { method: "POST" });
    document.getElementById("chat-box").innerHTML = "";
    document.getElementById("upload-status").textContent = "";
    document.getElementById("fileUpload").value = "";
    addMessage("New chat session started. Upload notes or ask a question.", "bot");
}

async function uploadFile(event) {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    addMessage(`Uploading ${file.name}...`, "bot");

    const response = await fetch("/upload", {
        method: "POST",
        body: formData,
    });

    const data = await response.json();
    const status = document.getElementById("upload-status");

    if (data.success) {
        status.textContent = `Uploaded: ${data.filename}`;
        status.classList.remove("error");
        status.classList.add("success");
        addMessage(`Uploaded ${data.filename} successfully.`, "bot");
    } else {
        status.textContent = data.error || "Upload failed.";
        status.classList.remove("success");
        status.classList.add("error");
        addMessage(`Upload error: ${data.error}`, "bot");
    }
}
