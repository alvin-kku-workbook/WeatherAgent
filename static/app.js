const chatBox = document.getElementById("chat-box");
const messageInput = document.getElementById("message-input");
const sendButton = document.getElementById("send-button");
const recordButton = document.getElementById("record-button");
const clearButton = document.getElementById("clear-button");
const statusText = document.getElementById("status-text");

let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;

function addMessage(role, text, toolCallsUsed = []) {
    const messageDiv = document.createElement("div");
    messageDiv.className = `message ${role}`;

    const bubbleDiv = document.createElement("div");
    bubbleDiv.className = "bubble";
    bubbleDiv.textContent = text;
    messageDiv.appendChild(bubbleDiv);

    if (toolCallsUsed.length > 0) {
        const toolDiv = document.createElement("div");
        toolDiv.className = "tool-badge";
        toolDiv.textContent = `도구 사용: ${toolCallsUsed.join(", ")}`;
        messageDiv.appendChild(toolDiv);
    }

    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message) {
        return;
    }

    addMessage("user", message);
    messageInput.value = "";
    statusText.textContent = "답변 생성 중...";
    sendButton.disabled = true;

    try {
        const response = await fetch("/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                message,
            }),
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "채팅 요청에 실패했습니다.");
        }

        addMessage("assistant", data.assistant_message, data.tool_calls_used || []);
        statusText.textContent = "대기 중";
    } catch (error) {
        addMessage("assistant", error.message);
        statusText.textContent = "오류 발생";
    } finally {
        sendButton.disabled = false;
    }
}

async function startRecording() {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    audioChunks = [];

    mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
            audioChunks.push(event.data);
        }
    };

    mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType || "audio/webm" });
        const formData = new FormData();
        formData.append("audio", audioBlob, "recording.webm");

        statusText.textContent = "음성을 텍스트로 변환 중...";
        recordButton.disabled = true;

        try {
            const response = await fetch("/transcribe", {
                method: "POST",
                body: formData,
            });
            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || "음성 변환에 실패했습니다.");
            }

            messageInput.value = data.transcript;
            statusText.textContent = "변환 완료. 내용을 확인하고 전송하세요.";
        } catch (error) {
            addMessage("assistant", error.message);
            statusText.textContent = "음성 변환 실패";
        } finally {
            recordButton.disabled = false;
            stream.getTracks().forEach((track) => track.stop());
        }
    };

    mediaRecorder.start();
    isRecording = true;
    recordButton.textContent = "녹음 중지";
    statusText.textContent = "녹음 중...";
}

function stopRecording() {
    if (mediaRecorder) {
        mediaRecorder.stop();
    }
    isRecording = false;
    recordButton.textContent = "녹음 시작";
}

sendButton.addEventListener("click", sendMessage);
messageInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
});

recordButton.addEventListener("click", async () => {
    try {
        if (!isRecording) {
            await startRecording();
        } else {
            stopRecording();
        }
    } catch (error) {
        addMessage("assistant", "마이크를 사용할 수 없습니다.");
        statusText.textContent = "녹음 사용 불가";
        recordButton.textContent = "녹음 시작";
        isRecording = false;
    }
});

clearButton.addEventListener("click", async () => {
    await fetch("/reset", { method: "POST" });
    chatBox.innerHTML = `
        <div class="message assistant">
            <div class="bubble">
                안녕하세요. 지역을 포함해서 날씨나 외출 준비를 물어보세요. 예: 서울 날씨 어때?
            </div>
        </div>
    `;
    messageInput.value = "";
    statusText.textContent = "대기 중";
});
