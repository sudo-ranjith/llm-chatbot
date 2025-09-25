const chatEl = document.getElementById("chat");
const inputEl = document.getElementById("user_input");
const sendBtn = document.getElementById("sendBtn");
const clearBtn = document.getElementById("clearBtn");

let sessionId = localStorage.getItem("session_id") || null;

// helper to append message
function appendMessage(role, text) {
  const wrapper = document.createElement("div");
  wrapper.className = `msg ${role === "user" ? "user" : "bot"}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerText = text;
  wrapper.appendChild(bubble);

  const meta = document.createElement("div");
  meta.className = "meta";
  const now = new Date();
  meta.innerText = now.toLocaleTimeString();
  wrapper.appendChild(meta);

  chatEl.appendChild(wrapper);
  chatEl.scrollTop = chatEl.scrollHeight;
}

// send message function
async function sendMessage() {
  const text = inputEl.value.trim();
  if (!text) return;
  inputEl.value = "";
  appendMessage("user", text);

  // prepare payload
  const payload = { message: text };
  if (sessionId) payload.session_id = sessionId;

  try {
    const resp = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await resp.json();
    if (data.session_id) {
      sessionId = data.session_id;
      localStorage.setItem("session_id", sessionId);
    }
    if (data.reply) {
      appendMessage("bot", data.reply);
    } else if (data.error) {
      appendMessage("bot", "Error: " + data.error);
      console.error(data.details || "");
    }
  } catch (err) {
    appendMessage("bot", "Network error: could not reach server.");
    console.error(err);
  }
}

// Enter to send, Shift+Enter for newline
inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

// button click
sendBtn.addEventListener("click", (e) => {
  e.preventDefault();
  sendMessage();
});

// clear conversation
clearBtn.addEventListener("click", async () => {
  if (!sessionId) {
    // nothing to clear
    chatEl.innerHTML = "";
    return;
  }
  // clear server session
  await fetch("/session/clear", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  });

  // clear local
  localStorage.removeItem("session_id");
  sessionId = null;
  chatEl.innerHTML = "";
});
