import os
import uuid

import anthropic
from flask import Flask, jsonify, render_template, request, session

from agent import SYSTEM_PROMPT, TOOLS, _dispatch_tool

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", os.urandom(24))

# Server-side conversation store: session_id -> list[message]
_conversations: dict = {}


def _serialize_content(content) -> list:
    """Convert Anthropic SDK content blocks to JSON-serializable dicts."""
    if not isinstance(content, list):
        return content
    out = []
    for block in content:
        if isinstance(block, dict):
            out.append(block)
        elif block.type == "text":
            out.append({"type": "text", "text": block.text})
        elif block.type == "tool_use":
            out.append({"type": "tool_use", "id": block.id, "name": block.name, "input": block.input})
    return out


@app.route("/")
def index():
    if "sid" not in session:
        session["sid"] = str(uuid.uuid4())
    return render_template("chat.html")


@app.route("/chat", methods=["POST"])
def chat():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return jsonify({"error": "ANTHROPIC_API_KEY is not set on the server."}), 500

    data = request.get_json(silent=True) or {}
    user_message = data.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "Empty message."}), 400

    sid = session.setdefault("sid", str(uuid.uuid4()))
    messages = _conversations.setdefault(sid, [])
    messages.append({"role": "user", "content": user_message})

    client = anthropic.Anthropic(api_key=api_key)

    try:
        while True:
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4096,
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                tools=TOOLS,
                messages=messages,
                extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
            )

            if response.stop_reason == "tool_use":
                tool_results = [
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": _dispatch_tool(block.name, block.input),
                    }
                    for block in response.content
                    if block.type == "tool_use"
                ]
                messages.append({"role": "assistant", "content": _serialize_content(response.content)})
                messages.append({"role": "user", "content": tool_results})
            else:
                reply = "".join(
                    block.text for block in response.content if hasattr(block, "text")
                )
                messages.append({"role": "assistant", "content": _serialize_content(response.content)})
                return jsonify({"reply": reply})

    except anthropic.APIError as exc:
        return jsonify({"error": f"API error: {exc}"}), 502
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/reset", methods=["POST"])
def reset():
    sid = session.get("sid")
    if sid:
        _conversations.pop(sid, None)
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
