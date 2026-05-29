import anthropic
import json
import os
import uuid
from datetime import datetime

SYSTEM_PROMPT = """You are an AI assistant for a professional research consultant. You help manage research projects, capture insights and notes, and support literature and topic exploration.

## Your Capabilities
- Help organize and track research projects
- Save and retrieve research notes, findings, and insights
- Suggest research directions and summarize topics on demand
- Maintain a structured knowledge base of notes and projects

## Your Tone
- Precise, analytical, and concise
- Ask clarifying questions when research scope is ambiguous
- Proactively suggest related angles or gaps in the research

Always use the available tools to save notes, manage projects, and retrieve information so nothing is lost."""

TOOLS = [
    {
        "name": "create_project",
        "description": "Creates a new research project with a title, description, and optional tags.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Short title of the research project"
                },
                "description": {
                    "type": "string",
                    "description": "Brief description of the project scope and goals"
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of topic tags (e.g. ['AI', 'climate', 'policy'])"
                }
            },
            "required": ["title", "description"]
        }
    },
    {
        "name": "list_projects",
        "description": "Lists all research projects currently on record.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "save_note",
        "description": "Saves a research note or finding, optionally linked to a project.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Short title or headline for the note"
                },
                "content": {
                    "type": "string",
                    "description": "The note body — findings, observations, quotes, or analysis"
                },
                "project_title": {
                    "type": "string",
                    "description": "Optional: title of the project this note belongs to"
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional topic tags for the note"
                }
            },
            "required": ["title", "content"]
        }
    },
    {
        "name": "list_notes",
        "description": "Lists saved research notes, optionally filtered by project title or tag.",
        "input_schema": {
            "type": "object",
            "properties": {
                "project_title": {
                    "type": "string",
                    "description": "Optional: filter notes by project title"
                },
                "tag": {
                    "type": "string",
                    "description": "Optional: filter notes by tag"
                }
            },
            "required": []
        }
    }
]

_PROJECTS_FILE = "projects.json"
_NOTES_FILE = "notes.json"


def _load_json(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)


def _save_json(path: str, data: list) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _tool_create_project(title: str, description: str, tags: list = None) -> str:
    projects = _load_json(_PROJECTS_FILE)
    project = {
        "id": str(uuid.uuid4())[:8],
        "created": datetime.now().isoformat(),
        "title": title,
        "description": description,
        "tags": tags or [],
    }
    projects.append(project)
    _save_json(_PROJECTS_FILE, projects)
    return f"Project created: [{project['id']}] \"{title}\"\nDescription: {description}\nTags: {', '.join(project['tags']) or 'none'}"


def _tool_list_projects() -> str:
    projects = _load_json(_PROJECTS_FILE)
    if not projects:
        return "No research projects on record."
    lines = [f"=== Research Projects ({len(projects)}) ==="]
    for p in projects:
        tag_str = f"  Tags: {', '.join(p['tags'])}" if p.get("tags") else ""
        lines.append(f"\n[{p['id']}] {p['created'][:10]} | {p['title']}\n  {p['description']}{tag_str}")
    return "\n".join(lines)


def _tool_save_note(title: str, content: str, project_title: str = "", tags: list = None) -> str:
    notes = _load_json(_NOTES_FILE)
    note = {
        "id": str(uuid.uuid4())[:8],
        "created": datetime.now().isoformat(),
        "title": title,
        "content": content,
        "project": project_title or "",
        "tags": tags or [],
    }
    notes.append(note)
    _save_json(_NOTES_FILE, notes)
    project_str = f" (project: {project_title})" if project_title else ""
    return f"Note saved: [{note['id']}] \"{title}\"{project_str}"


def _tool_list_notes(project_title: str = "", tag: str = "") -> str:
    notes = _load_json(_NOTES_FILE)
    if project_title:
        notes = [n for n in notes if project_title.lower() in n.get("project", "").lower()]
    if tag:
        notes = [n for n in notes if tag.lower() in [t.lower() for t in n.get("tags", [])]]
    if not notes:
        qualifier = f" for project '{project_title}'" if project_title else (f" tagged '{tag}'" if tag else "")
        return f"No notes found{qualifier}."
    lines = [f"=== Research Notes ({len(notes)}) ==="]
    for n in notes:
        project_str = f" | Project: {n['project']}" if n.get("project") else ""
        tag_str = f"  Tags: {', '.join(n['tags'])}" if n.get("tags") else ""
        lines.append(
            f"\n[{n['id']}] {n['created'][:10]}{project_str}\n  Title: {n['title']}\n  {n['content']}{tag_str}"
        )
    return "\n".join(lines)


_TOOL_HANDLERS = {
    "create_project": lambda args: _tool_create_project(**args),
    "list_projects": lambda _args: _tool_list_projects(),
    "save_note": lambda args: _tool_save_note(**args),
    "list_notes": lambda args: _tool_list_notes(**args),
}


def _dispatch_tool(name: str, args: dict) -> str:
    handler = _TOOL_HANDLERS.get(name)
    if not handler:
        return f"Unknown tool: {name}"
    try:
        return handler(args)
    except Exception as exc:
        return f"Tool error ({name}): {exc}"


def run_agent() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable is not set.")
        return

    client = anthropic.Anthropic(api_key=api_key)
    messages: list = []

    print("Research Consultant AI Agent")
    print("=" * 40)
    print("Hello! I'm your research AI assistant.")
    print("I can help manage projects, save notes, and explore research topics.")
    print("Type 'quit' or 'exit' to end the session.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        # Inner agentic loop — runs until the model stops calling tools
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
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result = _dispatch_tool(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })

                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})
            else:
                text = "".join(block.text for block in response.content if hasattr(block, "text"))
                messages.append({"role": "assistant", "content": response.content})
                print(f"\nAgent: {text}\n")
                break


if __name__ == "__main__":
    run_agent()
