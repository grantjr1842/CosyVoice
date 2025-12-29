#!/usr/bin/env python3
"""
Antigravity IDE Conversation Exporter

Exports all past conversations and their associated artifacts (brain folders)
to a structured output directory with markdown summaries and JSON metadata.

Usage:
    python export_conversations.py [--output-dir OUTPUT_DIR]
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

# Default paths
ANTIGRAVITY_DIR = Path.home() / ".gemini" / "antigravity"
CONVERSATIONS_DIR = ANTIGRAVITY_DIR / "conversations"
BRAIN_DIR = ANTIGRAVITY_DIR / "brain"
CODE_TRACKER_DIR = ANTIGRAVITY_DIR / "code_tracker"


def get_conversation_ids():
    """Get all conversation IDs from the conversations directory."""
    if not CONVERSATIONS_DIR.exists():
        return []
    return [f.stem for f in CONVERSATIONS_DIR.glob("*.pb")]


def get_brain_artifacts(conversation_id: str) -> dict:
    """Get all readable artifacts from a conversation's brain folder."""
    brain_folder = BRAIN_DIR / conversation_id
    artifacts = {}

    if not brain_folder.exists():
        return artifacts

    # Collect all markdown files and their content
    for md_file in brain_folder.glob("*.md"):
        if ".resolved" not in md_file.name:
            try:
                artifacts[md_file.name] = md_file.read_text(encoding="utf-8")
            except Exception as e:
                artifacts[md_file.name] = f"[Error reading file: {e}]"

    # Collect metadata files
    for meta_file in brain_folder.glob("*.metadata.json"):
        try:
            artifacts[meta_file.name] = json.loads(
                meta_file.read_text(encoding="utf-8")
            )
        except Exception as e:
            artifacts[meta_file.name] = {"error": str(e)}

    return artifacts


def get_code_tracker_files(pattern: str = "*") -> list[dict]:
    """Get files from the code tracker that match patent.

    Returns list of dicts with file paths and metadata.
    """
    active_dir = CODE_TRACKER_DIR / "active"
    if not active_dir.exists():
        return []

    results = []
    for repo_dir in active_dir.iterdir():
        if repo_dir.is_dir():
            for tracked_file in repo_dir.iterdir():
                results.append(
                    {
                        "repo": repo_dir.name,
                        "file": tracked_file.name,
                        "path": str(tracked_file),
                        "size": tracked_file.stat().st_size
                        if tracked_file.is_file()
                        else 0,
                    }
                )
    return results


def export_conversation(conversation_id: str, output_dir: Path):
    """Export a single conversation with all its artifacts."""
    conv_dir = output_dir / conversation_id
    conv_dir.mkdir(parents=True, exist_ok=True)

    # Get brain artifacts
    artifacts = get_brain_artifacts(conversation_id)

    # Export artifacts
    for name, content in artifacts.items():
        if isinstance(content, dict):
            (conv_dir / name).write_text(json.dumps(content, indent=2))
        else:
            (conv_dir / name).write_text(content)

    # Create summary file
    summary = {
        "conversation_id": conversation_id,
        "exported_at": datetime.now().isoformat(),
        "artifacts": list(artifacts.keys()),
        "pb_file": str(CONVERSATIONS_DIR / f"{conversation_id}.pb"),
    }
    (conv_dir / "_export_summary.json").write_text(json.dumps(summary, indent=2))

    return len(artifacts)


def create_master_index(output_dir: Path, conversation_ids: list[str]):
    """Create a master index markdown file summarizing all exports."""
    index_content = f"""# Antigravity Conversation Export
Exported: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Total Conversations: {len(conversation_ids)}

## Conversations

| ID | Artifacts |
|----|-----------|
"""
    for conv_id in sorted(conversation_ids):
        artifacts = get_brain_artifacts(conv_id)
        artifact_list = ", ".join(artifacts.keys()) if artifacts else "None"
        index_content += f"| [{conv_id[:8]}...](./{conv_id}/) | {artifact_list} |\n"

    # Add section for task/walkthrough summaries
    index_content += "\n## Artifact Summaries\n\n"

    for conv_id in sorted(conversation_ids):
        artifacts = get_brain_artifacts(conv_id)
        if artifacts:
            index_content += f"### {conv_id[:8]}...\n\n"

            if "task.md" in artifacts:
                task_preview = artifacts["task.md"][:500].replace("\n", "\n> ")
                index_content += f"**Task:**\n> {task_preview}\n\n"

            if "walkthrough.md" in artifacts:
                walk_preview = artifacts["walkthrough.md"][:500].replace("\n", "\n> ")
                index_content += f"**Walkthrough:**\n> {walk_preview}\n\n"

    (output_dir / "INDEX.md").write_text(index_content)


def main():
    parser = argparse.ArgumentParser(description="Export Antigravity IDE conversations")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "conversation_export",
        help="Output directory for exported conversations",
    )
    parser.add_argument(
        "--include-code-tracker",
        action="store_true",
        help="Include code tracker files in export",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Antigravity Conversation Exporter")
    print("=" * 40)
    print(f"Source: {ANTIGRAVITY_DIR}")
    print(f"Output: {output_dir}")
    print()

    # Get all conversation IDs
    conversation_ids = get_conversation_ids()
    print(f"Found {len(conversation_ids)} conversations")

    # Export each conversation
    total_artifacts = 0
    for i, conv_id in enumerate(conversation_ids, 1):
        num_artifacts = export_conversation(conv_id, output_dir)
        total_artifacts += num_artifacts
        print(
            f"  [{i}/{len(conversation_ids)}] {conv_id[:12]}... ({num_artifacts} artifacts)"
        )

    # Export code tracker if requested
    if args.include_code_tracker:
        code_tracker_dir = output_dir / "_code_tracker"
        code_tracker_dir.mkdir(exist_ok=True)

        tracker_files = get_code_tracker_files()
        print(f"\nExporting {len(tracker_files)} code tracker files...")

        for item in tracker_files:
            repo_dir = code_tracker_dir / item["repo"]
            repo_dir.mkdir(exist_ok=True)
            src = Path(item["path"])
            if src.exists() and src.is_file():
                try:
                    shutil.copy2(src, repo_dir / item["file"])
                except Exception as e:
                    print(f"  Warning: Could not copy {item['file']}: {e}")

    # Create master index
    create_master_index(output_dir, conversation_ids)

    print()
    print("✅ Export complete!")
    print(f"   Conversations: {len(conversation_ids)}")
    print(f"   Total artifacts: {total_artifacts}")
    print(f"   Output: {output_dir}")
    print(f"   Index: {output_dir / 'INDEX.md'}")


if __name__ == "__main__":
    main()
