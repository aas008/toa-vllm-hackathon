"""
Knowledge Base Query Tool

Searches an Obsidian-style wiki for vLLM concepts, techniques,
architectures, and optimization guidance. Returns relevant pages
to inform agent tuning/profiling decisions.
"""
from __future__ import annotations

import os
import re
from pathlib import Path


class KnowledgeBase:
    """Searchable index over an Obsidian wiki directory."""

    def __init__(self, wiki_path: str):
        self.root = Path(wiki_path)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Wiki not found: {wiki_path}")
        self._index: list[dict] = []
        self._build_index()

    def _build_index(self):
        """Scan all .md files and index by title, tags, and path."""
        for md_file in self.root.rglob("*.md"):
            rel = md_file.relative_to(self.root)
            # Skip raw sources, logs, templates, folds
            if any(p in rel.parts for p in (".raw", "log", "templates", "folds")):
                continue

            try:
                text = md_file.read_text(errors="replace")
            except Exception:
                continue

            title = rel.stem
            tags: list[str] = []
            doc_type = ""

            # Parse frontmatter
            if text.startswith("---"):
                end = text.find("---", 3)
                if end > 0:
                    fm = text[3:end]
                    m = re.search(r'^title:\s*(.+)$', fm, re.MULTILINE)
                    if m:
                        title = m.group(1).strip()
                    m = re.search(r'^tags:\s*\[(.+)\]', fm, re.MULTILINE)
                    if m:
                        tags = [t.strip() for t in m.group(1).split(",")]
                    m = re.search(r'^type:\s*(\S+)', fm, re.MULTILINE)
                    if m:
                        doc_type = m.group(1).strip()

            category = rel.parts[0] if len(rel.parts) > 1 else "root"

            self._index.append({
                "path": str(rel),
                "abs_path": str(md_file),
                "title": title,
                "tags": tags,
                "type": doc_type,
                "category": category,
                "size": len(text),
            })

    def query(self, topic: str, max_results: int = 3) -> str:
        """Search wiki for pages relevant to a topic.

        Scores pages by title match, tag match, path match, and content match.
        Returns concatenated page content for the top results.
        """
        topic_lower = topic.lower()
        terms = topic_lower.split()

        scored: list[tuple[float, dict]] = []
        for entry in self._index:
            score = 0.0
            title_lower = entry["title"].lower()
            path_lower = entry["path"].lower()
            tags_lower = " ".join(entry["tags"]).lower()

            # Exact title match
            if topic_lower == title_lower:
                score += 100
            # Title contains topic
            elif topic_lower in title_lower:
                score += 50
            # Topic terms in title
            for t in terms:
                if t in title_lower:
                    score += 10
                if t in tags_lower:
                    score += 8
                if t in path_lower:
                    score += 5

            # Boost concept/technique/architecture pages
            if entry["type"] in ("concept", "technique", "architecture"):
                score *= 1.5
            # Slight boost for non-index pages
            if entry["title"].lower() != "index" and entry["type"] != "index":
                score += 1

            if score > 0:
                scored.append((score, entry))

        # If no title/tag match, do content grep
        if not scored:
            for entry in self._index:
                try:
                    text = Path(entry["abs_path"]).read_text(errors="replace").lower()
                    count = sum(text.count(t) for t in terms)
                    if count > 0:
                        score = count * 2
                        if entry["type"] in ("concept", "technique", "architecture"):
                            score *= 1.5
                        scored.append((score, entry))
                except Exception:
                    continue

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:max_results]

        if not top:
            return f"No wiki pages found for '{topic}'. Available categories: {self._list_categories()}"

        parts = [f"=== KNOWLEDGE BASE: {len(top)} pages for '{topic}' ===\n"]
        for score, entry in top:
            try:
                text = Path(entry["abs_path"]).read_text(errors="replace")
                # Strip frontmatter for cleaner output
                if text.startswith("---"):
                    end = text.find("---", 3)
                    if end > 0:
                        text = text[end + 3:].strip()
                # Truncate very long pages
                if len(text) > 4000:
                    text = text[:4000] + "\n... [truncated]"
                parts.append(f"--- {entry['title']} ({entry['category']}/{entry['path']}) ---")
                parts.append(text)
                parts.append("")
            except Exception:
                continue

        return "\n".join(parts)

    def list_topics(self) -> str:
        """List all available topics grouped by category."""
        by_cat: dict[str, list[str]] = {}
        for entry in self._index:
            if entry["type"] == "index":
                continue
            cat = entry["category"]
            by_cat.setdefault(cat, []).append(entry["title"])

        lines = ["=== KNOWLEDGE BASE TOPICS ==="]
        for cat in sorted(by_cat.keys()):
            titles = sorted(by_cat[cat])
            lines.append(f"\n{cat}/ ({len(titles)} pages):")
            for t in titles:
                lines.append(f"  - {t}")
        return "\n".join(lines)

    def _list_categories(self) -> str:
        cats = set(e["category"] for e in self._index)
        return ", ".join(sorted(cats))
