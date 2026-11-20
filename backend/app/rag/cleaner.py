"""Text cleaning and normalization utilities for academic document ingestion."""

import re


def clean_text(text: str) -> str:
    """Cleans and normalizes extracted document text.

    - Normalizes Unicode spaces and line endings.
    - Repairs PDF column wrapping artifacts (e.g. 'Text\\n Books', 'Reference\\ns', '1\\n.\\n').
    - Removes trailing line whitespace while preserving list indentation.
    - Collapses multiple consecutive blank lines to at most one blank line (paragraph separator).
    - Preserves academic terminology, subject codes, percentages, dates, and tables.
    """
    if not text:
        return ""

    # Normalize various whitespace characters
    text = text.replace("\xa0", " ").replace("\u200b", "").replace("\u3000", " ")

    # Standardize line breaks
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Clean lines individually
    cleaned_lines: list[str] = []
    for line in text.split("\n"):
        stripped_line = line.rstrip()
        leading_spaces = len(stripped_line) - len(stripped_line.lstrip(" "))
        # Keep indentation up to 4 spaces for bullet lists / sub-items
        indent = " " * min(leading_spaces, 4)
        content = re.sub(r"[ \t]+", " ", stripped_line.strip())
        if content:
            cleaned_lines.append(f"{indent}{content}" if indent else content)
        else:
            cleaned_lines.append("")

    # Rejoin lines
    joined = "\n".join(cleaned_lines)

    # Collapse 3 or more consecutive newlines into 2
    collapsed = re.sub(r"\n{3,}", "\n\n", joined)

    # Normalize specific PDF extraction column-wrap artifacts
    # 1. 'Text\n Books' -> 'Text Books'
    collapsed = re.sub(r"(?i)\bText\s*\n+\s*Books\b", "Text Books", collapsed)

    # 2. 'Reference\ns' -> 'References'
    collapsed = re.sub(r"(?i)\bReference\s*\n+\s*s\b", "References", collapsed)

    # 3. Label followed by isolated colon line: 'Text Books\n:' -> 'Text Books:'
    collapsed = re.sub(
        r"(?i)\b(Text\s*Books?|References?|Course\s*Outcomes?|Unit\s+[IVXLCDM\d]+)\s*\n+\s*:",
        r"\1:",
        collapsed,
    )

    # 4. Broken numbering in lists: '1\n.\n' -> '1. '
    collapsed = re.sub(r"(\b\d+)\s*\n+\s*\.\s*\n+", r"\1. ", collapsed)

    # 5. Course code followed by isolated colon line: 'CSE205\n:\n' -> 'CSE205:\n'
    collapsed = re.sub(r"\b([A-Z]{3}\d{3})\s*\n+\s*:\s*\n+", r"\1:\n", collapsed)

    # 6. Hyphenated word wrap at end of line: 'algo-\nrithm' -> 'algorithm'
    collapsed = re.sub(r"(\b[A-Za-z]{3,})-\n+([A-Za-z]{2,}\b)", r"\1\2", collapsed)

    return collapsed.strip()
