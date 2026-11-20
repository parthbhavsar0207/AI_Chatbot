import { useState, useMemo, type ReactNode } from 'react'
import { Check, Copy } from 'lucide-react'
import type { CitationItem } from '../services/chatApi'

interface MarkdownRendererProps {
  content: string
  citations?: CitationItem[]
}

const INLINE_TOKEN_REGEX = /(\[\[cite:[a-zA-Z0-9_\-\.]+\]\]|`[^`]+`|\*\*[^*]+\*\*|__[^_]+__|(?<!\w)\*[^*\n]+\*(?!\w)|(?<!\w)_[^_\n]+_(?!\w))/g

function renderInline(
  text: string,
  citationMap: Map<string, { index: number; citation: CitationItem }>,
  keyPrefix = 'inline'
): ReactNode[] {
  if (!text) return []

  const parts = text.split(INLINE_TOKEN_REGEX)
  return parts.map((part, idx) => {
    const key = `${keyPrefix}-${idx}`
    if (!part) return null

    // Citation tag [[cite:chunk_id]]
    if (part.startsWith('[[cite:') && part.endsWith(']]')) {
      const chunkId = part.slice(7, -2).trim()
      const match = citationMap.get(chunkId)
      if (match) {
        const sourceLabel = match.citation.source_file || match.citation.title || `Source ${match.index}`
        return (
          <span
            key={key}
            className="inline-citation"
            title={`${sourceLabel} (Chunk ID: ${chunkId})`}
            aria-label={`Source ${match.index}: ${sourceLabel}`}
          >
            [{match.index}]
          </span>
        )
      }
      return (
        <span
          key={key}
          className="inline-citation"
          title={`Chunk ID: ${chunkId}`}
          aria-label={`Source chunk: ${chunkId}`}
        >
          [{chunkId.slice(0, 8)}]
        </span>
      )
    }

    // Inline code
    if (part.startsWith('`') && part.endsWith('`') && part.length >= 2) {
      return (
        <code key={key} className="inline-code">
          {part.slice(1, -1)}
        </code>
      )
    }

    // Bold (**text** or __text__)
    if ((part.startsWith('**') && part.endsWith('**') && part.length >= 4) ||
        (part.startsWith('__') && part.endsWith('__') && part.length >= 4)) {
      const inner = part.slice(2, -2)
      return (
        <strong key={key}>
          {renderInline(inner, citationMap, `${key}-b`)}
        </strong>
      )
    }

    // Italic (*text* or _text_)
    if ((part.startsWith('*') && part.endsWith('*') && part.length >= 2) ||
        (part.startsWith('_') && part.endsWith('_') && part.length >= 2)) {
      const inner = part.slice(1, -1)
      return (
        <em key={key}>
          {renderInline(inner, citationMap, `${key}-i`)}
        </em>
      )
    }

    return part
  })
}

function CodeBlock({ language, code }: { language: string; code: string }) {
  const [copied, setCopied] = useState(false)

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(code)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch {
      // Fallback or ignore clipboard errors
    }
  }

  const langLabel = language ? language.toUpperCase() : 'CODE'

  return (
    <div className="code-block-wrapper">
      <div className="code-block-header">
        <span className="code-block-lang">{langLabel}</span>
        <button
          type="button"
          onClick={handleCopy}
          className="code-copy-btn"
          aria-label="Copy code to clipboard"
        >
          {copied ? <Check size={12} aria-hidden="true" /> : <Copy size={12} aria-hidden="true" />}
          <span>{copied ? 'Copied' : 'Copy'}</span>
        </button>
      </div>
      <pre className="code-block-pre">
        <code>{code}</code>
      </pre>
    </div>
  )
}

function isTableDivider(line: string): boolean {
  const trimmed = line.trim()
  if (!trimmed.startsWith('|') || !trimmed.endsWith('|')) return false
  const cells = trimmed.split('|').slice(1, -1)
  return cells.length > 0 && cells.every(c => /^[\s:\-]+$/.test(c))
}

function parseTable(lines: string[]): { headers: string[]; rows: string[][] } | null {
  if (lines.length < 2) return null
  const headerLine = lines[0].trim()
  const dividerLine = lines[1].trim()

  if (!isTableDivider(dividerLine)) return null

  const headers = headerLine.split('|').slice(1, -1).map(c => c.trim())
  const rows: string[][] = []

  for (let i = 2; i < lines.length; i++) {
    const rowLine = lines[i].trim()
    if (rowLine.startsWith('|') && rowLine.endsWith('|')) {
      const cells = rowLine.split('|').slice(1, -1).map(c => c.trim())
      rows.push(cells)
    }
  }

  return { headers, rows }
}

export function MarkdownRenderer({ content, citations = [] }: MarkdownRendererProps) {
  const citationMap = useMemo(() => {
    const map = new Map<string, { index: number; citation: CitationItem }>()
    citations.forEach((c, idx) => {
      map.set(c.chunk_id, { index: idx + 1, citation: c })
    })
    return map
  }, [citations])

  const renderedBlocks = useMemo(() => {
    if (!content) return null

    const lines = content.replace(/\r\n/g, '\n').split('\n')
    const blocks: ReactNode[] = []
    let i = 0

    while (i < lines.length) {
      const line = lines[i]
      const trimmed = line.trim()

      if (!trimmed) {
        i++
        continue
      }

      // 1. Fenced Code Block
      if (trimmed.startsWith('```')) {
        const lang = trimmed.slice(3).trim()
        const codeLines: string[] = []
        i++
        while (i < lines.length && !lines[i].trim().startsWith('```')) {
          codeLines.push(lines[i])
          i++
        }
        if (i < lines.length) i++ // Skip closing ```
        blocks.push(
          <CodeBlock
            key={`code-${i}`}
            language={lang}
            code={codeLines.join('\n')}
          />
        )
        continue
      }

      // 2. Markdown Table
      if (trimmed.startsWith('|') && trimmed.endsWith('|')) {
        const tableLines: string[] = []
        while (i < lines.length && lines[i].trim().startsWith('|') && lines[i].trim().endsWith('|')) {
          tableLines.push(lines[i])
          i++
        }
        const parsedTable = parseTable(tableLines)
        if (parsedTable) {
          blocks.push(
            <div key={`table-${i}`} className="markdown-table-wrapper">
              <table className="markdown-table">
                <thead>
                  <tr>
                    {parsedTable.headers.map((h, colIdx) => (
                      <th key={colIdx}>{renderInline(h, citationMap, `th-${i}-${colIdx}`)}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {parsedTable.rows.map((row, rIdx) => (
                    <tr key={rIdx}>
                      {row.map((cell, cIdx) => (
                        <td key={cIdx}>{renderInline(cell, citationMap, `td-${i}-${rIdx}-${cIdx}`)}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )
          continue
        }
        // If not a valid table, fall through to process tableLines as normal lines
        for (const tLine of tableLines) {
          blocks.push(<p key={`tbl-fallback-${blocks.length}`}>{renderInline(tLine, citationMap, `tbl-f-${blocks.length}`)}</p>)
        }
        continue
      }

      // 3. Headings
      if (trimmed.startsWith('#### ')) {
        blocks.push(<h4 key={`h4-${i}`}>{renderInline(trimmed.slice(5), citationMap, `h4-${i}`)}</h4>)
        i++
        continue
      }
      if (trimmed.startsWith('### ')) {
        blocks.push(<h3 key={`h3-${i}`}>{renderInline(trimmed.slice(4), citationMap, `h3-${i}`)}</h3>)
        i++
        continue
      }
      if (trimmed.startsWith('## ')) {
        blocks.push(<h2 key={`h2-${i}`}>{renderInline(trimmed.slice(3), citationMap, `h2-${i}`)}</h2>)
        i++
        continue
      }
      if (trimmed.startsWith('# ')) {
        blocks.push(<h1 key={`h1-${i}`}>{renderInline(trimmed.slice(2), citationMap, `h1-${i}`)}</h1>)
        i++
        continue
      }

      // 4. Blockquotes
      if (trimmed.startsWith('> ')) {
        const quoteLines: string[] = []
        while (i < lines.length && lines[i].trim().startsWith('> ')) {
          quoteLines.push(lines[i].trim().slice(2))
          i++
        }
        blocks.push(
          <blockquote key={`quote-${i}`}>
            {renderInline(quoteLines.join(' '), citationMap, `quote-${i}`)}
          </blockquote>
        )
        continue
      }

      // 5. Unordered Lists
      if (/^[-*+]\s+/.test(trimmed)) {
        const listItems: string[] = []
        while (i < lines.length && /^[-*+]\s+/.test(lines[i].trim())) {
          listItems.push(lines[i].trim().replace(/^[-*+]\s+/, ''))
          i++
        }
        blocks.push(
          <ul key={`ul-${i}`}>
            {listItems.map((item, idx) => (
              <li key={idx}>{renderInline(item, citationMap, `li-${i}-${idx}`)}</li>
            ))}
          </ul>
        )
        continue
      }

      // 6. Ordered Lists
      if (/^\d+[\.\)]\s+/.test(trimmed)) {
        const listItems: string[] = []
        while (i < lines.length && /^\d+[\.\)]\s+/.test(lines[i].trim())) {
          listItems.push(lines[i].trim().replace(/^\d+[\.\)]\s+/, ''))
          i++
        }
        blocks.push(
          <ol key={`ol-${i}`}>
            {listItems.map((item, idx) => (
              <li key={idx}>{renderInline(item, citationMap, `oli-${i}-${idx}`)}</li>
            ))}
          </ol>
        )
        continue
      }

      // 7. Regular Paragraph
      const pLines: string[] = []
      while (
        i < lines.length &&
        lines[i].trim() &&
        !lines[i].trim().startsWith('```') &&
        !lines[i].trim().startsWith('|') &&
        !lines[i].trim().startsWith('#') &&
        !lines[i].trim().startsWith('> ') &&
        !/^[-*+]\s+/.test(lines[i].trim()) &&
        !/^\d+[\.\)]\s+/.test(lines[i].trim())
      ) {
        pLines.push(lines[i].trim())
        i++
      }

      if (pLines.length > 0) {
        blocks.push(
          <p key={`p-${i}`}>
            {renderInline(pLines.join(' '), citationMap, `p-${i}`)}
          </p>
        )
      }
    }

    return blocks
  }, [content, citationMap])

  return <div className="markdown-content">{renderedBlocks}</div>
}
