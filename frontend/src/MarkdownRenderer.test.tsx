import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import { MarkdownRenderer } from './components/MarkdownRenderer'
import type { CitationItem } from './services/chatApi'

afterEach(() => {
  cleanup()
})

describe('MarkdownRenderer', () => {
  const mockCitations: CitationItem[] = [
    {
      chunk_id: 'chunk_cse202_overview',
      document_id: 'doc_1',
      source_file: 'CSE202.pdf',
      title: 'Object Oriented Programming Syllabus',
    },
    {
      chunk_id: 'chunk_exam_dates',
      document_id: 'doc_2',
      source_file: 'Academic_Calendar.pdf',
      title: 'Examination Schedule 2026',
    },
  ]

  it('renders headings correctly', () => {
    const content = '# Heading 1\n## Heading 2\n### Heading 3\n#### Heading 4'
    render(<MarkdownRenderer content={content} />)

    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('Heading 1')
    expect(screen.getByRole('heading', { level: 2 })).toHaveTextContent('Heading 2')
    expect(screen.getByRole('heading', { level: 3 })).toHaveTextContent('Heading 3')
    expect(screen.getByRole('heading', { level: 4 })).toHaveTextContent('Heading 4')
  })

  it('renders bold and italic formatting without raw markdown asterisks', () => {
    const content = 'This is **bold concept** and this is *italic emphasis*.'
    render(<MarkdownRenderer content={content} />)

    const strongEl = screen.getByText('bold concept')
    expect(strongEl.tagName).toBe('STRONG')

    const emEl = screen.getByText('italic emphasis')
    expect(emEl.tagName).toBe('EM')

    expect(screen.queryByText(/\*\*/)).not.toBeInTheDocument()
  })

  it('renders unordered and ordered lists', () => {
    const content = '- Item Alpha\n- Item Beta\n- Item Gamma\n\n1. First step\n2. Second step'
    render(<MarkdownRenderer content={content} />)

    expect(screen.getByText('Item Alpha')).toBeInTheDocument()
    expect(screen.getByText('Item Beta')).toBeInTheDocument()
    expect(screen.getByText('First step')).toBeInTheDocument()
    expect(screen.getByText('Second step')).toBeInTheDocument()

    const uls = screen.getAllByRole('list')
    expect(uls.length).toBe(2)
  })

  it('renders markdown tables with horizontal scroll wrapper', () => {
    const content = `| Exam Type | Start Date | End Date |
| :--- | :--- | :--- |
| Mid-Term | 15 Oct 2026 | 22 Oct 2026 |
| End-Term | 10 Dec 2026 | 24 Dec 2026 |`

    const { container } = render(<MarkdownRenderer content={content} />)

    expect(screen.getByRole('table')).toBeInTheDocument()
    expect(screen.getByText('Exam Type')).toBeInTheDocument()
    expect(screen.getByText('Mid-Term')).toBeInTheDocument()
    expect(screen.getByText('End-Term')).toBeInTheDocument()
    expect(screen.getByText('10 Dec 2026')).toBeInTheDocument()

    const wrapper = container.querySelector('.markdown-table-wrapper')
    expect(wrapper).toBeInTheDocument()
  })

  it('renders fenced code blocks with language badge and code content', () => {
    const content = '```cpp\n#include <iostream>\nint main() { return 0; }\n```'
    const { container } = render(<MarkdownRenderer content={content} />)

    expect(screen.getByText('CPP')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Copy code to clipboard' })).toBeInTheDocument()
    expect(container.querySelector('code')).toHaveTextContent('#include <iostream>')
  })

  it('transforms [[cite:chunk_id]] tags into clean inline citation pills', () => {
    const content = 'CSE202 covers classes and objects [[cite:chunk_cse202_overview]]. Exams start in December [[cite:chunk_exam_dates]].'
    render(<MarkdownRenderer content={content} citations={mockCitations} />)

    expect(screen.getByText('[1]')).toBeInTheDocument()
    expect(screen.getByText('[2]')).toBeInTheDocument()
    expect(screen.queryByText(/\[\[cite:/)).not.toBeInTheDocument()

    const citePill = screen.getByText('[1]')
    expect(citePill).toHaveAttribute('title', expect.stringContaining('CSE202.pdf'))
  })

  it('renders simple conversational greetings as clean paragraphs', () => {
    const content = 'Hello! How can I help you with your academic questions today?'
    render(<MarkdownRenderer content={content} />)

    expect(screen.getByText(content)).toBeInTheDocument()
    expect(screen.queryByRole('heading')).not.toBeInTheDocument()
    expect(screen.queryByRole('list')).not.toBeInTheDocument()
  })
})
