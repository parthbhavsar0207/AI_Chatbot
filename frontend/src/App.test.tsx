import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import type { ChatAnswerResponse, EvidenceState } from './services/chatApi'

afterEach(() => { cleanup(); vi.unstubAllGlobals() })

function response(state: EvidenceState): ChatAnswerResponse {
  return {
    query: 'Tell me about the policy',
    answer: `Backend answer for ${state}.`,
    evidence_state: state,
    citations: state === 'sufficient' ? [{ chunk_id: 'chunk-abc', document_id: 'doc-123', source_file: 'CSE205.pdf', category: 'syllabus', course_code: 'CSE205', title: 'Prerequisites' }] : [],
    source_metadata: [],
    missing_information: state === 'partial' ? ['The minimum grade is not specified.'] : null,
    conflict_details: state === 'conflicting' ? { explanation: 'Two official documents differ.' } : null,
    model_identifier: 'deterministic_engine',
    retrieval_latency_ms: 3,
    reranking_latency_ms: 4,
    generation_latency_ms: 0,
    total_latency_ms: 7,
    citation_validation: { status: 'bypassed', valid_citations_count: 0, invalid_citations: [], substantive_statements_count: 0, cited_statements_count: 0, structural_coverage_passed: true, semantic_grounding_note: 'Structural only.' },
    citation_validation_status: 'bypassed',
    timestamp: '2026-01-01T00:00:00Z',
  }
}

async function submitQuestion() {
  const user = userEvent.setup()
  await user.type(screen.getByRole('textbox', { name: 'Your question' }), 'Tell me about the policy')
  await user.click(screen.getByRole('button', { name: 'Send message' }))
}

describe('chat interface', () => {
  it('does not submit blank or whitespace input', async () => {
    const fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)
    render(<App />)
    expect(screen.getByRole('button', { name: 'Send message' })).toBeDisabled()
    fireEvent.change(screen.getByRole('textbox', { name: 'Your question' }), { target: { value: '   ' } })
    expect(screen.getByRole('button', { name: 'Send message' })).toBeDisabled()
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it.each(['sufficient', 'partial', 'insufficient', 'conflicting'] as EvidenceState[])('renders %s evidence response from the API', async (state) => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => response(state) }))
    render(<App />)
    await submitQuestion()
    expect(await screen.findByText(`Backend answer for ${state}.`)).toBeInTheDocument()
    expect(screen.getByText(new RegExp(`${state} evidence`, 'i'))).toBeInTheDocument()
    if (state === 'sufficient') {
      expect(screen.getByLabelText('Sources')).toHaveTextContent('CSE205.pdf')
      expect(screen.getByText('Prerequisites')).toBeInTheDocument()
    }
    if (state === 'partial') expect(screen.getByText('The minimum grade is not specified.')).toBeInTheDocument()
    if (state === 'conflicting') expect(screen.getByText('Two official documents differ.')).toBeInTheDocument()
  })

  it('shows loading feedback, clears the chat, and sends the exact request shape', async () => {
    let finish!: (value: unknown) => void
    const fetchMock = vi.fn().mockReturnValue(new Promise((resolve) => { finish = resolve }))
    vi.stubGlobal('fetch', fetchMock)
    render(<App />)
    await submitQuestion()
    expect(screen.getByLabelText('Assistant is thinking')).toBeInTheDocument()
    expect(fetchMock).toHaveBeenCalledWith(expect.stringMatching(/\/api\/v1\/chat\/answer$/), expect.objectContaining({ method: 'POST', body: JSON.stringify({ query: 'Tell me about the policy', top_k: 5 }) }))
    finish({ ok: true, json: async () => response('insufficient') })
    await screen.findByText('Backend answer for insufficient.')
    await userEvent.setup().click(screen.getByRole('button', { name: 'Clear conversation' }))
    expect(screen.queryByRole('log', { name: 'Conversation' })).not.toBeInTheDocument()
  })

  it.each([422, 500, 502, 503, 504])('shows the backend detail for HTTP %i', async (status) => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false, status, json: async () => ({ detail: `Backend ${status} detail` }) }))
    render(<App />)
    await submitQuestion()
    expect(await screen.findByRole('alert')).toHaveTextContent(`Backend ${status} detail`)
  })

  it('shows FastAPI validation messages when a 422 detail is an error list', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false, status: 422, json: async () => ({ detail: [{ msg: 'Field required' }] }) }))
    render(<App />)
    await submitQuestion()
    expect(await screen.findByRole('alert')).toHaveTextContent('Field required')
  })

  it('surfaces the standard connection error when fetch fails', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('offline')))
    render(<App />)
    await submitQuestion()
    expect(await screen.findByRole('alert')).toHaveTextContent('Could not reach the chatbot backend')
  })
})
