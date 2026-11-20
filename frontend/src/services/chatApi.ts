export type ChatAnswerRequest = {
  query: string
  top_k?: number
  category?: string | null
  course_code?: string | null
  source_file?: string | null
}

export type EvidenceState = 'sufficient' | 'partial' | 'insufficient' | 'conflicting'

export type CitationItem = {
  chunk_id: string
  document_id: string
  source_file?: string | null
  category?: string | null
  course_code?: string | null
  title?: string | null
}

export type SourceMetadataItem = {
  source_file: string
  authority_level?: string | null
  document_origin?: string | null
  institutional_authenticity?: string | null
  verification_status?: string | null
}

export type CitationValidationResult = {
  status: 'valid' | 'regenerated_valid' | 'failed_rejected' | 'bypassed'
  valid_citations_count: number
  invalid_citations: string[]
  substantive_statements_count: number
  cited_statements_count: number
  structural_coverage_passed: boolean
  semantic_grounding_note: string
}

export type ChatAnswerResponse = {
  query: string
  answer: string
  evidence_state: EvidenceState
  citations: CitationItem[]
  source_metadata: SourceMetadataItem[]
  missing_information?: string[] | null
  conflict_details?: Record<string, unknown> | null
  model_identifier: string
  retrieval_latency_ms: number
  reranking_latency_ms: number
  generation_latency_ms: number
  total_latency_ms: number
  citation_validation: CitationValidationResult
  citation_validation_status: CitationValidationResult['status']
  timestamp: string
}

const baseUrl = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/$/, '')

const statusMessage: Record<number, string> = {
  422: 'The question could not be validated. Check it and try again.',
  500: 'The assistant encountered a server error. Please try again shortly.',
  502: 'The local language model returned an invalid response. Please try again.',
  503: 'The chatbot service or local model is unavailable. Check that the backend and Ollama are running.',
  504: 'The local model took too long to respond. Please try again.',
}

export class ChatApiError extends Error {
  constructor(message: string, public readonly status?: number) {
    super(message)
    this.name = 'ChatApiError'
  }
}

export async function askQuestion(query: string, signal?: AbortSignal): Promise<ChatAnswerResponse> {
  let response: Response
  try {
    response = await fetch(`${baseUrl}/api/v1/chat/answer`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify({ query: query.trim(), top_k: 5 } satisfies ChatAnswerRequest),
      signal,
    })
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') throw error
    throw new ChatApiError('Could not reach the chatbot backend. Check that it is running and try again.')
  }

  if (!response.ok) {
    let detail: string | undefined
    try {
      const body: { detail?: unknown } = await response.json()
      if (typeof body.detail === 'string') detail = body.detail
      else if (Array.isArray(body.detail)) {
        const messages = body.detail.flatMap((item: unknown) => {
          if (typeof item === 'object' && item !== null && 'msg' in item && typeof item.msg === 'string') return [item.msg]
          return []
        })
        if (messages.length) detail = messages.join(' ')
      }
    } catch { /* The documented error body may be unavailable. */ }
    throw new ChatApiError(detail || statusMessage[response.status] || `The request failed (HTTP ${response.status}).`, response.status)
  }

  try {
    return await response.json() as ChatAnswerResponse
  } catch {
    throw new ChatApiError('The chatbot backend returned an unreadable response.')
  }
}
