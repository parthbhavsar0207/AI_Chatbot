import { BookOpen, FileText } from 'lucide-react'
import type { ChatAnswerResponse } from '../services/chatApi'
import { EvidenceBadge } from './EvidenceBadge'
import { MarkdownRenderer } from './MarkdownRenderer'

function displayValue(value: unknown): string {
  if (typeof value === 'string') return value
  if (value === null || value === undefined) return ''
  return JSON.stringify(value, null, 2)
}

export function AssistantAnswer({ response }: { response: ChatAnswerResponse }) {
  return (
    <div className="assistant-content">
      <div className="answer-meta"><EvidenceBadge state={response.evidence_state} /></div>
      <div className="answer-text">
        <MarkdownRenderer content={response.answer} citations={response.citations} />
      </div>

      {response.missing_information?.length ? (
        <section className="response-note" aria-label="Missing information">
          <h3><CircleDotIcon />Information not available</h3>
          <ul>{response.missing_information.map((item, i) => <li key={`${i}-${item}`}>{item}</li>)}</ul>
        </section>
      ) : null}

      {response.conflict_details && Object.keys(response.conflict_details).length > 0 ? (
        <section className="response-note conflict-note" aria-label="Conflict details">
          <h3><CircleDotIcon />Conflict details</h3>
          <dl>{Object.entries(response.conflict_details).map(([key, value]) => (
            <div key={key}><dt>{key.replaceAll('_', ' ')}</dt><dd>{displayValue(value)}</dd></div>
          ))}</dl>
        </section>
      ) : null}

      {response.citations.length > 0 ? (
        <section className="citations" aria-label="Sources">
          <h3><BookOpen size={15} aria-hidden="true" />Sources</h3>
          <ul>{response.citations.map((citation, i) => {
            const source = citation.source_file || response.source_metadata.find((item) => item.source_file)?.source_file
            const provenance = response.source_metadata.find((item) => item.source_file === source)
            const provenanceLabel = provenance && [provenance.authority_level, provenance.institutional_authenticity, provenance.verification_status].filter(Boolean).join(' · ')
            return (
              <li key={`${citation.chunk_id}-${i}`}>
                <span className="citation-num">[{i + 1}]</span>
                <FileText size={15} aria-hidden="true" />
                <span className="citation-copy">
                  <strong>{citation.title || source || citation.chunk_id}</strong>
                  <span>{[source && citation.title ? source : null, citation.course_code, citation.category, provenanceLabel].filter(Boolean).join(' · ')}</span>
                </span>
              </li>
            )
          })}</ul>
        </section>
      ) : null}
    </div>
  )
}

function CircleDotIcon() {
  return <span className="note-dot" aria-hidden="true" />
}
