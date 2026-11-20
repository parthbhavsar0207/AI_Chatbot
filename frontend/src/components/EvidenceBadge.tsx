import { CheckCircle2, CircleHelp, CircleSlash, TriangleAlert } from 'lucide-react'
import type { EvidenceState } from '../services/chatApi'

const details: Record<EvidenceState, { label: string; className: string; Icon: typeof CheckCircle2 }> = {
  sufficient: { label: 'Sufficient evidence', className: 'evidence-sufficient', Icon: CheckCircle2 },
  partial: { label: 'Partial evidence', className: 'evidence-partial', Icon: CircleHelp },
  insufficient: { label: 'Insufficient evidence', className: 'evidence-insufficient', Icon: CircleSlash },
  conflicting: { label: 'Conflicting evidence', className: 'evidence-conflicting', Icon: TriangleAlert },
}

export function EvidenceBadge({ state }: { state: EvidenceState }) {
  const { label, className, Icon } = details[state]
  return <span className={`evidence-badge ${className}`}><Icon size={14} aria-hidden="true" />{label}</span>
}
