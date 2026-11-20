import { useEffect, useRef, useState, type FormEvent } from 'react'
import { ArrowUp, BookOpen, GraduationCap, RotateCcw, Sparkles } from 'lucide-react'
import { AssistantAnswer } from './components/AssistantAnswer'
import { ChatApiError, askQuestion, type ChatAnswerResponse } from './services/chatApi'

type Message =
  | { id: string; role: 'user'; text: string }
  | { id: string; role: 'assistant'; response: ChatAnswerResponse }

export default function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [failedQuery, setFailedQuery] = useState<string | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)
  const idRef = useRef(0)
  const nextId = () => `message-${++idRef.current}`

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' }) }, [messages, loading, error])

  async function runQuery(query: string) {
    setError(null)
    setLoading(true)
    try {
      const response = await askQuestion(query)
      setMessages((current) => [...current, { id: nextId(), role: 'assistant', response }])
      setFailedQuery(null)
    } catch (cause) {
      setFailedQuery(query)
      setError(cause instanceof ChatApiError ? cause.message : 'Something went wrong while contacting the assistant. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  async function sendMessage(event?: FormEvent) {
    event?.preventDefault()
    const query = input.trim()
    if (!query || loading) return
    setMessages((current) => [...current, { id: nextId(), role: 'user', text: query }])
    setInput('')
    await runQuery(query)
  }

  function clearConversation() {
    if (loading) return
    setMessages([])
    setError(null)
    setFailedQuery(null)
    setInput('')
  }

  const hasConversation = messages.length > 0

  return <div className="app-shell">
    <header className="topbar">
      <a className="brand" href="#home" aria-label="LPU Academic Assistant home">
        <span className="brand-mark"><GraduationCap size={22} strokeWidth={2.1} /></span>
        <span><strong>LPU Academic Assistant</strong><small>CSE · Artificial Intelligence &amp; Machine Learning</small></span>
      </a>
      <div className="topbar-right">
        <span className="service-label"><span className="status-dot" />Student support</span>
        {hasConversation && <button className="clear-button" onClick={clearConversation} disabled={loading} aria-label="Clear conversation"><RotateCcw size={15} /><span>Clear chat</span></button>}
      </div>
    </header>

    <main className="chat-layout">
      <section className={`chat-panel ${hasConversation ? 'has-messages' : 'is-empty'}`} aria-label="Chat with the academic assistant">
        {!hasConversation ? <div className="welcome">
          <span className="welcome-icon"><BookOpen size={24} /></span>
          <p className="eyebrow">YOUR ACADEMIC GUIDE</p>
          <h1>What can I help you<br className="desktop-break" /> find today?</h1>
          <p className="welcome-copy">Ask about course syllabi, academic regulations, examinations, and more. Answers are grounded in available university documents.</p>
          <div className="suggestions" aria-label="Example questions">
            {['What are the prerequisites for CSE205?', 'What is the attendance requirement?', 'Where can I find exam information?'].map((question) => <button key={question} onClick={() => setInput(question)}>{question}</button>)}
          </div>
        </div> : <div className="messages" role="log" aria-live="polite" aria-label="Conversation">
          {messages.map((message) => <article className={`message-row ${message.role}`} key={message.id}>
            {message.role === 'assistant' && <span className="assistant-avatar" aria-hidden="true"><GraduationCap size={17} /></span>}
            <div className={`message-bubble ${message.role === 'user' ? 'user-bubble' : 'assistant-bubble'}`}>
              {message.role === 'user' ? <p>{message.text}</p> : <AssistantAnswer response={message.response} />}
            </div>
          </article>)}
          {loading && <article className="message-row assistant" aria-label="Assistant is thinking">
            <span className="assistant-avatar" aria-hidden="true"><GraduationCap size={17} /></span>
            <div className="message-bubble assistant-bubble loading-bubble"><span className="loading-orb"><Sparkles size={15} /></span><span><strong>Checking university sources</strong><small>It may take a few seconds to prepare a grounded answer.</small></span><span className="typing-dots" aria-hidden="true"><i /><i /><i /></span></div>
          </article>}
          {error && <div className="error-banner" role="alert"><span>{error}</span><button onClick={() => failedQuery && !loading ? void runQuery(failedQuery) : setError(null)}>Try again</button></div>}
          <div ref={bottomRef} />
        </div>}

        {!hasConversation && error && <div className="error-banner initial-error" role="alert"><span>{error}</span><button onClick={() => setError(null)}>Dismiss</button></div>}

        <div className="composer-wrap">
          <form className="composer" onSubmit={sendMessage}>
            <textarea value={input} onChange={(event) => setInput(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); void sendMessage() } }} placeholder="Ask an academic question…" aria-label="Your question" rows={1} disabled={loading} />
            <button className="send-button" type="submit" aria-label="Send message" disabled={!input.trim() || loading}><ArrowUp size={19} strokeWidth={2.4} /></button>
          </form>
          <p className="composer-note">Answers are generated from indexed university documents. Always verify important decisions with official university sources.</p>
        </div>
      </section>
    </main>
  </div>
}
