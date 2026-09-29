// Per-turn latency breakdown reported by the server (ms from the turn's t0:
// end of user speech for voice turns, message sent for text turns).
const COLUMNS = [
  ['stt_done', 'STT'],
  ['rag_done', 'RAG'],
  ['llm_first_token', 'LLM 1st tok'],
  ['llm_first_text', 'LLM 1st text'],
  ['tts_first_audio', 'TTS 1st audio'],
  ['avatar_first_frame', 'Avatar 1st frame'],
  ['client_first_audio', 'Heard (client)'],
  ['llm_done', 'LLM done'],
]

export default function MetricsPanel({ metrics }) {
  if (!metrics.length) return null
  const rows = metrics.slice(-6).reverse()
  return (
    <details className="metrics" open>
      <summary>Latency (ms from end of user input)</summary>
      <div className="table-wrap">
        <table data-testid="metrics">
          <thead>
            <tr>
              <th>Turn</th>
              {COLUMNS.map(([, label]) => <th key={label}>{label}</th>)}
              <th>tok/s</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((m) => (
              <tr key={`${m.turn_id}-${m.ts}`} className={m.interrupted ? 'interrupted' : ''}>
                <td>{m.turn_id}{m.interrupted ? '*' : ''}</td>
                {COLUMNS.map(([key]) => <td key={key}>{m.ms_from_t0?.[key] ?? '–'}</td>)}
                <td>{m.llm?.tokens_per_s ? m.llm.tokens_per_s.toFixed(0) : '–'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  )
}
