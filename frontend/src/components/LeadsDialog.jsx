import { useEffect, useRef } from 'react'

// Leads (owner link only): what customers told her about themselves and what they need, written down
// after each conversation (backend leads.py), newest first, and a CSV download for a spreadsheet or CRM.
const COLUMNS = [['time', 'When'], ['name', 'Name'], ['email', 'Email'], ['phone', 'Phone'], ['need', 'Needs'],
  ['interest', 'Interest'], ['next_step', 'Next step'], ['timeline', 'Timeline'], ['budget', 'Budget'], ['notes', 'Notes']]

export default function LeadsDialog({ open, onClose, conv }) {
  const ref = useRef(null)

  useEffect(() => {
    const dialog = ref.current
    if (!dialog) return
    if (open && !dialog.open) {
      conv.loadLeads()
      dialog.showModal()
    } else if (!open && dialog.open) {
      dialog.close()
    }
  }, [open, conv])

  const rows = conv.leads?.leads ?? []
  const download = () => {
    const blob = new Blob([conv.leads.csv], { type: 'text/csv' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = `leads-${new Date().toISOString().slice(0, 10)}.csv`
    a.click()
    setTimeout(() => URL.revokeObjectURL(a.href), 1000)
  }

  return (
    <dialog ref={ref} className="settings leads" closedby="any" aria-labelledby="leads-title" onClose={onClose}
      data-testid="leads">
      <h2 id="leads-title">Leads</h2>
      {!conv.leads ? <p className="muted">Loading…</p> : rows.length === 0 ? (
        <p className="muted" data-testid="leads-empty">No leads yet. After a customer's conversation, what they said about
          themselves and what they need appears here.</p>
      ) : (
        <div className="table-wrap">
          <table data-testid="leads-table">
            <thead><tr>{COLUMNS.map(([, h]) => <th key={h}>{h}</th>)}</tr></thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.session}>{COLUMNS.map(([k]) => <td key={k}>{r[k]}</td>)}</tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <div className="actions">
        <span className="muted">Stored only on this PC (data\leads.jsonl).</span>
        <button type="button" className="secondary" onClick={conv.loadLeads}>Refresh</button>
        <button type="button" onClick={download} disabled={!rows.length} data-testid="leads-download">Download CSV</button>
        <button type="button" className="secondary" onClick={onClose}>Close</button>
      </div>
    </dialog>
  )
}
