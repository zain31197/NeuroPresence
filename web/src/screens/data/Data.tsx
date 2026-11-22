import { CheckCircle2, Database, Download, HardDrive, ImageIcon, KeyRound, Layers, Link2, Rows3, TriangleAlert, type LucideIcon } from 'lucide-react'
import { AnimatePresence, motion } from 'motion/react'
import { useEffect, useState } from 'react'
import { CountUp, EASE_OUT } from '../../components/motion'
import { PageHeader } from '../../components/PageHeader'
import { Button, buttonClass } from '../../components/ui/Button'
import { Card, CardHeader } from '../../components/ui/Card'
import { useToast } from '../../components/ui/Toast'
import { bytes } from '../../lib/account'
import { api, ApiError, EXPORT_PATH, type DataOverview, type DataTable } from '../../lib/api'
import { cn } from '../../lib/cn'
import { Diagram } from './Diagram'

/** Everything the database keeps about the account: how it is built, how many rows are theirs, and a copy to take away. */
export function Data() {
  const toast = useToast()
  const [data, setData] = useState<DataOverview | null>(null)
  const [busy, setBusy] = useState(false)
  const [chosen, setChosen] = useState<string | null>(null)

  const load = async () => {
    setBusy(true)
    try {
      setData(await api.data())
    } catch (error) {
      toast.problem(error instanceof ApiError ? error.message : 'The database could not be read.')
    } finally {
      setBusy(false)
    }
  }
  useEffect(() => {
    document.title = 'Your data · NeuroPresence'
    load()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const rows = data?.tables.reduce((sum, table) => sum + table.rows, 0) ?? 0
  const choose = (name: string) => {
    setChosen((now) => (now === name ? null : name))
    document.getElementById(`table-${name}`)?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
  }

  return (
    <main className="mx-auto max-w-[1180px] px-4 py-6 sm:px-6 lg:px-8 lg:py-7">
      <PageHeader
        title="Your data"
        actions={
          <>
            <Button busy={busy} onClick={load}>
              Read again
            </Button>
            <a href={EXPORT_PATH} download className={buttonClass('primary')}>
              <Download className="size-3.5" />
              Export my data
            </a>
          </>
        }
      >
        What the database on this computer keeps about you, read from the file itself. Every row here belongs to your account, and goes with
        it if you delete it.
      </PageHeader>

      <div className="mt-6 grid grid-cols-2 gap-px overflow-hidden rounded-card border border-line bg-line shadow-card lg:grid-cols-4">
        <Figure icon={Rows3} label="Rows that are yours" value={data ? rows : null} format={(v) => Math.round(v).toLocaleString('en-GB')} note={data ? `in ${data.tables.length} tables` : ''} index={0} />
        <Figure icon={HardDrive} label="The database" value={data?.bytes ?? null} format={(v) => bytes(Math.round(v))} note={data?.file ?? ''} index={1} />
        <Figure icon={ImageIcon} label="Your pictures in it" value={data?.pictures_bytes ?? null} format={(v) => bytes(Math.round(v))} note="stored as they are, not as links" index={2} />
        <Figure icon={Layers} label="Schema" value={data?.schema ?? null} format={(v) => `version ${Math.round(v)}`} note="upgraded in place" index={3} />
      </div>

      <Card className="mt-4 flex flex-wrap gap-x-8 gap-y-3 px-5 py-4">
        <Health ok={data ? data.integrity === 'ok' : null} text={data ? (data.integrity === 'ok' ? 'File intact' : `File: ${data.integrity}`) : 'Checking the file'} />
        <Health ok={data ? data.foreign_keys : null} text="Foreign keys enforced" />
        <Health ok={data ? data.orphans === 0 : null} text={data && data.orphans > 0 ? `${data.orphans} rows without an owner` : 'No row without its parent'} />
        <Health ok={data ? data.journal === 'wal' : null} text="Write-ahead log, so reading never waits" />
        <span className="ml-auto inline-flex items-center gap-2 font-mono text-[12px] text-ink-500">
          <Database className="size-3.5" />
          SQLite {data?.sqlite ?? ''}
        </span>
      </Card>

      <Card className="mt-4 overflow-hidden">
        <CardHeader
          title="How it is built"
          hint="Each box is a table, with how many of its rows are yours. A line is a foreign key: solid goes when its parent is deleted, dashed is only cleared. Point at a table to follow its ties."
        />
        <div className="px-3 pb-3 sm:px-5">{data ? <Diagram tables={data.tables} chosen={chosen} onChoose={choose} /> : <div className="h-[420px] animate-pulse rounded-panel bg-ink-50" />}</div>
      </Card>

      <div className="mt-4 grid gap-3 lg:grid-cols-2">
        {data?.tables.map((table) => (
          <TableCard key={table.name} table={table} open={chosen === table.name} onToggle={() => setChosen(chosen === table.name ? null : table.name)} />
        ))}
      </div>
    </main>
  )
}

function Figure({ icon: Icon, label, value, format, note, index }: { icon: LucideIcon; label: string; value: number | null; format: (v: number) => string; note: string; index: number }) {
  return (
    <motion.div className="bg-surface px-5 py-4" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.45, ease: EASE_OUT, delay: index * 0.06 }}>
      <p className="flex items-center gap-2 text-[12.5px] font-medium text-ink-500">
        <Icon className="size-3.5 text-ink-400" />
        {label}
      </p>
      <p className="display mt-1.5 text-[26px] leading-none font-semibold text-ink-950">{value === null ? '–' : <CountUp value={value} format={format} />}</p>
      <p className="mt-1.5 truncate text-[12px] text-ink-400">{note}</p>
    </motion.div>
  )
}

function Health({ ok, text }: { ok: boolean | null; text: string }) {
  return (
    <span className="inline-flex items-center gap-2 text-[13px] text-ink-700">
      {ok === null ? <span className="size-4 animate-pulse rounded-full bg-ink-100" /> : ok ? <CheckCircle2 className="size-4 text-good" /> : <TriangleAlert className="size-4 text-warning" />}
      {text}
    </span>
  )
}

function TableCard({ table, open, onToggle }: { table: DataTable; open: boolean; onToggle: () => void }) {
  return (
    <div id={`table-${table.name}`} className={cn('scroll-mt-6 overflow-hidden rounded-card border bg-surface shadow-card transition-colors duration-200', open ? 'border-accent-300' : 'border-line')}>
      <button type="button" aria-expanded={open} onClick={onToggle} className="flex w-full items-start gap-4 px-5 py-4 text-left transition-colors hover:bg-ink-50/70">
        <div className="min-w-0 flex-1">
          <p className="font-mono text-[13.5px] font-semibold text-ink-950">{table.name}</p>
          <p className="mt-0.5 text-[12.5px] leading-snug text-ink-500">{table.purpose}</p>
        </div>
        <div className="shrink-0 text-right">
          <p className="display text-[20px] leading-none font-semibold text-ink-950 tabular-nums">{table.rows.toLocaleString('en-GB')}</p>
          <p className="mt-1 text-[11.5px] text-ink-400">{table.rows === 1 ? 'row' : 'rows'}</p>
        </div>
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} transition={{ duration: 0.28, ease: EASE_OUT }}>
            <ul className="divide-y divide-line border-t border-line">
              {table.columns.map((column) => (
                <li key={column.name} className="flex items-center gap-2.5 px-5 py-2 text-[12.5px]">
                  {column.key ? <KeyRound className="size-3.5 shrink-0 text-warning" aria-label="Primary key" /> : column.references ? <Link2 className="size-3.5 shrink-0 text-accent-600" aria-label="Foreign key" /> : <span className="size-3.5 shrink-0" />}
                  <span className="font-mono text-ink-900">{column.name}</span>
                  {column.references && (
                    <span className="truncate text-[12px] text-accent-700">
                      {column.references.table}.{column.references.column}
                    </span>
                  )}
                  <span className="ml-auto shrink-0 font-mono text-[11px] text-ink-400">
                    {column.type.toLowerCase()}
                    {column.required ? '' : ', may be empty'}
                  </span>
                </li>
              ))}
            </ul>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
