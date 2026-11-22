import { motion } from 'motion/react'
import { useMemo, useState } from 'react'
import { EASE_OUT } from '../../components/motion'
import type { DataTable } from '../../lib/api'
import { cn } from '../../lib/cn'

// Where each table stands. users is the middle: everything else belongs to one.
const PLACE: Record<string, Place> = {
  logins: [0, 0],
  login_attempts: [0, 206],
  user_settings: [0, 394],
  enrolment_history: [0, 558],
  users: [310, 226],
  pictures: [310, 490],
  sessions: [620, 52],
  session_figures: [930, 0],
  events: [930, 234],
  consent_checks: [930, 468],
}
const WIDTH = 210
const HEAD = 38
const ROW = 24
const MOST_ROWS = 5 // of a table: its keys first, then what comes
const SIZE = { width: 1140, height: 730 }

type Column = DataTable['columns'][number]
type Place = [x: number, y: number]

/** Where each table stands. One this file does not know yet gets a place in a row underneath, so none is left out. */
export function places(names: string[]): Record<string, Place> {
  const unknown = names.filter((name) => !PLACE[name])
  const below = unknown.map((name, n): [string, Place] => [name, [(n % 4) * 310, SIZE.height + 30 + Math.floor(n / 4) * 240]])
  return { ...PLACE, ...Object.fromEntries(below) }
}

/** The columns that are drawn: the key, the references, then the first others, and how many are left out. */
export function drawnColumns(table: DataTable): { columns: Column[]; more: number } {
  const keys = table.columns.filter((column) => column.key || column.references)
  const others = table.columns.filter((column) => !column.key && !column.references)
  const columns = [...keys, ...others].slice(0, Math.max(MOST_ROWS, keys.length))
  return { columns, more: table.columns.length - columns.length }
}

const boxHeight = (table: DataTable) => {
  const { columns, more } = drawnColumns(table)
  return HEAD + columns.length * ROW + (more ? ROW : 6)
}

/** The database's tables and how they refer to one another, drawn from its own structure. Point at one to see its ties. */
export function Diagram({ tables, chosen, onChoose }: { tables: DataTable[]; chosen: string | null; onChoose: (name: string) => void }) {
  const [pointed, setPointed] = useState<string | null>(null)
  const lit = pointed ?? chosen
  const place = useMemo(() => places(tables.map((table) => table.name)), [tables])
  const height = Math.max(SIZE.height, ...tables.map((table) => place[table.name][1] + boxHeight(table) + 8))
  const ties = useMemo(
    () =>
      tables.flatMap((table) =>
        drawnColumns(table).columns.flatMap((column, row) => {
          const target = column.references && tables.find((other) => other.name === column.references!.table)
          if (!column.references || !target) return []
          const [x, y] = place[table.name]
          const [tx, ty] = place[target.name]
          const fromY = y + HEAD + row * ROW + ROW / 2
          const toY = ty + HEAD + ROW / 2
          const right = tx + WIDTH / 2 > x + WIDTH / 2 + 10
          const level = Math.abs(tx - x) < 20
          const fromX = right ? x + WIDTH : x
          const toX = level ? tx : right ? tx : tx + WIDTH
          const bend = level ? fromX - 44 : (fromX + toX) / 2
          return [
            {
              key: `${table.name}.${column.name}`,
              from: table.name,
              to: target.name,
              path: `M${fromX},${fromY} C${bend},${fromY} ${bend},${toY} ${toX},${toY}`,
              x: fromX,
              y: fromY,
              cascade: column.references.on_delete === 'CASCADE',
            },
          ]
        }),
      ),
    [tables, place],
  )

  return (
    <div className="overflow-x-auto pb-2">
      <svg
        viewBox={`-26 -10 ${SIZE.width + 40} ${height + 10}`}
        className="block h-auto w-full min-w-[860px]"
        role="img"
        aria-label={`Entity relationship diagram of ${tables.length} tables`}
      >
        {ties.map((tie, index) => {
          const on = lit !== null && (tie.from === lit || tie.to === lit)
          return (
            <g key={tie.key} className="transition-opacity duration-200" opacity={lit === null || on ? 1 : 0.18}>
              <motion.path
                d={tie.path}
                fill="none"
                stroke={on ? 'var(--color-accent-500)' : 'var(--color-ink-300)'}
                strokeWidth={on ? 2 : 1.4}
                strokeDasharray={tie.cascade ? undefined : '5 4'}
                initial={tie.cascade ? { pathLength: 0, opacity: 0 } : { opacity: 0 }}
                animate={tie.cascade ? { pathLength: 1, opacity: 1 } : { opacity: 1 }}
                transition={{ duration: 0.8, ease: EASE_OUT, delay: 0.35 + index * 0.05 }}
              />
              <circle cx={tie.x} cy={tie.y} r={on ? 4 : 3} fill={on ? 'var(--color-accent-500)' : 'var(--color-ink-400)'} />
            </g>
          )
        })}
        {tables.map((table, index) => {
          const [x, y] = place[table.name]
          const { columns, more } = drawnColumns(table)
          const near = lit === null || lit === table.name || ties.some((tie) => (tie.from === lit && tie.to === table.name) || (tie.to === lit && tie.from === table.name))
          return (
            <motion.g
              key={table.name}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: near ? 1 : 0.35, y: 0 }}
              transition={{ duration: 0.45, ease: EASE_OUT, delay: lit === null ? index * 0.05 : 0 }}
            >
              <g
                transform={`translate(${x} ${y})`}
                tabIndex={0}
                role="button"
                aria-label={`${table.name}, ${table.rows} rows`}
                className="cursor-pointer outline-none"
                onPointerEnter={() => setPointed(table.name)}
                onPointerLeave={() => setPointed(null)}
                onFocus={() => setPointed(table.name)}
                onBlur={() => setPointed(null)}
                onClick={() => onChoose(table.name)}
                onKeyDown={(event) => (event.key === 'Enter' || event.key === ' ') && onChoose(table.name)}
              >
                <rect width={WIDTH} height={boxHeight(table)} rx="12" fill="var(--color-surface)" stroke={lit === table.name ? 'var(--color-accent-500)' : 'var(--color-line-strong)'} strokeWidth={lit === table.name ? 2 : 1} />
                <path d={`M0,12 a12,12 0 0 1 12,-12 h${WIDTH - 24} a12,12 0 0 1 12,12 v${HEAD - 12} h-${WIDTH} z`} fill={table.name === 'users' ? 'var(--color-ink-950)' : 'var(--color-ink-50)'} />
                <text x="13" y="24" className={cn('text-[13px] font-semibold', table.name === 'users' ? 'fill-white' : 'fill-ink-950')}>
                  {table.name}
                </text>
                <text x={WIDTH - 13} y="24" textAnchor="end" className={cn('font-mono text-[11px]', table.name === 'users' ? 'fill-white/70' : 'fill-ink-500')}>
                  {table.rows.toLocaleString('en-GB')}
                </text>
                {columns.map((column, row) => (
                  <g key={column.name} transform={`translate(0 ${HEAD + row * ROW})`}>
                    <text x="13" y="16" className={cn('font-mono text-[11.5px]', column.key ? 'fill-ink-950 font-semibold' : column.references ? 'fill-accent-700' : 'fill-ink-600')}>
                      {column.name}
                    </text>
                    <text x={WIDTH - 13} y="16" textAnchor="end" className="fill-ink-400 font-mono text-[10px]">
                      {column.key ? 'KEY' : column.references ? 'REF' : column.type.toLowerCase()}
                    </text>
                  </g>
                ))}
                {more > 0 && (
                  <text x="13" y={HEAD + columns.length * ROW + 16} className="fill-ink-400 text-[11px]">
                    and {more} more
                  </text>
                )}
              </g>
            </motion.g>
          )
        })}
      </svg>
    </div>
  )
}
