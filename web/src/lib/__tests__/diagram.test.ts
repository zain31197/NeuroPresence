import { describe, expect, it } from 'vitest'
import { drawnColumns, places } from '../../screens/data/Diagram'
import type { DataTable } from '../api'

const column = (name: string, more: Partial<DataTable['columns'][number]> = {}) => ({ name, type: 'TEXT', key: false, required: false, references: null, ...more })
const table = (columns: DataTable['columns']): DataTable => ({ name: 't', purpose: '', rows: 0, columns })

describe('the diagram of the database', () => {
  it('draws the keys first and says how many columns it leaves out', () => {
    const owner = column('user_id', { references: { table: 'users', column: 'id', on_delete: 'CASCADE' } })
    const { columns, more } = drawnColumns(table([column('id', { key: true }), column('a'), column('b'), column('c'), column('d'), owner]))
    expect(columns.map((c) => c.name)).toEqual(['id', 'user_id', 'a', 'b', 'c'])
    expect(more).toBe(1)
  })

  it('never leaves out a reference, whatever the room', () => {
    const to = (name: string) => column(name, { references: { table: 'users', column: 'id', on_delete: 'CASCADE' } })
    expect(drawnColumns(table([column('id', { key: true }), to('a'), to('b'), to('c'), to('d'), to('e')])).columns).toHaveLength(6)
  })

  it('gives a table it does not know a place of its own', () => {
    const found = places(['users', 'new_one', 'new_two'])
    expect(found.users).toBeDefined()
    expect(found.new_one).not.toEqual(found.new_two)
    expect(found.new_one[1]).toBeGreaterThan(found.enrolment_history[1])
  })
})
