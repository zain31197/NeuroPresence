import * as RadixSelect from '@radix-ui/react-select'
import { Check, ChevronsUpDown } from 'lucide-react'
import type { ReactNode } from 'react'
import { cn } from '../../lib/cn'

export interface Option {
  value: string
  label: string
  icon?: ReactNode
}

interface Props {
  value: string
  onValueChange: (value: string) => void
  /** Options, optionally split into titled groups. */
  groups: { title?: string; options: Option[] }[]
  disabled?: boolean
  label: string
  className?: string
}

export function Select({ value, onValueChange, groups, disabled, label, className }: Props) {
  return (
    <RadixSelect.Root value={value} onValueChange={onValueChange} disabled={disabled}>
      <RadixSelect.Trigger
        aria-label={label}
        className={cn(
          'inline-flex h-9 min-w-[176px] items-center justify-between gap-2 rounded-control border border-line-strong',
          'bg-surface px-3 text-[13.5px] font-medium text-ink-900 shadow-control transition-colors hover:bg-ink-50',
          'disabled:pointer-events-none disabled:opacity-45 data-[state=open]:bg-ink-50',
          className,
        )}
      >
        <RadixSelect.Value />
        <RadixSelect.Icon>
          <ChevronsUpDown className="size-3.5 text-ink-400" />
        </RadixSelect.Icon>
      </RadixSelect.Trigger>
      <RadixSelect.Portal>
        <RadixSelect.Content
          position="popper"
          sideOffset={6}
          className="z-50 min-w-[var(--radix-select-trigger-width)] overflow-hidden rounded-panel border border-line bg-surface p-1 shadow-raised"
        >
          <RadixSelect.Viewport>
            {groups.map((group, index) => (
              <RadixSelect.Group key={group.title ?? index}>
                {group.title && (
                  <RadixSelect.Label className="label-caps px-2.5 pt-2 pb-1 text-ink-400">{group.title}</RadixSelect.Label>
                )}
                {group.options.map((option) => (
                  <RadixSelect.Item
                    key={option.value}
                    value={option.value}
                    className={cn(
                      'flex h-8 cursor-pointer items-center gap-2 rounded-[8px] pr-8 pl-2.5 text-[13.5px] text-ink-800 outline-none select-none',
                      'data-[highlighted]:bg-ink-100 data-[highlighted]:text-ink-950',
                    )}
                  >
                    {option.icon}
                    <RadixSelect.ItemText>{option.label}</RadixSelect.ItemText>
                    <RadixSelect.ItemIndicator className="absolute right-3">
                      <Check className="size-3.5 text-ink-950" strokeWidth={2.5} />
                    </RadixSelect.ItemIndicator>
                  </RadixSelect.Item>
                ))}
              </RadixSelect.Group>
            ))}
          </RadixSelect.Viewport>
        </RadixSelect.Content>
      </RadixSelect.Portal>
    </RadixSelect.Root>
  )
}
