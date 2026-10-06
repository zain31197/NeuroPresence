import * as RadixSwitch from '@radix-ui/react-switch'
import { cn } from '../../lib/cn'

interface Props {
  checked: boolean
  onCheckedChange: (checked: boolean) => void
  disabled?: boolean
  /** Read out by screen readers; the visible label sits beside the switch. */
  label: string
  id?: string
}

export function Switch({ checked, onCheckedChange, disabled, label, id }: Props) {
  return (
    <RadixSwitch.Root
      id={id}
      checked={checked}
      onCheckedChange={onCheckedChange}
      disabled={disabled}
      aria-label={label}
      className={cn(
        'relative h-[22px] w-[38px] shrink-0 rounded-full transition-colors duration-200',
        'bg-ink-200 data-[state=checked]:bg-ink-950 disabled:opacity-45',
      )}
    >
      <RadixSwitch.Thumb
        className={cn(
          'block size-[18px] translate-x-[2px] rounded-full bg-white transition-transform duration-200',
          'shadow-[0_1px_2px_rgb(11_13_18/0.3)] data-[state=checked]:translate-x-[18px]',
        )}
      />
    </RadixSwitch.Root>
  )
}
