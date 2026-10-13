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
        'group relative h-[22px] w-[38px] shrink-0 rounded-full transition-[background-color,box-shadow] duration-300 ease-out-soft',
        'bg-ink-200 hover:bg-ink-300 data-[state=checked]:bg-ink-950 data-[state=checked]:hover:bg-ink-800 disabled:opacity-45',
        'shadow-[inset_0_1px_2px_rgb(11_13_18/0.12)]',
      )}
    >
      <RadixSwitch.Thumb
        className={cn(
          'block size-[18px] translate-x-[2px] rounded-full bg-white transition-[transform,width] duration-300 ease-out-soft',
          'shadow-[0_1px_2px_rgb(11_13_18/0.3),0_2px_6px_-1px_rgb(11_13_18/0.25)] data-[state=checked]:translate-x-[18px]',
          'group-active:w-[21px] data-[state=checked]:group-active:translate-x-[15px]',
        )}
      />
    </RadixSwitch.Root>
  )
}
