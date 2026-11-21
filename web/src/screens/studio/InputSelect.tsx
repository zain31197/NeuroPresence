import { Clapperboard, Mic, Video } from 'lucide-react'
import { Select } from '../../components/ui/Select'
import type { InputOption } from '../../lib/api'

/** The inputs a session can be driven by, in three groups: cameras, the voice with the camera off, recorded clips. */
export function InputSelect({ value, onValueChange, inputs, disabled }: { value: string; onValueChange: (id: string) => void; inputs: InputOption[]; disabled?: boolean }) {
  return (
    <Select
      label="Input"
      value={value}
      onValueChange={onValueChange}
      disabled={disabled || inputs.length === 0}
      groups={[
        {
          title: 'Cameras',
          options: inputs
            .filter((option) => option.kind === 'camera')
            .map((option) => ({ value: option.id, label: option.label, icon: <Video className="size-3.5 text-ink-500" /> })),
        },
        {
          title: 'Camera off',
          options: inputs
            .filter((option) => option.kind === 'voice')
            .map((option) => ({ value: option.id, label: option.label, icon: <Mic className="size-3.5 text-ink-500" /> })),
        },
        {
          title: 'Recorded clips',
          options: inputs
            .filter((option) => option.kind === 'sample')
            .map((option) => ({ value: option.id, label: option.label, icon: <Clapperboard className="size-3.5 text-ink-500" /> })),
        },
      ].filter((group) => group.options.length > 0)}
    />
  )
}
