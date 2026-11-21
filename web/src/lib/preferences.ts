import { useCallback, useEffect, useState } from 'react'
import { api, type InputOption } from './api'

/** Which input to offer when nothing is remembered, or the remembered one is no longer there (a camera unplugged). */
export function fallbackInput(inputs: InputOption[]): string {
  return inputs.find((option) => option.kind === 'camera')?.id ?? inputs[0]?.id ?? 'camera:0'
}

/** The input a session was last started with, as the engine remembers it (the settings table), and a way to change it. */
export function usePreferredInput(inputs: InputOption[]): [string, (id: string) => void] {
  const [chosen, setChosen] = useState<string | null>(null)

  useEffect(() => {
    api.preferences().then(
      (saved) => saved.input && setChosen((current) => current ?? saved.input!),
      () => undefined, // an engine without preferences: the first camera is offered
    )
  }, [])

  const choose = useCallback((id: string) => {
    setChosen(id)
    api.setPreferences({ input: id }).catch(() => undefined)
  }, [])

  const known = chosen !== null && (inputs.length === 0 || inputs.some((option) => option.id === chosen))
  return [known ? chosen : fallbackInput(inputs), choose]
}
