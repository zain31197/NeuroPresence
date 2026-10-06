import { useEffect, useState } from 'react'
import { Switch } from '../../components/ui/Switch'
import { useToast } from '../../components/ui/Toast'
import { Tooltip } from '../../components/ui/Tooltip'
import { api, ApiError } from '../../lib/api'
import { useStudio } from '../../lib/studio'

/** A switch for every feature the engine registers, so each can be tested live, on and off. */
export function FeatureSwitches() {
  const { status } = useStudio()
  const toast = useToast()
  // What the person just chose, shown at once; the engine's own answer replaces it a moment later.
  const [chosen, setChosen] = useState<Record<string, boolean>>({})
  const features = status?.features ?? []

  useEffect(() => {
    setChosen((pending) => {
      const still = Object.fromEntries(
        Object.entries(pending).filter(([key, enabled]) => features.find((feature) => feature.key === key)?.enabled !== enabled),
      )
      return Object.keys(still).length === Object.keys(pending).length ? pending : still
    })
  }, [features])

  const toggle = async (key: string, enabled: boolean) => {
    setChosen((pending) => ({ ...pending, [key]: enabled }))
    try {
      await api.setFeature(key, enabled)
    } catch (error) {
      setChosen(({ [key]: _dropped, ...rest }) => rest)
      toast.problem(error instanceof ApiError ? error.message : 'The setting could not be changed.')
    }
  }

  return (
    <ul className="flex flex-wrap items-center gap-x-5 gap-y-2" aria-label="Features">
      {features.map((feature) => (
        <li key={feature.key} className="flex items-center gap-2.5">
          <Switch
            id={`feature-${feature.key}`}
            label={feature.label}
            checked={chosen[feature.key] ?? feature.enabled}
            onCheckedChange={(enabled) => void toggle(feature.key, enabled)}
          />
          <Tooltip content={feature.description} side="bottom">
            <label htmlFor={`feature-${feature.key}`} className="cursor-pointer text-[13.5px] font-medium text-ink-900">
              {feature.label}
            </label>
          </Tooltip>
        </li>
      ))}
    </ul>
  )
}
