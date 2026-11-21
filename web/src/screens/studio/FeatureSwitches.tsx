import { Activity, Cast, Crosshair, Fingerprint, Gauge, Move3d, ScanFace, Sparkles, Timer, type LucideIcon } from 'lucide-react'
import { motion } from 'motion/react'
import { useEffect, useState } from 'react'
import { Switch } from '../../components/ui/Switch'
import { useToast } from '../../components/ui/Toast'
import { Tooltip } from '../../components/ui/Tooltip'
import { api, ApiError } from '../../lib/api'
import { cn } from '../../lib/cn'
import { useStudio } from '../../lib/studio'

const ICONS: Record<string, LucideIcon> = {
  reenactment: Sparkles,
  steady_crop: Crosshair,
  steady_keypoints: Activity,
  even_motion: Gauge,
  even_delivery: Cast,
  natural_range: Move3d,
  identity_guard: Fingerprint,
  delay_watchdog: Timer,
  tracking_overlay: ScanFace,
}

/**
 * A switch for every feature the engine registers, so each can be tested live, on and off. Each is a tile
 * that says what the feature does; the full description is on the tile's title, and in its tooltip.
 */
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
    <ul className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4" aria-label="Features">
      {features.map((feature) => {
        const on = chosen[feature.key] ?? feature.enabled
        const Icon = ICONS[feature.key] ?? Sparkles
        return (
          <li key={feature.key}>
            <motion.label
              htmlFor={`feature-${feature.key}`}
              whileTap={{ scale: 0.985 }}
              className={cn(
                'flex h-full cursor-pointer gap-3 rounded-panel border p-3.5 transition-[background-color,border-color,box-shadow] duration-300',
                on ? 'border-accent-200 bg-accent-50/60 shadow-[0_6px_18px_-12px_rgb(80_98_240/0.6)]' : 'border-line bg-surface hover:bg-ink-50',
              )}
            >
              <span
                className={cn(
                  'grid size-8 shrink-0 place-items-center rounded-[10px] transition-colors duration-300',
                  on ? 'bg-accent-500 text-white' : 'bg-ink-100 text-ink-500',
                )}
              >
                <Icon className="size-4" strokeWidth={2} />
              </span>
              <span className="min-w-0 flex-1">
                <span className="flex items-center justify-between gap-2">
                  <Tooltip content={feature.description} side="top">
                    <span className="text-[13.5px] font-semibold text-ink-950">{feature.label}</span>
                  </Tooltip>
                  <Switch id={`feature-${feature.key}`} label={feature.label} checked={on} onCheckedChange={(enabled) => void toggle(feature.key, enabled)} />
                </span>
                <span className="mt-1 line-clamp-2 text-[12px] leading-snug text-ink-500" title={feature.description}>
                  {feature.description}
                </span>
              </span>
            </motion.label>
          </li>
        )
      })}
    </ul>
  )
}
