interface Props {
  value: number | null
  /** The limit the value is measured against; the full width of the track. */
  limit: number
  /** What the meter shows, for screen readers. */
  label: string
  valueText: string
}

/** One ratio against a limit. The track is a lighter step of the fill, so the whole bar reads as one thing. */
export function Meter({ value, limit, label, valueText }: Props) {
  const share = value === null ? 0 : Math.min(1, Math.max(0, value / limit))
  return (
    <div
      role="meter"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={limit}
      aria-valuenow={value ?? 0}
      aria-valuetext={valueText}
      className="h-2 overflow-hidden rounded-full bg-accent-100"
    >
      <div
        className="h-full rounded-full bg-accent-600 transition-[width] duration-500 ease-out"
        style={{ width: `${share * 100}%`, minWidth: value ? 6 : 0 }}
      />
    </div>
  )
}
