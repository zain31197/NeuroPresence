import { Check, Copy, Download, KeyRound } from 'lucide-react'
import { motion } from 'motion/react'
import { useState } from 'react'
import { Button } from './ui/Button'
import { EASE_OUT } from './motion'

/**
 * A recovery code, shown once: it sets a new password when the old one is forgotten, since no email is ever sent.
 * Going on needs a tick that it has been kept.
 */
export function RecoveryCode({ code, email, onDone, done = 'Open the studio' }: { code: string; email: string; onDone: () => void; done?: string }) {
  const [kept, setKept] = useState(false)
  const [copied, setCopied] = useState(false)

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(code)
      setCopied(true)
      window.setTimeout(() => setCopied(false), 2000)
    } catch {
      /* the code stays on screen to copy by hand */
    }
  }
  const download = () => {
    const text = `NeuroPresence recovery code\n\nAccount: ${email}\nCode:    ${code}\n\nIt sets a new password if you forget yours. It works once, and you get a new one then.\nAnyone who has it can do the same: keep it somewhere safe.\n`
    const url = URL.createObjectURL(new Blob([text], { type: 'text/plain' }))
    const link = document.createElement('a')
    link.href = url
    link.download = 'neuropresence-recovery-code.txt'
    link.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div>
      <span className="grid size-12 place-items-center rounded-[15px] bg-accent-50 text-accent-700">
        <KeyRound className="size-5" strokeWidth={1.9} />
      </span>
      <h1 className="display mt-5 text-[26px] leading-tight font-semibold text-ink-950">Keep this recovery code</h1>
      <p className="mt-2 text-[14px] leading-relaxed text-ink-600">
        Your account exists only on this computer, so no email can reset your password. This code can. It is shown once and works once.
      </p>

      <div className="mt-6 rounded-card border border-line bg-ink-50 px-3 py-5 sm:px-5">
        <p className="flex flex-wrap justify-center gap-x-1 gap-y-2 font-mono text-[15px] font-semibold text-ink-950 sm:gap-x-1.5 sm:text-[17px]" aria-label={`Recovery code ${code}`}>
          {code.split('-').map((group, index) => (
            <motion.span
              key={`${group}-${index}`}
              className="rounded-[9px] border border-line bg-surface px-1.5 py-1.5 tracking-wide shadow-control sm:px-2.5"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, ease: EASE_OUT, delay: 0.1 + index * 0.07 }}
            >
              {group}
            </motion.span>
          ))}
        </p>
        <div className="mt-4 flex justify-center gap-2">
          <Button size="sm" icon={copied ? <Check className="size-3.5" /> : <Copy className="size-3.5" />} onClick={copy}>
            {copied ? 'Copied' : 'Copy'}
          </Button>
          <Button size="sm" icon={<Download className="size-3.5" />} onClick={download}>
            Save as a file
          </Button>
        </div>
      </div>

      <label className="mt-5 flex cursor-pointer items-start gap-2.5 text-[13.5px] text-ink-700">
        <input type="checkbox" checked={kept} onChange={(event) => setKept(event.target.checked)} className="mt-0.5 size-4 accent-[var(--color-ink-950)]" />
        I have kept the code somewhere safe.
      </label>
      <Button variant="primary" size="lg" className="mt-5 w-full" disabled={!kept} onClick={onDone}>
        {done}
      </Button>
    </div>
  )
}
