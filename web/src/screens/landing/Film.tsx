import { Play } from 'lucide-react'
import { AnimatePresence, motion } from 'motion/react'
import { useRef, useState } from 'react'
import { GlowBorder } from '../../components/fx'
import { Reveal } from '../../components/motion'

// The launch video lives in web/public/media. Video files are kept out of git (.gitignore: *.mp4), so a fresh
// copy of the project may not have it: then the section does not show at all.
const VIDEO = '/media/neuropresence-launch.mp4'
const STILL = '/media/neuropresence-launch-poster.jpg'

/**
 * The 60-second launch video, in a frame with a moving edge of light. It shows its end card until it is
 * played, and plays only when asked, with sound and the browser's own controls.
 */
export function Film({ children }: { children: (video: React.ReactNode) => React.ReactNode }) {
  const video = useRef<HTMLVideoElement>(null)
  const [available, setAvailable] = useState(true)
  const [playing, setPlaying] = useState(false)

  if (!available) return null

  const play = () => {
    setPlaying(true)
    void video.current?.play().catch(() => setPlaying(false))
  }

  return children(
    <Reveal>
      <GlowBorder className="shadow-hero" radius={22}>
        <div className="relative aspect-video overflow-hidden rounded-[22px] bg-ink-950">
          <video
            ref={video}
            className="size-full object-cover"
            src={VIDEO}
            poster={STILL}
            preload="metadata"
            playsInline
            controls={playing}
            onError={() => setAvailable(false)}
            onPlay={() => setPlaying(true)}
            onEnded={() => setPlaying(false)}
          />
          <AnimatePresence>
            {!playing && (
              <motion.button
                type="button"
                onClick={play}
                aria-label="Play the 60-second video"
                className="group absolute inset-0 cursor-pointer bg-[linear-gradient(160deg,rgb(11_13_18/0.10),transparent_35%)] transition-colors"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0, transition: { duration: 0.2 } }}
              >
                {/* The still frame is the video's end card, with its text in the middle and at the bottom: the
                    control sits in the empty top corner, and the whole picture can be clicked. */}
                <span className="absolute top-5 left-5 inline-flex items-center gap-3 rounded-full bg-surface/95 py-2 pr-5 pl-2 shadow-raised backdrop-blur transition-transform duration-300 group-hover:scale-[1.03]">
                  <span className="relative grid size-11 place-items-center">
                    {[0, 1].map((ring) => (
                      <motion.span
                        key={ring}
                        aria-hidden="true"
                        className="absolute inset-0 rounded-full border-2 border-accent-400"
                        animate={{ scale: [1, 1.6], opacity: [0.6, 0] }}
                        transition={{ duration: 2.2, ease: 'easeOut', repeat: Infinity, delay: ring * 1.1 }}
                      />
                    ))}
                    <span className="relative grid size-11 place-items-center rounded-full bg-accent-500 text-white shadow-[0_10px_24px_-8px_rgb(80_98_240/0.7)]">
                      <Play className="ml-0.5 size-5 fill-current" strokeWidth={0} />
                    </span>
                  </span>
                  <span className="text-left">
                    <span className="block text-[14.5px] font-semibold text-ink-950">Watch the tour</span>
                    <span className="block font-mono text-[11.5px] text-ink-500">1:00 · sound on</span>
                  </span>
                </span>
              </motion.button>
            )}
          </AnimatePresence>
        </div>
      </GlowBorder>
    </Reveal>,
  )
}
