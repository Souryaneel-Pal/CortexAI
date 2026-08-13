/**
 * Integrated-camera capture for the facial modality.
 *
 * Opens the device camera, shows a live preview, and hands back a still frame
 * as a `File` so the caller treats it exactly like an uploaded photo — the
 * assessment pipeline needs no special case for camera input.
 *
 * Two things this is careful about, because both are easy to get wrong and
 * user-visible:
 *
 *   1. **The stream is always stopped.** A `MediaStream` whose tracks are not
 *      explicitly stopped leaves the camera (and its indicator light) running
 *      after the preview closes, which reads as the app spying. Tracks are
 *      stopped on capture, on cancel, and on unmount.
 *   2. **Permission and availability failures are explained.** `getUserMedia`
 *      rejects with a handful of distinct errors that mean very different
 *      things to a user — denied vs. no camera vs. in use by another app vs.
 *      an insecure origin — so each gets its own message instead of a generic
 *      "could not start camera".
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import { MaterialIcon } from './MaterialIcon'

interface CameraCaptureProps {
  /** Receives the captured still as a PNG File. */
  onCapture: (file: File) => void
  onClose: () => void
}

/** Square capture: the facial encoder consumes a centre-cropped 48x48. */
const CAPTURE_SIZE = 480

function describeCameraError(error: unknown): string {
  // Browsers disagree on the exact name, so match on both name and message.
  const name = (error as { name?: string })?.name ?? ''
  if (!window.isSecureContext) {
    return 'The camera needs a secure context. Use http://localhost or serve the app over HTTPS.'
  }
  switch (name) {
    case 'NotAllowedError':
    case 'SecurityError':
      return 'Camera access was blocked. Allow it for this site in your browser settings, then try again.'
    case 'NotFoundError':
    case 'OverconstrainedError':
      return 'No camera was found on this device.'
    case 'NotReadableError':
      return 'The camera is already in use by another application. Close it and try again.'
    default:
      return error instanceof Error ? error.message : 'Could not start the camera.'
  }
}

export function CameraCapture({ onCapture, onClose }: CameraCaptureProps) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [ready, setReady] = useState(false)

  const stopStream = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
  }, [])

  useEffect(() => {
    let cancelled = false

    async function start() {
      if (!navigator.mediaDevices?.getUserMedia) {
        setError('This browser does not support camera capture.')
        return
      }
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        })
        if (cancelled) {
          // The component unmounted while the permission prompt was open --
          // release the camera immediately rather than leaking the stream.
          stream.getTracks().forEach((track) => track.stop())
          return
        }
        streamRef.current = stream
        if (videoRef.current) {
          videoRef.current.srcObject = stream
          await videoRef.current.play().catch(() => undefined)
        }
        setReady(true)
      } catch (startError) {
        if (!cancelled) setError(describeCameraError(startError))
      }
    }

    void start()
    return () => {
      cancelled = true
      stopStream()
    }
  }, [stopStream])

  function capture() {
    const video = videoRef.current
    if (!video || !video.videoWidth) return

    // Centre-crop to a square before scaling: the encoder wants a face, and
    // squashing a 16:9 frame to square would distort every facial proportion
    // the model was trained on.
    const side = Math.min(video.videoWidth, video.videoHeight)
    const sx = (video.videoWidth - side) / 2
    const sy = (video.videoHeight - side) / 2

    const canvas = document.createElement('canvas')
    canvas.width = CAPTURE_SIZE
    canvas.height = CAPTURE_SIZE
    const context = canvas.getContext('2d')
    if (!context) {
      setError('Could not read a frame from the camera.')
      return
    }
    context.drawImage(video, sx, sy, side, side, 0, 0, CAPTURE_SIZE, CAPTURE_SIZE)

    canvas.toBlob((blob) => {
      if (!blob) {
        setError('Could not encode the captured frame.')
        return
      }
      stopStream()
      onCapture(new File([blob], `camera-capture-${Date.now()}.png`, { type: 'image/png' }))
    }, 'image/png')
  }

  function cancel() {
    stopStream()
    onClose()
  }

  if (error) {
    return (
      <div className="flex flex-col gap-md rounded-lg border border-error/40 bg-error/10 p-md" role="alert">
        <div className="flex items-start gap-xs">
          <MaterialIcon name="videocam_off" className="mt-[2px] text-[18px] text-error" />
          <span className="flex-1 font-label-sm text-label-sm text-error">{error}</span>
        </div>
        <button
          type="button"
          onClick={cancel}
          className="self-start rounded-md border border-outline-variant bg-surface-container-highest px-md py-sm font-label-md text-label-md text-on-surface transition-all hover:bg-primary hover:text-on-primary"
        >
          Close
        </button>
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-md">
      <div className="relative overflow-hidden rounded-lg border border-outline-variant bg-surface-container-low">
        <video
          ref={videoRef}
          playsInline
          muted
          // Mirrored so it behaves like a mirror, which is what people expect
          // of a self-view. The capture canvas draws the unmirrored frame.
          className="h-[220px] w-full scale-x-[-1] object-cover"
        />
        {!ready && (
          <div className="absolute inset-0 flex items-center justify-center gap-xs bg-surface-container-low">
            <MaterialIcon name="progress_activity" className="animate-spin text-[18px] text-primary" />
            <span className="font-label-sm text-label-sm text-on-surface-variant">Starting camera…</span>
          </div>
        )}
      </div>
      <div className="flex items-center gap-sm">
        <button
          type="button"
          onClick={capture}
          disabled={!ready}
          className="flex flex-1 items-center justify-center gap-xs rounded-md bg-primary px-md py-sm font-label-md text-label-md text-on-primary transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
        >
          <MaterialIcon name="photo_camera" className="text-[18px]" />
          Capture Photo
        </button>
        <button
          type="button"
          onClick={cancel}
          className="rounded-md border border-outline-variant bg-surface-container-highest px-md py-sm font-label-md text-label-md text-on-surface transition-all hover:bg-surface-container"
        >
          Cancel
        </button>
      </div>
    </div>
  )
}
