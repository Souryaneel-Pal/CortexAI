/**
 * Client-side audio normalisation.
 *
 * Why this exists: the server can only decode what `libsndfile` understands
 * (WAV, FLAC, OGG, MP3, AIFF, CAF, …). It cannot read **M4A/AAC, MP4, or
 * WEBM** — which is exactly what macOS Voice Memos, iPhone recordings, and
 * in-browser `MediaRecorder` produce. `torchaudio` would cover those via
 * `torchcodec`, but that needs FFmpeg's shared libraries installed system-wide;
 * without them the request dies with an `OSError: Could not load ...
 * libtorchcodec_core*.dylib` and the user sees a 500.
 *
 * The browser already has decoders for every format it can play, so we decode
 * here instead of adding an FFmpeg dependency to every deployment. As a bonus
 * the upload shrinks a lot: the encoders want 16 kHz mono, and a 48 kHz stereo
 * source is 6x larger than it needs to be over the wire.
 *
 * Output is a 16 kHz mono 16-bit PCM WAV, which `libsndfile` reads natively.
 */

/** What the speech encoder consumes (`SpeechEmotionDataset`, wav2vec2). */
export const TARGET_SAMPLE_RATE = 16000

/** The model truncates/pads to 4 s, so sending more is wasted bytes. */
const MAX_DURATION_SECONDS = 4

export class AudioDecodeError extends Error {}

function encodeWav16BitMono(samples: Float32Array, sampleRate: number): ArrayBuffer {
  const buffer = new ArrayBuffer(44 + samples.length * 2)
  const view = new DataView(buffer)

  const writeAscii = (offset: number, text: string) => {
    for (let i = 0; i < text.length; i += 1) view.setUint8(offset + i, text.charCodeAt(i))
  }

  const dataBytes = samples.length * 2
  writeAscii(0, 'RIFF')
  view.setUint32(4, 36 + dataBytes, true)
  writeAscii(8, 'WAVE')
  writeAscii(12, 'fmt ')
  view.setUint32(16, 16, true) // PCM chunk size
  view.setUint16(20, 1, true) // format = PCM
  view.setUint16(22, 1, true) // channels = mono
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * 2, true) // byte rate (mono, 2 bytes/sample)
  view.setUint16(32, 2, true) // block align
  view.setUint16(34, 16, true) // bits per sample
  writeAscii(36, 'data')
  view.setUint32(40, dataBytes, true)

  let offset = 44
  for (let i = 0; i < samples.length; i += 1) {
    // Clamp before scaling: decoded float samples can exceed [-1, 1] slightly
    // on lossy sources, and wrapping there would produce audible clicks.
    const clamped = Math.max(-1, Math.min(1, samples[i]))
    view.setInt16(offset, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true)
    offset += 2
  }
  return buffer
}

function arrayBufferToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer)
  let binary = ''
  // Chunked: String.fromCharCode(...bytes) blows the argument limit on
  // anything longer than a second or two of audio.
  const CHUNK = 0x8000
  for (let i = 0; i < bytes.length; i += CHUNK) {
    binary += String.fromCharCode(...bytes.subarray(i, i + CHUNK))
  }
  return btoa(binary)
}

/**
 * Decode any browser-playable audio file and return base64 16 kHz mono WAV.
 *
 * Throws `AudioDecodeError` if the browser itself cannot decode the file (a
 * corrupt file, or a video container with no audio track), which the caller
 * turns into a message naming the file rather than a stack trace.
 */
export async function decodeToWav16kMonoBase64(file: File): Promise<string> {
  const arrayBuffer = await file.arrayBuffer()

  // `webkitAudioContext` keeps older Safari working.
  const Ctx: typeof AudioContext =
    window.AudioContext ?? (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext
  if (!Ctx) {
    throw new AudioDecodeError('This browser has no Web Audio support, so the clip cannot be converted.')
  }

  const decodeContext = new Ctx()
  let decoded: AudioBuffer
  try {
    // `decodeAudioData` is what gives us M4A/AAC/MP3/WEBM support for free.
    decoded = await decodeContext.decodeAudioData(arrayBuffer.slice(0))
  } catch (cause) {
    // Logged rather than passed as `Error(msg, { cause })`, which needs an
    // ES2022 lib target this project does not set.
    console.warn('decodeAudioData failed', cause)
    throw new AudioDecodeError(
      `"${file.name}" could not be decoded. It may be corrupt, or a video file with no audio track.`,
    )
  } finally {
    void decodeContext.close()
  }

  const frames = Math.min(decoded.duration, MAX_DURATION_SECONDS) * TARGET_SAMPLE_RATE
  // OfflineAudioContext does the sample-rate conversion, so we never hand-roll
  // resampling (and never alias a 48 kHz source down to 16 kHz).
  const offline = new OfflineAudioContext(1, Math.ceil(frames), TARGET_SAMPLE_RATE)
  const source = offline.createBufferSource()
  source.buffer = decoded
  source.connect(offline.destination)
  source.start()

  const rendered = await offline.startRendering()
  return arrayBufferToBase64(encodeWav16BitMono(rendered.getChannelData(0), TARGET_SAMPLE_RATE))
}
