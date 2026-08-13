interface MaterialIconProps {
  name: string
  className?: string
  filled?: boolean
  style?: React.CSSProperties
}

/** Thin wrapper around the Material Symbols Outlined icon font used across the approved designs. */
export function MaterialIcon({ name, className = '', filled = false, style }: MaterialIconProps) {
  return (
    <span
      className={`material-symbols-outlined ${filled ? 'fill' : ''} ${className}`.trim()}
      style={style}
      aria-hidden="true"
    >
      {name}
    </span>
  )
}
