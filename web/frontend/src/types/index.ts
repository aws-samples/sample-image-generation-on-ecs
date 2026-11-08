// Frame types matching backend
export const FrameType = {
  INTERMEDIATE: 0x00,
  FINAL: 0x01,
  ERROR: 0x02,
  COMPLETE: 0x03,
  METADATA: 0x04,
} as const

export interface GenerationParams {
  prompt: string
  steps?: number
  seed?: number
  width?: number
  height?: number
  denoise_viz?: boolean
  backend?: string
}

export interface Frame {
  frame_type: number
  step_index: number
  payload: Uint8Array
  metadata?: Record<string, unknown>
}

export interface GalleryImage {
  id: string
  prompt: string
  width?: number
  height?: number
  seed?: number
  steps: number
  backend?: string
  denoise_viz?: boolean
  elapsed_s?: number
  created_at: string
  deleted: boolean
  url: string
  frames?: { step: number; url: string }[]
}

export interface GalleryResponse {
  images: GalleryImage[]
  next_cursor: string | null
  count: number
}
