import { useState, useRef, useCallback, useEffect } from 'react'
import { Trash2, X, Download, ChevronLeft, ChevronRight, Maximize2, ExternalLink } from 'lucide-react'
import { useGallery, useDeleteImage } from '@/hooks/useGallery.ts'
import { cn } from '@/lib/utils.ts'
import type { GalleryImage } from '@/types'

function FullscreenView({ image, onClose }: { image: GalleryImage; onClose: () => void }) {
  useEffect(() => {
    // Capture-phase + stopPropagation so Escape closes fullscreen only,
    // without also reaching the gallery's own Escape handler (which would
    // close the modal underneath in the same keypress).
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.stopPropagation()
        onClose()
      }
    }
    window.addEventListener('keydown', handler, true)
    return () => window.removeEventListener('keydown', handler, true)
  }, [onClose])

  return (
    <div
      className="fixed inset-0 z-[60] flex items-center justify-center bg-ink/95 cursor-zoom-out"
      onClick={onClose}
    >
      <img src={image.url} alt={image.prompt} className="max-w-[95vw] max-h-[95vh] object-contain" />
      <button onClick={onClose} className="absolute top-4 right-4 p-2 bg-surface/90 rounded-full shadow-soft hover:shadow-card transition-shadow">
        <X className="w-5 h-5 text-ink" />
      </button>
    </div>
  )
}

function ImageModal({
  image,
  onClose,
  onDelete,
  onPrev,
  onNext,
  hasPrev,
  hasNext,
}: {
  image: GalleryImage
  onClose: () => void
  onDelete: () => void
  onPrev: () => void
  onNext: () => void
  hasPrev: boolean
  hasNext: boolean
}) {
  const [isFullscreen, setIsFullscreen] = useState(false)

  return (
    <>
      <div className="fixed inset-0 z-50 flex items-center justify-center p-6 bg-ink/80 backdrop-blur-md" onClick={onClose}>
        <div
          className="relative w-fit min-w-[320px] max-w-[min(90vw,64rem)] max-h-[90vh] flex flex-col card overflow-hidden"
          onClick={(e) => e.stopPropagation()}
        >
          <div className="flex items-center justify-end p-3 border-b border-border shrink-0">
            <button onClick={onClose} className="p-2 hover:bg-accent-light rounded-full transition-colors">
              <X className="w-5 h-5 text-muted" />
            </button>
          </div>

          <div className="overflow-y-auto">
            <div className="relative flex items-center justify-center p-6 bg-surface-raised">
              {hasPrev && (
                <button onClick={onPrev} className="absolute left-3 top-1/2 -translate-y-1/2 p-2 bg-surface border border-border rounded-full shadow-soft hover:shadow-card transition-shadow">
                  <ChevronLeft className="w-5 h-5 text-ink" />
                </button>
              )}
              <img
                src={image.url}
                alt={image.prompt}
                onClick={() => setIsFullscreen(true)}
                className="max-w-[min(90vw,60rem)] max-h-[55vh] object-contain rounded-2xl cursor-zoom-in"
              />
              {hasNext && (
                <button onClick={onNext} className="absolute right-3 top-1/2 -translate-y-1/2 p-2 bg-surface border border-border rounded-full shadow-soft hover:shadow-card transition-shadow">
                  <ChevronRight className="w-5 h-5 text-ink" />
                </button>
              )}
            </div>

            <div className="p-5 border-t border-border space-y-4">
              <p className="text-sm text-ink leading-relaxed">{image.prompt}</p>
              <table className="w-full text-xs">
                <tbody className="text-muted">
                  <tr className="border-b border-border/50">
                    <td className="py-1.5 font-semibold text-ink/70 w-24">Width</td>
                    <td className="py-1.5">{image.width || '?'}px</td>
                  </tr>
                  <tr className="border-b border-border/50">
                    <td className="py-1.5 font-semibold text-ink/70 w-24">Height</td>
                    <td className="py-1.5">{image.height || '?'}px</td>
                  </tr>
                  <tr className="border-b border-border/50">
                    <td className="py-1.5 font-semibold text-ink/70">Engine</td>
                    <td className="py-1.5 capitalize">{image.backend || 'neuron'}</td>
                  </tr>
                  <tr className="border-b border-border/50">
                    <td className="py-1.5 font-semibold text-ink/70">Steps</td>
                    <td className="py-1.5">{image.steps}</td>
                  </tr>
                  {image.seed != null && (
                    <tr className="border-b border-border/50">
                      <td className="py-1.5 font-semibold text-ink/70">Seed</td>
                      <td className="py-1.5 font-mono">{image.seed}</td>
                    </tr>
                  )}
                  <tr className="border-b border-border/50">
                    <td className="py-1.5 font-semibold text-ink/70">Denoise</td>
                    <td className="py-1.5">{image.denoise_viz != null ? (image.denoise_viz ? 'Yes' : 'No') : '—'}</td>
                  </tr>
                  {image.elapsed_s != null && (
                    <tr className="border-b border-border/50">
                      <td className="py-1.5 font-semibold text-ink/70">Time</td>
                      <td className="py-1.5">{image.elapsed_s}s</td>
                    </tr>
                  )}
                  <tr>
                    <td className="py-1.5 font-semibold text-ink/70">Created</td>
                    <td className="py-1.5">{new Date(image.created_at).toLocaleString(undefined, { month: 'short', day: 'numeric', year: 'numeric', hour: '2-digit', minute: '2-digit' })}</td>
                  </tr>
                </tbody>
              </table>
              <div className="flex gap-2 pt-1">
                <button onClick={() => setIsFullscreen(true)} className="btn-secondary text-xs py-1.5 px-3.5 inline-flex items-center gap-1.5">
                  <Maximize2 className="w-3.5 h-3.5" /> Fullscreen
                </button>
                <a href={image.url} target="_blank" rel="noopener noreferrer" className="btn-secondary text-xs py-1.5 px-3.5 inline-flex items-center gap-1.5">
                  <ExternalLink className="w-3.5 h-3.5" /> Open in new tab
                </a>
                <a href={image.url} download={`zimage-${image.id}.png`} className="btn-secondary text-xs py-1.5 px-3.5 inline-flex items-center gap-1.5">
                  <Download className="w-3.5 h-3.5" /> Download
                </a>
                <button onClick={onDelete} className="text-xs py-1.5 px-3.5 inline-flex items-center gap-1.5 text-red-500 bg-red-50 rounded-full hover:bg-red-100 transition-colors font-medium">
                  <Trash2 className="w-3.5 h-3.5" /> Delete
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
      {isFullscreen && <FullscreenView image={image} onClose={() => setIsFullscreen(false)} />}
    </>
  )
}

function DeleteConfirmModal({ isOpen, onClose, onConfirm, isDeleting }: {
  isOpen: boolean; onClose: () => void; onConfirm: (hard: boolean) => void; isDeleting: boolean
}) {
  if (!isOpen) return null
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-ink/60 backdrop-blur-sm">
      <div className="card p-6 max-w-sm w-full">
        <h3 className="text-lg font-bold mb-2">Delete this image?</h3>
        <p className="text-sm text-muted mb-6">This will mark the image as deleted.</p>
        <div className="flex gap-3">
          <button onClick={onClose} disabled={isDeleting} className="btn-secondary flex-1">Cancel</button>
          <button onClick={() => onConfirm(false)} disabled={isDeleting} className="btn-primary flex-1 !bg-red-500 !hover:bg-red-600">
            {isDeleting ? 'Deleting...' : 'Delete'}
          </button>
        </div>
      </div>
    </div>
  )
}

export default function Gallery() {
  const { images, isLoading, isFetchingNextPage, hasNextPage, fetchNextPage } = useGallery()
  const deleteMutation = useDeleteImage()
  const [selectedImage, setSelectedImage] = useState<GalleryImage | null>(null)
  const [imageToDelete, setImageToDelete] = useState<GalleryImage | null>(null)
  const observerRef = useRef<IntersectionObserver | null>(null)

  const lastImageRef = useCallback(
    (node: HTMLDivElement | null) => {
      if (isFetchingNextPage) return
      if (observerRef.current) observerRef.current.disconnect()
      observerRef.current = new IntersectionObserver((entries) => {
        if (entries[0].isIntersecting && hasNextPage) fetchNextPage()
      })
      if (node) observerRef.current.observe(node)
    },
    [isFetchingNextPage, hasNextPage, fetchNextPage]
  )

  const handleDelete = (hard: boolean) => {
    if (imageToDelete) {
      deleteMutation.mutate({ id: imageToDelete.id, hard })
      setImageToDelete(null)
      if (selectedImage?.id === imageToDelete.id) setSelectedImage(null)
    }
  }

  useEffect(() => {
    if (!selectedImage) return
    const handler = (e: KeyboardEvent) => {
      const idx = images.indexOf(selectedImage)
      if (e.key === 'ArrowLeft' && idx > 0) setSelectedImage(images[idx - 1])
      if (e.key === 'ArrowRight' && idx < images.length - 1) setSelectedImage(images[idx + 1])
      if (e.key === 'Escape') setSelectedImage(null)
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [selectedImage, images])

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-6 w-6 border-2 border-accent border-t-transparent" />
      </div>
    )
  }

  return (
    <div>
      <div className="flex items-baseline justify-between mb-8">
        <h2 className="font-display text-3xl font-extrabold tracking-tight">Gallery</h2>
        <span className="text-sm text-muted font-medium">{images.length} images</span>
      </div>

      {images.length === 0 ? (
        <div className="text-center py-20">
          <p className="text-xl text-muted font-semibold">Nothing here yet</p>
          <p className="text-sm text-muted/70 mt-2">Create your first image to see it here</p>
        </div>
      ) : (
        <>
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 grid-flow-row gap-4">
            {images.map((image, index) => (
              <div
                key={image.id}
                ref={index === images.length - 1 ? lastImageRef : undefined}
                className={cn(
                  "relative group aspect-square rounded-2xl overflow-hidden bg-surface-raised cursor-pointer",
                  "shadow-soft hover:shadow-card hover:scale-[1.02] transition-all duration-200",
                  image.deleted && "opacity-40"
                )}
                onClick={() => setSelectedImage(image)}
              >
                <img src={image.url} alt={image.prompt} className="w-full h-full object-cover" loading="lazy" />
                <div className="absolute inset-0 bg-gradient-to-t from-ink/70 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-200">
                  <div className="absolute bottom-0 left-0 right-0 p-3">
                    <p className="text-white text-xs font-medium leading-relaxed line-clamp-2">{image.prompt}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {hasNextPage && (
            <div className="mt-10 text-center">
              <button onClick={() => fetchNextPage()} disabled={isFetchingNextPage} className="btn-secondary">
                {isFetchingNextPage ? 'Loading...' : 'Load more'}
              </button>
            </div>
          )}
        </>
      )}

      {selectedImage && (
        <ImageModal
          image={selectedImage}
          onClose={() => setSelectedImage(null)}
          onDelete={() => { setImageToDelete(selectedImage); setSelectedImage(null) }}
          hasPrev={images.indexOf(selectedImage) > 0}
          hasNext={images.indexOf(selectedImage) < images.length - 1}
          onPrev={() => { const i = images.indexOf(selectedImage); if (i > 0) setSelectedImage(images[i - 1]) }}
          onNext={() => { const i = images.indexOf(selectedImage); if (i < images.length - 1) setSelectedImage(images[i + 1]) }}
        />
      )}

      <DeleteConfirmModal isOpen={!!imageToDelete} onClose={() => setImageToDelete(null)} onConfirm={handleDelete} isDeleting={deleteMutation.isPending} />
    </div>
  )
}
