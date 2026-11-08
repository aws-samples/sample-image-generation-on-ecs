/** Public API client (gallery, generate, config). Shares interceptors with admin-api. */

import type { GalleryImage, GalleryResponse } from '@/types'

import { createAuthedAxios } from './http'

export const api = createAuthedAxios('/api')

export async function getHealth() {
  const { data } = await api.get('/health')
  return data
}

export async function getGallery(cursor?: string): Promise<GalleryResponse> {
  const { data } = await api.get('/gallery', { params: { cursor } })
  return data
}

export async function getImage(id: string): Promise<GalleryImage> {
  const { data } = await api.get(`/gallery/${id}`)
  return data
}

export async function deleteImage(id: string, hard = false): Promise<void> {
  await api.delete(`/gallery/${id}`, { params: { hard } })
}
