import { useInfiniteQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { getGallery, deleteImage } from '@/lib/api.ts'

export function useGallery() {
  const infiniteQuery = useInfiniteQuery({
    queryKey: ['gallery'],
    queryFn: ({ pageParam }) => getGallery(pageParam),
    getNextPageParam: (lastPage) => lastPage.next_cursor,
    initialPageParam: undefined as string | undefined,
    refetchInterval: 30_000,
  })

  const images = infiniteQuery.data?.pages.flatMap(page => page.images) ?? []

  return {
    images,
    isLoading: infiniteQuery.isLoading,
    isFetchingNextPage: infiniteQuery.isFetchingNextPage,
    hasNextPage: infiniteQuery.hasNextPage,
    fetchNextPage: infiniteQuery.fetchNextPage,
  }
}

export function useDeleteImage() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({ id, hard }: { id: string; hard?: boolean }) => 
      deleteImage(id, hard),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['gallery'] })
    },
  })
}
