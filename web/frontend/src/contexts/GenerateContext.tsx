import { createContext, useContext, useState, type ReactNode } from 'react'

interface GenerateState {
  prompt: string
  setPrompt: (v: string) => void
  seedMode: 'random' | 'fixed'
  setSeedMode: (v: 'random' | 'fixed') => void
  seed: string
  setSeed: (v: string) => void
  backend: string
  setBackend: (v: string) => void
  width: number
  setWidth: (v: number) => void
  height: number
  setHeight: (v: number) => void
  denoiseViz: boolean
  setDenoiseViz: (v: boolean) => void
}

const GenerateContext = createContext<GenerateState | null>(null)

export function useGenerateState() {
  const ctx = useContext(GenerateContext)
  if (!ctx) throw new Error('useGenerateState must be used within GenerateProvider')
  return ctx
}

export function GenerateProvider({ children }: { children: ReactNode }) {
  const [prompt, setPrompt] = useState('')
  const [seedMode, setSeedMode] = useState<'random' | 'fixed'>('random')
  const [seed, setSeed] = useState('')
  const [backend, setBackend] = useState('gpu')
  const [width, setWidth] = useState(1024)
  const [height, setHeight] = useState(1024)
  const [denoiseViz, setDenoiseViz] = useState(true)

  return (
    <GenerateContext.Provider
      value={{
        prompt, setPrompt,
        seedMode, setSeedMode,
        seed, setSeed,
        backend, setBackend,
        width, setWidth,
        height, setHeight,
        denoiseViz, setDenoiseViz,
      }}
    >
      {children}
    </GenerateContext.Provider>
  )
}
