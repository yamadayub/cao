import { Metadata } from 'next'
import { LandingPageClient } from './LandingPageClient'

export const metadata: Metadata = {
  title: 'Cao - AI顔シミュレーション',
  description: '理想の顔写真とあなたの顔写真を組み合わせて、顔全体・パーツ別の変化をシミュレーション。AIによる高品質なFace Swapで自然な仕上がり。',
}

export default function Home() {
  return <LandingPageClient />
}
