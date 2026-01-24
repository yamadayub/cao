import { Metadata } from 'next'
import { LandingPageClient } from './LandingPageClient'

export const metadata: Metadata = {
  title: 'Cao - AI顔シミュレーション',
  description: 'AIを使った顔分析とモーフィングシミュレーション。理想の自分を、AIでシミュレーション。',
}

export default function Home() {
  return <LandingPageClient />
}
