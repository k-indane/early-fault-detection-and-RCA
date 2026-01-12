import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Live Fault Monitor',
  description: 'Simulates a real-time hybrid machine learning model to detect, classify, and assist root cause sourcing of manufacturing faults on Tennessee Eastman Process Simulated Dataset',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
