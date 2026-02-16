import type { ReactNode } from 'react';

type Props = {
  children: ReactNode;
};

// Root layout is a passthrough - locale-specific layout handles html/body/providers
export default function RootLayout({ children }: Props) {
  return children;
}
