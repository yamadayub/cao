import '@testing-library/react';
import '@testing-library/jest-dom/vitest';
import { vi } from 'vitest';

// Mock @clerk/nextjs globally
vi.mock('@clerk/nextjs', () => ({
  useAuth: () => ({
    isLoaded: true,
    isSignedIn: false,
    userId: null,
  }),
  useClerk: () => ({
    loaded: true,
  }),
  useUser: () => ({
    isLoaded: true,
    isSignedIn: false,
    user: null,
  }),
  UserButton: () => null,
  SignInButton: ({ children }: { children: React.ReactNode }) => children,
  SignUpButton: ({ children }: { children: React.ReactNode }) => children,
  ClerkProvider: ({ children }: { children: React.ReactNode }) => children,
}));
