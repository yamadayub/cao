import { clerkMiddleware, createRouteMatcher } from '@clerk/nextjs/server';
import createIntlMiddleware from 'next-intl/middleware';
import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';
import { routing } from '@/i18n/routing';

// next-intl middleware for locale detection and routing
const intlMiddleware = createIntlMiddleware(routing);

// Protected routes requiring authentication (with optional locale prefix)
const isProtectedRoute = createRouteMatcher([
  '/mypage(.*)',
  '/:locale/mypage(.*)',
]);

// Clerkキーが設定されているかチェック
const hasClerkKey = !!process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY;

// Clerkキーがない場合のシンプルなミドルウェア
function simpleMiddleware(req: NextRequest) {
  // /mypageへのアクセスは/sign-inにリダイレクト
  if (isProtectedRoute(req)) {
    return NextResponse.redirect(new URL('/sign-in', req.url));
  }
  // Run intl middleware for locale handling
  return intlMiddleware(req);
}

// Clerk + intl middleware
const authMiddleware = clerkMiddleware(async (auth, req) => {
  if (isProtectedRoute(req)) {
    const session = await auth();
    if (!session.userId) {
      return session.redirectToSignIn();
    }
  }

  // Run intl middleware for locale handling
  return intlMiddleware(req);
});

export default function middleware(req: NextRequest) {
  // Clerkキーがない場合はシンプルなミドルウェアを使用
  if (!hasClerkKey) {
    return simpleMiddleware(req);
  }
  // @ts-expect-error clerkMiddleware types don't match NextMiddleware exactly
  return authMiddleware(req, {});
}

export const config = {
  matcher: [
    // Skip Next.js internals and all static files
    '/((?!_next|[^?]*\\.(?:html?|css|js(?!on)|jpe?g|webp|png|gif|svg|ttf|woff2?|ico|csv|docx?|xlsx?|zip|webmanifest)).*)',
    // Always run for API routes
    '/(api|trpc)(.*)',
  ],
};
