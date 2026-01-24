import { clerkMiddleware, createRouteMatcher } from '@clerk/nextjs/server';
import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

// 認証が必要なルート
const isProtectedRoute = createRouteMatcher(['/mypage(.*)']);

// 公開ルート（認証不要）
const isPublicRoute = createRouteMatcher([
  '/',
  '/simulate(.*)',
  '/s/(.*)',
  '/sign-in(.*)',
  '/sign-up(.*)',
  '/terms(.*)',
  '/privacy(.*)',
]);

// Clerkキーが設定されているかチェック
const hasClerkKey = !!process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY;

// Clerkキーがない場合のシンプルなミドルウェア
function simpleMiddleware(req: NextRequest) {
  // /mypageへのアクセスは/sign-inにリダイレクト
  if (isProtectedRoute(req)) {
    return NextResponse.redirect(new URL('/sign-in', req.url));
  }
  return NextResponse.next();
}

// Clerkミドルウェア
const authMiddleware = clerkMiddleware(async (auth, req) => {
  if (isProtectedRoute(req)) {
    const session = await auth();
    if (!session.userId) {
      return session.redirectToSignIn();
    }
  }
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
