import Link from 'next/link'

export function Footer() {
  return (
    <footer className="bg-gray-50 border-t border-gray-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex flex-col md:flex-row items-center justify-between gap-4">
          {/* Logo */}
          <Link href="/" className="text-xl font-bold text-blue-600">
            Cao
          </Link>

          {/* Links */}
          <nav className="flex items-center gap-6 text-sm text-gray-600">
            <Link href="/terms" className="hover:text-gray-900 transition-colors">
              利用規約
            </Link>
            <Link href="/privacy" className="hover:text-gray-900 transition-colors">
              プライバシーポリシー
            </Link>
            <a
              href="mailto:support@cao.app"
              className="hover:text-gray-900 transition-colors"
            >
              お問い合わせ
            </a>
          </nav>

          {/* Copyright */}
          <p className="text-sm text-gray-500">
            &copy; {new Date().getFullYear()} Cao. All rights reserved.
          </p>
        </div>
      </div>
    </footer>
  )
}
