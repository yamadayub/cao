import Link from 'next/link'

export function Footer() {
  return (
    <footer className="bg-neutral-50 border-t border-neutral-100">
      <div className="max-w-6xl mx-auto px-6 py-16">
        <div className="flex flex-col items-center gap-10">
          {/* Logo */}
          <Link
            href="/"
            className="font-serif text-2xl font-medium text-primary-700 tracking-tight hover:text-primary-800 transition-colors duration-300"
          >
            Cao
          </Link>

          {/* Links */}
          <nav className="flex items-center gap-8 text-sm">
            <Link
              href="/terms"
              className="text-neutral-500 hover:text-primary-700 transition-colors duration-300"
            >
              利用規約
            </Link>
            <span className="text-neutral-300">|</span>
            <Link
              href="/privacy"
              className="text-neutral-500 hover:text-primary-700 transition-colors duration-300"
            >
              プライバシーポリシー
            </Link>
            <span className="text-neutral-300">|</span>
            <a
              href="mailto:support@cao.app"
              className="text-neutral-500 hover:text-primary-700 transition-colors duration-300"
            >
              お問い合わせ
            </a>
          </nav>

          {/* Copyright */}
          <p className="text-xs text-neutral-400 tracking-wide">
            &copy; {new Date().getFullYear()} Cao. All rights reserved.
          </p>
        </div>
      </div>
    </footer>
  )
}
