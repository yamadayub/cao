import Link from 'next/link'

/**
 * プライバシーポリシーページ (SCR-008)
 *
 * 参照: functional-spec.md セクション 3.1
 */
export default function PrivacyPage() {
  return (
    <main className="min-h-screen bg-gray-50">
      {/* ヘッダー */}
      <header className="bg-white shadow-sm">
        <div className="max-w-4xl mx-auto px-4 py-4 flex items-center justify-between">
          <Link href="/" className="text-xl font-bold text-gray-900 hover:text-gray-700">
            Cao
          </Link>
          <nav className="flex gap-4">
            <Link
              href="/sign-in"
              className="text-sm text-gray-600 hover:text-gray-900 transition-colors"
            >
              ログイン
            </Link>
          </nav>
        </div>
      </header>

      {/* メインコンテンツ */}
      <div className="max-w-3xl mx-auto px-4 py-8">
        <article className="bg-white rounded-xl shadow-sm p-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-8">プライバシーポリシー</h1>

          <div className="prose prose-gray max-w-none">
            <p className="text-sm text-gray-500 mb-6">最終更新日: 2025年1月24日</p>

            <p className="text-gray-700 leading-relaxed mb-8">
              Cao（以下「当サービス」といいます）は、ユーザーの皆様のプライバシーを尊重し、個人情報の保護に努めています。本プライバシーポリシーでは、当サービスにおける個人情報の取り扱いについて説明します。
            </p>

            <section className="mb-8">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">1. 収集する情報</h2>
              <p className="text-gray-700 leading-relaxed mb-4">
                当サービスでは、以下の情報を収集することがあります。
              </p>

              <h3 className="text-lg font-medium text-gray-800 mb-2">1.1 ユーザーが提供する情報</h3>
              <ul className="list-disc pl-6 text-gray-700 space-y-2 mb-4">
                <li>メールアドレス（利用登録時）</li>
                <li>顔画像（シミュレーション利用時）</li>
              </ul>

              <h3 className="text-lg font-medium text-gray-800 mb-2">1.2 自動的に収集される情報</h3>
              <ul className="list-disc pl-6 text-gray-700 space-y-2">
                <li>アクセスログ（IPアドレス、ブラウザ情報等）</li>
                <li>Cookie情報</li>
                <li>利用状況に関する情報</li>
              </ul>
            </section>

            <section className="mb-8">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">2. 情報の利用目的</h2>
              <p className="text-gray-700 leading-relaxed mb-4">
                収集した情報は、以下の目的で利用します。
              </p>
              <ul className="list-disc pl-6 text-gray-700 space-y-2">
                <li>サービスの提供・運営</li>
                <li>ユーザー認証</li>
                <li>シミュレーション結果の保存・共有機能の提供</li>
                <li>サービスの改善・新機能の開発</li>
                <li>カスタマーサポートの提供</li>
                <li>不正利用の防止</li>
              </ul>
            </section>

            <section className="mb-8">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">3. 顔画像の取り扱い</h2>
              <p className="text-gray-700 leading-relaxed mb-4">
                当サービスでは、顔画像を以下のように取り扱います。
              </p>

              <h3 className="text-lg font-medium text-gray-800 mb-2">3.1 未登録ユーザーの場合</h3>
              <p className="text-gray-700 leading-relaxed mb-4">
                アップロードされた画像は、シミュレーション処理のためにのみ使用され、セッション終了後にサーバーから削除されます。サーバーに永続的に保存されることはありません。
              </p>

              <h3 className="text-lg font-medium text-gray-800 mb-2">3.2 登録ユーザーの場合</h3>
              <p className="text-gray-700 leading-relaxed">
                シミュレーション結果を保存した場合、画像はユーザーアカウントに紐づけて保存されます。ユーザーは、マイページからいつでも保存した画像を削除することができます。
              </p>
            </section>

            <section className="mb-8">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">4. 情報の共有</h2>
              <p className="text-gray-700 leading-relaxed mb-4">
                当サービスは、以下の場合を除き、ユーザーの個人情報を第三者に提供することはありません。
              </p>
              <ul className="list-disc pl-6 text-gray-700 space-y-2">
                <li>ユーザーの同意がある場合</li>
                <li>法令に基づく場合</li>
                <li>人の生命、身体または財産の保護のために必要がある場合</li>
                <li>サービス提供に必要な委託先（クラウドサービス事業者等）への共有</li>
              </ul>
            </section>

            <section className="mb-8">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">5. 情報の保護</h2>
              <p className="text-gray-700 leading-relaxed">
                当サービスは、収集した情報を適切に管理し、不正アクセス、紛失、破壊、改ざん、漏洩などを防止するため、合理的な安全対策を講じています。通信はTLS/SSLにより暗号化され、データは適切に保護されたサーバーに保存されます。
              </p>
            </section>

            <section className="mb-8">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">6. Cookieの使用</h2>
              <p className="text-gray-700 leading-relaxed">
                当サービスでは、ユーザー体験の向上、アクセス解析、認証状態の維持のためにCookieを使用しています。ブラウザの設定によりCookieを無効にすることができますが、一部の機能が利用できなくなる場合があります。
              </p>
            </section>

            <section className="mb-8">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">7. ユーザーの権利</h2>
              <p className="text-gray-700 leading-relaxed mb-4">
                ユーザーは、自身の個人情報について以下の権利を有します。
              </p>
              <ul className="list-disc pl-6 text-gray-700 space-y-2">
                <li>個人情報へのアクセス</li>
                <li>個人情報の訂正</li>
                <li>個人情報の削除</li>
                <li>アカウントの削除</li>
              </ul>
              <p className="text-gray-700 leading-relaxed mt-4">
                これらの権利を行使する場合は、お問い合わせ窓口までご連絡ください。
              </p>
            </section>

            <section className="mb-8">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">8. 同意状態の保存</h2>
              <p className="text-gray-700 leading-relaxed">
                当サービスでは、利用規約への同意状態をブラウザのローカルストレージに保存しています。この情報はユーザーのブラウザにのみ保存され、サーバーには送信されません。ブラウザのデータを消去した場合、再度同意が必要になります。
              </p>
            </section>

            <section className="mb-8">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">9. 未成年者の利用</h2>
              <p className="text-gray-700 leading-relaxed">
                当サービスは、18歳未満の方の利用を推奨していません。18歳未満の方が当サービスを利用する場合は、保護者の同意のもとでご利用ください。
              </p>
            </section>

            <section className="mb-8">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">10. プライバシーポリシーの変更</h2>
              <p className="text-gray-700 leading-relaxed">
                当サービスは、必要に応じて本プライバシーポリシーを変更することがあります。重要な変更がある場合は、サービス上でお知らせします。
              </p>
            </section>

            <section className="mb-8">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">11. お問い合わせ</h2>
              <p className="text-gray-700 leading-relaxed">
                プライバシーに関するご質問やご意見がございましたら、以下の窓口までお問い合わせください。
              </p>
              <p className="text-gray-700 leading-relaxed mt-4">
                メールアドレス: support@cao.app
              </p>
            </section>
          </div>

          {/* フッターリンク */}
          <div className="mt-8 pt-6 border-t border-gray-200">
            <div className="flex flex-wrap gap-4 text-sm">
              <Link href="/" className="text-blue-600 hover:text-blue-800">
                トップページに戻る
              </Link>
              <Link href="/terms" className="text-blue-600 hover:text-blue-800">
                利用規約
              </Link>
              <Link href="/simulate" className="text-blue-600 hover:text-blue-800">
                シミュレーションを試す
              </Link>
            </div>
          </div>
        </article>
      </div>

      {/* フッター */}
      <footer className="bg-white border-t border-gray-200 mt-12">
        <div className="max-w-4xl mx-auto px-4 py-6">
          <div className="flex flex-wrap justify-center gap-6 text-sm text-gray-600">
            <Link href="/terms" className="hover:text-gray-900">
              利用規約
            </Link>
            <Link href="/privacy" className="hover:text-gray-900">
              プライバシーポリシー
            </Link>
          </div>
          <p className="mt-4 text-center text-sm text-gray-500">
            &copy; 2025 Cao. All rights reserved.
          </p>
        </div>
      </footer>
    </main>
  )
}
