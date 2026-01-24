import Link from 'next/link'

/**
 * 利用規約ページ (SCR-007)
 *
 * 参照: functional-spec.md セクション 3.1
 * 参照: business-spec.md セクション 4.5
 */
export default function TermsPage() {
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
          <h1 className="text-3xl font-bold text-gray-900 mb-8">利用規約</h1>

          <div className="prose prose-gray max-w-none">
            <p className="text-sm text-gray-500 mb-6">最終更新日: 2025年1月24日</p>

            <section className="mb-8">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">第1条（適用）</h2>
              <p className="text-gray-700 leading-relaxed">
                本利用規約（以下「本規約」といいます）は、Cao（以下「当サービス」といいます）が提供するAI顔シミュレーションサービスの利用条件を定めるものです。ユーザーの皆様には、本規約に同意のうえ、当サービスをご利用いただきます。
              </p>
            </section>

            <section className="mb-8">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">第2条（定義）</h2>
              <ul className="list-disc pl-6 text-gray-700 space-y-2">
                <li>「ユーザー」とは、当サービスを利用するすべての方をいいます。</li>
                <li>「コンテンツ」とは、ユーザーがアップロードした画像、生成されたシミュレーション結果等をいいます。</li>
                <li>「本サービス」とは、当サービスが提供するAI顔シミュレーション機能及び関連サービスをいいます。</li>
              </ul>
            </section>

            <section className="mb-8">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">第3条（サービスの内容）</h2>
              <p className="text-gray-700 leading-relaxed mb-4">
                当サービスは、ユーザーがアップロードした顔画像を元に、AIを用いた顔のシミュレーション画像を生成するサービスを提供します。
              </p>
              <p className="text-gray-700 leading-relaxed">
                シミュレーション結果は参考情報としてご利用ください。実際の美容施術の効果を保証するものではありません。
              </p>
            </section>

            <section className="mb-8">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">第4条（利用登録）</h2>
              <ol className="list-decimal pl-6 text-gray-700 space-y-2">
                <li>当サービスの一部機能は、利用登録なしでご利用いただけます。</li>
                <li>シミュレーション結果の保存・共有機能を利用するには、利用登録が必要です。</li>
                <li>利用登録の際には、正確な情報を入力してください。</li>
              </ol>
            </section>

            <section className="mb-8">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">第5条（アップロード画像について）</h2>
              <ol className="list-decimal pl-6 text-gray-700 space-y-2">
                <li>ユーザーは、自身が権利を有する画像、または使用許諾を得た画像のみをアップロードしてください。</li>
                <li>他者の肖像権を侵害する画像のアップロードは禁止します。</li>
                <li>未登録ユーザーがアップロードした画像は、セッション終了後にサーバーから削除されます。</li>
                <li>登録ユーザーがアップロードした画像は、ユーザーアカウントに紐づけて保存されます。</li>
              </ol>
            </section>

            <section className="mb-8">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">第6条（禁止事項）</h2>
              <p className="text-gray-700 leading-relaxed mb-4">
                ユーザーは、当サービスの利用にあたり、以下の行為を行ってはなりません。
              </p>
              <ul className="list-disc pl-6 text-gray-700 space-y-2">
                <li>法令または公序良俗に反する行為</li>
                <li>他者の権利（肖像権、プライバシー権、著作権等）を侵害する行為</li>
                <li>当サービスの運営を妨害する行為</li>
                <li>不正アクセスやシステムへの攻撃行為</li>
                <li>当サービスの逆アセンブル、逆コンパイル、リバースエンジニアリング</li>
                <li>営利目的での無断利用</li>
                <li>その他、当サービスが不適切と判断する行為</li>
              </ul>
            </section>

            <section className="mb-8">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">第7条（免責事項）</h2>
              <ol className="list-decimal pl-6 text-gray-700 space-y-2">
                <li>当サービスは、シミュレーション結果の正確性・完全性を保証するものではありません。</li>
                <li>当サービスの利用により生じた損害について、当サービスは一切の責任を負いません。</li>
                <li>当サービスは、予告なくサービス内容の変更、中断、終了を行う場合があります。</li>
              </ol>
            </section>

            <section className="mb-8">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">第8条（知的財産権）</h2>
              <p className="text-gray-700 leading-relaxed">
                当サービスに関する著作権、商標権その他の知的財産権は、当サービスまたは正当な権利者に帰属します。ユーザーがアップロードした画像の著作権は、ユーザーに帰属します。
              </p>
            </section>

            <section className="mb-8">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">第9条（個人情報の取り扱い）</h2>
              <p className="text-gray-700 leading-relaxed">
                当サービスにおける個人情報の取り扱いについては、
                <Link href="/privacy" className="text-blue-600 hover:text-blue-800 underline">
                  プライバシーポリシー
                </Link>
                をご確認ください。
              </p>
            </section>

            <section className="mb-8">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">第10条（規約の変更）</h2>
              <p className="text-gray-700 leading-relaxed">
                当サービスは、必要に応じて本規約を変更することがあります。変更後の利用規約は、当サービス上に掲示した時点で効力を生じるものとします。
              </p>
            </section>

            <section className="mb-8">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">第11条（準拠法・管轄裁判所）</h2>
              <p className="text-gray-700 leading-relaxed">
                本規約の解釈にあたっては、日本法を準拠法とします。当サービスに関して紛争が生じた場合には、東京地方裁判所を第一審の専属的合意管轄裁判所とします。
              </p>
            </section>
          </div>

          {/* フッターリンク */}
          <div className="mt-8 pt-6 border-t border-gray-200">
            <div className="flex flex-wrap gap-4 text-sm">
              <Link href="/" className="text-blue-600 hover:text-blue-800">
                トップページに戻る
              </Link>
              <Link href="/privacy" className="text-blue-600 hover:text-blue-800">
                プライバシーポリシー
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
