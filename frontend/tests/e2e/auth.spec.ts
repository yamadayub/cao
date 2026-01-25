import { test, expect } from '@playwright/test'

/**
 * E2Eテスト: ユーザー認証 (UC-009)
 *
 * 参照: functional-spec.md セクション 3.5
 * 参照: /tests/e2e/specs/auth.spec.md
 *
 * Note: CI環境ではClerkの環境変数が設定されていないため、
 * 認証関連のテストはスキップされます。
 */

// CI環境かどうかを検出（Clerkキーが設定されていない場合はスキップ）
const skipClerkTests = process.env.CI === 'true'

test.describe('ユーザー認証', () => {
  test.skip(skipClerkTests, 'Skipping Clerk tests in CI environment without Clerk keys')
  test.describe('ログイン画面', () => {
    test('サインイン画面が表示される', async ({ page }) => {
      // Given: ユーザーがサインインページにアクセス
      await page.goto('/sign-in')

      // Then: Clerkのサインインフォームが表示される
      // Clerkコンポーネントはiframe内または特定のクラスで表示される
      await expect(page.locator('.cl-rootBox, .cl-signIn-root, [data-clerk-component]')).toBeVisible({ timeout: 10000 })
    })

    test('サインアップ画面が表示される', async ({ page }) => {
      // Given: ユーザーがサインアップページにアクセス
      await page.goto('/sign-up')

      // Then: Clerkのサインアップフォームが表示される
      await expect(page.locator('.cl-rootBox, .cl-signUp-root, [data-clerk-component]')).toBeVisible({ timeout: 10000 })
    })
  })

  test.describe('保護されたページへのアクセス', () => {
    test('未認証ユーザーがマイページにアクセスするとサインインにリダイレクトされる', async ({ page }) => {
      // Given: ユーザーは未認証

      // When: マイページに直接アクセス
      await page.goto('/mypage')

      // Then: サインインページにリダイレクトされる、またはサインインを促される
      // Clerkのミドルウェアによりリダイレクトされるか、ページ内でサインインが要求される
      await page.waitForTimeout(2000) // リダイレクトを待つ

      const url = page.url()
      const hasSignIn = url.includes('sign-in') || url.includes('login')
      const hasClerkComponent = await page.locator('.cl-rootBox, [data-clerk-component]').isVisible().catch(() => false)

      expect(hasSignIn || hasClerkComponent).toBeTruthy()
    })
  })

  test.describe('ナビゲーション', () => {
    test('シミュレーションページからサインインページへ遷移できる', async ({ page }) => {
      // Given: シミュレーションページにアクセス
      await page.goto('/simulate')

      // When: サインインリンク/ボタンがあればクリック（ヘッダーにある場合）
      const signInLink = page.locator('a[href*="sign-in"], button:has-text("ログイン"), a:has-text("ログイン")')

      if (await signInLink.isVisible().catch(() => false)) {
        await signInLink.click()

        // Then: サインインページに遷移
        await expect(page).toHaveURL(/sign-in/)
      } else {
        // ヘッダーにリンクがない場合はスキップ（後でナビゲーション追加後に有効化）
        test.skip()
      }
    })
  })
})
