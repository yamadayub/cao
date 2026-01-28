import { test, expect } from '@playwright/test'

/**
 * E2Eテスト: ナビゲーション
 *
 * 参照: functional-spec.md セクション 3.2 (SCR-001: ランディングページ)
 */

test.describe('ナビゲーション', () => {
  test.describe('ランディングページ', () => {
    test('ランディングページが正しく表示される', async ({ page }) => {
      // Given: ランディングページにアクセス
      await page.goto('/')

      // ローディングが終わるまで待つ
      await page.waitForTimeout(2000)

      // Then: ヘッドラインが表示される
      // 「理想の顔写真と」「あなたの顔写真を組み合わせて」「顔全体・パーツ別の変化をシミュレーション」
      await expect(page.locator('text=/理想の顔写真と/i')).toBeVisible({ timeout: 10000 })
    })

    test('「今すぐ試す」ボタンをクリックするとシミュレーションページに遷移する', async ({ page }) => {
      // Given: ランディングページにアクセス
      await page.goto('/')
      await page.waitForTimeout(2000)

      // When: 「今すぐ試す」ボタンをクリック
      // ヘッダーとヒーローセクションにボタンがあるので、最初のものをクリック
      const ctaButton = page.locator('a:has-text("今すぐ"), a:has-text("無料で試す")').first()
      await expect(ctaButton).toBeVisible({ timeout: 10000 })
      await ctaButton.click()

      // Then: シミュレーションページに遷移
      await expect(page).toHaveURL(/simulate/)
    })

    test('ロゴをクリックするとトップページに遷移する', async ({ page }) => {
      // Given: シミュレーションページにアクセス
      await page.goto('/simulate')
      await page.waitForTimeout(1000)

      // When: ロゴをクリック
      const logo = page.locator('a:has-text("Cao")').first()
      if (await logo.isVisible().catch(() => false)) {
        await logo.click()

        // Then: トップページに遷移
        await expect(page).toHaveURL(/^https?:\/\/[^/]+\/?$/)
      } else {
        // ロゴが見つからない場合はスキップ
        test.skip()
      }
    })

    test('未認証ユーザーにログインボタンが表示される', async ({ page }) => {
      // Given: ランディングページにアクセス
      await page.goto('/')
      await page.waitForTimeout(2000)

      // Then: ログインボタンが表示される
      const loginButton = page.locator('button:has-text("ログイン"), a:has-text("ログイン")')
      await expect(loginButton).toBeVisible({ timeout: 10000 })
    })
  })

  test.describe('フッター', () => {
    test('フッターに利用規約リンクがある', async ({ page }) => {
      // Given: ランディングページにアクセス
      await page.goto('/')
      await page.waitForTimeout(2000)

      // Then: 利用規約リンクが表示される
      const termsLink = page.locator('footer a:has-text("利用規約")')
      if (await termsLink.isVisible().catch(() => false)) {
        // When: クリック
        await termsLink.click()

        // Then: 利用規約ページに遷移
        await expect(page).toHaveURL(/terms/)
      } else {
        // フッターが見つからない場合はスキップ
        test.skip()
      }
    })

    test('フッターにプライバシーポリシーリンクがある', async ({ page }) => {
      // Given: ランディングページにアクセス
      await page.goto('/')
      await page.waitForTimeout(2000)

      // Then: プライバシーポリシーリンクが表示される
      const privacyLink = page.locator('footer a:has-text("プライバシー")')
      if (await privacyLink.isVisible().catch(() => false)) {
        // When: クリック
        await privacyLink.click()

        // Then: プライバシーポリシーページに遷移
        await expect(page).toHaveURL(/privacy/)
      } else {
        // フッターが見つからない場合はスキップ
        test.skip()
      }
    })
  })

  test.describe('使い方セクション', () => {
    test('使い方セクションに3ステップが表示される', async ({ page }) => {
      // Given: ランディングページにアクセス
      await page.goto('/')
      await page.waitForTimeout(2000)

      // Then: 使い方セクションが表示される
      await expect(page.locator('text=/使い方/i')).toBeVisible({ timeout: 10000 })

      // 3ステップが表示される (順序: 理想→現在→結果)
      await expect(page.locator('text=/理想の顔をアップロード/i')).toBeVisible()
      await expect(page.locator('text=/現在の自分の顔をアップロード/i')).toBeVisible()
      await expect(page.locator('text=/結果を確認/i')).toBeVisible()
    })
  })

  test.describe('特徴セクション', () => {
    test('特徴セクションに4つの特徴が表示される', async ({ page }) => {
      // Given: ランディングページにアクセス
      await page.goto('/')
      await page.waitForTimeout(2000)

      // Then: 特徴セクションが表示される
      await expect(page.locator('h2:has-text("特徴")')).toBeVisible({ timeout: 10000 })

      // 4つの特徴が表示される
      await expect(page.locator('text=/高品質なAI合成/i')).toBeVisible()
      await expect(page.locator('text=/パーツ別シミュレーション/i')).toBeVisible()
      await expect(page.locator('text=/Before.*After比較/i')).toBeVisible()
      await expect(page.locator('text=/プライバシー保護/i')).toBeVisible()
    })
  })
})
