import { test, expect } from '@playwright/test'

/**
 * E2Eテスト: 利用規約同意 (UC-001)
 *
 * 参照: /tests/e2e/specs/terms-agreement.spec.md
 * 参照: functional-spec.md セクション 3.3 利用規約同意モーダル
 *
 * Note: The simulate page starts on step 1 (ideal image upload),
 * so we use ideal-image-dropzone for these tests.
 */
test.describe('利用規約同意', () => {
  /**
   * シナリオ1: 初回アクセス時の利用規約同意ダイアログ表示
   */
  test('未同意ユーザーが画像アップロードエリアをクリックするとモーダルが表示される', async ({ page }) => {
    // Given: ローカルストレージがクリアされた状態
    await page.goto('/simulate')
    await page.evaluate(() => localStorage.clear())
    await page.reload()

    // Wait for the page to be ready
    await expect(page.getByTestId('ideal-image-dropzone')).toBeVisible()

    // When: 「理想の顔」アップロードエリアをクリック (step 1)
    await page.getByTestId('ideal-image-dropzone').click()

    // Then: 利用規約同意モーダルが表示される
    await expect(page.getByTestId('terms-agreement-modal')).toBeVisible()

    // モーダル内の要素が正しく表示されている
    await expect(page.getByTestId('terms-agreement-modal-title')).toContainText('利用規約への同意')
    await expect(page.getByTestId('terms-agreement-modal-checkbox')).toBeVisible()
    await expect(page.getByTestId('terms-agreement-modal-agree-button')).toBeVisible()

    // 同意ボタンは非活性状態
    await expect(page.getByTestId('terms-agreement-modal-agree-button')).toBeDisabled()
  })

  /**
   * シナリオ2: 利用規約への同意成功
   */
  test('チェックボックスにチェック後、同意ボタンで同意できる', async ({ page }) => {
    // Given: 未同意状態でシミュレーション作成画面にアクセス
    await page.goto('/simulate')
    await page.evaluate(() => localStorage.clear())
    await page.reload()

    // Wait for the page to be ready
    await expect(page.getByTestId('ideal-image-dropzone')).toBeVisible()

    // モーダルを開く
    await page.getByTestId('ideal-image-dropzone').click()
    await expect(page.getByTestId('terms-agreement-modal')).toBeVisible()

    // When: チェックボックスをクリックして同意にチェック
    await page.getByTestId('terms-agreement-modal-checkbox').check()

    // Then: 同意ボタンが活性化される
    await expect(page.getByTestId('terms-agreement-modal-agree-button')).toBeEnabled()

    // When: 同意ボタンをクリック
    await page.getByTestId('terms-agreement-modal-agree-button').click()

    // Then: モーダルが閉じる
    await expect(page.getByTestId('terms-agreement-modal')).not.toBeVisible()

    // ローカルストレージに同意情報が保存されている
    const hasAgreed = await page.evaluate(() => {
      return localStorage.getItem('cao_terms_agreed')
    })
    expect(hasAgreed).toBeTruthy()
  })

  /**
   * シナリオ3: 利用規約ページへのリンク遷移
   */
  test('「利用規約を読む」リンクが利用規約ページを開く', async ({ page, context }) => {
    // Given: 利用規約同意モーダルが表示されている
    await page.goto('/simulate')
    await page.evaluate(() => localStorage.clear())
    await page.reload()

    // Wait for the page to be ready
    await expect(page.getByTestId('ideal-image-dropzone')).toBeVisible()

    await page.getByTestId('ideal-image-dropzone').click()
    await expect(page.getByTestId('terms-agreement-modal')).toBeVisible()

    // When: 「利用規約を読む」リンクをクリック
    const newPagePromise = context.waitForEvent('page')
    await page.getByTestId('terms-agreement-modal-read-terms-link').click()

    // Then: 新しいタブで利用規約ページが開く
    const newPage = await newPagePromise
    await newPage.waitForLoadState()
    expect(newPage.url()).toContain('/terms')
  })

  /**
   * シナリオ4: 同意済みユーザーの再アクセス
   */
  test('同意済みユーザーはモーダルが表示されない', async ({ page }) => {
    // Given: ローカルストレージに同意情報がある状態
    await page.goto('/simulate')
    await page.evaluate(() => {
      // JSON形式で同意状態を保存（useTermsAgreementフックが期待する形式）
      localStorage.setItem('cao_terms_agreed', JSON.stringify({
        agreed: true,
        version: '1.0',
        timestamp: new Date().toISOString()
      }))
    })
    await page.reload()

    // Wait for the page to be ready
    await expect(page.getByTestId('ideal-image-dropzone')).toBeVisible()

    // Then: 利用規約同意バナーが表示されない
    await expect(page.getByTestId('terms-agreement-banner')).not.toBeVisible()

    // When: アップロードエリアをクリック（理想の顔 - step 1）
    await page.getByTestId('ideal-image-dropzone').click()

    // Then: モーダルは表示されない（ファイル選択ダイアログが開く想定）
    await expect(page.getByTestId('terms-agreement-modal')).not.toBeVisible()
  })

  /**
   * シナリオ5: 同意せずに閉じようとした場合（モーダル外クリック）
   */
  test('モーダル外をクリックしてもモーダルは閉じる', async ({ page }) => {
    // Given: 利用規約同意モーダルが表示されている
    await page.goto('/simulate')
    await page.evaluate(() => localStorage.clear())
    await page.reload()

    // Wait for the page to be ready
    await expect(page.getByTestId('ideal-image-dropzone')).toBeVisible()

    await page.getByTestId('ideal-image-dropzone').click()
    await expect(page.getByTestId('terms-agreement-modal')).toBeVisible()

    // When: モーダルの外側（オーバーレイ）をクリック
    // モーダルはdiv[data-testid="terms-agreement-modal"]がオーバーレイを兼ねている
    await page.getByTestId('terms-agreement-modal').click({ position: { x: 10, y: 10 } })

    // Then: モーダルが閉じる（このアプリの実装ではオーバーレイクリックで閉じる）
    await expect(page.getByTestId('terms-agreement-modal')).not.toBeVisible()

    // 同意情報は保存されていない
    const hasAgreed = await page.evaluate(() => {
      return localStorage.getItem('cao_terms_agreed')
    })
    expect(hasAgreed).toBeFalsy()
  })

  /**
   * 未同意時に利用規約同意バナーが表示される
   */
  test('未同意時に利用規約同意バナーが表示される', async ({ page }) => {
    // Given: ローカルストレージがクリアされた状態
    await page.goto('/simulate')
    await page.evaluate(() => localStorage.clear())
    await page.reload()

    // Then: 利用規約同意バナーが表示される
    await expect(page.getByTestId('terms-agreement-banner')).toBeVisible()
    await expect(page.getByTestId('terms-agreement-banner')).toContainText('利用規約への同意')
  })
})
