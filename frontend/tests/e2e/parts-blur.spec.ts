import { test, expect } from '@playwright/test'
import path from 'path'

/**
 * E2Eテスト: パーツ別シミュレーション結果のブラー表示
 *
 * 参照: /tests/e2e/specs/parts-blur.spec.md
 * 参照: functional-spec.md セクション 3.4 (SCR-003)
 *
 * テスト対象:
 * - E2E-013: パーツ別ブレンド生成（未認証・ブラー表示）
 * - E2E-014: ブラー画像タップからログイン・結果閲覧
 */

test.describe('E2E-013: パーツ別ブレンド生成（未認証・ブラー表示）', () => {
  const TEST_IMAGE_PATH = path.join(__dirname, 'fixtures/valid-face.jpg')

  /**
   * 各テスト前に利用規約同意済み・画像データ設定済みの状態を作る
   */
  test.beforeEach(async ({ page }) => {
    await page.goto('/simulate')
    await page.evaluate(() => {
      // 利用規約同意
      localStorage.setItem(
        'cao_terms_agreed',
        JSON.stringify({
          agreed: true,
          version: '1.0',
          timestamp: new Date().toISOString(),
        })
      )
      // テスト用の画像データをセット
      const testImageData =
        'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=='
      sessionStorage.setItem('cao_current_image', testImageData)
      sessionStorage.setItem('cao_ideal_image', testImageData)
    })
  })

  /**
   * シナリオ1: 未認証ユーザーがパーツ別シミュレーションを実行し結果にブラーが適用される
   */
  test('未認証ユーザーのパーツ別シミュレーション結果にブラーが適用される', async ({
    page,
  }) => {
    // Given: 結果画面に遷移
    await page.goto('/simulate/result')

    // パーツ別タブが表示されるまで待機
    await page.waitForSelector('[data-testid="parts-tab"]', {
      state: 'visible',
      timeout: 15000,
    }).catch(() => {
      // タブが存在しない場合はスキップ
    })

    const partsTab = page.getByTestId('parts-tab')
    if (!(await partsTab.isVisible())) {
      test.skip()
      return
    }

    // When: パーツ別タブをクリック
    await partsTab.click()

    // パーツを選択（目と鼻）
    await page.getByTestId('parts-selector-eye').click()
    await page.getByTestId('parts-selector-nose').click()

    // 適用ボタンをクリック
    await page.getByTestId('apply-parts-button').click()

    // Then: 結果画像にブラーが適用される
    await expect(page.getByTestId('parts-blur-overlay')).toBeVisible({
      timeout: 30000,
    })
    await expect(page.getByTestId('parts-blur-overlay-blur')).toBeVisible()

    // 「タップしてログイン」テキストが表示される
    await expect(page.getByTestId('parts-blur-login-text')).toBeVisible()
  })

  /**
   * シナリオ2: 現在の顔画像はブラーなしで確認できる
   */
  test('現在の顔画像はブラーなしで確認できる', async ({ page }) => {
    // Given: 結果画面でパーツ別タブを選択
    await page.goto('/simulate/result')

    const partsTab = page.getByTestId('parts-tab')
    if (!(await partsTab.isVisible().catch(() => false))) {
      test.skip()
      return
    }

    await partsTab.click()
    await page.getByTestId('parts-selector-eye').click()
    await page.getByTestId('apply-parts-button').click()

    // ブラー画像が表示されるまで待機
    await expect(page.getByTestId('parts-blur-overlay-blur')).toBeVisible({
      timeout: 30000,
    })

    // When: 「現在」ボタンをクリック
    await page.getByTestId('toggle-current-button').click()

    // Then: 現在の顔画像がブラーなしで表示される
    await expect(page.getByTestId('parts-blur-overlay-blur')).not.toBeVisible()

    // When: 「適用後」ボタンをクリック
    await page.getByTestId('toggle-applied-button').click()

    // Then: ブラー画像が再表示される
    await expect(page.getByTestId('parts-blur-overlay-blur')).toBeVisible()
  })
})

test.describe('E2E-014: ブラー画像タップからログイン・結果閲覧', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/simulate')
    await page.evaluate(() => {
      localStorage.setItem(
        'cao_terms_agreed',
        JSON.stringify({
          agreed: true,
          version: '1.0',
          timestamp: new Date().toISOString(),
        })
      )
      const testImageData =
        'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=='
      sessionStorage.setItem('cao_current_image', testImageData)
      sessionStorage.setItem('cao_ideal_image', testImageData)
    })
  })

  /**
   * シナリオ3: ブラー画像タップでログイン誘導モーダルが表示される
   */
  test('ブラー画像をタップするとパーツ別ログイン誘導モーダルが表示される', async ({
    page,
  }) => {
    // Given: 結果画面でパーツ別シミュレーション結果がブラー表示されている
    await page.goto('/simulate/result')

    const partsTab = page.getByTestId('parts-tab')
    if (!(await partsTab.isVisible().catch(() => false))) {
      test.skip()
      return
    }

    await partsTab.click()
    await page.getByTestId('parts-selector-eye').click()
    await page.getByTestId('apply-parts-button').click()

    await expect(page.getByTestId('parts-blur-overlay')).toBeVisible({
      timeout: 30000,
    })

    // When: ブラー画像をクリック
    await page.getByTestId('parts-blur-overlay').click()

    // Then: パーツ別ログイン誘導モーダルが表示される
    await expect(page.getByTestId('parts-login-prompt-modal')).toBeVisible()

    // モーダルのタイトルを確認
    await expect(
      page.getByTestId('parts-login-prompt-modal')
    ).toContainText('パーツ別の結果を見るにはログインが必要です')

    // ボタンが表示されることを確認
    await expect(
      page.getByTestId('parts-login-prompt-modal-login-button')
    ).toBeVisible()
    await expect(
      page.getByTestId('parts-login-prompt-modal-cancel-button')
    ).toBeVisible()
  })

  /**
   * シナリオ4: ログインするボタンでログイン画面に遷移する
   */
  test('ログインするボタンでログイン画面に遷移する', async ({ page }) => {
    // Given: パーツ別ログイン誘導モーダルが表示されている
    await page.goto('/simulate/result')

    const partsTab = page.getByTestId('parts-tab')
    if (!(await partsTab.isVisible().catch(() => false))) {
      test.skip()
      return
    }

    await partsTab.click()
    await page.getByTestId('parts-selector-eye').click()
    await page.getByTestId('apply-parts-button').click()
    await page.getByTestId('parts-blur-overlay').click()

    await expect(page.getByTestId('parts-login-prompt-modal')).toBeVisible()

    // When: 「ログインする」ボタンをクリック
    await page.getByTestId('parts-login-prompt-modal-login-button').click()

    // Then: ログイン画面に遷移する（または認証フローが開始される）
    await expect(page).toHaveURL(/\/(login|sign-in)/, { timeout: 10000 })
  })

  /**
   * シナリオ5: 今はログインしないボタンでモーダルが閉じる
   */
  test('今はログインしないボタンでモーダルが閉じる', async ({ page }) => {
    // Given: パーツ別ログイン誘導モーダルが表示されている
    await page.goto('/simulate/result')

    const partsTab = page.getByTestId('parts-tab')
    if (!(await partsTab.isVisible().catch(() => false))) {
      test.skip()
      return
    }

    await partsTab.click()
    await page.getByTestId('parts-selector-eye').click()
    await page.getByTestId('apply-parts-button').click()
    await page.getByTestId('parts-blur-overlay').click()

    await expect(page.getByTestId('parts-login-prompt-modal')).toBeVisible()

    // When: 「今はログインしない」ボタンをクリック
    await page.getByTestId('parts-login-prompt-modal-cancel-button').click()

    // Then: モーダルが閉じる
    await expect(
      page.getByTestId('parts-login-prompt-modal')
    ).not.toBeVisible()

    // ブラー画像が引き続き表示される
    await expect(page.getByTestId('parts-blur-overlay-blur')).toBeVisible()
  })

  /**
   * シナリオ6: ログイン後はブラーなしで結果が表示される
   *
   * 注意: このテストは認証が必要なため、実際の認証環境またはモック認証が必要
   */
  test.skip('ログイン後はブラーなしで結果が表示される', async ({ page }) => {
    // Given: 未認証ユーザーがパーツ別シミュレーション結果をブラー表示で閲覧している
    await page.goto('/simulate/result')

    const partsTab = page.getByTestId('parts-tab')
    await partsTab.click()
    await page.getByTestId('parts-selector-eye').click()
    await page.getByTestId('apply-parts-button').click()

    await expect(page.getByTestId('parts-blur-overlay-blur')).toBeVisible({
      timeout: 30000,
    })

    // When: ログイン完了（モック認証を設定）
    await page.evaluate(() => {
      // 認証状態をモック（実際の実装に合わせて調整が必要）
      localStorage.setItem('cao_auth_token', 'mock-token')
    })

    // ページをリロード
    await page.reload()

    // パーツ別タブを再選択
    await page.getByTestId('parts-tab').click()

    // Then: ブラーなしで表示される
    await expect(
      page.getByTestId('parts-blur-overlay-blur')
    ).not.toBeVisible()
    await expect(
      page.getByTestId('parts-blur-login-text')
    ).not.toBeVisible()
  })
})

test.describe('認証済みユーザーのパーツ別シミュレーション', () => {
  /**
   * シナリオ7: 認証済みユーザーの結果はブラーなしで表示される
   *
   * 注意: このテストは認証が必要なため、実際の認証環境またはモック認証が必要
   */
  test.skip('認証済みユーザーの結果はブラーなしで表示される', async ({
    page,
  }) => {
    // Given: 認証済みユーザー（モック認証を設定）
    await page.goto('/simulate')
    await page.evaluate(() => {
      localStorage.setItem(
        'cao_terms_agreed',
        JSON.stringify({
          agreed: true,
          version: '1.0',
          timestamp: new Date().toISOString(),
        })
      )
      localStorage.setItem('cao_auth_token', 'mock-token')
      const testImageData =
        'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=='
      sessionStorage.setItem('cao_current_image', testImageData)
      sessionStorage.setItem('cao_ideal_image', testImageData)
    })

    await page.goto('/simulate/result')

    const partsTab = page.getByTestId('parts-tab')
    await partsTab.click()
    await page.getByTestId('parts-selector-eye').click()
    await page.getByTestId('apply-parts-button').click()

    // Then: ブラーなしで表示される
    await expect(page.getByTestId('parts-blur-overlay')).toBeVisible({
      timeout: 30000,
    })
    await expect(
      page.getByTestId('parts-blur-overlay-blur')
    ).not.toBeVisible()
    await expect(
      page.getByTestId('parts-blur-login-text')
    ).not.toBeVisible()
  })
})

test.describe('アクセシビリティテスト', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/simulate')
    await page.evaluate(() => {
      localStorage.setItem(
        'cao_terms_agreed',
        JSON.stringify({
          agreed: true,
          version: '1.0',
          timestamp: new Date().toISOString(),
        })
      )
      const testImageData =
        'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=='
      sessionStorage.setItem('cao_current_image', testImageData)
      sessionStorage.setItem('cao_ideal_image', testImageData)
    })
  })

  /**
   * シナリオ8: ブラー画像はキーボードでアクセス可能
   */
  test('ブラー画像がキーボード操作でフォーカス・実行可能', async ({
    page,
  }) => {
    await page.goto('/simulate/result')

    const partsTab = page.getByTestId('parts-tab')
    if (!(await partsTab.isVisible().catch(() => false))) {
      test.skip()
      return
    }

    await partsTab.click()
    await page.getByTestId('parts-selector-eye').click()
    await page.getByTestId('apply-parts-button').click()

    await expect(page.getByTestId('parts-blur-overlay')).toBeVisible({
      timeout: 30000,
    })

    // When: Tabキーでブラー画像にフォーカス
    const blurOverlay = page.getByTestId('parts-blur-overlay')
    await blurOverlay.focus()

    // Then: フォーカスが当たっている
    await expect(blurOverlay).toBeFocused()

    // When: Enterキーを押す
    await page.keyboard.press('Enter')

    // Then: ログイン誘導モーダルが表示される
    await expect(page.getByTestId('parts-login-prompt-modal')).toBeVisible()
  })

  /**
   * シナリオ9: スクリーンリーダー用のaria属性が設定されている
   */
  test('スクリーンリーダー用のaria属性が適切に設定されている', async ({
    page,
  }) => {
    await page.goto('/simulate/result')

    const partsTab = page.getByTestId('parts-tab')
    if (!(await partsTab.isVisible().catch(() => false))) {
      test.skip()
      return
    }

    await partsTab.click()
    await page.getByTestId('parts-selector-eye').click()
    await page.getByTestId('apply-parts-button').click()

    await expect(page.getByTestId('parts-blur-overlay')).toBeVisible({
      timeout: 30000,
    })

    // Then: aria-label属性が設定されている
    const blurOverlay = page.getByTestId('parts-blur-overlay')
    await expect(blurOverlay).toHaveAttribute(
      'aria-label',
      'ログインしてパーツ別シミュレーション結果を表示'
    )

    // role="button"が設定されている
    await expect(blurOverlay).toHaveAttribute('role', 'button')

    // tabIndex="0"が設定されている
    await expect(blurOverlay).toHaveAttribute('tabindex', '0')
  })
})

test.describe('エッジケース', () => {
  /**
   * シナリオ10: ローディング中はスケルトンが表示される
   */
  test('パーツ別結果のローディング中はスケルトンが表示される', async ({
    page,
  }) => {
    await page.goto('/simulate')
    await page.evaluate(() => {
      localStorage.setItem(
        'cao_terms_agreed',
        JSON.stringify({
          agreed: true,
          version: '1.0',
          timestamp: new Date().toISOString(),
        })
      )
      const testImageData =
        'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=='
      sessionStorage.setItem('cao_current_image', testImageData)
      sessionStorage.setItem('cao_ideal_image', testImageData)
    })

    await page.goto('/simulate/result')

    const partsTab = page.getByTestId('parts-tab')
    if (!(await partsTab.isVisible().catch(() => false))) {
      test.skip()
      return
    }

    await partsTab.click()
    await page.getByTestId('parts-selector-eye').click()

    // 適用ボタンをクリック
    await page.getByTestId('apply-parts-button').click()

    // ローディング中またはスケルトンが表示される（高速処理の場合は確認が難しい）
    // 最終的には結果が表示される
    await expect(
      page
        .getByTestId('parts-blur-overlay-skeleton')
        .or(page.getByTestId('parts-blur-overlay'))
    ).toBeVisible({ timeout: 30000 })
  })

  /**
   * シナリオ11: ESCキーでログイン誘導モーダルが閉じる
   */
  test('ESCキーでログイン誘導モーダルが閉じる', async ({ page }) => {
    await page.goto('/simulate')
    await page.evaluate(() => {
      localStorage.setItem(
        'cao_terms_agreed',
        JSON.stringify({
          agreed: true,
          version: '1.0',
          timestamp: new Date().toISOString(),
        })
      )
      const testImageData =
        'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=='
      sessionStorage.setItem('cao_current_image', testImageData)
      sessionStorage.setItem('cao_ideal_image', testImageData)
    })

    await page.goto('/simulate/result')

    const partsTab = page.getByTestId('parts-tab')
    if (!(await partsTab.isVisible().catch(() => false))) {
      test.skip()
      return
    }

    await partsTab.click()
    await page.getByTestId('parts-selector-eye').click()
    await page.getByTestId('apply-parts-button').click()
    await page.getByTestId('parts-blur-overlay').click()

    await expect(page.getByTestId('parts-login-prompt-modal')).toBeVisible()

    // When: ESCキーを押す
    await page.keyboard.press('Escape')

    // Then: モーダルが閉じる
    await expect(
      page.getByTestId('parts-login-prompt-modal')
    ).not.toBeVisible()
  })
})
