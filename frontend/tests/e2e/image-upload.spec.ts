import { test, expect } from '@playwright/test'
import path from 'path'

/**
 * E2Eテスト: 画像アップロード (UC-002, UC-003)
 *
 * 参照: /tests/e2e/specs/image-upload.spec.md
 * 参照: functional-spec.md セクション 3.3 SCR-002
 */
test.describe('画像アップロード', () => {
  // テスト用画像ファイルのパス（テスト用のフィクスチャを配置する想定）
  const TEST_IMAGE_PATH = path.join(__dirname, 'fixtures/valid-face.jpg')

  /**
   * 各テスト前に利用規約に同意済みの状態を作る
   */
  test.beforeEach(async ({ page }) => {
    await page.goto('/simulate')
    // 利用規約に同意済みの状態をセット
    await page.evaluate(() => {
      localStorage.setItem('cao_terms_agreed', JSON.stringify({ agreed: true, version: '1.0', timestamp: new Date().toISOString() }))
    })
    await page.reload()
  })

  /**
   * シナリオ1: 現在の顔画像の正常アップロード
   */
  test('現在の顔画像をアップロードできる', async ({ page }) => {
    // Given: シミュレーション作成画面が表示されている
    await expect(page.getByTestId('current-image-dropzone')).toBeVisible()

    // プレビューがないことを確認
    await expect(page.getByTestId('current-image-preview')).not.toBeVisible()

    // When: ファイル選択でJPEG画像をアップロード
    const fileInput = page.getByTestId('current-image-input')
    await fileInput.setInputFiles(TEST_IMAGE_PATH)

    // Then: プレビュー画像が表示される
    await expect(page.getByTestId('current-image-preview')).toBeVisible()

    // 削除ボタンが表示される
    await expect(page.getByTestId('current-image-remove-button')).toBeVisible()
  })

  /**
   * シナリオ2: 理想の顔画像の正常アップロード
   */
  test('理想の顔画像をアップロードできる', async ({ page }) => {
    // Given: シミュレーション作成画面が表示されている
    await expect(page.getByTestId('ideal-image-dropzone')).toBeVisible()

    // When: ファイル選択でJPEG画像をアップロード
    const fileInput = page.getByTestId('ideal-image-input')
    await fileInput.setInputFiles(TEST_IMAGE_PATH)

    // Then: プレビュー画像が表示される
    await expect(page.getByTestId('ideal-image-preview')).toBeVisible()
  })

  /**
   * シナリオ4: アップロード済み画像の削除
   */
  test('アップロード済み画像を削除できる', async ({ page }) => {
    // Given: 現在の顔画像がアップロード済み
    const fileInput = page.getByTestId('current-image-input')
    await fileInput.setInputFiles(TEST_IMAGE_PATH)
    await expect(page.getByTestId('current-image-preview')).toBeVisible()

    // When: 削除ボタンをクリック
    await page.getByTestId('current-image-remove-button').click()

    // Then: プレビュー画像が消える
    await expect(page.getByTestId('current-image-preview')).not.toBeVisible()

    // 選択ボタンが再表示される
    await expect(page.getByTestId('current-image-select-button')).toBeVisible()
  })

  /**
   * シナリオ5: 両方の画像アップロード完了で生成ボタンが活性化
   */
  test('両方の画像をアップロードすると生成ボタンが活性化される', async ({ page }) => {
    // Given: 生成ボタンは非活性
    await expect(page.getByTestId('generate-button')).toBeDisabled()

    // When: 現在の顔画像をアップロード
    const currentInput = page.getByTestId('current-image-input')
    await currentInput.setInputFiles(TEST_IMAGE_PATH)
    await expect(page.getByTestId('current-image-preview')).toBeVisible()

    // まだ非活性
    await expect(page.getByTestId('generate-button')).toBeDisabled()

    // When: 理想の顔画像をアップロード
    const idealInput = page.getByTestId('ideal-image-input')
    await idealInput.setInputFiles(TEST_IMAGE_PATH)
    await expect(page.getByTestId('ideal-image-preview')).toBeVisible()

    // Then: 生成ボタンが活性化
    await expect(page.getByTestId('generate-button')).toBeEnabled()
  })

  /**
   * 画像を削除すると生成ボタンが非活性化される
   */
  test('画像を削除すると生成ボタンが非活性化される', async ({ page }) => {
    // Given: 両方の画像がアップロード済みで生成ボタンが活性
    const currentInput = page.getByTestId('current-image-input')
    const idealInput = page.getByTestId('ideal-image-input')
    await currentInput.setInputFiles(TEST_IMAGE_PATH)
    await idealInput.setInputFiles(TEST_IMAGE_PATH)
    await expect(page.getByTestId('generate-button')).toBeEnabled()

    // When: 現在の顔画像を削除
    await page.getByTestId('current-image-remove-button').click()

    // Then: 生成ボタンが非活性化
    await expect(page.getByTestId('generate-button')).toBeDisabled()
  })
})

/**
 * 画像バリデーションエラーのテスト
 *
 * 注意: これらのテストはテスト用の不正な画像ファイルを配置する必要がある
 * 実際のCIではfixturesディレクトリに適切なテストファイルを用意する
 */
test.describe('画像バリデーションエラー', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/simulate')
    await page.evaluate(() => {
      localStorage.setItem('cao_terms_agreed', JSON.stringify({ agreed: true, version: '1.0', timestamp: new Date().toISOString() }))
    })
    await page.reload()
  })

  /**
   * E1: 不正な画像形式（GIF）
   *
   * 注意: このテストはGIFファイルがfixturesディレクトリに存在する場合のみ動作
   */
  test.skip('GIF形式の画像はエラーになる', async ({ page }) => {
    // このテストはテスト用のGIFファイルが必要
    const gifPath = path.join(__dirname, 'fixtures/image.gif')

    // When: GIF画像をアップロード
    const fileInput = page.getByTestId('current-image-input')
    await fileInput.setInputFiles(gifPath)

    // Then: エラーメッセージが表示される
    await expect(page.getByTestId('current-image-error')).toBeVisible()
    await expect(page.getByTestId('current-image-error')).toContainText(
      'JPEG、PNG形式の画像をアップロードしてください'
    )
  })

  /**
   * E3: ファイルサイズ超過
   *
   * 注意: このテストは10MB超のファイルがfixturesディレクトリに存在する場合のみ動作
   */
  test.skip('10MB超の画像はエラーになる', async ({ page }) => {
    const largePath = path.join(__dirname, 'fixtures/large-file.jpg')

    // When: 大きなサイズの画像をアップロード
    const fileInput = page.getByTestId('current-image-input')
    await fileInput.setInputFiles(largePath)

    // Then: エラーメッセージが表示される
    await expect(page.getByTestId('current-image-error')).toBeVisible()
    await expect(page.getByTestId('current-image-error')).toContainText(
      '画像サイズは10MB以下にしてください'
    )
  })
})
