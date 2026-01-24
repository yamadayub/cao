import { defineConfig, devices } from '@playwright/test'

/**
 * Playwright E2Eテスト設定
 *
 * 参照: functional-spec.md セクション7 テスト要件
 * 参照: /tests/e2e/specs/*.spec.md E2Eテストシナリオ
 */
export default defineConfig({
  // テストファイルの配置ディレクトリ
  testDir: './tests/e2e',

  // テストファイルのパターン
  testMatch: '**/*.spec.ts',

  // 各テストのタイムアウト（30秒）
  timeout: 30 * 1000,

  // テスト実行時の期待値チェックのタイムアウト（5秒）
  expect: {
    timeout: 5 * 1000,
  },

  // 並列実行設定
  fullyParallel: true,

  // CI環境ではリトライしない
  retries: process.env.CI ? 2 : 0,

  // CI環境では並列度を制限
  workers: process.env.CI ? 1 : undefined,

  // レポーター設定
  reporter: [
    ['html', { outputFolder: 'playwright-report' }],
    ['list'],
  ],

  // 共通設定
  use: {
    // ベースURL（環境変数で上書き可能）
    baseURL: process.env.PLAYWRIGHT_BASE_URL || 'http://localhost:3000',

    // トレース（テスト失敗時に収集）
    trace: 'on-first-retry',

    // スクリーンショット（テスト失敗時のみ）
    screenshot: 'only-on-failure',

    // ビデオ（テスト失敗時のみ）
    video: 'on-first-retry',

    // タイムゾーン
    timezoneId: 'Asia/Tokyo',

    // ロケール
    locale: 'ja-JP',
  },

  // テスト対象ブラウザ（CIではChromiumのみ、ローカルでは全ブラウザ）
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
    // Safari/Firefox（ローカル環境のみ）
    ...(!process.env.CI
      ? [
          {
            name: 'webkit',
            use: { ...devices['Desktop Safari'] },
          },
        ]
      : []),
  ],

  // 開発サーバー設定（外部URLを使用する場合は無効化）
  webServer: process.env.PLAYWRIGHT_BASE_URL
    ? undefined
    : {
        command: 'pnpm dev',
        port: 3000,
        // テスト時は常に新しいサーバーを起動（正しい環境変数が読み込まれるように）
        reuseExistingServer: false,
        timeout: 120 * 1000,
      },
})
