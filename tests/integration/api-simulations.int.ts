/**
 * シミュレーションCRUD API - 結合テスト
 *
 * 対象エンドポイント:
 * - POST /api/v1/simulations (保存)
 * - GET /api/v1/simulations (一覧取得)
 * - GET /api/v1/simulations/{id} (詳細取得)
 * - DELETE /api/v1/simulations/{id} (削除)
 * - POST /api/v1/simulations/{id}/share (共有URL生成)
 * - GET /api/v1/shared/{token} (共有シミュレーション取得)
 *
 * 参照: functional-spec.md セクション 2.3.5 - 2.3.10
 */

import { describe, it, expect, beforeAll, afterAll, beforeEach } from 'vitest'

// テスト用の設定
const API_BASE_URL = process.env.API_BASE_URL || 'http://localhost:8000'
const SIMULATIONS_ENDPOINT = `${API_BASE_URL}/api/v1/simulations`

// 認証トークン（テスト用）
// TODO: 実際のテストではSupabase認証を使用してトークンを取得する
let authToken: string = ''

// テストで作成したシミュレーションのIDを追跡
let createdSimulationIds: string[] = []

// リクエストヘルパー
const authHeaders = () => ({
  Authorization: `Bearer ${authToken}`,
  'Content-Type': 'application/json',
})

// テストデータ
const createSimulationPayload = () => ({
  current_image: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==',
  ideal_image: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==',
  result_images: [
    { progress: 0, image: 'data:image/png;base64,iVBORw0KG...' },
    { progress: 0.25, image: 'data:image/png;base64,iVBORw0KG...' },
    { progress: 0.5, image: 'data:image/png;base64,iVBORw0KG...' },
    { progress: 0.75, image: 'data:image/png;base64,iVBORw0KG...' },
    { progress: 1.0, image: 'data:image/png;base64,iVBORw0KG...' },
  ],
  settings: {
    selected_progress: 0.5,
    notes: 'テスト用シミュレーション',
  },
})

// 認証セットアップ（テスト実行前）
beforeAll(async () => {
  // TODO: Supabase認証でテストユーザーのトークンを取得
  // authToken = await getTestUserToken()
  authToken = process.env.TEST_AUTH_TOKEN || 'test-token'
})

// テスト後のクリーンアップ
afterAll(async () => {
  // 作成したシミュレーションを削除
  for (const id of createdSimulationIds) {
    try {
      await fetch(`${SIMULATIONS_ENDPOINT}/${id}`, {
        method: 'DELETE',
        headers: authHeaders(),
      })
    } catch (e) {
      console.warn('Cleanup failed for simulation:', id)
    }
  }
})

describe('POST /api/v1/simulations - シミュレーション保存', () => {
  describe('正常系', () => {
    it('認証済みユーザーはシミュレーションを保存できる', async () => {
      const payload = createSimulationPayload()

      const response = await fetch(SIMULATIONS_ENDPOINT, {
        method: 'POST',
        headers: authHeaders(),
        body: JSON.stringify(payload),
      })

      expect(response.status).toBe(200)

      const json = await response.json()

      expect(json.success).toBe(true)
      expect(json.data).toBeDefined()
      expect(json.data.id).toBeDefined()
      expect(json.data.user_id).toBeDefined()
      expect(json.data.current_image_url).toBeDefined()
      expect(json.data.ideal_image_url).toBeDefined()
      expect(json.data.result_images).toBeDefined()
      expect(Array.isArray(json.data.result_images)).toBe(true)
      expect(json.data.settings).toBeDefined()
      expect(json.data.created_at).toBeDefined()
      expect(json.data.updated_at).toBeDefined()

      // クリーンアップ用にIDを保存
      createdSimulationIds.push(json.data.id)
    })

    it('settings は省略可能', async () => {
      const payload = {
        current_image: 'data:image/png;base64,...',
        ideal_image: 'data:image/png;base64,...',
        result_images: [
          { progress: 0, image: 'data:image/png;base64,...' },
          { progress: 1.0, image: 'data:image/png;base64,...' },
        ],
      }

      const response = await fetch(SIMULATIONS_ENDPOINT, {
        method: 'POST',
        headers: authHeaders(),
        body: JSON.stringify(payload),
      })

      expect(response.status).toBe(200)

      const json = await response.json()

      expect(json.success).toBe(true)
      expect(json.data.settings).toBeDefined()

      createdSimulationIds.push(json.data.id)
    })
  })

  describe('異常系', () => {
    it('未認証リクエストは UNAUTHORIZED を返す', async () => {
      const payload = createSimulationPayload()

      const response = await fetch(SIMULATIONS_ENDPOINT, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      })

      expect(response.status).toBe(401)

      const json = await response.json()

      expect(json.success).toBe(false)
      expect(json.error.code).toBe('UNAUTHORIZED')
      expect(json.error.message).toBe('Authentication required')
    })

    it('current_image がない場合は VALIDATION_ERROR を返す', async () => {
      const payload = {
        ideal_image: 'data:image/png;base64,...',
        result_images: [],
      }

      const response = await fetch(SIMULATIONS_ENDPOINT, {
        method: 'POST',
        headers: authHeaders(),
        body: JSON.stringify(payload),
      })

      expect(response.status).toBe(400)

      const json = await response.json()

      expect(json.success).toBe(false)
      expect(json.error.code).toBe('VALIDATION_ERROR')
    })

    it('ideal_image がない場合は VALIDATION_ERROR を返す', async () => {
      const payload = {
        current_image: 'data:image/png;base64,...',
        result_images: [],
      }

      const response = await fetch(SIMULATIONS_ENDPOINT, {
        method: 'POST',
        headers: authHeaders(),
        body: JSON.stringify(payload),
      })

      expect(response.status).toBe(400)

      const json = await response.json()

      expect(json.success).toBe(false)
      expect(json.error.code).toBe('VALIDATION_ERROR')
    })

    it('result_images がない場合は VALIDATION_ERROR を返す', async () => {
      const payload = {
        current_image: 'data:image/png;base64,...',
        ideal_image: 'data:image/png;base64,...',
      }

      const response = await fetch(SIMULATIONS_ENDPOINT, {
        method: 'POST',
        headers: authHeaders(),
        body: JSON.stringify(payload),
      })

      expect(response.status).toBe(400)

      const json = await response.json()

      expect(json.success).toBe(false)
      expect(json.error.code).toBe('VALIDATION_ERROR')
    })
  })
})

describe('GET /api/v1/simulations - シミュレーション一覧取得', () => {
  describe('正常系', () => {
    it('認証済みユーザーは自分のシミュレーション一覧を取得できる', async () => {
      const response = await fetch(SIMULATIONS_ENDPOINT, {
        method: 'GET',
        headers: authHeaders(),
      })

      expect(response.status).toBe(200)

      const json = await response.json()

      expect(json.success).toBe(true)
      expect(json.data).toBeDefined()
      expect(json.data.simulations).toBeDefined()
      expect(Array.isArray(json.data.simulations)).toBe(true)
      expect(json.data.pagination).toBeDefined()
      expect(json.data.pagination.total).toBeDefined()
      expect(json.data.pagination.limit).toBeDefined()
      expect(json.data.pagination.offset).toBeDefined()
      expect(json.data.pagination.has_more).toBeDefined()
    })

    it('limit パラメータで取得件数を制限できる', async () => {
      const response = await fetch(`${SIMULATIONS_ENDPOINT}?limit=5`, {
        method: 'GET',
        headers: authHeaders(),
      })

      const json = await response.json()

      expect(json.success).toBe(true)
      expect(json.data.pagination.limit).toBe(5)
      expect(json.data.simulations.length).toBeLessThanOrEqual(5)
    })

    it('offset パラメータでページングできる', async () => {
      const response = await fetch(`${SIMULATIONS_ENDPOINT}?offset=10`, {
        method: 'GET',
        headers: authHeaders(),
      })

      const json = await response.json()

      expect(json.success).toBe(true)
      expect(json.data.pagination.offset).toBe(10)
    })

    it('sort パラメータで並び順を指定できる', async () => {
      const response = await fetch(
        `${SIMULATIONS_ENDPOINT}?sort=created_at:asc`,
        {
          method: 'GET',
          headers: authHeaders(),
        }
      )

      expect(response.status).toBe(200)

      const json = await response.json()

      expect(json.success).toBe(true)
    })

    it('シミュレーション一覧にはサムネイルURLが含まれる', async () => {
      // まずシミュレーションを作成
      const payload = createSimulationPayload()
      const createResponse = await fetch(SIMULATIONS_ENDPOINT, {
        method: 'POST',
        headers: authHeaders(),
        body: JSON.stringify(payload),
      })
      const created = await createResponse.json()
      createdSimulationIds.push(created.data.id)

      // 一覧取得
      const response = await fetch(SIMULATIONS_ENDPOINT, {
        method: 'GET',
        headers: authHeaders(),
      })

      const json = await response.json()

      if (json.data.simulations.length > 0) {
        const simulation = json.data.simulations[0]
        expect(simulation.id).toBeDefined()
        expect(simulation.thumbnail_url).toBeDefined()
        expect(simulation.created_at).toBeDefined()
        expect(simulation.is_public).toBeDefined()
      }
    })
  })

  describe('異常系', () => {
    it('未認証リクエストは UNAUTHORIZED を返す', async () => {
      const response = await fetch(SIMULATIONS_ENDPOINT, {
        method: 'GET',
      })

      expect(response.status).toBe(401)

      const json = await response.json()

      expect(json.success).toBe(false)
      expect(json.error.code).toBe('UNAUTHORIZED')
    })

    it('limit が上限（100）を超える場合は制限される', async () => {
      const response = await fetch(`${SIMULATIONS_ENDPOINT}?limit=200`, {
        method: 'GET',
        headers: authHeaders(),
      })

      const json = await response.json()

      expect(json.success).toBe(true)
      expect(json.data.pagination.limit).toBeLessThanOrEqual(100)
    })
  })
})

describe('GET /api/v1/simulations/{id} - シミュレーション詳細取得', () => {
  let testSimulationId: string

  beforeAll(async () => {
    // テスト用シミュレーションを作成
    const payload = createSimulationPayload()
    const response = await fetch(SIMULATIONS_ENDPOINT, {
      method: 'POST',
      headers: authHeaders(),
      body: JSON.stringify(payload),
    })
    const json = await response.json()
    testSimulationId = json.data.id
    createdSimulationIds.push(testSimulationId)
  })

  describe('正常系', () => {
    it('自分のシミュレーション詳細を取得できる', async () => {
      const response = await fetch(
        `${SIMULATIONS_ENDPOINT}/${testSimulationId}`,
        {
          method: 'GET',
          headers: authHeaders(),
        }
      )

      expect(response.status).toBe(200)

      const json = await response.json()

      expect(json.success).toBe(true)
      expect(json.data.id).toBe(testSimulationId)
      expect(json.data.current_image_url).toBeDefined()
      expect(json.data.ideal_image_url).toBeDefined()
      expect(json.data.result_images).toBeDefined()
      expect(json.data.settings).toBeDefined()
      expect(json.data.share_token).toBeDefined()
      expect(json.data.is_public).toBeDefined()
    })
  })

  describe('異常系', () => {
    it('存在しないIDは NOT_FOUND を返す', async () => {
      const fakeId = '00000000-0000-0000-0000-000000000000'

      const response = await fetch(`${SIMULATIONS_ENDPOINT}/${fakeId}`, {
        method: 'GET',
        headers: authHeaders(),
      })

      expect(response.status).toBe(404)

      const json = await response.json()

      expect(json.success).toBe(false)
      expect(json.error.code).toBe('NOT_FOUND')
      expect(json.error.message).toBe('Simulation not found')
    })

    it('他ユーザーのシミュレーションは UNAUTHORIZED を返す', async () => {
      // TODO: 別ユーザーのシミュレーションIDでテスト
      // このテストは複数ユーザーのセットアップが必要
      console.warn('Multi-user test requires additional setup, skipping')
    })

    it('未認証リクエストは UNAUTHORIZED を返す', async () => {
      const response = await fetch(
        `${SIMULATIONS_ENDPOINT}/${testSimulationId}`,
        {
          method: 'GET',
        }
      )

      expect(response.status).toBe(401)

      const json = await response.json()

      expect(json.success).toBe(false)
      expect(json.error.code).toBe('UNAUTHORIZED')
    })
  })
})

describe('DELETE /api/v1/simulations/{id} - シミュレーション削除', () => {
  describe('正常系', () => {
    it('自分のシミュレーションを削除できる', async () => {
      // 削除用のシミュレーションを作成
      const payload = createSimulationPayload()
      const createResponse = await fetch(SIMULATIONS_ENDPOINT, {
        method: 'POST',
        headers: authHeaders(),
        body: JSON.stringify(payload),
      })
      const created = await createResponse.json()
      const simulationId = created.data.id

      // 削除
      const deleteResponse = await fetch(
        `${SIMULATIONS_ENDPOINT}/${simulationId}`,
        {
          method: 'DELETE',
          headers: authHeaders(),
        }
      )

      expect(deleteResponse.status).toBe(200)

      const json = await deleteResponse.json()

      expect(json.success).toBe(true)
      expect(json.data.deleted).toBe(true)
      expect(json.data.id).toBe(simulationId)

      // 削除確認
      const getResponse = await fetch(
        `${SIMULATIONS_ENDPOINT}/${simulationId}`,
        {
          method: 'GET',
          headers: authHeaders(),
        }
      )

      expect(getResponse.status).toBe(404)
    })
  })

  describe('異常系', () => {
    it('存在しないIDは NOT_FOUND を返す', async () => {
      const fakeId = '00000000-0000-0000-0000-000000000000'

      const response = await fetch(`${SIMULATIONS_ENDPOINT}/${fakeId}`, {
        method: 'DELETE',
        headers: authHeaders(),
      })

      expect(response.status).toBe(404)

      const json = await response.json()

      expect(json.success).toBe(false)
      expect(json.error.code).toBe('NOT_FOUND')
    })

    it('未認証リクエストは UNAUTHORIZED を返す', async () => {
      const fakeId = '00000000-0000-0000-0000-000000000000'

      const response = await fetch(`${SIMULATIONS_ENDPOINT}/${fakeId}`, {
        method: 'DELETE',
      })

      expect(response.status).toBe(401)
    })
  })
})

describe('POST /api/v1/simulations/{id}/share - 共有URL生成', () => {
  let testSimulationId: string

  beforeAll(async () => {
    const payload = createSimulationPayload()
    const response = await fetch(SIMULATIONS_ENDPOINT, {
      method: 'POST',
      headers: authHeaders(),
      body: JSON.stringify(payload),
    })
    const json = await response.json()
    testSimulationId = json.data.id
    createdSimulationIds.push(testSimulationId)
  })

  describe('正常系', () => {
    it('共有URLを生成できる', async () => {
      const response = await fetch(
        `${SIMULATIONS_ENDPOINT}/${testSimulationId}/share`,
        {
          method: 'POST',
          headers: authHeaders(),
        }
      )

      expect(response.status).toBe(200)

      const json = await response.json()

      expect(json.success).toBe(true)
      expect(json.data.share_token).toBeDefined()
      expect(json.data.share_url).toBeDefined()
      expect(json.data.share_url).toContain('cao.app/s/')
      expect(json.data.expires_at).toBeNull() // 無期限
    })

    it('生成された共有URLは有効なトークンを含む', async () => {
      const response = await fetch(
        `${SIMULATIONS_ENDPOINT}/${testSimulationId}/share`,
        {
          method: 'POST',
          headers: authHeaders(),
        }
      )

      const json = await response.json()

      // トークンが URL の一部として含まれていることを確認
      expect(json.data.share_url).toContain(json.data.share_token)
    })
  })

  describe('異常系', () => {
    it('存在しないシミュレーションは NOT_FOUND を返す', async () => {
      const fakeId = '00000000-0000-0000-0000-000000000000'

      const response = await fetch(`${SIMULATIONS_ENDPOINT}/${fakeId}/share`, {
        method: 'POST',
        headers: authHeaders(),
      })

      expect(response.status).toBe(404)
    })

    it('未認証リクエストは UNAUTHORIZED を返す', async () => {
      const response = await fetch(
        `${SIMULATIONS_ENDPOINT}/${testSimulationId}/share`,
        {
          method: 'POST',
        }
      )

      expect(response.status).toBe(401)
    })
  })
})

describe('GET /api/v1/shared/{token} - 共有シミュレーション取得', () => {
  let shareToken: string

  beforeAll(async () => {
    // シミュレーションを作成して共有URLを生成
    const payload = createSimulationPayload()
    const createResponse = await fetch(SIMULATIONS_ENDPOINT, {
      method: 'POST',
      headers: authHeaders(),
      body: JSON.stringify(payload),
    })
    const created = await createResponse.json()
    createdSimulationIds.push(created.data.id)

    const shareResponse = await fetch(
      `${SIMULATIONS_ENDPOINT}/${created.data.id}/share`,
      {
        method: 'POST',
        headers: authHeaders(),
      }
    )
    const shared = await shareResponse.json()
    shareToken = shared.data.share_token
  })

  describe('正常系', () => {
    it('共有トークンでシミュレーション結果を取得できる（認証不要）', async () => {
      const response = await fetch(`${API_BASE_URL}/api/v1/shared/${shareToken}`)

      expect(response.status).toBe(200)

      const json = await response.json()

      expect(json.success).toBe(true)
      expect(json.data.result_images).toBeDefined()
      expect(Array.isArray(json.data.result_images)).toBe(true)
      expect(json.data.created_at).toBeDefined()
    })

    it('共有データには result_images と created_at のみ含まれる', async () => {
      const response = await fetch(`${API_BASE_URL}/api/v1/shared/${shareToken}`)

      const json = await response.json()

      expect(json.data.result_images).toBeDefined()
      expect(json.data.created_at).toBeDefined()

      // 元画像やユーザー情報は含まれない（プライバシー保護）
      expect(json.data.current_image_url).toBeUndefined()
      expect(json.data.ideal_image_url).toBeUndefined()
      expect(json.data.user_id).toBeUndefined()
    })
  })

  describe('異常系', () => {
    it('無効なトークンは NOT_FOUND を返す', async () => {
      const response = await fetch(
        `${API_BASE_URL}/api/v1/shared/invalid-token-123`
      )

      expect(response.status).toBe(404)

      const json = await response.json()

      expect(json.success).toBe(false)
      expect(json.error.code).toBe('NOT_FOUND')
      expect(json.error.message).toBe('Shared simulation not found')
    })

    it('空のトークンは NOT_FOUND を返す', async () => {
      const response = await fetch(`${API_BASE_URL}/api/v1/shared/`)

      expect([404, 400]).toContain(response.status)
    })
  })
})
