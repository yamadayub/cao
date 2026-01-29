/**
 * 保留アクション管理のテスト
 *
 * UC-012: ログインして直前のアクションを継続
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import {
  savePendingAction,
  getPendingAction,
  clearPendingAction,
  PendingAction,
  PendingActionType,
} from '@/lib/pending-action'

// sessionStorage のモック
const mockSessionStorage = (() => {
  let store: Record<string, string> = {}
  return {
    getItem: vi.fn((key: string) => store[key] || null),
    setItem: vi.fn((key: string, value: string) => {
      store[key] = value
    }),
    removeItem: vi.fn((key: string) => {
      delete store[key]
    }),
    clear: vi.fn(() => {
      store = {}
    }),
  }
})()

Object.defineProperty(window, 'sessionStorage', {
  value: mockSessionStorage,
})

describe('保留アクション管理', () => {
  beforeEach(() => {
    mockSessionStorage.clear()
    vi.clearAllMocks()
  })

  afterEach(() => {
    mockSessionStorage.clear()
  })

  describe('savePendingAction', () => {
    it('パーツブラーアクションを保存できる', () => {
      const action: PendingAction = {
        type: 'parts-blur',
        viewMode: 'parts',
        partsSelection: {
          left_eye: true,
          right_eye: true,
          left_eyebrow: false,
          right_eyebrow: false,
          nose: true,
          lips: false,
        },
      }

      savePendingAction(action)

      expect(mockSessionStorage.setItem).toHaveBeenCalledWith(
        'cao_pending_action',
        expect.any(String)
      )

      const savedData = JSON.parse(
        mockSessionStorage.setItem.mock.calls[0][1]
      )
      expect(savedData.type).toBe('parts-blur')
      expect(savedData.viewMode).toBe('parts')
      expect(savedData.partsSelection.left_eye).toBe(true)
      expect(savedData.partsSelection.nose).toBe(true)
    })

    it('ダウンロードアクションを保存できる', () => {
      const action: PendingAction = {
        type: 'download',
        viewMode: 'morph',
      }

      savePendingAction(action)

      const savedData = JSON.parse(
        mockSessionStorage.setItem.mock.calls[0][1]
      )
      expect(savedData.type).toBe('download')
      expect(savedData.viewMode).toBe('morph')
    })

    it('保存アクションを保存できる', () => {
      const action: PendingAction = {
        type: 'save',
        viewMode: 'parts',
      }

      savePendingAction(action)

      const savedData = JSON.parse(
        mockSessionStorage.setItem.mock.calls[0][1]
      )
      expect(savedData.type).toBe('save')
    })

    it('共有アクションを保存できる', () => {
      const action: PendingAction = {
        type: 'share',
        viewMode: 'morph',
      }

      savePendingAction(action)

      const savedData = JSON.parse(
        mockSessionStorage.setItem.mock.calls[0][1]
      )
      expect(savedData.type).toBe('share')
    })
  })

  describe('getPendingAction', () => {
    it('保存されたアクションを取得できる', () => {
      const action: PendingAction = {
        type: 'parts-blur',
        viewMode: 'parts',
        partsSelection: {
          left_eye: true,
          right_eye: false,
          left_eyebrow: false,
          right_eyebrow: false,
          nose: false,
          lips: true,
        },
      }

      mockSessionStorage.getItem.mockReturnValue(JSON.stringify(action))

      const result = getPendingAction()

      expect(result).not.toBeNull()
      expect(result?.type).toBe('parts-blur')
      expect(result?.viewMode).toBe('parts')
      expect(result?.partsSelection?.left_eye).toBe(true)
      expect(result?.partsSelection?.lips).toBe(true)
    })

    it('アクションが保存されていない場合はnullを返す', () => {
      mockSessionStorage.getItem.mockReturnValue(null)

      const result = getPendingAction()

      expect(result).toBeNull()
    })

    it('不正なJSONの場合はnullを返す', () => {
      mockSessionStorage.getItem.mockReturnValue('invalid json')

      const result = getPendingAction()

      expect(result).toBeNull()
    })
  })

  describe('clearPendingAction', () => {
    it('保存されたアクションを削除できる', () => {
      clearPendingAction()

      expect(mockSessionStorage.removeItem).toHaveBeenCalledWith(
        'cao_pending_action'
      )
    })
  })

  describe('アクションタイプの検証', () => {
    const validTypes: PendingActionType[] = ['parts-blur', 'download', 'save', 'share']

    validTypes.forEach((type) => {
      it(`${type} は有効なアクションタイプである`, () => {
        const action: PendingAction = {
          type,
          viewMode: 'morph',
        }

        savePendingAction(action)

        const savedData = JSON.parse(
          mockSessionStorage.setItem.mock.calls[0][1]
        )
        expect(savedData.type).toBe(type)
      })
    })
  })
})
