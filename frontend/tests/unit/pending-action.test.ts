/**
 * 保留アクション管理のテスト
 *
 * UC-012: ログインして直前のアクションを継続
 * 業務仕様書 7.3: ログイン時の画像復元
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import {
  savePendingAction,
  getPendingAction,
  clearPendingAction,
  saveSimulationImages,
  getSimulationImages,
  clearSimulationImages,
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

  describe('saveSimulationImages', () => {
    it('swappedImageを保存できる', () => {
      const swappedImage = 'data:image/png;base64,swapped123'

      saveSimulationImages({ swappedImage })

      expect(mockSessionStorage.setItem).toHaveBeenCalledWith(
        'cao_swapped_image',
        swappedImage
      )
    })

    it('partsBlendImageを保存できる', () => {
      const partsBlendImage = 'data:image/png;base64,partsblend456'

      saveSimulationImages({ partsBlendImage })

      expect(mockSessionStorage.setItem).toHaveBeenCalledWith(
        'cao_parts_blend_image',
        partsBlendImage
      )
    })

    it('両方の画像を同時に保存できる', () => {
      const swappedImage = 'data:image/png;base64,swapped123'
      const partsBlendImage = 'data:image/png;base64,partsblend456'

      saveSimulationImages({ swappedImage, partsBlendImage })

      expect(mockSessionStorage.setItem).toHaveBeenCalledWith(
        'cao_swapped_image',
        swappedImage
      )
      expect(mockSessionStorage.setItem).toHaveBeenCalledWith(
        'cao_parts_blend_image',
        partsBlendImage
      )
    })

    it('nullの場合は保存しない', () => {
      saveSimulationImages({ swappedImage: null, partsBlendImage: null })

      expect(mockSessionStorage.setItem).not.toHaveBeenCalled()
    })
  })

  describe('getSimulationImages', () => {
    it('保存された画像を取得できる', () => {
      const swappedImage = 'data:image/png;base64,swapped123'
      const partsBlendImage = 'data:image/png;base64,partsblend456'

      mockSessionStorage.getItem
        .mockReturnValueOnce(swappedImage)
        .mockReturnValueOnce(partsBlendImage)

      const result = getSimulationImages()

      expect(result.swappedImage).toBe(swappedImage)
      expect(result.partsBlendImage).toBe(partsBlendImage)
    })

    it('保存されていない場合はnullを返す', () => {
      mockSessionStorage.getItem.mockReturnValue(null)

      const result = getSimulationImages()

      expect(result.swappedImage).toBeNull()
      expect(result.partsBlendImage).toBeNull()
    })
  })

  describe('clearSimulationImages', () => {
    it('保存された画像を削除できる', () => {
      clearSimulationImages()

      expect(mockSessionStorage.removeItem).toHaveBeenCalledWith('cao_swapped_image')
      expect(mockSessionStorage.removeItem).toHaveBeenCalledWith('cao_parts_blend_image')
    })
  })

  /**
   * 業務仕様書 7.3: ログイン時の画像復元
   * 統合テスト: パーツブラーからログイン後の画像復元フロー
   */
  describe('ログイン時の画像復元フロー', () => {
    it('パーツブラーアクションと画像を保存し、ログイン後に復元できる', () => {
      // Step 1: ユーザーがブラー画像をクリック
      const swappedImage = 'data:image/png;base64,swappedface'
      const partsBlendImage = 'data:image/png;base64,partsresult'
      const partsSelection = {
        eyes: true,
        nose: true,
        lips: false,
      }

      // 保留アクションを保存
      const action: PendingAction = {
        type: 'parts-blur',
        viewMode: 'parts',
        partsSelection,
        partsViewMode: 'applied',
      }
      savePendingAction(action)

      // 画像を保存
      saveSimulationImages({ swappedImage, partsBlendImage })

      // 保存されたことを確認
      const savedAction = JSON.parse(mockSessionStorage.setItem.mock.calls[0][1])
      expect(savedAction.type).toBe('parts-blur')
      expect(savedAction.viewMode).toBe('parts')
      expect(savedAction.partsViewMode).toBe('applied')

      expect(mockSessionStorage.setItem).toHaveBeenCalledWith('cao_swapped_image', swappedImage)
      expect(mockSessionStorage.setItem).toHaveBeenCalledWith('cao_parts_blend_image', partsBlendImage)
    })

    it('保存されたアクションにpartsViewMode: applied が含まれていること', () => {
      const action: PendingAction = {
        type: 'parts-blur',
        viewMode: 'parts',
        partsSelection: { eyes: true, nose: false, lips: false },
        partsViewMode: 'applied',
      }

      savePendingAction(action)

      const savedData = JSON.parse(mockSessionStorage.setItem.mock.calls[0][1])

      // ログイン後の復元に必要な情報がすべて含まれていることを確認
      expect(savedData.type).toBe('parts-blur')
      expect(savedData.viewMode).toBe('parts')
      expect(savedData.partsViewMode).toBe('applied')
      expect(savedData.partsSelection).toEqual({ eyes: true, nose: false, lips: false })
    })
  })
})
