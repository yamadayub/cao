/**
 * ImageUploaderコンポーネント - 単体テスト
 *
 * 対象: 画像アップロードUIコンポーネント
 * 参照: functional-spec.md セクション 3.3 (SCR-002)
 */

import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { ImageUploader, type ImageUploaderProps } from '@/components/features/ImageUploader'

// FileReaderのモック
class MockFileReader {
  result: string | null = null
  onload: ((this: FileReader, ev: ProgressEvent<FileReader>) => unknown) | null = null

  readAsDataURL(_blob: Blob) {
    this.result = 'data:image/jpeg;base64,test'
    setTimeout(() => {
      if (this.onload) {
        this.onload({ target: { result: this.result } } as unknown as ProgressEvent<FileReader>)
      }
    }, 0)
  }
}

// グローバルFileReaderをモック
vi.stubGlobal('FileReader', MockFileReader)

// ファイルオブジェクトのモック作成ヘルパー
const createMockFile = (
  name: string,
  type: string,
  sizeInBytes: number
): File => {
  const blob = new Blob([''], { type })
  Object.defineProperty(blob, 'size', { value: sizeInBytes })
  Object.defineProperty(blob, 'name', { value: name })
  return blob as File
}

describe('ImageUploader', () => {
  const defaultProps: ImageUploaderProps = {
    label: 'テスト画像',
    previewUrl: null,
    onFileSelect: vi.fn(),
    onFileRemove: vi.fn(),
    onValidationError: vi.fn(),
  }

  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('初期表示', () => {
    it('ラベルが表示される', () => {
      render(<ImageUploader {...defaultProps} />)
      expect(screen.getByText('テスト画像')).toBeInTheDocument()
    })

    it('画像選択ボタンが表示される', () => {
      render(<ImageUploader {...defaultProps} />)
      expect(screen.getByRole('button', { name: '画像を選択' })).toBeInTheDocument()
    })

    it('ドロップエリアが表示される', () => {
      render(<ImageUploader {...defaultProps} />)
      expect(screen.getByText('ドラッグ&ドロップ')).toBeInTheDocument()
    })

    it('プレビューがない場合はプレースホルダーが表示される', () => {
      render(<ImageUploader {...defaultProps} />)
      expect(screen.getByText('または下のボタンで選択')).toBeInTheDocument()
    })
  })

  describe('プレビュー表示', () => {
    it('プレビューURLがある場合は画像が表示される', () => {
      render(
        <ImageUploader
          {...defaultProps}
          previewUrl="data:image/jpeg;base64,test"
        />
      )
      const img = screen.getByAltText('テスト画像のプレビュー')
      expect(img).toBeInTheDocument()
      expect(img).toHaveAttribute('src', 'data:image/jpeg;base64,test')
    })

    it('プレビューがある場合は削除ボタンが表示される', () => {
      render(
        <ImageUploader
          {...defaultProps}
          previewUrl="data:image/jpeg;base64,test"
        />
      )
      expect(screen.getByRole('button', { name: '削除' })).toBeInTheDocument()
    })

    it('プレビューがある場合は画像選択ボタンは表示されない', () => {
      render(
        <ImageUploader
          {...defaultProps}
          previewUrl="data:image/jpeg;base64,test"
        />
      )
      expect(screen.queryByRole('button', { name: '画像を選択' })).not.toBeInTheDocument()
    })
  })

  describe('ファイル選択', () => {
    it('有効なJPEGファイルを選択するとonFileSelectが呼ばれる', async () => {
      const onFileSelect = vi.fn()
      render(<ImageUploader {...defaultProps} onFileSelect={onFileSelect} />)

      const file = createMockFile('test.jpg', 'image/jpeg', 1024)
      const input = document.querySelector('input[type="file"]') as HTMLInputElement

      fireEvent.change(input, { target: { files: [file] } })

      await waitFor(() => {
        expect(onFileSelect).toHaveBeenCalledWith(file, 'data:image/jpeg;base64,test')
      })
    })

    it('有効なPNGファイルを選択するとonFileSelectが呼ばれる', async () => {
      const onFileSelect = vi.fn()
      render(<ImageUploader {...defaultProps} onFileSelect={onFileSelect} />)

      const file = createMockFile('test.png', 'image/png', 1024)
      const input = document.querySelector('input[type="file"]') as HTMLInputElement

      fireEvent.change(input, { target: { files: [file] } })

      await waitFor(() => {
        expect(onFileSelect).toHaveBeenCalledWith(file, 'data:image/jpeg;base64,test')
      })
    })

    it('無効な形式のファイルを選択するとonValidationErrorが呼ばれる', async () => {
      const onValidationError = vi.fn()
      render(<ImageUploader {...defaultProps} onValidationError={onValidationError} />)

      const file = createMockFile('test.gif', 'image/gif', 1024)
      const input = document.querySelector('input[type="file"]') as HTMLInputElement

      fireEvent.change(input, { target: { files: [file] } })

      await waitFor(() => {
        expect(onValidationError).toHaveBeenCalledWith(
          'JPEG、PNG形式の画像をアップロードしてください'
        )
      })
    })

    it('サイズが大きすぎるファイルを選択するとonValidationErrorが呼ばれる', async () => {
      const onValidationError = vi.fn()
      render(<ImageUploader {...defaultProps} onValidationError={onValidationError} />)

      const file = createMockFile('test.jpg', 'image/jpeg', 15 * 1024 * 1024) // 15MB
      const input = document.querySelector('input[type="file"]') as HTMLInputElement

      fireEvent.change(input, { target: { files: [file] } })

      await waitFor(() => {
        expect(onValidationError).toHaveBeenCalledWith(
          '画像サイズは10MB以下にしてください'
        )
      })
    })
  })

  describe('ドラッグ&ドロップ', () => {
    it('ドラッグオーバー時にスタイルが変わる', () => {
      render(<ImageUploader {...defaultProps} testId="uploader" />)
      const dropzone = screen.getByTestId('uploader-dropzone')

      fireEvent.dragOver(dropzone)

      expect(screen.getByText('ここにドロップ')).toBeInTheDocument()
    })

    it('ドラッグリーブ時にスタイルが戻る', () => {
      render(<ImageUploader {...defaultProps} testId="uploader" />)
      const dropzone = screen.getByTestId('uploader-dropzone')

      fireEvent.dragOver(dropzone)
      fireEvent.dragLeave(dropzone)

      expect(screen.getByText('ドラッグ&ドロップ')).toBeInTheDocument()
    })

    it('有効なファイルをドロップするとonFileSelectが呼ばれる', async () => {
      const onFileSelect = vi.fn()
      render(
        <ImageUploader
          {...defaultProps}
          onFileSelect={onFileSelect}
          testId="uploader"
        />
      )
      const dropzone = screen.getByTestId('uploader-dropzone')

      const file = createMockFile('test.jpg', 'image/jpeg', 1024)
      const dataTransfer = {
        files: [file],
      }

      fireEvent.drop(dropzone, { dataTransfer })

      await waitFor(() => {
        expect(onFileSelect).toHaveBeenCalledWith(file, 'data:image/jpeg;base64,test')
      })
    })
  })

  describe('削除機能', () => {
    it('削除ボタンをクリックするとonFileRemoveが呼ばれる', () => {
      const onFileRemove = vi.fn()
      render(
        <ImageUploader
          {...defaultProps}
          previewUrl="data:image/jpeg;base64,test"
          onFileRemove={onFileRemove}
        />
      )

      const removeButton = screen.getByRole('button', { name: '削除' })
      fireEvent.click(removeButton)

      expect(onFileRemove).toHaveBeenCalled()
    })
  })

  describe('エラー表示', () => {
    it('エラーがある場合はエラーメッセージが表示される', () => {
      render(
        <ImageUploader
          {...defaultProps}
          error="テストエラーメッセージ"
          testId="uploader"
        />
      )

      expect(screen.getByText('テストエラーメッセージ')).toBeInTheDocument()
    })

    it('エラーメッセージにはrole="alert"が設定される', () => {
      render(
        <ImageUploader
          {...defaultProps}
          error="テストエラーメッセージ"
          testId="uploader"
        />
      )

      expect(screen.getByRole('alert')).toHaveTextContent('テストエラーメッセージ')
    })
  })

  describe('無効状態', () => {
    it('disabledの場合、ボタンが無効化される', () => {
      render(<ImageUploader {...defaultProps} disabled />)

      const button = screen.getByRole('button', { name: '画像を選択' })
      expect(button).toBeDisabled()
    })

    it('disabledの場合、ファイル入力が無効化される', () => {
      render(<ImageUploader {...defaultProps} disabled />)

      const input = document.querySelector('input[type="file"]') as HTMLInputElement
      expect(input).toBeDisabled()
    })
  })

  describe('アクセシビリティ', () => {
    it('ドロップエリアにaria-labelが設定される', () => {
      render(<ImageUploader {...defaultProps} testId="uploader" />)
      const dropzone = screen.getByTestId('uploader-dropzone')

      expect(dropzone).toHaveAttribute('aria-label', 'テスト画像をアップロード')
    })

    it('キーボード操作でファイル選択ダイアログを開ける', () => {
      render(<ImageUploader {...defaultProps} testId="uploader" />)
      const dropzone = screen.getByTestId('uploader-dropzone')
      const input = document.querySelector('input[type="file"]') as HTMLInputElement
      const clickSpy = vi.spyOn(input, 'click')

      fireEvent.keyDown(dropzone, { key: 'Enter' })

      expect(clickSpy).toHaveBeenCalled()
    })

    it('スペースキーでもファイル選択ダイアログを開ける', () => {
      render(<ImageUploader {...defaultProps} testId="uploader" />)
      const dropzone = screen.getByTestId('uploader-dropzone')
      const input = document.querySelector('input[type="file"]') as HTMLInputElement
      const clickSpy = vi.spyOn(input, 'click')

      fireEvent.keyDown(dropzone, { key: ' ' })

      expect(clickSpy).toHaveBeenCalled()
    })
  })
})
