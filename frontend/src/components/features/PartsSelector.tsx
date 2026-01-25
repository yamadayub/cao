'use client'

import { memo } from 'react'
import type { PartsSelection } from '@/lib/api/types'
import { PARTS_DISPLAY_NAMES } from '@/lib/api/types'

interface PartsSelectorProps {
  /** 現在の選択状態 */
  selection: PartsSelection
  /** 選択変更ハンドラ */
  onChange: (selection: PartsSelection) => void
  /** 無効状態 */
  disabled?: boolean
  /** テスト用ID */
  testId?: string
}

/**
 * パーツ選択コンポーネント
 *
 * 各顔パーツのON/OFFを切り替えるトグルボタン群
 */
export const PartsSelector = memo(function PartsSelector({
  selection,
  onChange,
  disabled = false,
  testId = 'parts-selector',
}: PartsSelectorProps) {
  const handleToggle = (part: keyof PartsSelection) => {
    onChange({
      ...selection,
      [part]: !selection[part],
    })
  }

  // パーツをグループ分け（上段: 目・眉、下段: 鼻・唇）
  const eyeGroup: (keyof PartsSelection)[] = ['left_eyebrow', 'left_eye', 'right_eye', 'right_eyebrow']
  const faceGroup: (keyof PartsSelection)[] = ['nose', 'lips']

  const renderPartButton = (part: keyof PartsSelection) => {
    const isSelected = selection[part]
    return (
      <button
        key={part}
        type="button"
        onClick={() => handleToggle(part)}
        disabled={disabled}
        className={`
          px-4 py-2 text-sm font-medium rounded-full transition-all duration-200
          focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500
          ${
            isSelected
              ? 'bg-primary-700 text-white shadow-md'
              : 'bg-white text-neutral-600 border border-neutral-300 hover:bg-neutral-50'
          }
          ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
        `}
        data-testid={`${testId}-${part}`}
        aria-pressed={isSelected}
      >
        {PARTS_DISPLAY_NAMES[part]}
      </button>
    )
  }

  const hasAnySelection = Object.values(selection).some(Boolean)

  return (
    <div className="space-y-4" data-testid={testId}>
      {/* ラベル */}
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium text-neutral-700">
          ブレンドするパーツを選択
        </h3>
        {hasAnySelection && (
          <span className="text-xs text-primary-600">
            {Object.values(selection).filter(Boolean).length}個選択中
          </span>
        )}
      </div>

      {/* 目・眉グループ */}
      <div className="flex flex-wrap gap-2 justify-center">
        {eyeGroup.map(renderPartButton)}
      </div>

      {/* 鼻・唇グループ */}
      <div className="flex flex-wrap gap-2 justify-center">
        {faceGroup.map(renderPartButton)}
      </div>

      {/* 全選択/全解除ボタン */}
      <div className="flex justify-center gap-2 pt-2">
        <button
          type="button"
          onClick={() =>
            onChange({
              left_eye: true,
              right_eye: true,
              left_eyebrow: true,
              right_eyebrow: true,
              nose: true,
              lips: true,
            })
          }
          disabled={disabled}
          className="text-xs text-primary-600 hover:text-primary-800 underline disabled:opacity-50 disabled:cursor-not-allowed"
          data-testid={`${testId}-select-all`}
        >
          全て選択
        </button>
        <span className="text-neutral-300">|</span>
        <button
          type="button"
          onClick={() =>
            onChange({
              left_eye: false,
              right_eye: false,
              left_eyebrow: false,
              right_eyebrow: false,
              nose: false,
              lips: false,
            })
          }
          disabled={disabled}
          className="text-xs text-neutral-500 hover:text-neutral-700 underline disabled:opacity-50 disabled:cursor-not-allowed"
          data-testid={`${testId}-clear-all`}
        >
          全て解除
        </button>
      </div>
    </div>
  )
})
