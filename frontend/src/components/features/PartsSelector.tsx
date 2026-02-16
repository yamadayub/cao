'use client'

import { memo } from 'react'
import { useTranslations } from 'next-intl'
import type { PartsSelection } from '@/lib/api/types'

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
 * シンプル3択: 目・鼻・唇
 */
export const PartsSelector = memo(function PartsSelector({
  selection,
  onChange,
  disabled = false,
  testId = 'parts-selector',
}: PartsSelectorProps) {
  const t = useTranslations('result')
  const tp = useTranslations('parts')

  const handleToggle = (part: keyof PartsSelection) => {
    onChange({
      ...selection,
      [part]: !selection[part],
    })
  }

  // パーツリスト（目・鼻・唇の3択）
  const parts: (keyof PartsSelection)[] = ['eyes', 'nose', 'lips']

  const renderPartButton = (part: keyof PartsSelection) => {
    const isSelected = selection[part]
    return (
      <button
        key={part}
        type="button"
        onClick={() => handleToggle(part)}
        disabled={disabled}
        className={`
          px-6 py-3 text-base font-medium rounded-full transition-all duration-200
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
        {tp(part)}
      </button>
    )
  }

  const hasAnySelection = Object.values(selection).some(Boolean)

  return (
    <div className="space-y-4" data-testid={testId}>
      {/* ラベル */}
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium text-neutral-700">
          {t('parts.selectParts')}
        </h3>
        {hasAnySelection && (
          <span className="text-xs text-primary-600">
            {t('parts.selectedCount', { count: Object.values(selection).filter(Boolean).length })}
          </span>
        )}
      </div>

      {/* パーツボタン（横並び） */}
      <div className="flex flex-wrap gap-3 justify-center">
        {parts.map(renderPartButton)}
      </div>

      {/* 全選択/全解除ボタン */}
      <div className="flex justify-center gap-2 pt-2">
        <button
          type="button"
          onClick={() =>
            onChange({
              eyes: true,
              nose: true,
              lips: true,
            })
          }
          disabled={disabled}
          className="text-xs text-primary-600 hover:text-primary-800 underline disabled:opacity-50 disabled:cursor-not-allowed"
          data-testid={`${testId}-select-all`}
        >
          {t('parts.selectAll')}
        </button>
        <span className="text-neutral-300">|</span>
        <button
          type="button"
          onClick={() =>
            onChange({
              eyes: false,
              nose: false,
              lips: false,
            })
          }
          disabled={disabled}
          className="text-xs text-neutral-500 hover:text-neutral-700 underline disabled:opacity-50 disabled:cursor-not-allowed"
          data-testid={`${testId}-clear-all`}
        >
          {t('parts.clearAll')}
        </button>
      </div>
    </div>
  )
})
