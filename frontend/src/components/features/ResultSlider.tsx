'use client'

import { useCallback, useMemo } from 'react'

/**
 * スライダーのスナップポイント
 */
const SNAP_POINTS = [0, 0.25, 0.5, 0.75, 1.0]

export interface ResultSliderProps {
  /** 現在の変化度 (0.0 - 1.0) */
  value: number
  /** 値変更時のコールバック */
  onChange: (value: number) => void
  /** 無効状態 */
  disabled?: boolean
  /** テスト用のdata-testid */
  testId?: string
}

/**
 * シミュレーション結果のスライダーコンポーネント
 *
 * 5段階のスナップポイント（0%, 25%, 50%, 75%, 100%）で変化度を調整
 */
export function ResultSlider({
  value,
  onChange,
  disabled = false,
  testId,
}: ResultSliderProps) {
  /**
   * スライダーの値を最も近いスナップポイントにスナップ
   */
  const snapToNearest = useCallback((rawValue: number): number => {
    let closestPoint = SNAP_POINTS[0]
    let minDistance = Math.abs(rawValue - closestPoint)

    for (const point of SNAP_POINTS) {
      const distance = Math.abs(rawValue - point)
      if (distance < minDistance) {
        minDistance = distance
        closestPoint = point
      }
    }

    return closestPoint
  }, [])

  /**
   * スライダー変更ハンドラ
   */
  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const rawValue = parseFloat(e.target.value)
      const snappedValue = snapToNearest(rawValue)
      onChange(snappedValue)
    },
    [onChange, snapToNearest]
  )

  /**
   * 現在の値のインデックスを計算
   */
  const currentIndex = useMemo(() => {
    return SNAP_POINTS.indexOf(value)
  }, [value])

  /**
   * パーセンテージ表示用
   */
  const percentageDisplay = useMemo(() => {
    return `${Math.round(value * 100)}%`
  }, [value])

  return (
    <div className="w-full max-w-lg mx-auto" data-testid={testId}>
      {/* スライダーコンテナ */}
      <div className="relative pt-1">
        {/* ラベル（現在/理想） */}
        <div className="flex justify-between mb-2">
          <span className="text-sm font-medium text-gray-600">現在</span>
          <span className="text-sm font-medium text-gray-600">理想</span>
        </div>

        {/* スライダートラック背景 */}
        <div className="relative h-2 bg-gray-200 rounded-full">
          {/* アクティブトラック */}
          <div
            className="absolute h-full bg-blue-500 rounded-full transition-all duration-150"
            style={{ width: `${value * 100}%` }}
          />

          {/* スナップポイントマーカー */}
          <div className="absolute inset-0 flex justify-between items-center px-0">
            {SNAP_POINTS.map((point, index) => (
              <button
                key={point}
                type="button"
                onClick={() => !disabled && onChange(point)}
                disabled={disabled}
                className={`
                  w-4 h-4 rounded-full border-2 transition-all duration-150
                  ${
                    index <= currentIndex
                      ? 'bg-blue-500 border-blue-600'
                      : 'bg-white border-gray-300'
                  }
                  ${
                    disabled
                      ? 'cursor-not-allowed'
                      : 'cursor-pointer hover:scale-110'
                  }
                  ${
                    point === value
                      ? 'ring-2 ring-blue-300 ring-offset-2'
                      : ''
                  }
                  focus:outline-none focus:ring-2 focus:ring-blue-400 focus:ring-offset-2
                `}
                aria-label={`${Math.round(point * 100)}%に設定`}
                data-testid={testId ? `${testId}-point-${Math.round(point * 100)}` : undefined}
              />
            ))}
          </div>
        </div>

        {/* ネイティブスライダー（透明、インタラクション用） */}
        <input
          type="range"
          min="0"
          max="1"
          step="0.01"
          value={value}
          onChange={handleChange}
          disabled={disabled}
          className="absolute inset-0 w-full h-8 opacity-0 cursor-pointer disabled:cursor-not-allowed"
          aria-label="変化度を調整"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(value * 100)}
          aria-valuetext={percentageDisplay}
          data-testid={testId ? `${testId}-input` : undefined}
        />

        {/* パーセンテージラベル */}
        <div className="flex justify-between mt-3">
          {SNAP_POINTS.map((point) => (
            <span
              key={point}
              className={`
                text-xs font-medium transition-colors duration-150
                ${point === value ? 'text-blue-600' : 'text-gray-500'}
              `}
            >
              {Math.round(point * 100)}%
            </span>
          ))}
        </div>
      </div>

      {/* 現在の変化度表示 */}
      <div className="mt-4 text-center">
        <p className="text-lg font-semibold text-gray-800">
          現在の変化度:{' '}
          <span className="text-blue-600" data-testid={testId ? `${testId}-value` : undefined}>
            {percentageDisplay}
          </span>
        </p>
      </div>
    </div>
  )
}
