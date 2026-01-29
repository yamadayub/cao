'use client';

import { useState, useCallback } from 'react';
import { generateShareImage, shareImage, ShareResult } from '@/lib/share';

interface ShareButtonProps {
  /** 変更前画像（base64） */
  beforeImage: string;
  /** 変更後画像（base64） */
  afterImage: string;
  /** 適用されたパーツ */
  appliedParts?: string[];
  /** 追加のクラス名 */
  className?: string;
  /** テスト用ID */
  testId?: string;
}

type ShareState = 'idle' | 'generating' | 'sharing' | 'success' | 'error';

/**
 * シンプルなシェアボタン
 *
 * - モバイル: Web Share APIでネイティブシェア
 * - デスクトップ: クリップボードにコピー
 */
export function ShareButton({
  beforeImage,
  afterImage,
  appliedParts = [],
  className = '',
  testId,
}: ShareButtonProps) {
  const [state, setState] = useState<ShareState>('idle');
  const [message, setMessage] = useState('');

  const handleShare = useCallback(async () => {
    setState('generating');
    setMessage('');

    try {
      // シェア画像を生成
      const imageBlob = await generateShareImage(beforeImage, afterImage);

      setState('sharing');

      // シェア実行
      const result: ShareResult = await shareImage(imageBlob, appliedParts);

      switch (result) {
        case 'shared':
          setState('success');
          setMessage('シェアしました！');
          break;
        case 'copied':
          setState('success');
          setMessage('画像をコピーしました！SNSに貼り付けてシェア');
          break;
        case 'cancelled':
          setState('idle');
          break;
        case 'failed':
          setState('error');
          setMessage('シェアに失敗しました');
          break;
      }

      // 成功メッセージを3秒後にリセット
      if (result === 'shared' || result === 'copied') {
        setTimeout(() => {
          setState('idle');
          setMessage('');
        }, 3000);
      }
    } catch (error) {
      console.error('Share error:', error);
      setState('error');
      setMessage('画像の生成に失敗しました');

      setTimeout(() => {
        setState('idle');
        setMessage('');
      }, 3000);
    }
  }, [beforeImage, afterImage, appliedParts]);

  const isProcessing = state === 'generating' || state === 'sharing';

  return (
    <button
      type="button"
      onClick={handleShare}
      disabled={isProcessing}
      data-testid={testId}
      className={`
        w-full py-3 px-4 rounded-xl font-medium text-sm
        flex items-center justify-center gap-2
        transition-all duration-200
        focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500
        ${
          isProcessing
            ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
            : state === 'success'
            ? 'bg-green-500 text-white'
            : state === 'error'
            ? 'bg-red-500 text-white'
            : 'bg-gray-900 text-white hover:bg-gray-800 active:scale-[0.98]'
        }
        ${className}
      `}
    >
      {isProcessing ? (
        <>
          <svg
            className="animate-spin h-4 w-4"
            viewBox="0 0 24 24"
            fill="none"
          >
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
            />
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
            />
          </svg>
          {state === 'generating' ? '画像を生成中...' : 'シェア中...'}
        </>
      ) : state === 'success' ? (
        <>
          <svg
            className="w-4 h-4"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M5 13l4 4L19 7"
            />
          </svg>
          {message}
        </>
      ) : state === 'error' ? (
        <>
          <svg
            className="w-4 h-4"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
          {message}
        </>
      ) : (
        <>
          <svg
            className="w-4 h-4"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.368 2.684 3 3 0 00-5.368-2.684z"
            />
          </svg>
          シェアする
        </>
      )}
    </button>
  );
}
