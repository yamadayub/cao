'use client';

import { useState, useCallback } from 'react';
import { useTranslations } from 'next-intl';
import { generateShareImage, shareImage, ShareResult, ShareImageType } from '@/lib/share';

interface ShareButtonProps {
  /** 変更前画像（base64） */
  beforeImage: string;
  /** 変更後画像（base64） */
  afterImage: string;
  /** ログイン済みかどうか */
  isSignedIn: boolean;
  /** ログインが必要な時に呼ばれるコールバック */
  onLoginRequired: () => void;
  /** 追加のクラス名 */
  className?: string;
  /** テスト用ID */
  testId?: string;
}

type ShareState = 'idle' | 'selecting' | 'generating' | 'sharing' | 'success' | 'error';

/**
 * シェアボタン（タイプ選択ダイアログ付き）
 *
 * - シェアタイプ選択: Before/After比較 または 結果のみ
 * - モバイル: Web Share APIでネイティブシェア
 * - デスクトップ: クリップボードにコピー
 */
export function ShareButton({
  beforeImage,
  afterImage,
  isSignedIn,
  onLoginRequired,
  className = '',
  testId,
}: ShareButtonProps) {
  const t = useTranslations('modals');
  const [state, setState] = useState<ShareState>('idle');
  const [message, setMessage] = useState('');

  const openShareTypeDialog = useCallback(() => {
    // 未ログインの場合はログインを要求
    if (!isSignedIn) {
      onLoginRequired();
      return;
    }
    setState('selecting');
    setMessage('');
  }, [isSignedIn, onLoginRequired]);

  const closeDialog = useCallback(() => {
    setState('idle');
    setMessage('');
  }, []);

  const handleShare = useCallback(async (shareType: ShareImageType) => {
    setState('generating');
    setMessage('');

    try {
      // シェア画像を生成
      const imageBlob = await generateShareImage(beforeImage, afterImage, shareType);

      setState('sharing');

      // シェア実行
      const result: ShareResult = await shareImage(imageBlob);

      switch (result) {
        case 'shared':
          setState('success');
          setMessage(t('snsShare.shared'));
          break;
        case 'copied':
          setState('success');
          setMessage(t('snsShare.copied'));
          break;
        case 'cancelled':
          setState('idle');
          break;
        case 'failed':
          setState('error');
          setMessage(t('snsShare.failed'));
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
      setMessage(t('snsShare.generateFailed'));

      setTimeout(() => {
        setState('idle');
        setMessage('');
      }, 3000);
    }
  }, [beforeImage, afterImage, t]);

  const isProcessing = state === 'generating' || state === 'sharing';

  return (
    <>
      {/* メインボタン */}
      <button
        type="button"
        onClick={openShareTypeDialog}
        disabled={isProcessing}
        data-testid={testId}
        className={`
          w-full py-3 px-4 rounded-xl font-medium text-sm
          flex items-center justify-center gap-2
          transition-all duration-200
          focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-400
          ${
            isProcessing
              ? 'bg-primary-100 text-primary-400 cursor-not-allowed'
              : state === 'success'
              ? 'bg-primary-600 text-white'
              : state === 'error'
              ? 'bg-red-400 text-white'
              : 'bg-primary-600 text-white hover:bg-primary-700 active:scale-[0.98]'
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
            {state === 'generating' ? t('snsShare.generating') : t('snsShare.sharing')}
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
            {t('snsShare.shareButton')}
          </>
        )}
      </button>

      {/* シェアタイプ選択ダイアログ */}
      {state === 'selecting' && (
        <div
          className="fixed inset-0 bg-black/50 z-50 flex items-end sm:items-center justify-center"
          onClick={closeDialog}
        >
          <div
            className="bg-white w-full sm:max-w-md sm:rounded-2xl rounded-t-2xl p-6 animate-slide-up"
            onClick={(e) => e.stopPropagation()}
          >
            <h3 className="text-lg font-bold text-gray-900 mb-2">
              {t('snsShare.selectTitle')}
            </h3>
            <p className="text-sm text-gray-500 mb-6">
              {t('snsShare.selectDescription')}
            </p>

            <div className="space-y-3">
              {/* Before/After比較オプション */}
              <button
                type="button"
                onClick={() => handleShare('before_after')}
                data-testid="share-type-before-after"
                className="w-full p-4 rounded-xl border-2 border-gray-200 hover:border-primary-500 hover:bg-primary-50 transition-all text-left group"
              >
                <div className="flex items-center gap-4">
                  {/* プレビューアイコン */}
                  <div className="flex-shrink-0 w-16 h-10 bg-gray-100 rounded-lg flex items-center justify-center gap-1">
                    <div className="w-4 h-6 bg-gray-300 rounded" />
                    <svg className="w-3 h-3 text-gray-400" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M10.293 3.293a1 1 0 011.414 0l6 6a1 1 0 010 1.414l-6 6a1 1 0 01-1.414-1.414L14.586 11H3a1 1 0 110-2h11.586l-4.293-4.293a1 1 0 010-1.414z" clipRule="evenodd" />
                    </svg>
                    <div className="w-4 h-6 bg-primary-300 rounded" />
                  </div>
                  <div className="flex-1">
                    <div className="font-medium text-gray-900 group-hover:text-primary-700">
                      {t('snsShare.beforeAfter')}
                    </div>
                    <div className="text-sm text-gray-500">
                      {t('snsShare.beforeAfterDesc')}
                    </div>
                  </div>
                </div>
              </button>

              {/* 結果のみオプション */}
              <button
                type="button"
                onClick={() => handleShare('result_only')}
                data-testid="share-type-result-only"
                className="w-full p-4 rounded-xl border-2 border-gray-200 hover:border-primary-500 hover:bg-primary-50 transition-all text-left group"
              >
                <div className="flex items-center gap-4">
                  {/* プレビューアイコン */}
                  <div className="flex-shrink-0 w-16 h-10 bg-gray-100 rounded-lg flex items-center justify-center">
                    <div className="w-6 h-6 bg-primary-300 rounded" />
                  </div>
                  <div className="flex-1">
                    <div className="font-medium text-gray-900 group-hover:text-primary-700">
                      {t('snsShare.resultOnly')}
                    </div>
                    <div className="text-sm text-gray-500">
                      {t('snsShare.resultOnlyDesc')}
                    </div>
                  </div>
                </div>
              </button>
            </div>

            {/* キャンセルボタン */}
            <button
              type="button"
              onClick={closeDialog}
              className="w-full mt-4 py-3 text-sm font-medium text-gray-500 hover:text-gray-700 transition-colors"
            >
              {t('common.cancel')}
            </button>
          </div>
        </div>
      )}

      <style jsx>{`
        @keyframes slide-up {
          from {
            transform: translateY(100%);
            opacity: 0;
          }
          to {
            transform: translateY(0);
            opacity: 1;
          }
        }
        .animate-slide-up {
          animation: slide-up 0.2s ease-out;
        }
      `}</style>
    </>
  );
}
