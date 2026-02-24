'use client';

import { useState, useCallback, useRef } from 'react';
import { useTranslations } from 'next-intl';
import { generateShareImage, shareImage, ShareResult, ShareImageType } from '@/lib/share';
import { generateBlendVideo } from '@/lib/api/video';

interface ShareButtonProps {
  /** 変更前画像（base64） */
  beforeImage: string;
  /** 変更後画像（base64） */
  afterImage: string;
  /** 理想の顔画像（base64）— ブレンド動画用 */
  idealImage?: string;
  /** 認証トークン取得関数 — ブレンド動画用 */
  getAuthToken?: () => Promise<string | null>;
  /** ログイン済みかどうか */
  isSignedIn: boolean;
  /** ログインが必要な時に呼ばれるコールバック */
  onLoginRequired: () => void;
  /** 追加のクラス名 */
  className?: string;
  /** テスト用ID */
  testId?: string;
}

type ShareState = 'idle' | 'selecting' | 'generating' | 'sharing' | 'success' | 'error' | 'generating-video' | 'video-preview';

/**
 * 画像をロードしてHTMLImageElementを返す
 */
function loadImageElement(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}

/**
 * Ease-in-out補間（smooth step）
 */
function easeInOut(t: number): number {
  return t * t * (3 - 2 * t);
}

/**
 * Canvasにスライダーフレームを描画
 */
function drawSliderFrame(
  ctx: CanvasRenderingContext2D,
  beforeImg: HTMLImageElement,
  afterImg: HTMLImageElement,
  size: number,
  pos: number,
) {
  const splitX = pos * size;

  // Before画像（背景全体）
  ctx.drawImage(beforeImg, 0, 0, size, size);

  // After画像（左側クリップで徐々に表示）
  if (splitX > 0) {
    ctx.save();
    ctx.beginPath();
    ctx.rect(0, 0, splitX, size);
    ctx.clip();
    ctx.drawImage(afterImg, 0, 0, size, size);
    ctx.restore();
  }

  // スライダーライン
  if (pos > 0.01 && pos < 0.99) {
    ctx.save();
    ctx.shadowColor = 'rgba(0,0,0,0.4)';
    ctx.shadowBlur = 4;
    ctx.shadowOffsetX = 1;
    ctx.fillStyle = 'white';
    ctx.fillRect(splitX - 1.5, 0, 3, size);
    ctx.restore();

    // ハンドル（中央の円）
    const cy = size / 2;
    ctx.beginPath();
    ctx.arc(splitX, cy, size * 0.03, 0, Math.PI * 2);
    ctx.fillStyle = 'white';
    ctx.fill();
    ctx.strokeStyle = 'rgba(0,0,0,0.2)';
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  // Before/Afterラベル（左=After、右=Before）
  const labelSize = Math.max(12, size * 0.028);
  ctx.font = `600 ${labelSize}px sans-serif`;
  if (pos > 0.15) {
    const lx = 8;
    const ly = 8;
    const metrics = ctx.measureText('After');
    ctx.fillStyle = 'rgba(0,0,0,0.6)';
    ctx.fillRect(lx, ly, metrics.width + 12, labelSize + 8);
    ctx.fillStyle = 'white';
    ctx.fillText('After', lx + 6, ly + labelSize + 1);
  }
  if (pos < 0.85) {
    const label = 'Before';
    const metrics = ctx.measureText(label);
    const lx = size - metrics.width - 20;
    const ly = 8;
    ctx.fillStyle = 'rgba(0,0,0,0.6)';
    ctx.fillRect(lx, ly, metrics.width + 12, labelSize + 8);
    ctx.fillStyle = 'white';
    ctx.fillText(label, lx + 6, ly + labelSize + 1);
  }
}

/**
 * Canvasにクロスフェードフレームを描画
 */
function drawCrossfadeFrame(
  ctx: CanvasRenderingContext2D,
  beforeImg: HTMLImageElement,
  afterImg: HTMLImageElement,
  size: number,
  pos: number,
) {
  // Before画像（背景全体）
  ctx.globalAlpha = 1;
  ctx.drawImage(beforeImg, 0, 0, size, size);

  // After画像（透明度で徐々に表示）
  ctx.globalAlpha = pos;
  ctx.drawImage(afterImg, 0, 0, size, size);
  ctx.globalAlpha = 1;
}

/**
 * シェアボタン（タイプ選択ダイアログ付き）
 *
 * - シェアタイプ選択: Before/After比較、結果のみ、スライダー動画、モーフィング動画
 * - 動画: MediaRecorder APIでブラウザ内録画（サーバー不要）
 * - モバイル: Web Share APIでネイティブシェア
 * - デスクトップ: クリップボードにコピー
 */
export function ShareButton({
  beforeImage,
  afterImage,
  idealImage,
  getAuthToken,
  isSignedIn,
  onLoginRequired,
  className = '',
  testId,
}: ShareButtonProps) {
  const t = useTranslations('modals');
  const [state, setState] = useState<ShareState>('idle');
  const [message, setMessage] = useState('');
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

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
    // 動画のObject URLを解放
    if (videoUrl) {
      URL.revokeObjectURL(videoUrl);
    }
    setState('idle');
    setMessage('');
    setVideoUrl(null);
  }, [videoUrl]);

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

  /**
   * クライアント側で動画を生成
   * Canvas + MediaRecorder APIを使用（サーバー不要）
   * @param mode 'slider' | 'crossfade'
   */
  const handleVideoGenerate = useCallback(async (mode: 'slider' | 'crossfade') => {
    setState('generating-video');
    setMessage('');

    try {
      // 画像をロード
      const [beforeImg, afterImg] = await Promise.all([
        loadImageElement(beforeImage),
        loadImageElement(afterImage),
      ]);

      const SIZE = 540; // 動画解像度（540x540、SNS共有に十分）
      const canvas = document.createElement('canvas');
      canvas.width = SIZE;
      canvas.height = SIZE;
      const ctx = canvas.getContext('2d')!;

      // MIMEタイプを選択（Safari対応）
      const mimeType = MediaRecorder.isTypeSupported('video/webm;codecs=vp9')
        ? 'video/webm;codecs=vp9'
        : MediaRecorder.isTypeSupported('video/webm')
        ? 'video/webm'
        : 'video/mp4';

      // MediaRecorder開始
      const stream = canvas.captureStream(30);
      const recorder = new MediaRecorder(stream, { mimeType });
      const chunks: Blob[] = [];
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.push(e.data);
      };

      const videoPromise = new Promise<Blob>((resolve) => {
        recorder.onstop = () => {
          const baseMime = mimeType.split(';')[0];
          resolve(new Blob(chunks, { type: baseMime }));
        };
      });

      recorder.start();

      // リアルタイムでアニメーション描画（5秒 = 1ループ）
      const DURATION = 5000;
      await new Promise<void>((resolve) => {
        const startTime = performance.now();

        const draw = (timestamp: number) => {
          const elapsed = timestamp - startTime;
          if (elapsed >= DURATION) {
            // 最終フレーム描画
            const drawFn = mode === 'slider' ? drawSliderFrame : drawCrossfadeFrame;
            drawFn(ctx, beforeImg, afterImg, SIZE, 0);
            resolve();
            return;
          }

          // Timeline: Before(0.5s) → Slide(2s) → After(1.5s) → SlideBack(0.5s) → End(0.5s)
          const t = elapsed / DURATION;
          let pos: number;
          if (t < 0.10) pos = 0;
          else if (t < 0.50) pos = easeInOut((t - 0.10) / 0.40);
          else if (t < 0.80) pos = 1;
          else if (t < 0.90) pos = 1 - easeInOut((t - 0.80) / 0.10);
          else pos = 0;

          const drawFn = mode === 'slider' ? drawSliderFrame : drawCrossfadeFrame;
          drawFn(ctx, beforeImg, afterImg, SIZE, pos);
          requestAnimationFrame(draw);
        };

        requestAnimationFrame(draw);
      });

      recorder.stop();
      const videoBlob = await videoPromise;
      const url = URL.createObjectURL(videoBlob);
      setVideoUrl(url);
      setState('video-preview');
    } catch (error) {
      console.error('Video recording error:', error);
      setState('error');
      setMessage(t('snsShare.videoFailed'));

      setTimeout(() => {
        setState('idle');
        setMessage('');
      }, 3000);
    }
  }, [beforeImage, afterImage, t]);

  /**
   * 動画をWeb Share APIでシェア
   */
  const handleVideoShare = useCallback(async () => {
    if (!videoUrl) return;

    const response = await fetch(videoUrl);
    const blob = await response.blob();
    const ext = blob.type.includes('webm') ? 'webm' : 'mp4';
    const file = new File([blob], `cao-morph.${ext}`, { type: blob.type });

    try {
      if (navigator.share && navigator.canShare?.({ files: [file] })) {
        await navigator.share({
          files: [file],
          title: 'Cao - Before/After',
        });
        setState('success');
        setMessage(t('snsShare.shared'));
        setTimeout(() => {
          setState('idle');
          setMessage('');
          setVideoUrl(null);
        }, 3000);
      }
    } catch (e) {
      if ((e as Error).name === 'AbortError') {
        return;
      }
      console.error('Video share error:', e);
    }
  }, [videoUrl, t]);

  /**
   * 動画をダウンロード
   */
  const handleVideoDownload = useCallback(async () => {
    if (!videoUrl) return;

    const response = await fetch(videoUrl);
    const blob = await response.blob();
    const ext = blob.type.includes('webm') ? 'webm' : 'mp4';

    const link = document.createElement('a');
    link.href = videoUrl;
    link.download = `cao-morph.${ext}`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    setState('success');
    setMessage(t('snsShare.videoReady'));
    setTimeout(() => {
      setState('idle');
      setMessage('');
      setVideoUrl(null);
    }, 3000);
  }, [videoUrl, t]);

  /**
   * サーバー側でブレンドリビール動画を生成
   */
  const handleBlendVideoGenerate = useCallback(async () => {
    if (!idealImage || !getAuthToken) return;

    setState('generating-video');
    setMessage('');

    try {
      const token = await getAuthToken();
      if (!token) {
        setState('error');
        setMessage(t('snsShare.failed'));
        setTimeout(() => { setState('idle'); setMessage(''); }, 3000);
        return;
      }

      const result = await generateBlendVideo(beforeImage, idealImage, afterImage, token, 'flash');

      // サーバーから返されたURLがdata URIの場合はBlobに変換
      let blobUrl: string;
      if (result.video_url.startsWith('data:')) {
        const res = await fetch(result.video_url);
        const blob = await res.blob();
        blobUrl = URL.createObjectURL(blob);
      } else {
        // Supabase URL → fetch → blob
        const res = await fetch(result.video_url);
        const blob = await res.blob();
        blobUrl = URL.createObjectURL(blob);
      }

      setVideoUrl(blobUrl);
      setState('video-preview');
    } catch (error) {
      console.error('Blend video generation error:', error);
      setState('error');
      setMessage(t('snsShare.videoFailed'));
      setTimeout(() => { setState('idle'); setMessage(''); }, 3000);
    }
  }, [beforeImage, afterImage, idealImage, getAuthToken, t]);

  /**
   * Web Share APIでファイル共有がサポートされているかどうか
   */
  const canShareFiles = typeof navigator !== 'undefined' && !!navigator.share;

  const isProcessing = state === 'generating' || state === 'sharing' || state === 'generating-video';

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
            {state === 'generating-video' ? t('snsShare.generatingVideo') : state === 'generating' ? t('snsShare.generating') : t('snsShare.sharing')}
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

              {/* スライダー動画オプション */}
              <button
                type="button"
                onClick={() => handleVideoGenerate('slider')}
                data-testid="share-type-slider-video"
                className="w-full p-4 rounded-xl border-2 border-gray-200 hover:border-primary-500 hover:bg-primary-50 transition-all text-left group"
              >
                <div className="flex items-center gap-4">
                  <div className="flex-shrink-0 w-16 h-10 bg-gray-100 rounded-lg flex items-center justify-center gap-0.5">
                    <div className="w-4 h-6 bg-gray-300 rounded-sm" />
                    <div className="w-px h-7 bg-primary-400" />
                    <div className="w-4 h-6 bg-primary-300 rounded-sm" />
                  </div>
                  <div className="flex-1">
                    <div className="font-medium text-gray-900 group-hover:text-primary-700">
                      {t('snsShare.sliderVideo')}
                    </div>
                    <div className="text-sm text-gray-500">
                      {t('snsShare.sliderVideoDesc')}
                    </div>
                  </div>
                </div>
              </button>

              {/* モーフィング動画オプション */}
              <button
                type="button"
                onClick={() => handleVideoGenerate('crossfade')}
                data-testid="share-type-morph-video"
                className="w-full p-4 rounded-xl border-2 border-gray-200 hover:border-primary-500 hover:bg-primary-50 transition-all text-left group"
              >
                <div className="flex items-center gap-4">
                  <div className="flex-shrink-0 w-16 h-10 bg-gray-100 rounded-lg flex items-center justify-center">
                    <svg className="w-6 h-6 text-primary-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  </div>
                  <div className="flex-1">
                    <div className="font-medium text-gray-900 group-hover:text-primary-700">
                      {t('snsShare.morphVideo')}
                    </div>
                    <div className="text-sm text-gray-500">
                      {t('snsShare.morphVideoDesc')}
                    </div>
                  </div>
                </div>
              </button>

              {/* ブレンドリビール動画オプション（idealImage がある場合のみ表示） */}
              {idealImage && getAuthToken && (
                <button
                  type="button"
                  onClick={handleBlendVideoGenerate}
                  data-testid="share-type-blend-video"
                  className="w-full p-4 rounded-xl border-2 border-primary-200 bg-primary-50/50 hover:border-primary-500 hover:bg-primary-50 transition-all text-left group"
                >
                  <div className="flex items-center gap-4">
                    <div className="flex-shrink-0 w-16 h-10 bg-gradient-to-r from-gray-200 via-primary-200 to-primary-400 rounded-lg flex items-center justify-center">
                      <svg className="w-6 h-6 text-white drop-shadow" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z" />
                      </svg>
                    </div>
                    <div className="flex-1">
                      <div className="font-medium text-gray-900 group-hover:text-primary-700">
                        {t('snsShare.blendVideo')}
                      </div>
                      <div className="text-sm text-gray-500">
                        {t('snsShare.blendVideoDesc')}
                      </div>
                    </div>
                  </div>
                </button>
              )}
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

      {/* 動画生成中ダイアログ */}
      {state === 'generating-video' && (
        <div
          className="fixed inset-0 bg-black/50 z-50 flex items-end sm:items-center justify-center"
        >
          <div
            className="bg-white w-full sm:max-w-md sm:rounded-2xl rounded-t-2xl p-6 animate-slide-up"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex flex-col items-center py-8">
              <svg
                className="animate-spin h-10 w-10 text-primary-500 mb-4"
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
              <p className="text-gray-700 font-medium">{t('snsShare.generatingVideo')}</p>
              <p className="text-sm text-gray-400 mt-2">~5s</p>
            </div>
          </div>
        </div>
      )}

      {/* 動画プレビュー＆ダウンロードダイアログ */}
      {state === 'video-preview' && videoUrl && (
        <div
          className="fixed inset-0 bg-black/50 z-50 flex items-end sm:items-center justify-center"
          onClick={closeDialog}
        >
          <div
            className="bg-white w-full sm:max-w-md sm:rounded-2xl rounded-t-2xl p-6 animate-slide-up"
            onClick={(e) => e.stopPropagation()}
          >
            <h3 className="text-lg font-bold text-gray-900 mb-4">
              {t('snsShare.videoReady')}
            </h3>

            {/* 動画プレビュー */}
            <div className="relative w-full rounded-xl overflow-hidden bg-black mb-4" style={{ aspectRatio: '9/16', maxHeight: '400px' }}>
              <video
                ref={videoRef}
                src={videoUrl}
                autoPlay
                loop
                muted
                playsInline
                className="w-full h-full object-contain"
              />
            </div>

            {/* シェアボタン（Web Share API対応時） */}
            {canShareFiles && (
              <button
                type="button"
                onClick={handleVideoShare}
                className="w-full py-3 px-4 rounded-xl font-medium text-sm bg-primary-600 text-white hover:bg-primary-700 active:scale-[0.98] transition-all flex items-center justify-center gap-2"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.368 2.684 3 3 0 00-5.368-2.684z" />
                </svg>
                {t('snsShare.shareVideo')}
              </button>
            )}

            {/* 動画を保存ボタン */}
            <button
              type="button"
              onClick={handleVideoDownload}
              className={`w-full py-3 px-4 rounded-xl font-medium text-sm transition-all flex items-center justify-center gap-2 ${
                canShareFiles
                  ? 'mt-2 bg-white text-neutral-700 border border-neutral-300 hover:bg-neutral-50 hover:border-neutral-400'
                  : 'bg-primary-600 text-white hover:bg-primary-700 active:scale-[0.98]'
              }`}
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
              </svg>
              {t('snsShare.saveVideo')}
            </button>

            {/* キャンセルボタン */}
            <button
              type="button"
              onClick={closeDialog}
              className="w-full mt-3 py-3 text-sm font-medium text-gray-500 hover:text-gray-700 transition-colors"
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
