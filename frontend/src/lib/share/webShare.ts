/**
 * Web Share API ラッパー
 *
 * モバイル: ネイティブシェアシート
 * デスクトップ: クリップボードコピー
 */

import { getShareText, getShareUrl } from './textGenerator';

export type ShareResult = 'shared' | 'copied' | 'failed' | 'cancelled';

/**
 * 画像をクリップボードにコピー（フォールバック）
 */
async function copyImageToClipboard(imageBlob: Blob): Promise<boolean> {
  try {
    // ClipboardItem APIが使えるか確認
    if (typeof ClipboardItem !== 'undefined') {
      await navigator.clipboard.write([
        new ClipboardItem({
          'image/png': imageBlob,
        }),
      ]);
      return true;
    }
    return false;
  } catch (e) {
    console.error('Clipboard copy failed:', e);
    return false;
  }
}

/**
 * 画像をダウンロード（最終フォールバック）
 */
function downloadImage(imageBlob: Blob): void {
  const url = URL.createObjectURL(imageBlob);
  const link = document.createElement('a');
  link.href = url;
  link.download = 'cao-simulation.png';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

/**
 * Web Share APIで画像をシェア
 *
 * @param imageBlob - シェアする画像のBlob
 * @param appliedParts - 適用されたパーツ（テキスト生成用）
 * @returns シェア結果
 */
export async function shareImage(
  imageBlob: Blob,
  appliedParts: string[] = []
): Promise<ShareResult> {
  const shareText = getShareText(appliedParts);
  const shareUrl = getShareUrl();

  // 画像ファイルを作成
  const file = new File([imageBlob], 'cao-simulation.png', {
    type: 'image/png',
  });

  // Web Share APIが使えるか確認（files付き）
  if (
    typeof navigator !== 'undefined' &&
    navigator.canShare &&
    navigator.canShare({ files: [file] })
  ) {
    try {
      await navigator.share({
        text: shareText,
        url: shareUrl,
        files: [file],
      });
      return 'shared';
    } catch (e) {
      if (e instanceof Error && e.name === 'AbortError') {
        return 'cancelled';
      }
      console.error('Web Share API failed:', e);
      // フォールバックへ
    }
  }

  // フォールバック1: クリップボードにコピー
  const copied = await copyImageToClipboard(imageBlob);
  if (copied) {
    return 'copied';
  }

  // フォールバック2: ダウンロード
  downloadImage(imageBlob);
  return 'copied'; // ダウンロードもコピーと同様に扱う
}

/**
 * Web Share APIがサポートされているか確認
 */
export function isWebShareSupported(): boolean {
  return (
    typeof navigator !== 'undefined' &&
    typeof navigator.share === 'function' &&
    typeof navigator.canShare === 'function'
  );
}

/**
 * ファイル共有がサポートされているか確認
 */
export function isFileShareSupported(): boolean {
  if (!isWebShareSupported()) return false;

  try {
    const testFile = new File(['test'], 'test.png', { type: 'image/png' });
    return navigator.canShare({ files: [testFile] });
  } catch {
    return false;
  }
}
