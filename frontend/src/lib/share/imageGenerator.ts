/**
 * シェア画像生成（Canvas API）
 *
 * クライアントサイドで Before/After 比較画像を生成
 * URL埋め込みでブランド認知
 */

const CANVAS_WIDTH = 1200;
const CANVAS_HEIGHT = 630;

/**
 * Base64画像をImageElementとして読み込む
 */
function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = reject;
    // data:URLでない場合はそのまま、ある場合もそのまま
    img.src = src.startsWith('data:') ? src : src;
  });
}

/**
 * 画像を中央クロップして描画
 */
function drawCroppedImage(
  ctx: CanvasRenderingContext2D,
  img: HTMLImageElement,
  x: number,
  y: number,
  width: number,
  height: number
) {
  const imgAspect = img.width / img.height;
  const targetAspect = width / height;

  let srcX = 0;
  let srcY = 0;
  let srcW = img.width;
  let srcH = img.height;

  if (imgAspect > targetAspect) {
    // 画像が横長 → 左右をクロップ
    srcW = img.height * targetAspect;
    srcX = (img.width - srcW) / 2;
  } else {
    // 画像が縦長 → 上下をクロップ
    srcH = img.width / targetAspect;
    srcY = (img.height - srcH) / 2;
  }

  ctx.drawImage(img, srcX, srcY, srcW, srcH, x, y, width, height);
}

/**
 * 角丸の矩形パスを作成
 */
function roundedRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
  radius: number
) {
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.lineTo(x + width - radius, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
  ctx.lineTo(x + width, y + height - radius);
  ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
  ctx.lineTo(x + radius, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
  ctx.lineTo(x, y + radius);
  ctx.quadraticCurveTo(x, y, x + radius, y);
  ctx.closePath();
}

/**
 * シェア画像を生成
 *
 * @param beforeImage - 変更前画像（base64またはURL）
 * @param afterImage - 変更後画像（base64またはURL）
 * @returns PNG形式のBlob
 */
export async function generateShareImage(
  beforeImage: string,
  afterImage: string
): Promise<Blob> {
  const canvas = document.createElement('canvas');
  canvas.width = CANVAS_WIDTH;
  canvas.height = CANVAS_HEIGHT;
  const ctx = canvas.getContext('2d');

  if (!ctx) {
    throw new Error('Canvas context not available');
  }

  // 背景（グラデーション）
  const gradient = ctx.createLinearGradient(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
  gradient.addColorStop(0, '#FAFAFA');
  gradient.addColorStop(1, '#F0F0F0');
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

  // 画像を読み込み
  const [beforeImg, afterImg] = await Promise.all([
    loadImage(beforeImage),
    loadImage(afterImage),
  ]);

  // 画像サイズ・位置
  const imageSize = 380;
  const imageY = 60;
  const beforeX = 120;
  const afterX = 700;
  const radius = 16;

  // Before画像（角丸クリッピング）
  ctx.save();
  roundedRect(ctx, beforeX, imageY, imageSize, imageSize, radius);
  ctx.clip();
  drawCroppedImage(ctx, beforeImg, beforeX, imageY, imageSize, imageSize);
  ctx.restore();

  // Before画像の枠線
  ctx.strokeStyle = '#E5E5E5';
  ctx.lineWidth = 2;
  roundedRect(ctx, beforeX, imageY, imageSize, imageSize, radius);
  ctx.stroke();

  // After画像（角丸クリッピング）
  ctx.save();
  roundedRect(ctx, afterX, imageY, imageSize, imageSize, radius);
  ctx.clip();
  drawCroppedImage(ctx, afterImg, afterX, imageY, imageSize, imageSize);
  ctx.restore();

  // After画像の枠線
  ctx.strokeStyle = '#E5E5E5';
  ctx.lineWidth = 2;
  roundedRect(ctx, afterX, imageY, imageSize, imageSize, radius);
  ctx.stroke();

  // 矢印
  const arrowX = 560;
  const arrowY = imageY + imageSize / 2;
  ctx.fillStyle = '#9CA3AF';
  ctx.font = 'bold 64px Arial, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText('→', arrowX, arrowY);

  // ラベル
  ctx.fillStyle = '#6B7280';
  ctx.font = '500 20px Arial, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('Before', beforeX + imageSize / 2, imageY + imageSize + 30);
  ctx.fillText('After', afterX + imageSize / 2, imageY + imageSize + 30);

  // ブランドロゴ + URL（下部）
  const brandY = 560;

  // ロゴテキスト
  ctx.fillStyle = '#374151';
  ctx.font = 'bold 28px Arial, sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText('Cao', 120, brandY);

  // サブテキスト
  ctx.fillStyle = '#6B7280';
  ctx.font = '20px Arial, sans-serif';
  ctx.fillText('- 美容シミュレーション', 175, brandY);

  // URL（右寄せ）
  ctx.fillStyle = '#9CA3AF';
  ctx.font = '18px Arial, sans-serif';
  ctx.textAlign = 'right';
  ctx.fillText('cao-staging.style-elements.jp', CANVAS_WIDTH - 120, brandY);

  // Blobとして返す
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => {
        if (blob) {
          resolve(blob);
        } else {
          reject(new Error('Failed to create blob'));
        }
      },
      'image/png',
      0.9
    );
  });
}
