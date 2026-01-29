/**
 * シェアテキスト生成
 */

type PartType = 'full' | 'eyes' | 'nose' | 'lips' | 'eyebrows' | 'mixed';

const PART_MESSAGES: Record<PartType, string> = {
  full: '理想の顔にしたらこんな感じ！',
  eyes: '目を変えたらこんな感じ！',
  nose: '鼻を変えたらこんな感じ！',
  lips: '唇を変えたらこんな感じ！',
  eyebrows: '眉を変えたらこんな感じ！',
  mixed: 'パーツを変えたらこんな感じ！',
};

/**
 * 適用パーツからパーツタイプを判定
 */
function getPartType(appliedParts: string[]): PartType {
  if (!appliedParts || appliedParts.length === 0) {
    return 'full';
  }

  // 全パーツ適用の場合
  const allParts = ['left_eye', 'right_eye', 'nose', 'lips', 'left_eyebrow', 'right_eyebrow'];
  if (allParts.every((part) => appliedParts.includes(part))) {
    return 'full';
  }

  // 単一カテゴリの場合
  const hasEyes = appliedParts.some((p) => p.includes('eye') && !p.includes('eyebrow'));
  const hasNose = appliedParts.includes('nose');
  const hasLips = appliedParts.includes('lips');
  const hasEyebrows = appliedParts.some((p) => p.includes('eyebrow'));

  const categories = [hasEyes, hasNose, hasLips, hasEyebrows].filter(Boolean);

  if (categories.length === 1) {
    if (hasEyes) return 'eyes';
    if (hasNose) return 'nose';
    if (hasLips) return 'lips';
    if (hasEyebrows) return 'eyebrows';
  }

  return 'mixed';
}

/**
 * シェアテキストを生成
 */
export function getShareText(appliedParts: string[] = []): string {
  const partType = getPartType(appliedParts);
  const message = PART_MESSAGES[partType];

  return `${message}

Caoで美容シミュレーション
https://cao-staging.style-elements.jp

#Cao #美容シミュレーション`;
}

/**
 * シェアURLを取得
 */
export function getShareUrl(): string {
  return 'https://cao-staging.style-elements.jp';
}
