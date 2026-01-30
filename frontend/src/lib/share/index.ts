/**
 * シェア機能モジュール
 */

export { generateShareImage } from './imageGenerator';
export type { ShareImageType } from './imageGenerator';
export { shareImage, isWebShareSupported, isFileShareSupported } from './webShare';
export type { ShareResult } from './webShare';
export { getShareText, getShareUrl } from './textGenerator';
