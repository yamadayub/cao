# Multi-Agent TDD 開発設定

## 開発方針
- **テスト駆動開発（TDD）を厳守**
- 要件書 → テスト → 実装 の順序を必ず守る
- E2Eテストが通るまで実装を継続
- 仕様変更時は必ず要件書から更新する

---

## サブエージェント構成

### Spec Agent (仕様策定)
- **役割**: 業務仕様書・機能要件書の作成・更新
- **トリガー**: 新機能追加、仕様変更時
- **出力**: `/docs/business-spec.md`, `/docs/functional-spec.md`
- **Task指示例**:
```
以下の要件を元に業務仕様書と機能要件書を更新してください。
- 変更内容: [具体的な変更内容]
- 参照: /docs/business-spec.md, /docs/functional-spec.md
```

### Test Agent (テスト設計)
- **役割**: テストケース設計、テストコード生成
- **トリガー**: 要件書更新後
- **参照**: `/docs/functional-spec.md`
- **出力**: 
  - 単体テスト: `/tests/unit/*.test.ts`
  - 結合テスト: `/tests/integration/*.int.ts`
  - E2Eシナリオ: `/tests/e2e/specs/*.spec.md`
- **Task指示例**:
```
以下の機能要件に基づいてテストを作成してください。
- 参照: /docs/functional-spec.md の [該当セクション]
- 出力: 単体テスト、結合テスト、E2Eシナリオ
```

### Impl Agent (実装)
- **役割**: テストを満たすソースコード実装
- **トリガー**: テストコード生成後
- **終了条件**: 全テストがパス
- **Task指示例**:
```
以下のテストが通るように実装してください。
- テストファイル: /tests/unit/[feature].test.ts
- 終了条件: pnpm test:unit が全てパス
```

---

## Task Tool 使用ルール

### 新機能開発フロー
```
1. Spec Agent起動 (Task)
   コンテキスト: 機能要件
   出力: 仕様書

2. Test Agent起動 (Task)
   コンテキスト: 仕様書
   出力: テストコード

3. Impl Agent起動 (Task)
   コンテキスト: テストコード
   終了条件: ローカルテストパス

4. git push → GitHub Actions
   E2E失敗時: agent-browserでデバッグ
```

### バグ修正フロー
```
1. 失敗しているテストを特定
2. Test Agent: テストが正しいか確認・修正
3. Impl Agent: テストが通るまで実装修正
4. E2E確認
```

---

## E2Eテスト戦略

### agent-browser (AI探索・デバッグ用)
- 新規E2Eシナリオの探索・設計
- テスト失敗時のデバッグ
- Markdownシナリオの実行確認

### Playwright (CI/CD自動実行用)
- 確定したテストの高速実行
- 並列ブラウザテスト
- Trace/Video記録

---

## コマンド一覧

### テスト
```bash
pnpm test:unit        # 単体テスト (Vitest)
pnpm test:integration # 結合テスト (Vitest)
pnpm test:e2e         # E2Eテスト (Playwright)
pnpm test             # 全テスト実行
```

### E2E (agent-browser)
```bash
agent-browser open <url>      # ブラウザ起動
agent-browser snapshot        # 要素一覧取得
agent-browser click @e1       # 要素クリック
agent-browser fill @e2 "text" # 入力
agent-browser screenshot      # スクリーンショット
```

---

## ファイル構成
```
/
├── CLAUDE.md                 # この設定ファイル
├── .claude/skills/           # Claude Codeスキル
├── docs/                     # 仕様書
│   ├── business-spec.md      # 業務仕様書
│   └── functional-spec.md    # 機能要件書
├── src/                      # ソースコード
├── tests/
│   ├── unit/                 # 単体テスト
│   ├── integration/          # 結合テスト
│   └── e2e/
│       ├── specs/            # E2Eシナリオ (Markdown)
│       ├── playwright/       # Playwrightテスト
│       └── pages/            # Page Objects
└── .github/workflows/        # CI/CD設定
```
