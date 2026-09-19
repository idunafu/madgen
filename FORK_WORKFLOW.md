# このフォークの開発手順

このフォークの `main` は自分用の開発・利用の本体です。上流に出さない改善もここへ取り込みます。上流への PR は `upstream/main` から別ブランチを作り、必要な変更だけを切り出します。

## remote と作業場所

| 対象 | 用途 |
| --- | --- |
| `origin` (`idunafu/madgen`) | 自分のフォーク。push 先 |
| `upstream` (`0266st/madgen`) | 上流。fetch と PR の提出先 |
| `main` | 自分用の改善をまとめる。上流に合わせて reset しない |
| `codex/<topic>` | 上流へ提出する変更。PR ごとに worktree を持つ |

普段の作業フォルダは `main` のまま使います。PR の修正は別の worktree で行います。現在の対応は `git worktree list` と `git branch -vv` で確認できます。1つの PR に対してレビュー専用の別ブランチを増やさず、その PR のブランチへ修正を追加します。

## 新しい PR を切り出す

自分の `main` では、できる範囲で目的ごとにコミットを分けておきます。以下の `<topic>` と `<commit>` は実際の名前・コミットに置き換えます。

```powershell
git fetch upstream
git worktree add --no-track -b codex/<topic> ../madgen-pr-<topic> upstream/main
cd ../madgen-pr-<topic>
git cherry-pick <commit>
# 自分用の設定や他の変更への依存を整理し、CONTRIBUTING.md に沿って確認する
git push -u origin codex/<topic>
gh pr create --repo 0266st/madgen --base main --head idunafu:codex/<topic>
```

PR ブランチには自分用の `main` を merge しません。PR に別の独自改善が混ざるためです。必要な依存変更だけを明示的に切り出します。

## PR のレビュー修正を自分用 main にも反映する

レビュー修正は PR の worktree でコミットし、`origin` の同じブランチに push します。共同編集者の更新があれば、先に `git pull --ff-only` で取り込みます。

PR が自分用 `main` に取り込み済みなら、普段の作業フォルダで以下を実行します。未コミットの変更がある場合は先にコミットまたは退避します。

```powershell
git fetch origin
git merge --no-ff origin/codex/<topic>
# テスト・差分確認後
git push origin main
```

これは PR のコミットを自分用 `main` に取り込む操作です。上流の PR をマージ・クローズする操作ではありません。PR の履歴を途中で rebase / force-push した場合は、取り込み済みの履歴との差分を確認してから統合します。

## 上流の更新を取り込む

普段の作業フォルダで、未コミット変更を片付けてから実行します。

```powershell
git fetch upstream
git merge upstream/main
# 競合解決・テスト後
git push origin main
```

上流が squash merge した変更は、自分用 main とコミット ID が異なります。競合があれば変更内容を比較して解決します。自分用 main 全体の rebase や、上流への強制的な一致は行いません。

PR が上流にマージされ、必要な変更が自分用 main にも残っていることを確認したら、PR の worktree とブランチを削除します。未コミット・未 push の変更がないことを先に確認し、削除で force オプションが必要になった場合は理由を調べます。

## Python 環境

仮想環境は worktree ごとに独立しています。PR 作業のために普段使いの `.venv` を切り替える必要はありません。

- GPU を使う環境では `uv sync --extra lyrics --extra cu128`、実行も `uv run --extra lyrics --extra cu128 madgen ...` を使います。
- CPU 用は `cu128` の代わりに `cpu` を指定します。
- 既存環境を変更せずに検証するときは `uv run --no-sync pytest` / `uv run --no-sync ruff check` を使います。この検証は新規環境での依存解決成功を保証しません。
- パッケージの取得先は既存の設定に従います。解決に失敗しても index を無断で切り替えません。
