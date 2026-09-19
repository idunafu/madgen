# 歌詞合成の調査記録

2026-09-20: Praatエンジンの開発をmainから分離し、実装は `codex/praat` ブランチに保存した。この調査資料と測定結果・再現スクリプトはmainと `codex/praat` の両方に残す。

各報告にある「既定」「実装済み」やPraat用CLIオプションは実験時点の実装を指す。mainの歌詞合成は従来のWORLD経路であり、Praat依存やオプションは追加していない。Praatを使う再現スクリプトは `codex/praat` とローカルの入力・解析キャッシュが必要。報告内のPraat実装への相対リンクも同ブランチで参照する。

- [最新の評価と分岐の理由](praat-world-assessment.md): 息・滑舌の問題と、WORLDを比較基準にする判断。
- [mainのWORLD改善](world-improvements.md): 実使用コアの共有、子音調整、従来WORLDとの比較。
- [既存実装の照合・改善点](lyrics-implementation-review.md)
- [当初の設計調査](lyrics-rendering-design.md)
- [原音保持経路の実装・検証](source-renderer-implementation.md)
- [子音の回帰調査](consonant-regression.md)
- [Praatへの歌唱用整形の追加](praat-singing-processing.md)
- [歯擦音抑制の実装・比較](praat-deessing.md)

`research/` には測定値の要約と再現スクリプトを置く。比較音声やコーパスなどの大きなローカル成果物は `work/` に残し、Gitには含めない。
