# Praat経路への歌唱用整形の追加

追記: その後に残った歯擦音の刺さりには、[歯擦音抑制](praat-deessing.md)を追加した。以下の比較は抑制追加前の記録。

2026-09-20。原音保持経路で自然さが失われ、子音だけの修正では改善しなかったとの報告を受け、旧経路の母音持続・音量調整・接続の重なりをPraatにも追加した。

## 使い方

通常のコマンドで新しい処理が有効になる。明示する場合:

```powershell
uv run --no-sync --offline --cache-dir .uv-cache madgen render --db work/corpus_sample.sqlite --ust target/iwashi_madgen.ust --lyrics-renderer source --lyrics-source-style sustain --out-dir work/praat_singing
```

`--no-sync`は既存の依存環境を保ったまま使うための指定。新しいシステム依存・Python依存は追加していない。

| 設定 | 動作 |
| --- | --- |
| `--lyrics-renderer source --lyrics-source-style sustain` | 新しい既定。Praatで母音を持続させ、音量を揃え、連続音の末尾を重ねる |
| `--lyrics-renderer source --lyrics-source-style preserve` | 以前の原音保持経路。全区間・強弱を残す。子音の回帰修正も含む |
| `--lyrics-renderer world` | 旧WORLDの選択・合成経路。source-styleは使わない |

`--no-lyrics-stretch`を指定すると、コア抽出と伸縮はしない。sustainの音量調整と、素材に余裕がある場合の連続音への接続尾は残る。

## 実装

[source_synth.py](../src/madgen/source_synth.py)、[match.py](../src/madgen/match.py)、[render.py](../src/madgen/render.py)、[cli.py](../src/madgen/cli.py)。

### 1. 母音の有声コア

母音・撥音は共有解析のF0と音量から、区間内の最大音量から12 dB以内で有声判定された最長の連続部分を選ぶ。最低4フレーム（約20ms）必要とする。
選択時に使用範囲を確定し、この範囲のF0・伸縮量で採点する。合成側で使用範囲を切り直したりF0を推定し直したりしない。

前後の静かな減衰・息・無音を母音全長へ伸ばすのを避け、選んだコアをPraatのoverlap-addで音符長へ合わせる。コアが見つからない場合は元区間を維持し、存在しない有声音を生成しない。
素材の入口・出口も残したい用途にはpreserveを使う。

### 2. 母音と子音の音量

母音は20ms窓・約5ms間隔のエネルギーから緩やかな変動を補正する。局所ゲインは±12 dB以内で滑らかに補間し、ピッチ周期そのものへ追従しない。
その後、母音・撥音の目標RMSを-18 dBFS、子音を-24 dBFSとし、さらに既存のvelocity/127を反映する。USTのvelocity=100では、それぞれ約-20.08 / -26.08 dBFSとなる。フレーズのゲイン・EQ・ミックスのマスター調整はこの後に適用される。

無音を大きく増幅しないよう、最終の増幅は最大24 dBとし、RMSがほぼ0ならそのままにする。極端に静かな素材は目標音量に届かないことがある。
無声子音は原波形にゲインとフェードだけを適用し、移調やPraat伸縮は行わない。

この処理はsustainの明示的な合成方針。`target_rms_db`・`level_sustain`を計画に保存する。最終倍率は合成後の実波形から求めるため、計画の`gain`だけが全ゲインを表すわけではない。

### 3. 接続の重なり

楽譜上のサンプル位置が次の音素と正確に接するときだけ、最大20msの末尾を計画する。次の音素が短い場合はその長さの半分まで。
母音は重なりを含む長さへ合成し、無声子音は元区間内に存在する分だけをコピーする。短い子音を伸ばして尾を捏造することはしない。
出る音は末尾全体をフェードし、入る音は5msで立ち上げる。これは重ね合わせであり、両者のフェードを合わせて常に振幅1にする方式ではない。

休符・フレーズ終端には末尾を延ばさない。`overlap_samples`を含む出力長・時間対応表を選択前に決め、そのまま合成する。
接続尾は楽譜上の開始時刻を基準とする。短い無声子音を右寄せした場合、実波形同士が重なる長さは短くなることがある。
sustainは単位ごとの音量処理と重なりを使うので、preserveの連続素材一括合成は行わない。

### 4. 計画と互換性

JSONに`core_selected`、`target_rms_db`、`level_sustain`、`overlap_samples`、`render_plan_version=2`を追加した。
音声・F0解析キャッシュは変更せず再利用する。`source_start/end`は実使用コア、`output_length`は接続尾を含む長さ、`note_duration_sec`は元の楽譜長を表す。
共有F0・使用範囲・時間対応という設計は維持し、歌唱向けの整形を追加した。

## 旧素材固定の比較

[比較スクリプト](research/compare_praat_sustain.py)は、旧WORLDの`work/iwashi_lyrics_sample/plan.json`から素材IDを取り、同じ素材のまま処理を段階的に追加する。タイミングはUSTを再読込し、古いJSONの小数丸めを引き継がない。既存出力を上書きせず、DBは読み取り専用で開く。

```powershell
uv run --no-sync --offline --cache-dir .uv-cache python docs/research/compare_praat_sustain.py
```

出力: `work/praat-sustain-18-30/`。各WAVは18〜30秒の12秒間、ボーカルのみ。

| ファイル | 内容 | 母音RMSの10–90パーセンタイル |
| --- | --- | --- |
| `00-world-existing.wav` | 以前の実出力 | 前回調査で約-18.0〜-17.8 dBFS |
| `01-praat-preserve.wav` | 旧選択素材をPraatの原音保持処理へ入力 | -35.03〜-27.82 dBFS |
| `02-praat-core.wav` | 有声コア抽出を追加 | -29.28〜-23.59 dBFS |
| `03-praat-core-level.wav` | さらに音量調整を追加 | -18.12〜-18.04 dBFS |
| `04-praat-core-level-overlap.wav` | さらに接続の重なりを追加 | -18.01〜-17.74 dBFS |

全Praat比較で素材IDは同じ。旧WORLDのジョブを別途再現して旧出力のマスター倍率を推定し、全Praat比較へ共通適用した。比較ごとに別々のピーク正規化はしていない。
Praat比較の全音声は有限値のみで、ピーク1.0未満を確認した。計画全文・共通倍率・測定値は`comparison.json`に保存。
リポジトリには[測定値の要約](research/praat-singing-results.json)も保存した。

音量の安定化は確認できたが、音質・聞き取り・自然さの優劣はこれだけでは確定しない。各中間音声は、コア・音量・重なりのどこで聞こえ方が変わるかを判断するためのもの。

## 検証と限界

伴奏付き全曲も、以前と同じ入力・トラック指定で`work/iwashi_lyrics_sample/praat_sustain/`へ出力した。`mix.wav`、`vocals.wav`、`accompaniment.wav`、各パート、計画を保存し、元の出力は維持。再選択を含む通常のCLI出力である。
約171秒・歌詞857ユニット、うちコア選択505、接続尾759。全音声が有限値で、ミックスのピークは約0.891（-1 dBFS）。USTユニット外の歌声は完全無音だった。歌声のactive loudnessは約-20.7 dB、伴奏調整は-11.8 dBとなり、旧出力のログ値と一致した。
`mix-18-30.wav` / `vocals-18-30.wav`に問題区間を切り出し、`verification.json`に確認値を保存した。

全117テスト・Ruff・`git diff --check`を通過。追加テストでは、有声コアからの持続、独立したFFTによる音程、母音と子音の6 dB差、緩やかな音量変動の低減、連続音だけの重なり、休符の完全無音、伸縮無効時の不足区間を確認した。WORLD合成関数を呼ぶと失敗する条件でもPraat持続テストを通している。

- WORLDの非周期性上限制限や、無声F0の補間による有声化は移植していない。Praatでは波形とパルスを使うため、同じパラメータ操作にはならない。
- 新CLIで全曲を実行すると素材も再選択される。ここに示した旧素材固定の比較とは条件が異なる。
- 固定60msの子音配分、USTのピッチ曲線、CV/VC単位の可変長選択、促音の高度な処理は今回の範囲外。
- 原音の強弱をそのまま残す目的と、異なる素材を均一な歌唱へ整える目的を、sustain/preserveで選べるようにした。
