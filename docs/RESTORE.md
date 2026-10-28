# バックアップと復元の手順

## 何がどこにあるか

- 毎日の実行と週末の研究は、`data/` を GitHub Actions のキャッシュで引き継いでいます。
- キャッシュは**7日間使われないと消えます**。
- 取り直せないデータは、毎日の実行が成功したあとに、private のリポジトリ **dakiishu-boop/kabuscout-backup** に控えています（`kabuscout/backup.py`）。
  - 控えているもの:
    - kabuscout.db（ペーパートレード・通知・モデルの記録）
    - scored/（全候補の予測）
    - flows/（信用残・空売り・品貸料）
    - tdnet_xbrl/（決算短信の数値）
    - jquants/
    - strategy_params.json（研究ループが採用した設定・影の候補）
    - 小さい結果の JSON
  - 株価・シグナル表・モデルのファイルは取り直せるので控えていません。
  - J-Quants 由来のファイルがあるので、このリポジトリは**公開しないでください**。
- 控えは毎回1つのコミットにまとめ、次のブランチに上書きで置きます。ある日の控えが壊れていても、別の枠に前の分が残ります。

  | ブランチ | 中身 |
  |---|---|
  | `latest` | 最新 |
  | `daily-mon`〜`daily-fri` | 曜日ごと（約1週間分） |
  | `weekly-0`〜`weekly-3` | 週ごと（約1ヶ月分） |
  | `monthly-01`〜`monthly-12` | 月ごと（約1年分） |

- **試し戻し**: 毎月1日に `backup-drill` ワークフローが動きます。`latest` を一時フォルダに戻して、次の点を確かめ、結果を Discord に送ります。
  - 指紋（sha256）
  - DB が壊れていないか（integrity_check）
  - 行数
  - parquet が読めるか

  手動でも実行できます。Actions → backup-drill → Run workflow を押し、`backup_branch` で確かめる枠を選びます。

## キャッシュが消えたとき（毎日の実行が「Cache not found」で失敗し続ける）

失敗し続けるのは正しい動きです（空から作り直して控えを上書きしないため）。次の順に戻します。

1. GitHub の **Actions → weekly-research → Run workflow** を開きます。
2. 次のように設定して実行します。
   - **command**: `prices`
   - **restore_from_backup**: チェックを入れる
   - **backup_branch**: 通常は `latest`。壊れていたら `daily-…` などの前の枠
3. この実行が行うこと（6時間の上限があるのは研究用のワークフローだけなので、株価の全期間の取り直しはこちらで行います）:
   1. キャッシュが無いことを確かめる（キャッシュがある日は戻しません）
   2. 控えを確かめてから `data/` に戻す
   3. 株価を全期間取り直す
   4. 成功したらキャッシュと控えを保存する
4. 次の平日の毎日の実行から元に戻ります。
   - シグナル表とモデルは、最初の実行で自動的に作り直されます。
   - ペーパートレードの記録・全候補の記録・需給データは控えから戻っています。

手元の Mac で戻す場合は、次のコマンドを使います（`data/` にペーパートレードの記録があるときは `--force` が必要）。

```bash
git clone git@github.com:dakiishu-boop/kabuscout-backup.git -b latest /tmp/kabuscout-backup
```

```bash
python -m kabuscout restore --from /tmp/kabuscout-backup
```

## 定時実行が止まったとき（60日間 push が無い）

- 公開リポジトリは、60日間動きが無いと GitHub が定時実行を止めます。
- 最後の push から45日を過ぎると、毎日の実行が Discord で知らせます。そのときは何か1つ commit して push してください。
- 止まってしまったら、次の順に戻します。
  1. Actions の各ワークフローで「Enable workflow」を押します。
  2. 止まって7日以上たっていたら、キャッシュも消えています。上の手順で戻します。

## 鍵

- 書き込みは、バックアップ用リポジトリだけに使える deploy key で行っています。
  - リポジトリ側: kabuscout-backup の Settings → Deploy keys の「kabuscout-actions」（書き込み可）
  - 本体側: Secrets の `BACKUP_DEPLOY_KEY`
- 鍵を作り直すときは、両方を入れ替えてください。
