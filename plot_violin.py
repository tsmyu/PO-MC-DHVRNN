import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def create_violin_plot():
    """
    angle_diffs.csvを読み込み、diff_measuredとdiff_predictedの
    バイオリンプロットを作成して保存します。
    """
    # --- 設定項目 ---
    # angle_diffs.csvが保存されているディレクトリパス
    # result_topview_direction.pyの`path`変数と同じ値を設定してください。
    result_dir = r"C:\Users\yota-\OneDrive - 同志社大学\PO-MC-DHVRNN_aoki\result\20251119_pd_400epoch"
    
    # --- 処理開始 ---
    csv_path = os.path.join(result_dir, 'angle_diffs.csv')
    output_path = os.path.join(result_dir, 'angle_diffs_split_violin_plot.png')

    # CSVファイルの読み込み
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません -> {csv_path}")
        print("先に result_topview_direction.py を実行して angle_diffs.csv を生成してください。")
        return

    # プロット用にデータを整形 (ワイドフォーマットからロングフォーマットへ)
    # 'diff_measured'と'diff_predicted'の列を1つの列にまとめる
    df_melted = pd.melt(df, value_vars=['diff_measured', 'diff_predicted'], var_name='Type', value_name='Angle Difference (degrees)')

    # 凡例のラベルを変更
    df_melted['Type'] = df_melted['Type'].replace({
        'diff_measured': '実測放射方向 vs. 進行方向',
        'diff_predicted': '予測放射方向 vs. 進行方向'
    })

    # バイオリンプロットの作成
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 6))

    # split=Trueで左右に分かれたバイオリンプロットを作成
    # x軸はカテゴリ分けに不要なため、空の文字列を渡して1つにまとめる
    sns.violinplot(x=pd.Series([""]*len(df_melted)), y='Angle Difference (degrees)', hue='Type', data=df_melted, split=True, inner="quartile", ax=ax, palette="pastel")

    # 個々のデータ点をstripplotで重ねて描画
    sns.stripplot(x=pd.Series([""]*len(df_melted)), y='Angle Difference (degrees)', hue='Type', data=df_melted, 
                  dodge=True, jitter=0.05, size=2, ax=ax, palette=['.3', '.3'])

    # グラフのタイトルとラベルを設定
    ax.set_title('Comparison of Angle Differences', fontsize=16)
    ax.set_xlabel('')
    ax.set_ylabel('Angle Difference (degrees)', fontsize=12)

    # Y軸の範囲を指定
    ax.set_ylim(-10, 100)

    # 凡例を1つにまとめる
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2])

    # グラフを保存して表示
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"バイオリンプロットを {output_path} に保存しました。")
    # グラフを表示
    plt.show()

if __name__ == '__main__':
    create_violin_plot()
