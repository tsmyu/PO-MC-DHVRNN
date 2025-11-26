import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def create_violin_plot_by_env(csv_path, output_image_path, filter_range=None, selected_envs=None, combine_envs=False, y_range=None):
    """
    CSVファイルからデータを読み込み、ENVの値ごとにangle_difference_degの
    バイオリンプロットを作成します。

    Args:
        csv_path (str): 入力CSVファイルのパス。
        output_image_path (str): 出力画像ファイルのパス。
        filter_range (tuple, optional): 角度差をフィルタリングする範囲 (min, max)。
                                        Noneの場合はフィルタリングしません。
        selected_envs (list, optional): 描画対象のENVのリスト。
                                        Noneの場合は全てのENVを描画します。
        combine_envs (bool, optional): Trueの場合、selected_envsで選択した
                                       全てのデータを1つのプロットにまとめます。
        y_range (tuple, optional): Y軸の表示範囲 (min, max)。
                                   Noneの場合は自動で設定します。
    """
    try:
        # CSVファイルを読み込みます。
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません -> {csv_path}")
        return

    # データのフィルタリング
    plot_title = 'ENVごとの角度差（angle_diff）の分布'
    if filter_range:
        min_angle, max_angle = filter_range
        original_count = len(df)
        df = df[(df['angle_diff'] >= min_angle) & (df['angle_diff'] <= max_angle)].copy()
        filtered_count = len(df)
        print(f"データをフィルタリングしました ({min_angle}°～{max_angle}°の範囲)。")
        print(f"元のデータ数: {original_count}")
        print(f"フィルタ後のデータ数: {filtered_count} ({filtered_count / original_count:.2%})")
        plot_title = f'ENVごとの角度差の分布 ({min_angle}°～{max_angle}°の範囲)'
    else:
        print("フィルタリングなしで全データをプロットします。")

    # ENVのフィルタリング
    if selected_envs:
        print(f"指定されたENVでフィルタリングします: {selected_envs}")
        df = df[df['ENV'].isin(selected_envs)].copy()
        # 描画順はselected_envsの指定順に従う
        env_types_for_plot = [env for env in selected_envs if env in df['ENV'].unique()]
        env_str = ", ".join(map(str, env_types_for_plot))
        plot_title = f'ENV {env_str} の' + plot_title.split('の', 1)[1]
    else:
        # ENV列のユニークな値を取得し、ソートします。
        env_types_for_plot = sorted(df['ENV'].unique())
    print(f"グラフ化対象のENV: {env_types_for_plot}")

    # --- 統計情報の計算と出力 ---
    print("\n--- 統計情報 ---")
    # ENVごとに統計情報を計算して表示
    stats = df.groupby('ENV')['angle_diff'].agg(['mean', 'std', 'count'])
    print("ENVごとの角度差 (angle_diff):")
    print(stats)

    if combine_envs and selected_envs:
        # 結合データ全体の統計情報を計算して表示
        mean_val = df['angle_diff'].mean()
        std_val = df['angle_diff'].std()
        print(f"\n対象: ENV {env_str} の結合データ (データ数: {len(df)})")
        print(f"  平均値: {mean_val:.2f}")
        print(f"  標準偏差: {std_val:.2f}")
    print("------------------\n")

    # グラフのスタイルとフォント設定
    sns.set_theme(style="whitegrid")
    # ご利用の環境に合わせて日本語フォントを指定してください
    # Windows: 'Yu Gothic', 'Meiryo'
    # macOS: 'Hiragino Sans'
    # Linux: 'IPAexGothic' (要インストール)
    plt.rcParams['font.family'] = 'Yu Gothic' 
    plt.rcParams['axes.unicode_minus'] = False # マイナス記号の文字化け防止

    # グラフのサイズ設定
    plt.figure(figsize=(12, 8))

    if combine_envs and selected_envs:
        # 選択したENVを1つにまとめてプロット
        combined_label = f'ENVs {env_str}'
        ax = sns.violinplot(y='angle_diff', data=df, palette='muted', inner="quartile")
        sns.stripplot(y='angle_diff', data=df, color='black', jitter=0.2, size=2, alpha=0.3, ax=ax)
        ax.set_title(plot_title + 'と生データのプロット', fontsize=16)
        ax.set_xlabel(combined_label, fontsize=12)
        ax.set_xticks([]) # X軸の目盛りを削除
    else:
        # ENVごとにプロット (従来の動作)
        # バイオリンプロットの作成
        # x軸に'ENV'、y軸に'angle_difference_deg'を指定します。
        # inner="quartile"で、バイオリンプロット内に四分位範囲を点線で描画します。
        ax = sns.violinplot(x='ENV', y='angle_diff', data=df, palette='muted', order=env_types_for_plot, inner="quartile")

        # stripplotで個々のデータ点を黒色で重ねて描画します。
        sns.stripplot(x='ENV', y='angle_diff', data=df, color='black', jitter=0.2, size=2, alpha=0.3, order=env_types_for_plot, ax=ax)
        ax.set_title(plot_title + 'と生データのプロット', fontsize=16)
        ax.set_xlabel('ENV', fontsize=12)

    ax.set_ylabel('角度差 (度)', fontsize=12)

    # Y軸の範囲を設定
    if y_range:
        ax.set_ylim(y_range)
        print(f"Y軸の範囲を {y_range} に設定しました。")


    # グラフを保存 (bbox_inches='tight'でラベルが切れないように調整)
    plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
    print(f"バイオリンプロットを {output_image_path} に保存しました。")

    # グラフを表示
    plt.show()

if __name__ == '__main__':
    # --- 設定項目 ---
    # 入力CSVファイルのパス (calculate_angle_diff.pyの出力ファイル)
    INPUT_CSV = r'./result/20251119_pd_400epoch/angle_diffs.csv'  # 入力ファイル
    
    # グラフに表示したいENVを指定します (例: [1, 4])
    # Noneにすると、存在する全てのENVをプロットします。
    ENVS_TO_PLOT = [1,2,3,4]

    # Trueにすると、ENVS_TO_PLOTで指定したENVのデータを1つにまとめてプロットします。
    # Falseの場合は、ENVごとに分けてプロットします。
    COMBINE_ENVS = False

    # Y軸の範囲を指定します (例: (0, 100))
    # Noneにすると、自動で範囲が設定されます。
    Y_AXIS_RANGE = (-20, 200)
    
    # --- 処理開始 ---
    
    # 出力ファイル名の生成
    if ENVS_TO_PLOT:
        if COMBINE_ENVS:
            env_suffix = '_env_combined_' + '_'.join(map(str, ENVS_TO_PLOT))
        else:
            env_suffix = '_env_' + '_'.join(map(str, ENVS_TO_PLOT))
    else:
        env_suffix = '_env_all'
    
    # 全てのデータを使ってグラフを作成
    output_image_path = f'./result/20251119_pd_400epoch/violin_plot{env_suffix}_all.png'
    print("\n--- 全データのグラフを作成します ---")
    create_violin_plot_by_env(INPUT_CSV, output_image_path, filter_range=None, selected_envs=ENVS_TO_PLOT, combine_envs=COMBINE_ENVS, y_range=Y_AXIS_RANGE)
