import pandas as pd
import numpy as np
import os

def calculate_angle_diff(input_csv_path, output_csv_path):
    """
    CSVファイルからデータを読み込み、進行方向と放射方向の角度差を計算して、
    新しいCSVファイルに保存します。

    Args:
        input_csv_path (str): 入力CSVファイルのパス。
        output_csv_path (str): 出力CSVファイルのパス。
    """
    try:
        # CSVファイルを読み込みます。ヘッダーがないことを想定しています。
        df = pd.read_csv(input_csv_path, header=None)
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません -> {input_csv_path}")
        return

    results = []
    # DataFrameを1行ずつ処理します。
    # 2行目から最終行までループし、i-1行目とi行目の座標から進行方向を計算します。
    for i in range(1, len(df)):
        # 1, 2列目からX, Y座標を取得
        # 6列目(インデックス5)の値が1の行のみを処理対象とする
        if df.iloc[i, 5] != 1:
            continue

        x_prev, y_prev = df.iloc[i-1, 0], df.iloc[i-1, 1]
        x_curr, y_curr = df.iloc[i, 0], df.iloc[i, 1]

        # 8列目から放射方向の角度を取得
        emission_angle_deg = df.iloc[i, 7]

        # 進行方向の計算
        # 9列目(Env)と10列目(Bat)の値を取得
        env_val = df.iloc[i, 8]
        bat_val = df.iloc[i, 9]

        dx = x_curr - x_prev
        dy = y_curr - y_prev

        # dxとdyが両方0の場合は、移動がないため角度を計算せずにスキップ
        if dx == 0 and dy == 0:
            continue

        # 進行方向の角度を計算（-180°から180°の範囲）
        travel_angle_deg = np.rad2deg(np.arctan2(dy, dx))

        # 角度差を計算
        angle_diff = abs(travel_angle_deg - emission_angle_deg)

        # 角度差が180°より大きい場合は、360°から引いて小さい方の角度を求める
        if angle_diff > 180:
            angle_diff = 360 - angle_diff

        results.append({
            'step': i,
            'x': x_curr,
            'y': y_curr,
            'travel_direction_deg': travel_angle_deg,
            'emission_direction_deg': emission_angle_deg,
            'angle_difference_deg': angle_diff,
            'ENV': env_val,
            'BAT': bat_val
        })

    # 結果をDataFrameに変換
    if results:
        results_df = pd.DataFrame(results)
        # CSVファイルに保存
        results_df.to_csv(output_csv_path, index=False)
        print(f"角度差の計算結果を {output_csv_path} に保存しました。")
    else:
        print("計算対象のデータがありませんでした。")


if __name__ == '__main__':
    # --- 設定項目 ---
    # 入力CSVファイルのパス
    INPUT_CSV = r'./aoki/dataset_pd_yubi_add.csv'
    # 出力CSVファイルのパス
    OUTPUT_CSV = r'./aoki/angle_diffs_from_dataset.csv'
    
    # --- 処理開始 ---
    calculate_angle_diff(INPUT_CSV, OUTPUT_CSV)
